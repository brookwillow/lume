"""Lume - macOS voice assistant with floating HUD overlay.

Hold Right Option key to speak, release to send.
Overlay appears on any screen showing ASR + agent results.
"""

import asyncio
import json
import os
import signal
import sys
import threading
import time

from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
from PyObjCTools import AppHelper
from pynput import keyboard

from .agent import Agent
from .voice import start_recording, stop_recording
from .output import speak, play_sound, stop_speaking, wait_for_playback
from .overlay import LumeOverlay
from .eval_logger import EvalLogger
from .config import load_config
from .clipboard_memory import ClipboardMemoryWatcher
from .memory import MemoryEngine, MemorySettings
from .skill_evolution import SkillEvolutionConfig, SkillEvolutionEngine
from .tools import execute_tool
from .stats_reporter import StatsReporter, StatsReporterSettings
from .asr_lexicon import ASRLexiconEngine, ASRLexiconSettings


# ─── Globals ───
agent = Agent()
eval_log = EvalLogger()
config = load_config()
skill_evolution = SkillEvolutionEngine(
    SkillEvolutionConfig.from_settings(config.skill_evolution)
)
memory_engine = MemoryEngine(MemorySettings.from_config(config.memory))
clipboard_watcher = ClipboardMemoryWatcher(memory_engine)
stats_reporter = StatsReporter(StatsReporterSettings.from_config(config.stats_report))
asr_lexicon = ASRLexiconEngine(ASRLexiconSettings.from_config(config.asr))
asr_lexicon.refresh_local_entities_background(reason="startup")
loop: asyncio.AbstractEventLoop = None
overlay: LumeOverlay = None
_processing = False
_recording_active = False
_active_mode = "agent"
_interaction_generation = 0
_state_lock = threading.Lock()
_current_task = None
_VOICE_MODE_KEY = keyboard.Key.alt_r
_INPUT_MODE_KEY = getattr(keyboard.Key, "cmd_r", None)

def _set_recording_active(val: bool):
    global _recording_active
    _recording_active = val


def _next_generation() -> int:
    global _interaction_generation
    with _state_lock:
        _interaction_generation += 1
        return _interaction_generation


def _current_generation() -> int:
    with _state_lock:
        return _interaction_generation


def _is_current_generation(generation: int) -> bool:
    return generation == _current_generation()


def _interrupt_current(reason: str = "interrupted"):
    global _current_task
    stop_speaking()
    task = _current_task
    if task is not None and not task.done():
        task.cancel()
    print(f"↯ Interrupted previous interaction: {reason}")


def _start_voice_capture(mode: str):
    global _active_mode
    if _recording_active:
        return
    if _processing:
        _interrupt_current("new recording")
    _next_generation()
    _active_mode = mode
    _set_recording_active(True)
    stop_speaking()
    AppHelper.callAfter(overlay.show_listening)
    try:
        start_recording(on_partial=_on_partial_asr, on_level=_on_audio_level)
    except Exception as exc:
        # This callback runs inside pynput's listener thread. Never let an
        # unavailable/stale CoreAudio device terminate the hotkey listener.
        _set_recording_active(False)
        print(f"❌ Microphone unavailable: {exc}")
        AppHelper.callAfter(overlay.show_error, "Microphone unavailable")
        return
    print("🎤 Recording..." if mode == "agent" else "⌨️  Smart input recording...")


# ─── Hotkey Handlers (called from pynput thread) ───

def on_press(key):
    if key == _VOICE_MODE_KEY:
        _start_voice_capture("agent")
    elif _INPUT_MODE_KEY is not None and key == _INPUT_MODE_KEY:
        _start_voice_capture("input")


def on_release(key):
    global _current_task
    if key == _VOICE_MODE_KEY or (
        _INPUT_MODE_KEY is not None and key == _INPUT_MODE_KEY
    ):
        if not _recording_active:
            return
        _set_recording_active(False)
        AppHelper.callAfter(overlay.show_processing)
        print("⏹  Processing...")
        generation = _current_generation()
        _current_task = asyncio.run_coroutine_threadsafe(
            _process_voice(_active_mode, generation),
            loop,
        )


# ─── Streaming ASR Callback (called from partial ASR thread) ───

def _on_audio_level(level: float):
    """Feed real-time audio level to overlay waveform."""
    AppHelper.callAfter(overlay.update_audio_level, level)


def _on_partial_asr(text: str):
    """Update overlay with partial ASR result during recording."""
    AppHelper.callAfter(overlay.show_transcription, text)


# ─── Async Processing (runs on asyncio thread) ───

async def _process_voice(mode: str = "agent", generation: int | None = None):
    global _processing, _current_task
    if generation is None:
        generation = _current_generation()
    if not _is_current_generation(generation):
        return
    _processing = True
    eval_log.start()
    try:
        t0 = time.perf_counter()

        # stop_recording() runs ASR and returns structured result
        asr = stop_recording(scene=mode)
        t1 = time.perf_counter()
        if not _is_current_generation(generation):
            print("── Interrupted ── stale ASR result ignored")
            return

        if not asr.text.strip():
            print(f"── ASR ── (no speech) [{t1-t0:.2f}s]")
            AppHelper.callAfter(overlay.show_error, "No speech detected")
            interaction = eval_log.abort("no speech")
            if interaction:
                memory_engine.observe_completed_interaction(interaction)
                skill_evolution.observe_completed_interaction(interaction)
            return

        print(f"── ASR ── [{t1-t0:.2f}s]")
        print(f'   "{asr.text}" [lang={asr.language} emotion={asr.emotion} event={asr.event}]')
        _print_hotword_status(asr)
        print()

        # Show final transcription on overlay
        AppHelper.callAfter(overlay.show_transcription, asr.text)
        # Give the recognized text one rendered frame before the agent replaces
        # it with its execution state. This is non-blocking and cancellable.
        await asyncio.sleep(0.35)
        if not _is_current_generation(generation):
            print("── Interrupted ── stale transcription ignored")
            return

        # Build message for LLM with ASR context
        meta_parts = []
        if asr.language:
            meta_parts.append(f"lang={asr.language}")
        if asr.emotion and asr.emotion != "NEUTRAL":
            meta_parts.append(f"emotion={asr.emotion}")
        if asr.event and asr.event != "Speech":
            meta_parts.append(f"event={asr.event}")

        if meta_parts:
            llm_input = f"[{', '.join(meta_parts)}] {asr.text}"
        else:
            llm_input = asr.text

        # Log ASR result
        eval_log.set_asr(
            text=asr.text, language=asr.language,
            emotion=asr.emotion, duration_ms=int((t1 - t0) * 1000),
            raw_text=asr.raw,
        )
        eval_log.set_user_message(llm_input)

        # LLM
        _reset_execution_display()
        if mode == "input":
            result = await agent.compose_input_text(llm_input, on_think=_on_think)
        else:
            result = await agent.run(llm_input, on_think=_on_think)
        t2 = time.perf_counter()
        if not _is_current_generation(generation):
            print("── Interrupted ── stale LLM result ignored")
            return

        print(f"── Response ──")
        print(f"   {result}")
        print()
        print(f"── Stats ── asr:{t1-t0:.2f}s  llm:{t2-t1:.2f}s  total:{t2-t0:.2f}s")
        s = agent.stats
        print(f"   model={s.model}  tokens={s.total_tokens}  ctx={s.context_tokens}  cache={s.cache_rate}")
        print()

        # Log and show result
        if not _is_current_generation(generation):
            print("── Interrupted ── stale response ignored")
            return
        eval_log.set_llm_turns(getattr(agent, "_last_turn_count", 0))
        eval_log.set_llm_duration(int((t2 - t1) * 1000))
        if mode == "input":
            output_start = time.perf_counter()
            insert_result = await asyncio.to_thread(execute_tool, "insert_text", {"text": result})
            output_ms = int((time.perf_counter() - output_start) * 1000)
            if not _is_current_generation(generation):
                print("── Interrupted ── stale insert result ignored")
                return
            print(f"── Insert ── {insert_result}")
            eval_log.set_output_duration(duration_ms=output_ms)
            AppHelper.callAfter(overlay.show_result, "已输入")
        else:
            # The result stays visible for the whole TTS playback. A separate
            # watcher avoids blocking the next interaction or cancellation.
            AppHelper.callAfter(overlay.show_result, result, None)
            tts = await speak(result)
            eval_log.set_output_duration(duration_ms=tts.total_ms, tts_duration_ms=tts.generation_ms)
            asyncio.create_task(_hide_result_after_tts(tts, generation))

        interaction = eval_log.finish(
            response=result,
            total_tokens=agent.stats.total_tokens,
            cache_rate=agent.stats.cache_rate,
            cache_hit_tokens=agent.stats.cache_hit,
            cache_miss_tokens=agent.stats.cache_miss,
        )
        if interaction:
            memory_engine.observe_completed_interaction(interaction)
            skill_evolution.observe_completed_interaction(interaction)
            asr_lexicon.observe_completed_interaction(interaction)

    except asyncio.CancelledError:
        print("── Interrupted ── current interaction cancelled")
        interaction = eval_log.abort("interrupted")
        if interaction and _is_current_generation(generation):
            memory_engine.observe_completed_interaction(interaction)
            skill_evolution.observe_completed_interaction(interaction)
        return

    except Exception as e:
        error_msg = f"Error: {e}"
        print(f"❌ {error_msg}")
        AppHelper.callAfter(overlay.show_error, str(e)[:80])
        interaction = eval_log.abort(str(e))
        if interaction:
            memory_engine.observe_completed_interaction(interaction)
            skill_evolution.observe_completed_interaction(interaction)

    finally:
        if _is_current_generation(generation):
            _processing = False
            _current_task = None


async def _hide_result_after_tts(tts, generation: int):
    """Hide only the result belonging to the TTS playback that just ended."""
    await wait_for_playback(tts)
    await asyncio.sleep(0.8)
    if _is_current_generation(generation):
        AppHelper.callAfter(overlay.hide)


def _on_think(event_type: str, data: dict):
    if event_type == "thinking":
        history = "\n".join(getattr(_on_think, "_completed_steps", [])[-2:])
        AppHelper.callAfter(overlay.show_thinking, history)
    elif event_type == "tool_call":
        tool_name = data['tool']
        print(f"── Tool Call ──")
        print(f"   {tool_name}({data['params']})")
        action = _tool_action_label(tool_name, data.get("params", {}))
        history = "\n".join(getattr(_on_think, "_completed_steps", [])[-2:])
        AppHelper.callAfter(overlay.show_action, action, history)
        _on_think._last_action = action
        _on_think._last_params = data.get("params", {})
        eval_log.tool_start()
    elif event_type == "tool_result":
        result_str = str(data["result"])
        print(f"── Tool Result ({len(result_str)} chars) ──")
        print(f"   {result_str[:300]}")
        if len(result_str) > 300:
            print("   ... (truncated)")
        print()
        params = getattr(_on_think, "_last_params", {})
        eval_log.tool_end(data["tool"], params, result_str)
        action = getattr(
            _on_think, "_last_action", _tool_action_label(data["tool"], params)
        )
        status = _tool_result_status(result_str)
        if status == "success":
            completed = "✓ " + action.replace("正在", "已")
        elif status == "failed":
            completed = "× " + action.replace("正在", "未能")
        else:
            completed = "! " + action.replace("正在", "已尝试")
        _on_think._completed_steps = [
            *getattr(_on_think, "_completed_steps", []), completed
        ][-2:]


def _reset_execution_display():
    """Reset the one-line execution history for a fresh user interaction."""
    _on_think._last_action = ""
    _on_think._completed_steps = []
    _on_think._last_params = {}


def _print_hotword_status(asr):
    """Print observable hotword state without claiming model attribution."""
    requested = getattr(asr, "hotwords_requested", [])
    if not requested:
        print("   🔥 Hotwords: none recalled")
        return
    preview = ", ".join(
        f"{item['text']}({item['weight']:.2f},{item.get('source', 'unknown')})"
        for item in requested[:8]
    )
    suffix = " …" if len(requested) > 8 else ""
    status = "accepted" if getattr(asr, "hotword_param_accepted", False) else "fallback"
    print(f"   🔥 Hotwords: requested={len(requested)} backend={status} [{preview}{suffix}]")
    matches = getattr(asr, "hotword_matches", [])
    if matches:
        source_by_term = {str(item.get("text", "")): item.get("source", "unknown") for item in requested}
        labels = []
        for item in matches:
            if item.get("kind") == "normalization_applied":
                canonical = str(item["canonical"])
                labels.append(f"{item['variant']}→{canonical}[{source_by_term.get(canonical, 'unknown')}]")
            else:
                canonical = str(item["canonical"])
                labels.append(f"{canonical}[{source_by_term.get(canonical, 'unknown')}]")
        print(f"   🎯 Hotword hits: {', '.join(labels)}")
    else:
        print("   🎯 Hotword hits: none")
    print("   ℹ️  backend=accepted means the hotword parameter was passed to ASR; model-level attribution is unavailable.")


def _tool_result_status(result: str) -> str:
    """Use standardized status, while retaining clear legacy tool outcomes."""
    try:
        value = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        lowered = str(result).strip().lower()
        if lowered.startswith(("error:", "failed", "file not found", "unknown tool")):
            return "failed"
        return "success"
    status = value.get("status") if isinstance(value, dict) else None
    return status if status in {"success", "failed", "unknown"} else "unknown"


def _tool_action_label(tool: str, params: dict) -> str:
    """Translate internal tool calls into short, user-facing execution states."""
    if tool == "music_control":
        action = params.get("action", "")
        return {
            "play": "正在继续播放",
            "pause": "正在暂停播放",
            "next": "正在切换下一首",
            "prev": "正在切换上一首",
        }.get(action, "正在控制播放")
    if tool == "clipboard":
        return "正在读取剪贴板" if params.get("action", "get") == "get" else "正在更新剪贴板"
    if tool == "read_notes":
        return "正在读取备忘录"
    if tool == "chrome":
        return "正在操作浏览器"
    if tool == "browser":
        return "正在操作网页"
    if tool == "lark":
        return "正在处理飞书内容"
    if tool == "notion":
        return "正在处理 Notion 内容"
    if tool == "load_skill":
        return "正在读取已学经验"
    if tool == "read_memory":
        return "正在读取长期记忆"
    if tool == "remember_memory":
        return "正在保存长期记忆"
    if tool in {"read_file", "list_dir", "search_files", "spotlight_search"}:
        return "正在查找相关内容"
    if tool in {"write_file", "replace_selected_text", "replace_before_cursor"}:
        return "正在更新内容"
    if tool == "insert_text":
        return "正在输入文字"
    if tool == "open_app":
        return "正在打开应用"
    if tool == "open_url":
        return "正在打开网页"
    if tool == "applescript":
        return "正在操作 macOS"
    if tool == "run_command":
        return "正在执行系统操作"
    return "正在执行操作"


# ─── Entry Point ───

_should_quit = False


def _shutdown(signum=None, frame=None):
    """Clean shutdown handler for SIGINT/SIGTERM."""
    global _should_quit
    _should_quit = True


def _check_quit(timer):
    """Periodic check - gives Python a chance to process signals."""
    if _should_quit:
        print("\n👋 Bye!")
        AppHelper.stopEventLoop()


def main():
    global loop, overlay

    print("🔥 Lume Voice Assistant")
    print("Hold Right Option (⌥) key to speak, release to send")
    print("Hold Right Command (⌘) to dictate text into the current cursor")
    print("Press Ctrl+C to quit\n")

    # Register signal handlers BEFORE AppKit takes over
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Initialize NSApplication (no dock icon)
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    # Create overlay (must be on main thread)
    overlay = LumeOverlay()

    # Start asyncio event loop in background thread
    loop = asyncio.new_event_loop()
    asyncio_thread = threading.Thread(target=loop.run_forever, daemon=True)
    asyncio_thread.start()

    # Preload ASR model in background
    from .voice import preload_model
    threading.Thread(target=preload_model, daemon=True).start()

    # Start keyboard listener (runs in its own thread)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    clipboard_watcher.start()
    stats_reporter.start()

    # Periodic timer to let Python process signals (every 0.3s)
    from Foundation import NSTimer, NSRunLoop, NSDefaultRunLoopMode
    timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(0.3, True, _check_quit)
    NSRunLoop.currentRunLoop().addTimer_forMode_(timer, NSDefaultRunLoopMode)

    # Run AppKit event loop on main thread
    try:
        AppHelper.runEventLoop(installInterrupt=False)
    finally:
        clipboard_watcher.stop()
        stats_reporter.stop()
        listener.stop()
        loop.call_soon_threadsafe(loop.stop)
        os._exit(0)


if __name__ == "__main__":
    main()
