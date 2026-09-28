"""Output: TTS via edge-tts + notification via osascript."""

import subprocess
import tempfile
import os
import time
import asyncio
from dataclasses import dataclass

import edge_tts

# Kill any currently playing TTS audio
_current_tts_proc: subprocess.Popen | None = None


@dataclass
class TTSResult:
    generation_ms: int = 0
    playback_start_ms: int = 0
    total_ms: int = 0
    process: subprocess.Popen | None = None


def stop_speaking():
    """Interrupt any currently playing TTS audio."""
    global _current_tts_proc
    if _current_tts_proc and _current_tts_proc.poll() is None:
        _current_tts_proc.terminate()
        _current_tts_proc = None


async def speak(text: str, voice: str = "zh-CN-XiaoxiaoNeural") -> TTSResult:
    """Speak text using Edge TTS (Microsoft neural voice)."""
    global _current_tts_proc
    start = time.perf_counter()

    # Stop previous TTS if still playing
    if _current_tts_proc and _current_tts_proc.poll() is None:
        _current_tts_proc.terminate()

    # Generate audio to temp file
    tmp = os.path.join(tempfile.gettempdir(), "lume_tts.mp3")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(tmp)
    generated = time.perf_counter()

    # Play with afplay (non-blocking)
    _current_tts_proc = subprocess.Popen(
        ["afplay", tmp],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    playback_started = time.perf_counter()
    return TTSResult(
        generation_ms=int((generated - start) * 1000),
        playback_start_ms=int((playback_started - generated) * 1000),
        total_ms=int((playback_started - start) * 1000),
        process=_current_tts_proc,
    )


async def wait_for_playback(result: TTSResult) -> int:
    """Wait for this specific TTS playback process without blocking the event loop."""
    global _current_tts_proc
    process = result.process
    if process is None:
        return 0
    start = time.perf_counter()
    await asyncio.to_thread(process.wait)
    if _current_tts_proc is process:
        _current_tts_proc = None
    return int((time.perf_counter() - start) * 1000)


def notify(title: str, message: str):
    """Show macOS notification."""
    # Escape for AppleScript
    title_escaped = title.replace('"', '\\"')
    msg_escaped = message.replace('"', '\\"')[:200]
    script = f'display notification "{msg_escaped}" with title "{title_escaped}"'
    subprocess.Popen(
        ["osascript", "-e", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def play_sound(sound: str = "Ping"):
    """Play a system sound (Ping, Pop, Blow, etc.)."""
    path = f"/System/Library/Sounds/{sound}.aiff"
    subprocess.Popen(
        ["afplay", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
