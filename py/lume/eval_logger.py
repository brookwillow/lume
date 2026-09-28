"""Interaction logger for Lume eval.

Records every voice interaction as a structured JSONL entry to ~/.lume/eval_log.jsonl.
Each entry captures the full pipeline: ASR → tool calls → final response → timing.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

_LOG_DIR = Path.home() / ".lume"
_LOG_FILE = _LOG_DIR / "eval_log.jsonl"


@dataclass
class ToolTrace:
    tool: str = ""
    params: dict = field(default_factory=dict)
    result: str = ""
    duration_ms: int = 0


@dataclass
class Interaction:
    # Timestamps
    timestamp: str = ""
    # ASR
    asr_text: str = ""
    asr_raw_text: str = ""
    asr_language: str = ""
    asr_emotion: str = ""
    asr_duration_ms: int = 0
    # Agent
    user_message: str = ""  # what was sent to LLM (may include metadata)
    tool_calls: list = field(default_factory=list)  # list of ToolTrace dicts
    llm_turns: int = 0
    final_response: str = ""
    # Timing
    llm_duration_ms: int = 0
    tts_duration_ms: int = 0
    output_duration_ms: int = 0
    total_duration_ms: int = 0
    # Token stats
    total_tokens: int = 0
    cache_rate: str = ""
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    # Outcome (for later annotation)
    success: bool | None = None  # None = not yet annotated
    notes: str = ""


class EvalLogger:
    """Records interactions for offline evaluation."""

    def __init__(self):
        self._current: Interaction | None = None
        self._start_time: float = 0
        self._tool_start: float = 0

    def start(self):
        """Begin recording a new interaction."""
        self._current = Interaction()
        self._start_time = time.time()
        self._current.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def set_asr(self, text: str, language: str, emotion: str, duration_ms: int, raw_text: str = ""):
        """Record ASR result."""
        if not self._current:
            return
        self._current.asr_text = text
        self._current.asr_raw_text = raw_text
        self._current.asr_language = language
        self._current.asr_emotion = emotion
        self._current.asr_duration_ms = duration_ms

    def set_user_message(self, message: str):
        """Record the full message sent to LLM."""
        if not self._current:
            return
        self._current.user_message = message

    def tool_start(self):
        """Mark the start of a tool call."""
        self._tool_start = time.time()

    def tool_end(self, tool: str, params: dict, result: str):
        """Record a completed tool call."""
        if not self._current:
            return
        duration = int((time.time() - self._tool_start) * 1000)
        # Truncate long results for log
        result_short = result[:1000] if len(result) > 1000 else result
        self._current.tool_calls.append(asdict(ToolTrace(
            tool=tool,
            params=params,
            result=result_short,
            duration_ms=duration,
        )))

    def set_llm_turns(self, turns: int):
        if self._current:
            self._current.llm_turns = turns

    def set_llm_duration(self, duration_ms: int):
        if self._current:
            self._current.llm_duration_ms = duration_ms

    def set_output_duration(self, duration_ms: int = 0, tts_duration_ms: int = 0):
        if self._current:
            self._current.output_duration_ms = duration_ms
            self._current.tts_duration_ms = tts_duration_ms

    def finish(
        self,
        response: str,
        total_tokens: int = 0,
        cache_rate: str = "",
        cache_hit_tokens: int = 0,
        cache_miss_tokens: int = 0,
    ) -> dict | None:
        """Finalize and write the interaction log."""
        if not self._current:
            return None
        self._current.final_response = response
        self._current.total_tokens = total_tokens
        self._current.cache_rate = cache_rate
        self._current.cache_hit_tokens = cache_hit_tokens
        self._current.cache_miss_tokens = cache_miss_tokens
        self._current.total_duration_ms = int((time.time() - self._start_time) * 1000)
        data = asdict(self._current)

        # Write to JSONL
        _LOG_DIR.mkdir(exist_ok=True)
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        self._current = None
        return data

    def abort(self, error: str = "") -> dict | None:
        """Record a failed interaction."""
        if not self._current:
            return None
        self._current.success = False
        self._current.notes = f"error: {error}"
        return self.finish(response="", total_tokens=0)
