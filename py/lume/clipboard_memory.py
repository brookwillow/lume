"""Clipboard watcher for long-term memory extraction."""

import hashlib
import subprocess
import threading
import time

from .memory import MemoryEngine


class ClipboardMemoryWatcher:
    """Poll clipboard changes and pass candidate text to MemoryEngine."""

    def __init__(self, memory_engine: MemoryEngine):
        self.memory_engine = memory_engine
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_hash = ""

    def start(self) -> None:
        settings = self.memory_engine.settings
        if not settings.enabled or not settings.clipboard_enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        settings = self.memory_engine.settings
        interval = max(0.5, float(settings.clipboard_poll_interval_seconds))
        while not self._stop.is_set():
            text = _read_clipboard()
            if text:
                text = text[: settings.clipboard_max_chars]
                digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
                if digest != self._last_hash:
                    self._last_hash = digest
                    self.memory_engine.observe_clipboard_text(text)
            self._stop.wait(interval)


def _read_clipboard() -> str:
    try:
        result = subprocess.run(
            ["pbpaste"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()
