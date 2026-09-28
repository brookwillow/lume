"""Periodic usage statistics reports for Lume."""

import json
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_LOG_FILE = Path.home() / ".lume" / "eval_log.jsonl"
_STATE_FILE = Path.home() / ".lume" / "stats_report_state.json"


@dataclass
class StatsReporterSettings:
    enabled: bool = True
    interval_hours: float = 3.0
    recipient_email: str = ""
    report_dir: Path = Path.home() / ".lume" / "reports" / "stats"

    @classmethod
    def from_config(cls, config: Any) -> "StatsReporterSettings":
        return cls(
            enabled=bool(getattr(config, "enabled", True)),
            interval_hours=float(getattr(config, "interval_hours", 3.0)),
            recipient_email=str(getattr(config, "recipient_email", "")),
            report_dir=Path(str(getattr(config, "report_dir", "~/.lume/reports/stats"))).expanduser(),
        )


class StatsReporter:
    def __init__(
        self,
        settings: StatsReporterSettings,
        log_file: Path = _LOG_FILE,
        state_file: Path = _STATE_FILE,
    ):
        self.settings = settings
        self.log_file = log_file
        self.state_file = state_file
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.settings.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run_loop(self) -> None:
        interval_seconds = max(60.0, self.settings.interval_hours * 3600.0)
        while not self._stop.wait(interval_seconds):
            try:
                self.run_due_report()
            except Exception as e:
                print(f"Stats report failed: {e}")

    def run_due_report(self, now: datetime | None = None) -> Path | None:
        now = now or datetime.now()
        interval = timedelta(hours=max(0.01, self.settings.interval_hours))
        start = self._load_last_report_end()
        if start is None:
            start = now - interval
        if now - start < interval:
            return None

        report = self.build_report(start, now)
        if report is None:
            self._save_last_report_end(now)
            return None

        path = self._write_report(report, now)
        self._send_email(
            subject=f"Lume usage report {start:%Y-%m-%d %H:%M} - {now:%Y-%m-%d %H:%M}",
            body=report,
        )
        self._save_last_report_end(now)
        return path

    def build_report(self, start: datetime, end: datetime) -> str | None:
        all_rows = _derive_request_metrics(self._read_logs())
        rows = [
            row for row in all_rows
            if start <= _parse_timestamp(str(row.get("timestamp", ""))) < end
        ]
        if not rows:
            return None

        request_count = len(rows)
        loop_total = sum(_int(row.get("llm_turns")) for row in rows)
        token_total = sum(_request_tokens(row) for row in rows)
        total_ms = sum(_int(row.get("total_duration_ms")) for row in rows)
        asr_ms = sum(_int(row.get("asr_duration_ms")) for row in rows)
        llm_ms = sum(_int(row.get("llm_duration_ms")) for row in rows)
        tool_ms = sum(_tool_duration_ms(row) for row in rows)
        tts_ms = sum(_int(row.get("tts_duration_ms")) for row in rows)
        output_ms = sum(_int(row.get("output_duration_ms")) for row in rows)
        cache_rate = _cache_rate(rows)

        lines = [
            "# Lume Usage Report",
            "",
            f"- Window: {start:%Y-%m-%d %H:%M:%S} - {end:%Y-%m-%d %H:%M:%S}",
            f"- Requests: {request_count}",
            f"- Cache: {cache_rate}{_cache_total_suffix(rows)}",
            *_data_notes(rows),
            "",
            "## Average Metrics",
            "",
            "| Metric | Average/request | Total |",
            "| --- | ---: | ---: |",
            f"| Loops | {_fmt_float(loop_total / request_count)} | {loop_total} |",
            f"| End-to-end time | {_fmt_ms(total_ms / request_count)} | {_fmt_ms(total_ms)} |",
            f"| ASR time | {_fmt_ms(asr_ms / request_count)} | {_fmt_ms(asr_ms)} |",
            f"| LLM time | {_fmt_ms(llm_ms / request_count)} | {_fmt_ms(llm_ms)} |",
            f"| Tool time | {_fmt_ms(tool_ms / request_count)} | {_fmt_ms(tool_ms)} |",
            f"| TTS generation time | {_fmt_ms(tts_ms / request_count)} | {_fmt_ms(tts_ms)} |",
            f"| Output time | {_fmt_ms(output_ms / request_count)} | {_fmt_ms(output_ms)} |",
            f"| Tokens | {_fmt_float(token_total / request_count)} | {token_total} |",
            "",
            "## Request Details",
            "",
            "| Time | Request | Loops | Total | ASR | LLM | Tools | TTS gen | Output | Tokens | Cache hit/miss | Cache |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]

        for row in rows:
            lines.append(
                "| "
                + " | ".join([
                    _escape_pipe(str(row.get("timestamp", ""))),
                    _escape_pipe(_short_text(str(row.get("asr_text") or row.get("user_message") or ""), 48)),
                    str(_int(row.get("llm_turns"))),
                    _fmt_ms(_int(row.get("total_duration_ms"))),
                    _fmt_ms(_int(row.get("asr_duration_ms"))),
                    _fmt_ms(_int(row.get("llm_duration_ms"))),
                    _fmt_ms(_tool_duration_ms(row)),
                    _fmt_ms(_int(row.get("tts_duration_ms"))),
                    _fmt_ms(_int(row.get("output_duration_ms"))),
                    str(_request_tokens(row)),
                    _cache_hit_miss(row),
                    _row_cache_rate(row),
                ])
                + " |"
            )

        lines.extend([
            "",
            "## Daily Summary",
            "",
            "| Date | Requests | Avg loops | Avg total | Avg LLM | Avg tools | Avg TTS | Tokens | Cache |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])

        for date, day_rows in sorted(_group_by_day(rows).items()):
            day_total_ms = sum(_int(row.get("total_duration_ms")) for row in day_rows)
            day_llm_ms = sum(_int(row.get("llm_duration_ms")) for row in day_rows)
            day_tool_ms = sum(_tool_duration_ms(row) for row in day_rows)
            day_tts_ms = sum(_int(row.get("tts_duration_ms")) for row in day_rows)
            day_count = len(day_rows)
            lines.append(
                "| "
                + " | ".join([
                    date,
                    str(day_count),
                    _fmt_float(sum(_int(row.get("llm_turns")) for row in day_rows) / day_count),
                    _fmt_ms(day_total_ms / day_count),
                    _fmt_ms(day_llm_ms / day_count),
                    _fmt_ms(day_tool_ms / day_count),
                    _fmt_ms(day_tts_ms / day_count),
                    str(sum(_request_tokens(row) for row in day_rows)),
                    _cache_rate(day_rows),
                ])
                + " |"
            )

        return "\n".join(lines)

    def _read_logs(self) -> list[dict]:
        if not self.log_file.exists():
            return []
        rows = []
        for line in self.log_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                rows.append(data)
        return rows

    def _write_report(self, report: str, now: datetime) -> Path:
        self.settings.report_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.report_dir / f"{now:%Y%m%d-%H%M%S}.md"
        path.write_text(report, encoding="utf-8")
        return path

    def _send_email(self, subject: str, body: str) -> None:
        if not self.settings.recipient_email:
            return
        script = f"""
tell application "Mail"
    set newMessage to make new outgoing message with properties {{subject:{_as_applescript_string(subject)}, content:{_as_applescript_string(body)}, visible:false}}
    tell newMessage
        make new to recipient at end of to recipients with properties {{address:{_as_applescript_string(self.settings.recipient_email)}}}
        send
    end tell
end tell
"""
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"Stats report email failed: {result.stderr.strip()}")

    def _load_last_report_end(self) -> datetime | None:
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        value = data.get("last_report_end")
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _save_last_report_end(self, value: datetime) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps({"last_report_end": value.isoformat()}, ensure_ascii=False),
            encoding="utf-8",
        )


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.min


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _tool_duration_ms(row: dict) -> int:
    calls = row.get("tool_calls", [])
    if not isinstance(calls, list):
        return 0
    return sum(_int(call.get("duration_ms")) for call in calls if isinstance(call, dict))


def _derive_request_metrics(rows: list[dict]) -> list[dict]:
    sorted_rows = sorted(rows, key=lambda row: _parse_timestamp(str(row.get("timestamp", ""))))
    derived = []
    prev_cumulative_tokens = 0
    for row in sorted_rows:
        item = dict(row)
        total_tokens = _int(item.get("total_tokens"))
        has_token_breakdown = "cache_hit_tokens" in item or "cache_miss_tokens" in item
        if has_token_breakdown:
            item["_request_tokens"] = total_tokens
            item["_token_note"] = "recorded"
        elif prev_cumulative_tokens and total_tokens >= prev_cumulative_tokens:
            item["_request_tokens"] = total_tokens - prev_cumulative_tokens
            item["_token_note"] = "estimated_delta"
        else:
            item["_request_tokens"] = total_tokens
            item["_token_note"] = "legacy_logged"
        if total_tokens:
            prev_cumulative_tokens = total_tokens
        derived.append(item)
    return derived


def _request_tokens(row: dict) -> int:
    return _int(row.get("_request_tokens", row.get("total_tokens")))


def _cache_hit_miss(row: dict) -> str:
    if "cache_hit_tokens" not in row and "cache_miss_tokens" not in row:
        return "-"
    return f"{_int(row.get('cache_hit_tokens'))}/{_int(row.get('cache_miss_tokens'))}"


def _cache_total_suffix(rows: list[dict]) -> str:
    if not any("cache_hit_tokens" in row or "cache_miss_tokens" in row for row in rows):
        return " (from logged cache rates)"
    return f" ({sum(_int(row.get('cache_hit_tokens')) for row in rows)} hit / {sum(_int(row.get('cache_miss_tokens')) for row in rows)} miss)"


def _data_notes(rows: list[dict]) -> list[str]:
    notes = []
    if any(row.get("_token_note") == "estimated_delta" for row in rows):
        notes.append("- Note: some token values are estimated from legacy cumulative logs.")
    if any(row.get("_token_note") == "legacy_logged" for row in rows):
        notes.append("- Note: some old rows predate per-request token logging; first legacy row may be cumulative.")
    if any("tts_duration_ms" not in row for row in rows):
        notes.append("- Note: TTS/output timing is available only for new rows after this update.")
    if any(_int(row.get("llm_duration_ms")) == 0 for row in rows):
        notes.append("- Note: LLM timing may be missing for old rows logged before LLM duration was added.")
    return notes


def _cache_rate(rows: list[dict]) -> str:
    hit = sum(_int(row.get("cache_hit_tokens")) for row in rows)
    miss = sum(_int(row.get("cache_miss_tokens")) for row in rows)
    if hit + miss > 0:
        return f"{hit / (hit + miss) * 100:.0f}%"
    rates = [_parse_percent(str(row.get("cache_rate", ""))) for row in rows]
    rates = [rate for rate in rates if rate is not None]
    if not rates:
        return "-"
    return f"{sum(rates) / len(rates):.0f}%"


def _row_cache_rate(row: dict) -> str:
    return _cache_rate([row])


def _parse_percent(value: str) -> float | None:
    if not value.endswith("%"):
        return None
    try:
        return float(value[:-1])
    except ValueError:
        return None


def _group_by_day(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        ts = _parse_timestamp(str(row.get("timestamp", "")))
        key = ts.strftime("%Y-%m-%d") if ts != datetime.min else "unknown"
        grouped.setdefault(key, []).append(row)
    return grouped


def _fmt_ms(value: float | int) -> str:
    seconds = float(value) / 1000.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remain = seconds - minutes * 60
    return f"{minutes}m{remain:.0f}s"


def _fmt_float(value: float) -> str:
    return f"{value:.2f}"


def _short_text(value: str, limit: int) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def _escape_pipe(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _as_applescript_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
