import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from lume.stats_reporter import StatsReporter, StatsReporterSettings


class StatsReporterTests(unittest.TestCase):
    def test_build_report_summarizes_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "eval_log.jsonl"
            rows = [
                {
                    "timestamp": "2026-07-01 09:00:00",
                    "asr_text": "暂停音乐",
                    "llm_turns": 2,
                    "asr_duration_ms": 700,
                    "llm_duration_ms": 1200,
                    "tts_duration_ms": 400,
                    "output_duration_ms": 450,
                    "total_duration_ms": 2200,
                    "tool_calls": [{"tool": "run_command", "duration_ms": 300}],
                    "total_tokens": 1000,
                    "cache_hit_tokens": 800,
                    "cache_miss_tokens": 200,
                },
                {
                    "timestamp": "2026-07-01 10:00:00",
                    "asr_text": "下一首",
                    "llm_turns": 1,
                    "asr_duration_ms": 600,
                    "llm_duration_ms": 900,
                    "tts_duration_ms": 300,
                    "output_duration_ms": 340,
                    "total_duration_ms": 1800,
                    "tool_calls": [],
                    "total_tokens": 500,
                    "cache_hit_tokens": 250,
                    "cache_miss_tokens": 250,
                },
            ]
            log_file.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                encoding="utf-8",
            )
            reporter = StatsReporter(
                StatsReporterSettings(report_dir=Path(tmp) / "reports", recipient_email=""),
                log_file=log_file,
                state_file=Path(tmp) / "state.json",
            )

            report = reporter.build_report(
                datetime(2026, 7, 1, 8, 0, 0),
                datetime(2026, 7, 1, 11, 0, 0),
            )

        self.assertIsNotNone(report)
        self.assertIn("- Requests: 2", report)
        self.assertIn("- Cache: 70%", report)
        self.assertIn("## Average Metrics", report)
        self.assertIn("| Loops | 1.50 | 3 |", report)
        self.assertIn("| End-to-end time | 2.0s | 4.0s |", report)
        self.assertIn("| ASR time | 0.7s | 1.3s |", report)
        self.assertIn("| LLM time | 1.1s | 2.1s |", report)
        self.assertIn("| Tool time | 0.1s | 0.3s |", report)
        self.assertIn("| TTS generation time | 0.3s | 0.7s |", report)
        self.assertIn("| Output time | 0.4s | 0.8s |", report)
        self.assertIn("| Tokens | 750.00 | 1500 |", report)
        self.assertIn("TTS gen", report)
        self.assertIn("Cache hit/miss", report)
        self.assertIn("| 2026-07-01 | 2 | 1.50 |", report)
        self.assertIn("暂停音乐", report)

    def test_run_due_report_writes_report_and_sends_email(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "eval_log.jsonl"
            log_file.write_text(
                json.dumps({
                    "timestamp": "2026-07-01 09:00:00",
                    "asr_text": "测试",
                    "llm_turns": 1,
                    "total_duration_ms": 1000,
                    "total_tokens": 10,
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            reporter = StatsReporter(
                StatsReporterSettings(
                    interval_hours=3,
                    recipient_email="user@example.com",
                    report_dir=Path(tmp) / "reports",
                ),
                log_file=log_file,
                state_file=Path(tmp) / "state.json",
            )

            with patch("lume.stats_reporter.subprocess.run") as run:
                run.return_value.returncode = 0
                path = reporter.run_due_report(datetime(2026, 7, 1, 12, 0, 0))

            self.assertTrue(path.exists())
            self.assertIn("Lume Usage Report", path.read_text(encoding="utf-8"))
            self.assertTrue((Path(tmp) / "state.json").exists())
            self.assertTrue(run.called)

    def test_run_due_report_skips_before_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            state_file.write_text(
                json.dumps({"last_report_end": "2026-07-01T10:00:00"}),
                encoding="utf-8",
            )
            reporter = StatsReporter(
                StatsReporterSettings(interval_hours=3, report_dir=Path(tmp), recipient_email=""),
                log_file=Path(tmp) / "missing.jsonl",
                state_file=state_file,
            )

            path = reporter.run_due_report(datetime(2026, 7, 1, 11, 0, 0))

        self.assertIsNone(path)

    def test_legacy_cumulative_tokens_are_reported_as_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "eval_log.jsonl"
            rows = [
                {"timestamp": "2026-07-01 09:00:00", "asr_text": "一", "llm_turns": 1, "total_tokens": 1000, "cache_rate": "80%"},
                {"timestamp": "2026-07-01 09:01:00", "asr_text": "二", "llm_turns": 1, "total_tokens": 1300, "cache_rate": "90%"},
            ]
            log_file.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                encoding="utf-8",
            )
            reporter = StatsReporter(
                StatsReporterSettings(report_dir=Path(tmp) / "reports", recipient_email=""),
                log_file=log_file,
                state_file=Path(tmp) / "state.json",
            )

            report = reporter.build_report(
                datetime(2026, 7, 1, 8, 0, 0),
                datetime(2026, 7, 1, 10, 0, 0),
            )

        self.assertIn("| Tokens | 650.00 | 1300 |", report)
        self.assertIn("| 2026-07-01 09:01:00 | 二 | 1 | 0.0s | 0.0s | 0.0s | 0.0s | 0.0s | 0.0s | 300 | - | 90% |", report)
        self.assertIn("estimated from legacy cumulative logs", report)


if __name__ == "__main__":
    unittest.main()
