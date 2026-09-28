import json
import tempfile
import unittest
from pathlib import Path

from lume.memory import MemoryEngine, MemoryExtraction, MemoryFact, MemorySettings


class MemoryTests(unittest.TestCase):
    def test_simple_task_is_not_memory_candidate(self):
        interaction = {
            "user_message": "打开 Safari",
            "tool_calls": [{"tool": "open_app", "params": {"name": "Safari"}, "result": "Opened Safari"}],
            "final_response": "已打开 Safari",
        }

        self.assertFalse(MemoryEngine.is_candidate(interaction))

    def test_high_confidence_preference_merges_into_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = MemorySettings(
                enabled=True,
                memory_dir=Path(tmp),
                min_confidence=0.9,
                sensitive_min_confidence=0.97,
            )
            engine = MemoryEngine(settings)
            extraction = MemoryExtraction(
                should_write=True,
                rationale="The user explicitly stated their preferred shell.",
                facts=[
                    MemoryFact(
                        key="preferred_shell",
                        value="zsh",
                        category="preference",
                        sensitivity="preference",
                        confidence=0.95,
                    )
                ],
            )

            engine.apply_extraction(extraction, source_id="test-run")

            profile = json.loads((Path(tmp) / "profile.json").read_text())
            self.assertEqual(profile["preferred_shell"]["value"], "zsh")
            self.assertIn("Preferred shell: zsh", engine.load_persistent_prompt())

    def test_sensitive_fact_requires_higher_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = MemorySettings(
                enabled=True,
                memory_dir=Path(tmp),
                min_confidence=0.9,
                sensitive_min_confidence=0.97,
            )
            engine = MemoryEngine(settings)
            extraction = MemoryExtraction(
                should_write=True,
                rationale="Email mentioned, but confidence is too low for sensitive data.",
                facts=[
                    MemoryFact(
                        key="email",
                        value="user@example.com",
                        category="identity",
                        sensitivity="sensitive",
                        confidence=0.94,
                    )
                ],
            )

            engine.apply_extraction(extraction, source_id="test-run")

            self.assertFalse((Path(tmp) / "profile.json").exists())

    def test_clipboard_text_builds_memory_candidate_interaction(self):
        interaction = MemoryEngine.clipboard_interaction("我的邮箱是 user@example.com")

        self.assertEqual(interaction["source"], "clipboard")
        self.assertIn("我的邮箱", interaction["user_message"])
        self.assertTrue(MemoryEngine.is_candidate(interaction))

    def test_plain_clipboard_text_is_not_memory_candidate(self):
        interaction = MemoryEngine.clipboard_interaction("临时复制的一段普通文字")

        self.assertFalse(MemoryEngine.is_candidate(interaction))

    def test_explicit_remember_request_is_memory_candidate(self):
        interaction = {
            "user_message": "记一下我下周要给产品团队同步语音助手进展",
            "tool_calls": [],
            "final_response": "好的。",
        }

        self.assertTrue(MemoryEngine.is_candidate(interaction))

    def test_explicit_remember_note_merges_into_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = MemorySettings(enabled=True, memory_dir=Path(tmp))
            engine = MemoryEngine(settings)
            extraction = MemoryExtraction(
                should_write=True,
                rationale="The user explicitly asked to remember this note.",
                facts=[
                    MemoryFact(
                        key="remembered_note_product_sync",
                        value="下周要给产品团队同步语音助手进展",
                        category="note",
                        sensitivity="profile",
                        confidence=0.95,
                    )
                ],
            )

            engine.apply_extraction(extraction, source_id="test-run")

            prompt = engine.load_persistent_prompt()
            self.assertIn("Remembered note product sync", prompt)
            self.assertIn("下周要给产品团队同步语音助手进展", prompt)

    def test_clipboard_commands_are_counted_for_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = MemorySettings(enabled=True, memory_dir=Path(tmp))
            engine = MemoryEngine(settings)

            engine.record_clipboard_patterns("git status\n临时普通文本")
            engine.record_clipboard_patterns("git status")

            prompt = engine.load_persistent_prompt()

            self.assertIn("Frequent clipboard commands/instructions", prompt)
            self.assertIn("git status", prompt)
            self.assertIn("count=2", prompt)
            self.assertNotIn("临时普通文本", prompt)

    def test_clipboard_patterns_skip_sensitive_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = MemorySettings(enabled=True, memory_dir=Path(tmp))
            engine = MemoryEngine(settings)

            engine.record_clipboard_patterns("export API_KEY=sk-secret")

            self.assertNotIn("API_KEY", engine.load_persistent_prompt())


if __name__ == "__main__":
    unittest.main()
