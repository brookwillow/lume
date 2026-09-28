import json
import tempfile
import unittest
from pathlib import Path

from lume.skill_evolution import (
    SkillEvolutionConfig,
    SkillEvolutionEngine,
    SkillReview,
)


class SkillEvolutionTests(unittest.TestCase):
    def test_simple_tool_only_interaction_is_not_candidate(self):
        interaction = {
            "user_message": "打开 Safari",
            "tool_calls": [{"tool": "open_app", "params": {"name": "Safari"}, "result": "Opened Safari"}],
            "final_response": "已打开 Safari",
            "success": None,
        }

        self.assertFalse(SkillEvolutionEngine.is_candidate(interaction))

    def test_explicit_save_skill_request_is_candidate(self):
        interaction = {
            "user_message": "把它打开关闭沉淀进去吧",
            "tool_calls": [{"tool": "run_command", "params": {"command": "open app"}, "result": "ok"}],
            "final_response": "已关闭网易云音乐。",
            "success": None,
        }

        self.assertTrue(SkillEvolutionEngine.is_candidate(interaction))

    def test_multi_step_tool_exploration_is_candidate_without_explicit_save(self):
        interaction = {
            "user_message": "再打开这个音乐播放器吧",
            "tool_calls": [
                {"tool": "run_command", "params": {"command": "open -a app"}, "result": "(exit code 1)"},
                {"tool": "run_command", "params": {"command": "mdfind -name 网易"}, "result": "/Applications/NeteaseMusic.app"},
                {"tool": "run_command", "params": {"command": "open app path"}, "result": "(no output)"},
            ],
            "final_response": "已打开网易云音乐。",
            "success": None,
        }

        self.assertTrue(SkillEvolutionEngine.is_candidate(interaction))

    def test_success_follow_up_can_trigger_learning_for_previous_tool_use(self):
        target = {
            "user_message": "试试下一首呢",
            "tool_calls": [{"tool": "run_command", "params": {"command": "key code 124"}, "result": "(no output)"}],
            "final_response": "已尝试用方向键切换到下一首。",
            "success": None,
        }
        follow_up = {
            "user_message": "这个可以了",
            "tool_calls": [],
            "final_response": "好的，已成功切换到下一首。",
            "success": None,
        }

        self.assertTrue(SkillEvolutionEngine.should_curate_with_follow_up(target, follow_up))

    def test_high_confidence_review_writes_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = SkillEvolutionConfig(
                enabled=True,
                min_confidence=0.92,
                skill_dir=root / "skills",
                run_dir=root / "runs",
            )
            engine = SkillEvolutionEngine(config=config)
            interaction = {
                "user_message": "修复 AppKit overlay 长文本显示",
                "tool_calls": [{"tool": "write_file", "params": {}, "result": "ok"}],
                "final_response": "已修复",
                "success": None,
            }
            review = SkillReview(
                should_write=True,
                confidence=0.96,
                skill_name="macos-appkit-overlay-text-wrapping",
                skill_markdown="# macOS AppKit Overlay Text Wrapping\n\nUse AppKit text measurement.",
                rationale="The user accepted the final behavior after correction.",
            )

            path = engine.write_skill(interaction, review)

            self.assertIsNotNone(path)
            self.assertTrue((path / "SKILL.md").exists())
            metadata = json.loads((path / "metadata.json").read_text())
            self.assertEqual(metadata["name"], "macos-appkit-overlay-text-wrapping")
            self.assertEqual(metadata["confidence"], 0.96)

    def test_maintenance_plan_can_deprecate_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = SkillEvolutionConfig(
                enabled=True,
                skill_dir=root / "skills",
                run_dir=root / "runs",
            )
            engine = SkillEvolutionEngine(config=config)
            skill_dir = config.skill_dir / "duplicate-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("# Duplicate\n", encoding="utf-8")
            (skill_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "name": "duplicate-skill",
                        "status": "active",
                        "confidence": 0.95,
                    }
                ),
                encoding="utf-8",
            )

            engine.apply_maintenance_plan(
                {
                    "actions": [
                        {
                            "action": "deprecate",
                            "skill": "duplicate-skill",
                            "reason": "Covered by a clearer skill.",
                        }
                    ]
                }
            )

            metadata = json.loads((skill_dir / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "deprecated")
            self.assertEqual(metadata["maintenance_reason"], "Covered by a clearer skill.")

    def test_persistent_prompt_contains_index_not_full_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = SkillEvolutionConfig(
                enabled=True,
                skill_dir=root / "skills",
                run_dir=root / "runs",
            )
            engine = SkillEvolutionEngine(config=config)
            skill_dir = config.skill_dir / "sample-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: sample-skill\ndescription: Use for sample tasks.\n---\n\n# Sample\n\nFULL WORKFLOW SECRET\n",
                encoding="utf-8",
            )
            (skill_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "name": "sample-skill",
                        "status": "active",
                        "confidence": 0.95,
                        "title": "sample-skill",
                    }
                ),
                encoding="utf-8",
            )

            prompt = engine.load_persistent_prompt()

            self.assertIn("sample-skill", prompt)
            self.assertIn("Use for sample tasks.", prompt)
            self.assertNotIn("FULL WORKFLOW SECRET", prompt)


if __name__ == "__main__":
    unittest.main()
