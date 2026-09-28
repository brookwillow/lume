import json
import tempfile
import unittest
from pathlib import Path

from lume.config import load_config
from lume.llm import DEEPSEEK_MODEL


class ConfigTests(unittest.TestCase):
    def test_stats_report_email_defaults_to_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(Path(tmp) / "missing.json")

        self.assertEqual(config.stats_report.recipient_email, "")

    def test_stats_report_email_loaded_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps({
                    "stats_report": {
                        "recipient_email": "user@example.com",
                        "interval_hours": 2,
                    }
                }),
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertEqual(config.stats_report.recipient_email, "user@example.com")
        self.assertEqual(config.stats_report.interval_hours, 2.0)

    def test_asr_corrections_loaded_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps({
                    "asr": {
                        "correction_enabled": True,
                        "corrections": {"Lume": ["卢米"]},
                    }
                }),
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertTrue(config.asr.correction_enabled)
        self.assertEqual(config.asr.corrections, {"Lume": ["卢米"]})

    def test_asr_lexicon_evolution_defaults_to_30_interactions(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(Path(tmp) / "missing.json")

        self.assertTrue(config.asr.lexicon_evolution_enabled)
        self.assertEqual(config.asr.curation_interval_interactions, 30)

    def test_local_entity_hotword_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(Path(tmp) / "missing.json")
        self.assertEqual(config.asr.local_entity_hotword_top_k, 20)
        self.assertEqual(config.asr.local_entity_index_file, "~/.lume/local_entities.json")

    def test_models_are_unified_even_with_legacy_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({
                "model": "deepseek-v4-pro",
                "skill_evolution": {"model": "deepseek-v4-pro"},
                "memory": {"model": "deepseek-v4-pro"},
                "asr": {"model": "deepseek-v4-flash"},
            }), encoding="utf-8")
            config = load_config(path)

        self.assertEqual(config.model, DEEPSEEK_MODEL)
        self.assertEqual(config.skill_evolution.model, DEEPSEEK_MODEL)
        self.assertEqual(config.memory.model, DEEPSEEK_MODEL)
        self.assertEqual(config.asr.model, DEEPSEEK_MODEL)


if __name__ == "__main__":
    unittest.main()
