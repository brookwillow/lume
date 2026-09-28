import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lume.asr_correction import normalize_asr_text
from lume.asr_lexicon import ASRLexiconEngine, ASRLexiconSettings, _has_task_evidence
from lume.config import ASRSettings


class ASRLexiconTests(unittest.TestCase):
    def test_capture_candidate_from_explicit_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = ASRLexiconEngine(ASRLexiconSettings(run_dir=Path(tmp), lexicon_file=Path(tmp) / "lexicon.json"))

            changed = engine.capture_candidates({"user_message": "不是 scale，是 skill"})

            candidates = json.loads((Path(tmp) / "candidates.json").read_text(encoding="utf-8"))

        self.assertTrue(changed)
        self.assertIn("skill\tscale", candidates)
        self.assertEqual(candidates["skill\tscale"]["count"], 1)

    def test_apply_review_writes_promoted_lexicon(self):
        with tempfile.TemporaryDirectory() as tmp:
            lexicon_file = Path(tmp) / "lexicon.json"
            engine = ASRLexiconEngine(
                ASRLexiconSettings(
                    run_dir=Path(tmp),
                    lexicon_file=lexicon_file,
                    min_confidence=0.96,
                )
            )

            path = engine.apply_review({
                "should_write": True,
                "entries": [
                    {"canonical": "Lume", "variant": "卢米", "confidence": 0.97},
                ],
            })

            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["entries"]["Lume"]["variants"], ["卢米"])

    def test_persistent_lexicon_is_used_by_normalizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            lexicon_file = Path(tmp) / "lexicon.json"
            lexicon_file.write_text(
                json.dumps({"entries": {"Lume": {"variants": ["卢米"]}}}, ensure_ascii=False),
                encoding="utf-8",
            )

            text = normalize_asr_text(
                "打开卢米",
                ASRSettings(lexicon_file=str(lexicon_file)),
            )

        self.assertEqual(text, "打开Lume")

    def test_curation_due_uses_30_interaction_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = ASRLexiconEngine(ASRLexiconSettings(run_dir=Path(tmp)))

            self.assertFalse(engine._curation_due(29))
            self.assertTrue(engine._curation_due(30))

    def test_task_resolved_hotword_is_persisted_and_recalled_by_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            lexicon_file = Path(tmp) / "lexicon.json"
            engine = ASRLexiconEngine(ASRLexiconSettings(run_dir=Path(tmp), lexicon_file=lexicon_file, local_entity_index_file=Path(tmp) / "entities.json"))
            engine.apply_review({"should_write": True, "entries": [{"canonical": "望京 SOHO", "variants": ["王晶搜猴"], "entity_type": "place", "contexts": ["navigation"], "confidence": 0.97, "base_weight": 0.9, "evidence_count": 2, "source": "task_execution"}]})
            hotwords = engine.recall_hotwords(scene="navigation", limit=5)
            data = json.loads(lexicon_file.read_text(encoding="utf-8"))
        self.assertEqual(hotwords[0]["text"], "望京 SOHO")
        self.assertEqual(data["entries"]["望京 SOHO"]["contexts"], ["navigation"])

    def test_high_confidence_canonical_without_variant_is_a_hotword(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = ASRLexiconEngine(ASRLexiconSettings(run_dir=Path(tmp), lexicon_file=Path(tmp) / "lexicon.json", local_entity_index_file=Path(tmp) / "entities.json"))
            engine.apply_review({"should_write": True, "entries": [{"canonical": "Alice", "confidence": 0.97, "contexts": ["communication"]}]})
            hotwords = engine.recall_hotwords(scene="communication")
        self.assertEqual([item["text"] for item in hotwords], ["Alice"])

    def test_successful_task_trace_can_trigger_curation_without_explicit_correction(self):
        self.assertTrue(_has_task_evidence({"tool_calls": [{"tool": "open_url"}], "success": True}))
        self.assertFalse(_has_task_evidence({"tool_calls": [], "success": True}))

    def test_hotword_curation_excludes_sensitive_long_term_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile.json"
            profile.write_text(json.dumps({
                "preferred_browser": {"value": "Chrome", "sensitivity": "preference"},
                "email": {"value": "user@example.com", "sensitivity": "sensitive"},
            }), encoding="utf-8")
            engine = ASRLexiconEngine(ASRLexiconSettings(run_dir=root, memory_profile_file=profile))
            memory = engine._load_hotword_memory()
        self.assertEqual(list(memory), ["preferred_browser"])

    def test_local_application_entities_are_recalled_without_promoting_to_lexicon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lexicon_file = root / "lexicon.json"
            entity_file = root / "entities.json"
            entity_file.write_text(json.dumps({"entities": [
                {"canonical": "豆包", "entity_type": "application", "contexts": ["agent"], "base_weight": 0.8, "source": "local_application_index"},
                {"canonical": "item two", "entity_type": "application", "contexts": ["agent"], "base_weight": 0.99, "source": "local_application_index"},
            ]}), encoding="utf-8")
            engine = ASRLexiconEngine(ASRLexiconSettings(run_dir=root, lexicon_file=lexicon_file, local_entity_index_file=entity_file, local_entity_hotword_top_k=10))
            hotwords = engine.recall_hotwords(scene="agent", limit=10)
        self.assertEqual([item["text"] for item in hotwords], ["豆包"])
        self.assertFalse(lexicon_file.exists())

    def test_refresh_writes_local_application_index_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entity_file = root / "entities.json"
            engine = ASRLexiconEngine(ASRLexiconSettings(run_dir=root, local_entity_index_file=entity_file))
            with patch("lume.tools.local_application_entities", return_value=[{"canonical": "Lark", "source": "local_application_index"}]):
                path = engine.refresh_local_entities(reason="test")
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["reason"], "test")
        self.assertEqual(data["entities"][0]["canonical"], "Lark")

    def test_hotword_recall_uses_cached_snapshot_between_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lexicon = root / "lexicon.json"
            lexicon.write_text(json.dumps({"entries": {"Lark": {"confidence": 0.97, "base_weight": 0.9}}}), encoding="utf-8")
            engine = ASRLexiconEngine(ASRLexiconSettings(run_dir=root, lexicon_file=lexicon, local_entity_index_file=root / "entities.json"))
            with patch.object(engine, "_load_lexicon", wraps=engine._load_lexicon) as load_lexicon:
                engine.recall_hotwords(scene="agent")
                engine.recall_hotwords(scene="agent")
        self.assertEqual(load_lexicon.call_count, 1)


if __name__ == "__main__":
    unittest.main()
