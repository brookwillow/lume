import unittest

from lume.config import ASRSettings
from lume.asr_correction import normalize_asr_text, normalize_asr_text_with_trace
from lume.asr_correction import normalize_hotword_aliases


class VoiceTests(unittest.TestCase):
    def test_normalizes_common_developer_terms(self):
        text = normalize_asr_text("把这个保存成scale，然后问一下克劳德和扣德克斯")

        self.assertIn("skill", text)
        self.assertIn("Claude", text)
        self.assertIn("Codex", text)

    def test_can_disable_asr_correction(self):
        text = normalize_asr_text(
            "保存成scale",
            ASRSettings(correction_enabled=False),
        )

        self.assertEqual(text, "保存成scale")

    def test_custom_correction_from_config(self):
        text = normalize_asr_text(
            "打开卢米",
            ASRSettings(corrections={"Lume": ["卢米"]}),
        )

        self.assertEqual(text, "打开Lume")

    def test_normalization_trace_records_actual_replacement(self):
        text, trace = normalize_asr_text_with_trace("打开卢米", ASRSettings(corrections={"Lume": ["卢米"]}))
        self.assertEqual(text, "打开Lume")
        self.assertEqual(trace, [{"variant": "卢米", "canonical": "Lume", "count": 1}])

    def test_hotword_alias_normalizes_unique_ascii_near_match(self):
        text, trace = normalize_hotword_aliases("在 chrom 中搜索", [{"text": "Google Chrome", "aliases": ["Google Chrome"]}])
        self.assertEqual(text, "在 Google Chrome 中搜索")
        self.assertEqual(trace, [{"variant": "chrom", "canonical": "Google Chrome", "count": 1}])

    def test_hotword_alias_does_not_fuzzy_correct_chinese(self):
        text, trace = normalize_hotword_aliases("打开豆宝", [{"text": "豆包", "aliases": ["豆包"]}])
        self.assertEqual(text, "打开豆宝")
        self.assertEqual(trace, [])

    def test_hotword_alias_keeps_complete_application_name_intact(self):
        text, trace = normalize_hotword_aliases("打开 Google Chrome", [{"text": "Google Chrome", "aliases": ["Google Chrome"]}])
        self.assertEqual(text, "打开 Google Chrome")
        self.assertEqual(trace, [])



if __name__ == "__main__":
    unittest.main()
