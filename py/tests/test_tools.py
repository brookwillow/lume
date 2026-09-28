import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import json

from lume import tools


class ToolTests(unittest.TestCase):
    def test_screenshot_returns_original_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            screenshot_dir = Path(tmp) / "screenshots"

            def fake_run(cmd, **kwargs):
                Path(cmd[-1]).write_bytes(b"png")

                class Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return Result()

            with patch.object(tools, "_SCREENSHOT_DIR", screenshot_dir), \
                 patch.object(tools.subprocess, "run", side_effect=fake_run):
                result = json.loads(tools.execute_tool("screenshot", {}))

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["display"], "main")
            self.assertEqual(result["mode"], "image")
            self.assertTrue(Path(result["path"]).exists())
            self.assertEqual(Path(result["path"]).stat().st_mode & 0o777, 0o600)

    def test_screenshot_analyze_uses_deepseek_visual_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fake_run(cmd, **kwargs):
                Path(cmd[-1]).write_bytes(b"png")

                class Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return Result()

            with patch.object(tools, "_SCREENSHOT_DIR", Path(tmp) / "screenshots"), \
                 patch.object(tools.subprocess, "run", side_effect=fake_run), \
                 patch.object(tools, "analyze_image", return_value="Safari displays a login error.") as analyze:
                result = json.loads(tools.execute_tool("screenshot", {"mode": "analyze", "prompt": "Explain the error."}))

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["mode"], "analyze")
            self.assertEqual(result["analysis"], "Safari displays a login error.")
            self.assertEqual(analyze.call_args.args[1], "Explain the error.")

    def test_insert_text_uses_clipboard_and_paste(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))

            class Result:
                returncode = 0
                stdout = "old clipboard"
                stderr = ""

            return Result()

        with patch.object(tools.subprocess, "run", side_effect=fake_run):
            result = tools.execute_tool("insert_text", {"text": "hello"})

        self.assertEqual(result, "Inserted text")
        self.assertEqual(calls[0][0], ["pbpaste"])
        self.assertEqual(calls[1][0], ["pbcopy"])
        self.assertEqual(calls[1][1]["input"], "hello")
        self.assertEqual(calls[2][0], ["osascript", "-e", 'tell application "System Events" to keystroke "v" using command down'])
        self.assertEqual(calls[3][0], ["pbcopy"])
        self.assertEqual(calls[3][1]["input"], "old clipboard")

    def test_read_notes_uses_read_only_applescript(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))

            class Result:
                returncode = 0
                stdout = "First note\n---NOTE---\nSecond note"
                stderr = ""

            return Result()

        with patch.object(tools.subprocess, "run", side_effect=fake_run):
            result = tools.execute_tool("read_notes", {"limit": 2})

        self.assertIn("[Note 1]", result)
        self.assertIn("First note", result)
        self.assertIn("[Note 2]", result)
        self.assertIn("Second note", result)
        self.assertEqual(calls[0][0][0], "osascript")
        self.assertNotIn("set the clipboard", calls[0][0][2])

    def test_get_selected_text_copies_and_restores_clipboard(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))

            class Result:
                returncode = 0
                stdout = "old clipboard" if cmd == ["pbpaste"] and len(calls) == 1 else "selected text"
                stderr = ""

            return Result()

        with patch.object(tools.subprocess, "run", side_effect=fake_run):
            result = tools.execute_tool("get_selected_text", {})

        self.assertEqual(result, "selected text")
        self.assertEqual(calls[1][0], ["osascript", "-e", 'tell application "System Events" to keystroke "c" using command down'])
        self.assertEqual(calls[-1][0], ["pbcopy"])
        self.assertEqual(calls[-1][1]["input"], "old clipboard")

    def test_select_before_cursor_then_replace_text(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))

            class Result:
                returncode = 0
                stdout = "old clipboard"
                stderr = ""

            return Result()

        with patch.object(tools.subprocess, "run", side_effect=fake_run):
            result = tools.execute_tool("replace_before_cursor", {"text": "new text", "scope": "line"})

        self.assertEqual(result, "Replaced text before cursor")
        self.assertIn("key code 123 using {command down, shift down}", calls[0][0][2])
        self.assertEqual(calls[2][1]["input"], "new text")

    def test_load_skill_reads_skill_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "demo-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("# Demo Skill\nFull workflow", encoding="utf-8")

            with patch.object(tools, "_SKILL_DIR", Path(tmp)):
                result = tools.execute_tool("load_skill", {"name": "demo-skill"})

        self.assertIn("# Demo Skill", result)
        self.assertIn("Full workflow", result)

    def test_remember_memory_writes_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(tools, "_MEMORY_DIR", Path(tmp)):
                result = tools.execute_tool(
                    "remember_memory",
                    {
                        "key": "remembered_note_project",
                        "value": "下周同步项目进展",
                        "category": "note",
                        "sensitivity": "profile",
                    },
                )
                profile = (Path(tmp) / "profile.json").read_text()

        self.assertEqual(result, "Memory saved")
        self.assertIn("下周同步项目进展", profile)

    def test_read_memory_returns_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "profile.json"
            p.write_text('{"preferred_browser":{"value":"Chrome"}}', encoding="utf-8")
            with patch.object(tools, "_MEMORY_DIR", Path(tmp)):
                result = tools.execute_tool("read_memory", {})

        self.assertIn("preferred_browser", result)

    def test_applescript_no_output_returns_unknown_status(self):
        def fake_run(cmd, **kwargs):
            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        with patch.object(tools.subprocess, "run", side_effect=fake_run):
            result = tools.execute_tool("applescript", {"script": 'tell app "System Events" to keystroke space'})

        data = json.loads(result)
        self.assertEqual(data["status"], "unknown")
        self.assertNotIn("verified", data)

    def test_run_command_osascript_no_output_returns_unknown_status(self):
        def fake_run(cmd, **kwargs):
            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        with patch.object(tools.subprocess, "run", side_effect=fake_run):
            result = tools.execute_tool(
                "run_command",
                {"command": "osascript -e 'tell application \"System Events\" to keystroke space'"},
            )

        data = json.loads(result)
        self.assertEqual(data["status"], "unknown")
        self.assertNotIn("verified", data)

    def test_run_command_failure_returns_structured_failed_status(self):
        def fake_run(cmd, **kwargs):
            class Result:
                returncode = 1
                stdout = ""
                stderr = "boom"

            return Result()

        with patch.object(tools.subprocess, "run", side_effect=fake_run):
            result = tools.execute_tool("run_command", {"command": "false"})

        data = json.loads(result)
        self.assertEqual(data["status"], "failed")
        self.assertIn("boom", data["stderr"])

    def test_spotlight_search_uses_supported_mdfind_arguments_and_limits_results(self):
        def fake_run(cmd, **kwargs):
            class Result:
                returncode = 0
                stdout = "\n".join(f"/tmp/result-{i}" for i in range(25))
                stderr = ""

            self.assertEqual(cmd, ["mdfind", "NeteaseMusic"])
            return Result()

        with patch.object(tools.subprocess, "run", side_effect=fake_run):
            result = tools.execute_tool("spotlight_search", {"query": "NeteaseMusic"})

        self.assertIn("/tmp/result-19", result)
        self.assertNotIn("/tmp/result-20", result)
        self.assertIn("5 more", result)

    def test_manage_application_resolves_localized_bundle_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "豆包.app"
            info = app / "Contents" / "Info.plist"
            info.parent.mkdir(parents=True)
            import plistlib
            info.write_bytes(plistlib.dumps({"CFBundleIdentifier": "com.example.doubao", "CFBundleDisplayName": "豆包", "CFBundleExecutable": "Doubao"}))
            with patch.object(tools, "_APP_SEARCH_ROOTS", (root,)), patch.object(tools, "_APP_INDEX_CACHE", (0.0, [])):
                data = json.loads(tools.execute_tool("manage_application", {"action": "resolve", "query": "豆包"}))
        self.assertEqual(data["status"], "resolved")
        self.assertEqual(data["candidates"][0]["bundle_id"], "com.example.doubao")

    def test_manage_application_requires_confirmation_for_uninstall(self):
        app = {"name": "Demo", "bundle_id": "com.example.demo", "path": "/Applications/Demo.app", "aliases": ["Demo"], "localized_names": ["Demo"], "version": "", "executable": "Demo", "score": 1.0}
        with patch.object(tools, "_find_local_applications", return_value=[app]):
            data = json.loads(tools.execute_tool("manage_application", {"action": "uninstall", "bundle_id": app["bundle_id"]}))
        self.assertEqual(data["status"], "needs_confirmation")

    def test_manage_application_reuses_matching_asr_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "豆包.app"
            info = app / "Contents" / "Info.plist"
            info.parent.mkdir(parents=True)
            import plistlib
            info.write_bytes(plistlib.dumps({"CFBundleIdentifier": "com.example.doubao", "CFBundleDisplayName": "豆包"}))
            lexicon = root / "asr_lexicon.json"
            lexicon.write_text(json.dumps({"entries": {"豆包": {"variants": ["豆宝"]}}}), encoding="utf-8")
            with patch.object(tools, "_APP_SEARCH_ROOTS", (root,)), patch.object(tools, "_ASR_LEXICON_FILE", lexicon), patch.object(tools, "_APP_INDEX_CACHE", (0.0, [])):
                data = json.loads(tools.execute_tool("manage_application", {"action": "resolve", "query": "豆宝浏览器"}))
        self.assertEqual(data["status"], "resolved")
        self.assertEqual(data["candidates"][0]["bundle_id"], "com.example.doubao")

    def test_chrome_execute_without_output_is_unverifiable(self):
        def fake_run(cmd, **kwargs):
            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        with patch.object(tools.subprocess, "run", side_effect=fake_run):
            result = tools.execute_tool("chrome", {"command": "execute 'document.title'"})

        data = json.loads(result)
        self.assertEqual(data["status"], "unknown")
        self.assertIn("observable", data["reason"])

    def test_browser_click_uses_safe_chrome_cli_arguments(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))

            class Result:
                returncode = 0
                stdout = '{"status":"success","action":"click","text":"登录"}'
                stderr = ""

            return Result()

        with patch.object(tools.subprocess, "run", side_effect=fake_run):
            result = tools.execute_tool("browser", {"action": "click", "text": "登录"})

        self.assertEqual(json.loads(result)["status"], "success")
        self.assertEqual(calls[0][0][:2], ["chrome-cli", "execute"])
        self.assertIn("document.querySelectorAll", calls[0][0][2])
        self.assertFalse(calls[0][1].get("shell", False))

    def test_browser_fill_requires_a_selector(self):
        result = tools.execute_tool("browser", {"action": "fill", "value": "hello"})
        self.assertEqual(json.loads(result)["status"], "failed")

    def test_browser_inspect_returns_cli_json(self):
        def fake_run(cmd, **kwargs):
            class Result:
                returncode = 0
                stdout = '{"status":"success","action":"inspect","items":[]}'
                stderr = ""

            return Result()

        with patch.object(tools.subprocess, "run", side_effect=fake_run):
            result = tools.execute_tool("browser", {"action": "inspect"})

        self.assertEqual(json.loads(result)["action"], "inspect")


if __name__ == "__main__":
    unittest.main()
