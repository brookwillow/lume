import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from lume.agent import (
    Agent,
    _MAX_TOOL_RESULT_CHARS_IN_CONTEXT,
    _active_skill_prompt,
    _build_system_prompt,
    _explicit_screenshot_mode,
    _needs_response_cleanup,
    _needs_time_context,
    _runtime_context,
)


class AgentContextTests(unittest.TestCase):
    def test_explicit_screenshot_request_prefers_builtin_analysis(self):
        self.assertEqual(_explicit_screenshot_mode("你直接截图看一下"), "analyze")
        self.assertEqual(_explicit_screenshot_mode("截个图给我"), "image")
        self.assertIsNone(_explicit_screenshot_mode("打开浏览器"))

    def test_dsml_tool_call_is_parsed(self):
        response = '''<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="run_command">
<｜｜DSML｜｜ parameter name="command" string="true">ls -t ~/Desktop/*.png</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>'''
        self.assertEqual(
            Agent()._extract_tool_call(response),
            {"tool": "run_command", "params": {"command": "ls -t ~/Desktop/*.png"}},
        )

    def test_direct_screenshot_request_runs_builtin_tool_before_model(self):
        agent = Agent()
        agent.history = []
        response = type("Response", (), {"content": "我已看到屏幕。", "model": "deepseek-flash", "usage": type("Usage", (), {"total_tokens": 1, "prompt_tokens": 1, "cache_hit": 0, "cache_miss": 0})()})()
        with patch("lume.agent.execute_tool", return_value='{"status":"success","analysis":"visible error"}') as execute, \
             patch("lume.agent.call_deepseek", new=AsyncMock(return_value=response)), \
             patch.object(agent, "_save_and_compress", new=AsyncMock()):
            result = asyncio.run(agent.run("你直接截图看一下"))

        self.assertEqual(result, "我已看到屏幕。")
        execute.assert_called_once_with("screenshot", {"mode": "analyze", "display": "main"})

    def test_runtime_context_keeps_short_tail_and_truncates_tool_result(self):
        history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        history.append({"role": "user", "content": "[Tool: run_command]\nResult:\n" + "x" * 2000})

        ctx = _runtime_context(history, 3)

        self.assertEqual(len(ctx), 3)
        self.assertEqual(ctx[0]["content"], "msg 8")
        self.assertIn("truncated", ctx[-1]["content"])
        self.assertLess(len(ctx[-1]["content"]), _MAX_TOOL_RESULT_CHARS_IN_CONTEXT + 120)

    def test_time_context_is_opt_in(self):
        self.assertFalse(_needs_time_context("打开音乐"))
        self.assertTrue(_needs_time_context("明天有什么会议"))
        self.assertNotIn("Current time:", _build_system_prompt(include_time=False))
        self.assertIn("Current time:", _build_system_prompt(include_time=True))

    def test_loaded_skill_is_promoted_to_binding_execution_knowledge(self):
        prompt = _active_skill_prompt(
            "netease-music",
            "Use /Applications/NeteaseMusic.app and application NeteaseMusic.",
        )

        self.assertIn("binding execution knowledge", prompt)
        self.assertIn("do not\ntranslate, substitute, or reconstruct", prompt)
        self.assertIn("/Applications/NeteaseMusic.app", prompt)

    def test_internal_or_overlong_final_output_is_marked_for_cleanup(self):
        self.assertTrue(_needs_response_cleanup('```bash\nosascript -e "..."\n```'))
        self.assertTrue(_needs_response_cleanup("Tool Result: {\"status\": \"success\"}"))
        self.assertTrue(_needs_response_cleanup("内容\n" * 4))
        self.assertFalse(_needs_response_cleanup("已完成整理并复制到剪贴板。"))

    def test_url_final_output_is_marked_for_model_rewrite(self):
        self.assertTrue(_needs_response_cleanup("请访问 https://example.com"))


if __name__ == "__main__":
    unittest.main()
