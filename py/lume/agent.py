import asyncio
import datetime
import html
import inspect
import json
import re
from pathlib import Path
from typing import Callable

from .config import load_config
from .llm import DEEPSEEK_MODEL, call_deepseek, Usage
from .memory import MemoryEngine, MemorySettings
from .skill_evolution import SkillEvolutionConfig, SkillEvolutionEngine
from .tools import execute_tool

# Persist history to ~/.lume/history.json
_HISTORY_DIR = Path.home() / ".lume"
_HISTORY_FILE = _HISTORY_DIR / "history.json"
_MAX_HISTORY = 50  # Keep last N messages to avoid unbounded growth
_AGENT_RECENT_MESSAGES = 8
_INPUT_RECENT_MESSAGES = 6
_MAX_TOOL_RESULT_CHARS_IN_CONTEXT = 600

# Context window management
_CONTEXT_WINDOW = 64000  # DeepSeek context window in tokens
_COMPRESS_THRESHOLD = 0.6  # Trigger compression at 60% usage
_COMPRESS_TARGET = 0.3  # Compress down to ~30% usage


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.5 chars per token for Chinese, ~4 chars for English."""
    if not text:
        return 0
    # Count Chinese characters
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - cn_chars
    return int(cn_chars / 1.5 + other_chars / 4)


def _estimate_messages_tokens(messages: list[dict], system_prompt: str = "") -> int:
    """Estimate total tokens for a message list + system prompt."""
    total = _estimate_tokens(system_prompt)
    for msg in messages:
        total += _estimate_tokens(msg.get("content", ""))
        total += 4  # role/formatting overhead per message
    return total


def _structured_tool_status(result: str) -> str:
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        return ""
    if not isinstance(data, dict):
        return ""
    status = data.get("status")
    return status if isinstance(status, str) else ""


def _runtime_context(history: list[dict], recent_messages: int) -> list[dict]:
    """Short dynamic tail for real-time control; full history stays on disk for learning."""
    return [_compact_context_message(msg) for msg in history[-recent_messages:]]


def _compact_context_message(msg: dict) -> dict:
    content = str(msg.get("content", ""))
    if content.startswith("[Tool:") and "\nResult:\n" in content:
        head, result = content.split("\nResult:\n", 1)
        if len(result) > _MAX_TOOL_RESULT_CHARS_IN_CONTEXT:
            result = result[:_MAX_TOOL_RESULT_CHARS_IN_CONTEXT] + f"\n... (truncated, {len(result)} chars total)"
        content = f"{head}\nResult:\n{result}"
    return {"role": msg.get("role", "user"), "content": content}


def _needs_time_context(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "今天", "明天", "昨天", "后天", "本周", "下周", "上周", "这个月", "下个月",
        "几点", "时间", "日期", "日程", "会议", "提醒", "calendar", "schedule",
        "today", "tomorrow", "yesterday", "date", "time",
    )
    return any(marker in lowered for marker in markers)


def _explicit_screenshot_mode(text: str) -> str | None:
    """Recognize direct screenshot requests that must not be delegated to a skill."""
    lowered = text.lower()
    has_screenshot = any(marker in lowered for marker in (
        "截图", "截屏", "截个图", "屏幕截图", "screen shot", "screenshot", "capture screen",
    ))
    if not has_screenshot:
        return None
    if any(marker in lowered for marker in ("看一下", "看看", "看下", "识别", "读一下", "内容", "错误", "什么", "分析", "look at", "analyze", "read it")):
        return "analyze"
    return "image"


def _active_skill_prompt(name: str, content: str) -> str:
    """Make a loaded, high-confidence skill binding for the current task only."""
    return f"""

Active verified skill: {name}
The skill below is binding execution knowledge for this task. Its verified paths,
application names, process names, commands, and scripts are literal: do not
translate, substitute, or reconstruct them. Follow its preflight before taking
the requested action. Do not conclude an application is unavailable until that
preflight has been attempted. If the skill does not cover the requested action,
say so instead of inventing a replacement workflow.

--- BEGIN VERIFIED SKILL ---
{content}
--- END VERIFIED SKILL ---
"""


def _needs_response_cleanup(text: str) -> bool:
    """Identify leaked tool traces or reasoning that must not reach voice/UI output."""
    value = str(text).strip()
    lowered = value.lower()
    internal_markers = (
        "```", "{\"tool\"", "'tool'", "tool result", "tool call",
        "run_command(", "applescript(", "osascript ", "traceback",
        "stdout:", "stderr:", "exit code", "<script",
    )
    url_or_markup = (
        re.search(r"(?i)\b(?:https?://|file://|www\.)", value)
        or re.search(r"\[[^\]]+\]\([^)]*\)", value)
    )
    if any(marker in lowered for marker in internal_markers) or url_or_markup:
        return True
    return len(value) > 360 or value.count("\n") > 2


_RESPONSE_CLEANUP_PROMPT = """Rewrite the candidate into the final response for a macOS voice assistant.
Return only a concise, natural-language answer in the user's language: one sentence when possible, never more than two.
The reply will be read aloud verbatim. Write a complete, fluent sentence that can be spoken naturally.
Do not include reasoning, commands, code blocks, JSON, tool names, raw output, implementation details, URLs, links, email addresses, file paths, or Markdown. Never read out an address; if a page was opened, simply say that it was opened.
Do not claim an unverified action succeeded. Preserve only the user-relevant outcome."""

SYSTEM_PROMPT = """You are Lume, a concise voice assistant on macOS.
Your responses are spoken aloud, so keep them SHORT — ideally one sentence, never more than two.
Do not repeat the question. Do not add filler words. Be direct and to the point.
Every final response is sent directly to TTS. Output a complete, natural sentence that sounds good when read aloud. Never include URLs, links, email addresses, file paths, Markdown, or raw special-symbol formatting; describe the outcome instead. If you opened a page, say it was opened rather than reading its address.
If a request matches a persistent learned skill index entry, call load_skill first and follow that full skill before taking action. Screenshot requests are the exception: the built-in screenshot tool has priority over every learned skill.

You can sense the user's speech metadata: [lang=xx] for language, [emotion=XX] for emotional state.
Respond in the same language the user speaks. Adapt tone if they sound upset or excited.
When the user explicitly asks you to remember, note, or keep something in mind, call remember_memory immediately with a concise key and exact value.

You have tools to interact with macOS. Respond with JSON to call a tool:
{"tool": "tool_name", "params": {...}}

File & Code:
- load_skill: {"name": "..."} — load the full content of a persistent learned skill before applying it
- remember_memory: {"key": "...", "value": "...", "category": "note|preference|profile|identity", "sensitivity": "profile|preference|sensitive"} — immediately save user-requested long-term memory
- read_memory: {} — read saved long-term memory
- read_file: {"path": "..."}
- write_file: {"path": "...", "content": "..."}
- run_command: {"command": "..."} — shell command
- list_dir: {"path": "..."}
- search_files: {"pattern": "...", "path": "..."}

macOS System:
- manage_application: {"action": "resolve|list|open|quit|minimize|focus|force_quit|dock_add|dock_remove|uninstall", "query": "app name", "bundle_id": "optional", "confirm": false} — the ONLY tool for app/Dock management. Always call action=resolve before a state-changing app action when the user supplied a name. Use the returned bundle_id for the next call. force_quit and uninstall require confirm=true; do not use run_command or applescript for app/Dock management.
- open_app: {"name": "Safari"} — launch/activate an app
- open_url: {"url": "https://..."} — open URL in default browser
- applescript: {"script": "..."} — run AppleScript (control any app)
- clipboard: {"action": "get"} or {"action": "set", "text": "..."}
- insert_text: {"text": "..."} — paste text at the current cursor location
- read_notes: {"limit": 2} — read recent Apple Notes content without changing clipboard
- get_selected_text: {} — read currently selected text
- replace_selected_text: {"text": "..."} — replace current selection
- replace_before_cursor: {"scope": "line|word", "text": "..."} — replace text before cursor
- system_info: {} — battery, wifi, volume, dark mode, frontmost app
- screenshot: {"mode": "image|analyze", "display": "main|all", "prompt": "optional analysis instruction"} — capture the current screen. Use mode=image to return the original PNG path plus `directory` and `filename`; use mode=analyze to send the image to DeepSeek and return its visual analysis. Use only when the user explicitly asks for a screenshot or when reading visible on-screen content is necessary to complete their request. Screenshots can contain sensitive information; do not capture or send them to DeepSeek speculatively.

Screenshot rule: for a direct request such as “截图看一下”“你直接截图看一下”“看看屏幕上的报错”, call screenshot with mode=analyze immediately. Never use applescript, shortcut, or run_command as a screenshot fallback, and do not load or follow a learned screenshot-capture skill.
- set_volume: {"level": 50} — 0-100
- music_control: {"action": "play|pause|next|prev|status"}
- spotlight_search: {"query": "..."} — find files via Spotlight
- notification: {"title": "...", "body": "..."}
- shortcut: {"name": "Shortcut Name", "input": "optional text"}

The applescript tool is very powerful — use it to control Calendar, Reminders, Mail, Safari, Finder, Notes, Messages, System Settings, and any scriptable app.
Never use applescript or run_command to manage applications, app windows, Dock entries, or uninstall apps; use manage_application instead.
If manage_application resolve returns not_found, you may make one different semantic alias query (for example, a homophone, English brand name, or shorter product name). Do not execute an app action until resolve returns a bundle_id.
Media keys work for most music apps via music_control tool.

Lark/Feishu (via lark-cli):
- lark: {"command": "...", "cwd": "optional working directory"} — run lark-cli commands. When sending a screenshot returned by `screenshot` with mode=image, pass its `directory` as `cwd` and `./<filename>` to `--image`.
  Examples:
  - "im +messages-send --chat-id oc_xxx --text 'message'" — send message
  - "calendar +agenda" — view upcoming events
  - "task +get-my-tasks" — list my tasks
  - "contact +search-user --query 'name'" — find user
  - "docs +get --url 'doc_url'" — read document
  - "drive +search --query 'keyword'" — search files
  - "im +search --query 'keyword'" — search chat history
  - "approval +my-pending" — pending approvals

Notion (via ntn CLI):
- notion: {"command": "..."} — run ntn commands
  Examples:
  - "pages get <page-id>" — read a page as Markdown
  - "pages create --content '# Title\nBody'" — create a page
  - "pages edit <page-id> --content '# Updated'" — edit a page
  - "datasources query <id>" — query a database
  - "datasources query <id> --filter '{...}'" — filtered query
  - "api v1/search query='keyword'" — search across workspace
  - "api v1/databases/<id>/query" — query database via API

Chrome Browser (via chrome-cli):
- chrome: {"command": "..."} — control Google Chrome
  Examples:
  - "list tabs" — list all open tabs
  - "info" — get active tab URL + title
  - "source" — get page HTML source of active tab
  - "open <url>" — open URL in new tab
  - "execute 'document.title'" — run JS in active tab
  - "execute 'document.body.innerText'" — get page text content
  - "activate -t <id>" — switch to a tab
  - "close" — close active tab
  - "reload" — reload active tab

Browser page actions (preferred for ordinary web interaction):
- browser: {"action": "inspect", "limit": 20} — list visible buttons, links, and form controls
- browser: {"action": "click", "text": "Submit", "index": 0} — click a visible control by its text; use selector when text is ambiguous
- browser: {"action": "click", "selector": "button[type=submit]"} — click a CSS selector
- browser: {"action": "fill", "selector": "#email", "value": "name@example.com"} — fill an input or textarea
- browser: {"action": "read", "selector": "main", "max_chars": 2000} — read visible text
- browser: {"action": "select", "selector": "#country", "text": "China"} — choose a select option
- browser: {"action": "scroll", "direction": "down", "amount": 600} — scroll the page
Use browser inspect before clicking ambiguous controls. It cannot bypass login, CAPTCHA, or browser permissions.

For `chrome execute`, the JavaScript must return a short, observable string (for example, "paused" or "no video"). If browser or chrome execute returns no result, tell the user to enable Chrome's View > Developer > Allow JavaScript from Apple Events before trying again.
If Chrome returns an `unknown` status, do not repeat the same probe or claim success. Try one materially different observable check, then explain that the action could not be verified.

After using tools, give the user ONLY the key takeaway in one short sentence.
Do NOT dump raw tool output. Extract the essential fact and say it concisely.
If the user asks to save an experience as a skill, do not type the word "skill" or "scale"; explain briefly that reusable experiences are saved by the background skill evolution system when enabled and sufficiently confident.
Some tools return JSON with a top-level "status": use "success" as completed, "failed" as failed, and "unknown" as attempted but not confirmable from the tool result. For "unknown", do not say the action succeeded; say it was attempted and ask the user to confirm if needed.
Examples of good responses after tool calls:
- "你有3个待办审批" (not the full list)
- "已发送给张三" (not the message content)
- "当前电量72%" (not all system info)
- "找到了5个相关文件" (not all file paths)
When done, respond without any JSON.

Examples of correct tool calls:
User: "明天有什么会"
→ {"tool": "lark", "params": {"command": "calendar +agenda --days 2"}}

User: "帮我把音量调到30"
→ {"tool": "set_volume", "params": {"level": 30}}

User: "我有什么待办任务"
→ {"tool": "lark", "params": {"command": "task +get-my-tasks"}}

User: "看看当前打开的网页是什么"
→ {"tool": "chrome", "params": {"command": "info"}}

User: "在notion里搜一下项目进度"
→ {"tool": "notion", "params": {"command": "api v1/search query='项目进度'"}}

User: "看看我屏幕上的报错"
→ {"tool": "screenshot", "params": {"mode": "analyze", "display": "main", "prompt": "Read the error message and explain the likely cause briefly in Chinese."}}

User: "暂停网易云音乐"
→ {"tool": "music_control", "params": {"action": "pause"}}

User: "网易云下一首"
→ {"tool": "music_control", "params": {"action": "next"}}

"""


_CONFIG = load_config()
_SKILL_EVOLUTION = SkillEvolutionEngine(
    SkillEvolutionConfig.from_settings(_CONFIG.skill_evolution)
)
_MEMORY = MemoryEngine(MemorySettings.from_config(_CONFIG.memory))


def _build_system_prompt(include_time: bool = False) -> str:
    """Build system prompt with stable content first for better context cache reuse."""
    memory_status = (
        "\nLong-term memory is enabled. Memory extraction happens asynchronously after each interaction; do not claim a fact is saved unless it is already present in Known long-term user memory."
        if _MEMORY.settings.enabled
        else "\nLong-term memory is disabled. Do not claim you will remember facts beyond the current conversation."
    )
    prompt = (
        SYSTEM_PROMPT
        + _MEMORY.load_persistent_prompt()
        + _SKILL_EVOLUTION.load_persistent_prompt()
        + memory_status
    )
    if include_time:
        now = datetime.datetime.now()
        prompt += f"\nCurrent time: {now.strftime('%Y-%m-%d %H:%M')} ({now.strftime('%A')})"
    return prompt


def _build_input_mode_prompt() -> str:
    return _build_system_prompt(include_time=False) + """

You are now in smart input method mode.
The user is asking you to produce text that will be inserted at the current cursor.
You may call tools when needed to gather context, read apps, inspect data, or edit already typed text.
By default tools must be read-only. Do not modify clipboard, files, apps, or system state.
Exception: use replace_selected_text or replace_before_cursor only when the user explicitly asks to rewrite already typed text.
Prefer read_notes for Apple Notes content.
When you have enough information, return ONLY the exact text to insert.
Do not explain. Do not wrap in quotes. Do not return JSON unless you are calling a tool.
If the user asks to polish, translate, continue, draft, or format text, output only the final composed text.
"""


class Stats:
    def __init__(self, model: str = DEEPSEEK_MODEL):
        self.total_tokens = 0
        self.context_tokens = 0
        self.cache_hit = 0
        self.cache_miss = 0
        self.model = model

    @property
    def cache_rate(self) -> str:
        total = self.cache_hit + self.cache_miss
        if total == 0:
            return "-"
        return f"{self.cache_hit / total * 100:.0f}%"


class Agent:
    def __init__(self):
        self.history: list[dict] = []
        self.model = DEEPSEEK_MODEL
        self.stats = Stats(self.model)
        self._load_history()

    def _load_history(self):
        """Load conversation history from disk."""
        if _HISTORY_FILE.exists():
            try:
                data = json.loads(_HISTORY_FILE.read_text())
                self.history = data if isinstance(data, list) else []
            except (json.JSONDecodeError, OSError):
                self.history = []

    def _save_history(self):
        """Persist conversation history to disk (keep last N messages)."""
        _HISTORY_DIR.mkdir(exist_ok=True)
        trimmed = self.history[-_MAX_HISTORY:]
        _HISTORY_FILE.write_text(json.dumps(trimmed, ensure_ascii=False, indent=None))

    async def _save_and_compress(self):
        """Persist bounded history; learning systems use eval/memory logs, not this prompt tail."""
        self._save_history()

    async def _clean_final_response(self, user_message: str, candidate: str) -> str:
        """Repair occasional model leakage without adding latency to normal replies."""
        if not _needs_response_cleanup(candidate):
            return candidate.strip()
        try:
            resp = await call_deepseek(
                _RESPONSE_CLEANUP_PROMPT,
                [
                    {"role": "user", "content": f"User request:\n{user_message}"},
                    {"role": "assistant", "content": f"Candidate response:\n{candidate}"},
                ],
                model=self.model,
            )
        except Exception:
            return candidate.strip()

        self.stats.model = resp.model
        self.stats.total_tokens += resp.usage.total_tokens
        self.stats.context_tokens = resp.usage.prompt_tokens
        self.stats.cache_hit += resp.usage.cache_hit
        self.stats.cache_miss += resp.usage.cache_miss
        cleaned = resp.content.strip()
        return cleaned if cleaned and not _needs_response_cleanup(cleaned) else candidate.strip()

    async def run(
        self, user_message: str, on_think: Callable | None = None
    ) -> str:
        self.stats = Stats(self.model)
        self.history.append({"role": "user", "content": user_message})
        messages = _runtime_context(self.history, _AGENT_RECENT_MESSAGES)
        system_prompt = _build_system_prompt(include_time=_needs_time_context(user_message))
        active_skill_name = ""
        active_skill_content = ""
        self._last_turn_count = 0

        screenshot_mode = _explicit_screenshot_mode(user_message)
        if screenshot_mode:
            tool_call = {"tool": "screenshot", "params": {"mode": screenshot_mode, "display": "main"}}
            if on_think:
                ret = on_think("tool_call", tool_call)
                if inspect.isawaitable(ret):
                    await ret
            result = await asyncio.to_thread(execute_tool, tool_call["tool"], tool_call["params"])
            if on_think:
                ret = on_think("tool_result", {"tool": tool_call["tool"], "result": result})
                if inspect.isawaitable(ret):
                    await ret
            assistant_msg = {"role": "assistant", "content": json.dumps(tool_call, ensure_ascii=False)}
            user_tool_msg = {"role": "user", "content": f'[Tool: screenshot]\nResult:\n{result}'}
            self.history.extend((assistant_msg, user_tool_msg))
            messages.extend((_compact_context_message(assistant_msg), _compact_context_message(user_tool_msg)))

        for i in range(10):  # max iterations
            self._last_turn_count = i + 1
            if on_think:
                ret = on_think("thinking", {"turn": i + 1})
                if inspect.isawaitable(ret):
                    await ret
            # Print LLM request
            print(f"── LLM Request (turn {i+1}) ──")
            for msg in messages[-3:]:  # last 3 messages for context
                role = msg["role"]
                content = msg["content"][:200]
                print(f"   [{role}] {content}")
            if len(messages) > 3:
                print(f"   ... ({len(messages)} runtime messages, {len(self.history)} persisted)")
            print()

            turn_system_prompt = system_prompt
            if active_skill_content:
                turn_system_prompt += _active_skill_prompt(
                    active_skill_name, active_skill_content
                )
            resp = await call_deepseek(turn_system_prompt, messages, model=self.model)

            # Print LLM response
            print(f"── LLM Response ──")
            print(f"   {resp.content[:500]}")
            print()

            # Update stats
            self.stats.model = resp.model
            self.stats.total_tokens += resp.usage.total_tokens
            self.stats.context_tokens = resp.usage.prompt_tokens
            self.stats.cache_hit = resp.usage.cache_hit
            self.stats.cache_miss = resp.usage.cache_miss

            # Try to extract tool call
            tool_call = self._extract_tool_call(resp.content)

            if tool_call is None:
                final_response = await self._clean_final_response(user_message, resp.content)
                self.history.append({"role": "assistant", "content": final_response})
                await self._save_and_compress()
                return final_response

            # Execute tool
            if on_think:
                ret = on_think("tool_call", tool_call)
                if inspect.isawaitable(ret):
                    await ret

            result = await asyncio.to_thread(execute_tool, tool_call["tool"], tool_call["params"])
            if tool_call["tool"] == "load_skill" and result.startswith("---"):
                active_skill_name = str(tool_call["params"].get("name", "loaded-skill"))
                active_skill_content = result

            # Truncate long tool output before feeding back to LLM
            if tool_call["tool"] != "load_skill" and len(result) > 2000:
                result = result[:2000] + f"\n... (truncated, {len(result)} total chars)"

            if on_think:
                ret = on_think("tool_result", {"tool": tool_call["tool"], "result": result})
                if inspect.isawaitable(ret):
                    await ret

            # Add recovery hint on structured tool status.
            tool_status = _structured_tool_status(result)
            if result.startswith("Error:"):
                result += "\n(Try a different approach or inform the user.)"
            if tool_status == "failed":
                result += "\n(Tool status is failed. Try a different approach or inform the user.)"
            elif tool_status == "unknown":
                result += "\n(Tool status is unknown. Do not claim success; say it was attempted and ask the user to confirm if needed.)"

            tool_msg = f'[Tool: {tool_call["tool"]}]\nResult:\n{result}'
            assistant_msg = {"role": "assistant", "content": resp.content}
            user_tool_msg = {"role": "user", "content": tool_msg}
            self.history.append(assistant_msg)
            self.history.append(user_tool_msg)
            messages.append(_compact_context_message(assistant_msg))
            messages.append(_compact_context_message(user_tool_msg))

        await self._save_and_compress()
        return "Reached maximum iterations."

    async def compose_input_text(
        self, user_message: str, on_think: Callable | None = None
    ) -> str:
        """Generate text for smart input mode, allowing tools before final text."""
        self.stats = Stats(self.model)
        messages = [
            *_runtime_context(self.history, _INPUT_RECENT_MESSAGES),
            {"role": "user", "content": user_message},
        ]
        system_prompt = _build_input_mode_prompt()
        if _needs_time_context(user_message):
            now = datetime.datetime.now()
            system_prompt += f"\nCurrent time: {now.strftime('%Y-%m-%d %H:%M')} ({now.strftime('%A')})"

        for i in range(6):
            self._last_turn_count = i + 1
            if on_think:
                ret = on_think("thinking", {"turn": i + 1})
                if inspect.isawaitable(ret):
                    await ret
            resp = await call_deepseek(system_prompt, messages, model=self.model)
            self.stats.model = resp.model
            self.stats.total_tokens += resp.usage.total_tokens
            self.stats.context_tokens = resp.usage.prompt_tokens
            self.stats.cache_hit = resp.usage.cache_hit
            self.stats.cache_miss = resp.usage.cache_miss

            text = resp.content.strip()
            tool_call = self._extract_tool_call(text)
            if tool_call is None:
                return text

            if not self._is_input_mode_tool_allowed(tool_call):
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": (
                        "That tool is not allowed in smart input mode because it can modify state. "
                        "Use read-only tools only, then return the exact text to insert."
                    ),
                })
                continue

            if on_think:
                ret = on_think("tool_call", tool_call)
                if inspect.isawaitable(ret):
                    await ret

            result = await asyncio.to_thread(execute_tool, tool_call["tool"], tool_call["params"])
            if len(result) > 2000:
                result = result[:2000] + f"\n... (truncated, {len(result)} total chars)"

            if on_think:
                ret = on_think("tool_result", {"tool": tool_call["tool"], "result": result})
                if inspect.isawaitable(ret):
                    await ret

            tool_status = _structured_tool_status(result)
            if result.startswith("Error:"):
                result += "\n(Try a different approach or return the best insertable text.)"
            if tool_status == "failed":
                result += "\n(Tool status is failed. Try a different approach or return the best insertable text.)"
            elif tool_status == "unknown":
                result += "\n(Tool status is unknown. Do not claim success; return a careful attempted-status message if needed.)"

            messages.append({"role": "assistant", "content": text})
            messages.append(_compact_context_message({"role": "user", "content": f'[Tool: {tool_call["tool"]}]\nResult:\n{result}'}))

        return ""

    # Tool name aliases for fuzzy matching
    _TOOL_ALIASES = {
        "load_skill": "load_skill", "loadSkill": "load_skill",
        "remember_memory": "remember_memory", "rememberMemory": "remember_memory",
        "read_memory": "read_memory", "readMemory": "read_memory",
        "open_url": "open_url", "openUrl": "open_url", "openurl": "open_url",
        "open_app": "open_app", "openApp": "open_app", "openapp": "open_app",
        "manage_application": "manage_application", "manageApplication": "manage_application", "app_manager": "manage_application",
        "read_file": "read_file", "readFile": "read_file",
        "write_file": "write_file", "writeFile": "write_file",
        "run_command": "run_command", "runCommand": "run_command",
        "list_dir": "list_dir", "listDir": "list_dir",
        "search_files": "search_files", "searchFiles": "search_files",
        "applescript": "applescript", "appleScript": "applescript",
        "clipboard": "clipboard",
        "insert_text": "insert_text", "insertText": "insert_text",
        "read_notes": "read_notes", "readNotes": "read_notes",
        "get_selected_text": "get_selected_text", "getSelectedText": "get_selected_text",
        "replace_selected_text": "replace_selected_text", "replaceSelectedText": "replace_selected_text",
        "replace_before_cursor": "replace_before_cursor", "replaceBeforeCursor": "replace_before_cursor",
        "set_volume": "set_volume", "setVolume": "set_volume",
        "music_control": "music_control", "musicControl": "music_control",
        "system_info": "system_info", "systemInfo": "system_info",
        "spotlight_search": "spotlight_search", "spotlightSearch": "spotlight_search",
        "browser": "browser", "web_browser": "browser", "browserAction": "browser",
    }

    _INPUT_MODE_READ_TOOLS = {
        "read_file",
        "list_dir",
        "search_files",
        "clipboard",
        "system_info",
        "spotlight_search",
        "manage_application",
        "lark",
        "notion",
        "chrome",
        "browser",
        "read_notes",
        "get_selected_text",
        "replace_selected_text",
        "replace_before_cursor",
    }

    def _is_input_mode_tool_allowed(self, tool_call: dict) -> bool:
        tool = tool_call.get("tool", "")
        if tool not in self._INPUT_MODE_READ_TOOLS:
            return False
        if tool == "clipboard":
            return tool_call.get("params", {}).get("action", "get") == "get"
        if tool == "manage_application":
            return tool_call.get("params", {}).get("action", "resolve") in {"resolve", "list"}
        if tool == "browser":
            return tool_call.get("params", {}).get("action", "") in {"inspect", "read"}
        if tool in {"replace_selected_text", "replace_before_cursor"}:
            params = tool_call.get("params", {})
            return bool(str(params.get("text", params.get("content", ""))).strip())
        return True

    def _extract_tool_call(self, text: str) -> dict | None:
        dsml_call = self._extract_dsml_tool_call(text)
        if dsml_call is not None:
            return dsml_call

        # Find "tool" marker
        match = re.search(r'"tool"\s*:\s*"', text)
        if not match:
            return None

        # Walk back to find opening brace
        start = text.rfind("{", 0, match.start())
        if start == -1:
            return None

        # Bracket-match for balanced braces
        depth = 0
        in_str = False
        escape = False
        end = -1

        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end == -1:
            return None

        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

        if "tool" not in obj:
            return None

        # Normalize tool name via aliases
        tool_name = obj["tool"]
        obj["tool"] = self._TOOL_ALIASES.get(tool_name, tool_name)

        # Tolerance: accept missing "params", or "arguments" as alias
        if "params" not in obj:
            if "arguments" in obj:
                obj["params"] = obj.pop("arguments")
            else:
                obj["params"] = {}

        return obj

    def _extract_dsml_tool_call(self, text: str) -> dict | None:
        """Accept DeepSeek's occasional DSML/XML tool-call representation."""
        invoke = re.search(r'<[^>]*\binvoke\s+name="([^"]+)"[^>]*>(.*?)</[^>]*\binvoke\s*>', text, re.DOTALL)
        if not invoke:
            return None
        params: dict[str, object] = {}
        for param in re.finditer(r'<[^>]*\bparameter\s+name="([^"]+)"([^>]*)>(.*?)</[^>]*\bparameter\s*>', invoke.group(2), re.DOTALL):
            name, attributes, value = param.group(1), param.group(2), html.unescape(param.group(3).strip())
            if 'json="true"' in attributes:
                try:
                    params[name] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass
            if 'number="true"' in attributes:
                try:
                    params[name] = float(value) if "." in value else int(value)
                    continue
                except ValueError:
                    pass
            if 'boolean="true"' in attributes:
                params[name] = value.lower() == "true"
                continue
            params[name] = value
        tool_name = self._TOOL_ALIASES.get(invoke.group(1), invoke.group(1))
        return {"tool": tool_name, "params": params}
