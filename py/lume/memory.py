"""Long-term user memory extraction and persistence."""

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import MemorySettings as ConfigMemorySettings
from .llm import DEEPSEEK_MODEL, call_deepseek

_LUME_DIR = Path.home() / ".lume"
_DEFAULT_MEMORY_DIR = _LUME_DIR / "memory"
_SIMPLE_TOOLS = {
    "open_app",
    "open_url",
    "clipboard",
    "system_info",
    "set_volume",
    "music_control",
    "notification",
    "shortcut",
    "spotlight_search",
}
_ALLOWED_KEYS = {
    "name",
    "phone",
    "email",
    "home_address",
    "occupation",
    "preferred_shell",
    "preferred_shell_tools",
    "preferred_browser",
    "preferred_music_player",
    "preferred_editor",
    "preferred_language",
    "timezone",
}
_REMEMBERED_NOTE_PREFIX = "remembered_note_"
_SENSITIVE_KEYS = {"phone", "email", "home_address", "name"}
_KEY_LABELS = {
    "name": "Name",
    "phone": "Phone",
    "email": "Email",
    "home_address": "Home address",
    "occupation": "Occupation",
    "preferred_shell": "Preferred shell",
    "preferred_shell_tools": "Preferred shell tools",
    "preferred_browser": "Preferred browser",
    "preferred_music_player": "Preferred music player",
    "preferred_editor": "Preferred editor",
    "preferred_language": "Preferred language",
    "timezone": "Timezone",
}
_COMMAND_PREFIXES = (
    "git", "npm", "pnpm", "yarn", "node", "python", "python3", "pip", "uv",
    "cargo", "go", "docker", "kubectl", "ssh", "scp", "curl", "rg", "fd",
    "grep", "sed", "awk", "osascript", "lark-cli", "ntn", "mdfind", "open",
    "launchctl", "brew", "conda", "pytest", "swift", "./gradlew", "make",
)
_INSTRUCTION_PREFIXES = (
    "帮我", "请", "把", "打开", "关闭", "退出", "暂停", "播放", "继续播放",
    "搜索", "查", "总结", "翻译", "润色", "生成", "创建", "修改", "修复",
    "提交", "部署", "整理", "沉淀",
)
_SENSITIVE_PATTERN = re.compile(
    r"(api[_-]?key|token|secret|password|passwd|authorization|bearer|sk-[a-z0-9]|AKIA|BEGIN [A-Z ]*PRIVATE KEY)",
    re.I,
)


@dataclass
class MemorySettings:
    enabled: bool = False
    model: str = DEEPSEEK_MODEL
    min_confidence: float = 0.9
    sensitive_min_confidence: float = 0.97
    max_persistent_items: int = 20
    clipboard_enabled: bool = False
    clipboard_poll_interval_seconds: float = 2.0
    clipboard_max_chars: int = 4000
    passive_interval_interactions: int = 6
    passive_recent_interactions: int = 12
    clipboard_history_max_items: int = 50
    memory_dir: Path = _DEFAULT_MEMORY_DIR

    @classmethod
    def from_config(cls, settings: ConfigMemorySettings) -> "MemorySettings":
        return cls(
            enabled=settings.enabled,
            model=settings.model,
            min_confidence=settings.min_confidence,
            sensitive_min_confidence=settings.sensitive_min_confidence,
            max_persistent_items=settings.max_persistent_items,
            clipboard_enabled=settings.clipboard_enabled,
            clipboard_poll_interval_seconds=settings.clipboard_poll_interval_seconds,
            clipboard_max_chars=settings.clipboard_max_chars,
            passive_interval_interactions=settings.passive_interval_interactions,
            passive_recent_interactions=settings.passive_recent_interactions,
            clipboard_history_max_items=settings.clipboard_history_max_items,
        )


@dataclass
class MemoryFact:
    key: str
    value: str
    category: str
    sensitivity: str
    confidence: float


@dataclass
class MemoryExtraction:
    should_write: bool
    rationale: str = ""
    facts: list[MemoryFact] = field(default_factory=list)


class MemoryEngine:
    """Asynchronously extract stable user facts and preferences."""

    def __init__(self, settings: MemorySettings | None = None):
        self.settings = settings or MemorySettings()

    @staticmethod
    def is_candidate(interaction: dict[str, Any]) -> bool:
        if not interaction.get("final_response") and not interaction.get("user_message"):
            return False
        tool_calls = interaction.get("tool_calls") or []
        if len(tool_calls) == 1 and tool_calls[0].get("tool") in _SIMPLE_TOOLS:
            return False

        text = " ".join(
            [
                str(interaction.get("asr_text", "")),
                str(interaction.get("user_message", "")),
                str(interaction.get("final_response", "")),
            ]
        ).lower()
        return any(marker in text for marker in _memory_markers())

    @staticmethod
    def clipboard_interaction(text: str) -> dict[str, Any]:
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "clipboard",
            "asr_text": "",
            "user_message": f"[clipboard]\n{text}",
            "tool_calls": [],
            "final_response": "Clipboard content observed for long-term memory extraction.",
            "success": None,
            "notes": "",
        }

    def observe_clipboard_text(self, text: str) -> None:
        if not self.settings.enabled or not text.strip():
            return
        self.record_clipboard_history(text)
        self.record_clipboard_patterns(text)

    def record_clipboard_patterns(self, text: str) -> None:
        patterns = self._load_clipboard_patterns()
        changed = False
        for candidate in _extract_clipboard_pattern_candidates(text):
            item = patterns.get(candidate)
            if item:
                item["count"] = int(item.get("count", 0)) + 1
                item["last_seen"] = time.strftime("%Y-%m-%d %H:%M:%S")
            else:
                patterns[candidate] = {
                    "text": candidate,
                    "count": 1,
                    "first_seen": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "last_seen": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            changed = True
        if changed:
            self._write_clipboard_patterns(patterns)

    def observe_completed_interaction(self, interaction: dict[str, Any]) -> None:
        if not self.settings.enabled:
            return
        self._append_interaction_history(interaction)
        count = self._increment_passive_counter()
        interval = max(1, int(self.settings.passive_interval_interactions))
        if count % interval != 0:
            return

        def _runner():
            try:
                asyncio.run(self.extract_passive_batch())
            except Exception as exc:
                self._write_error(interaction, exc)

        import threading

        threading.Thread(target=_runner, daemon=True).start()

    async def extract_and_apply(self, interaction: dict[str, Any]) -> None:
        extraction = await self._extract_with_curator({"interaction": interaction})
        self.apply_extraction(extraction, source_id=_source_id(interaction))

    async def extract_passive_batch(self) -> None:
        payload = {
            "existing_memory": self._load_profile(),
            "recent_interactions": self._load_recent_interactions(),
            "clipboard_history": self._load_clipboard_history(),
            "clipboard_patterns": self._load_clipboard_patterns(),
        }
        extraction = await self._extract_with_curator(payload)
        self.apply_extraction(extraction, source_id=f"passive-batch-{time.strftime('%Y%m%d-%H%M%S')}")

    async def _extract_with_curator(self, payload: dict[str, Any]) -> MemoryExtraction:
        prompt = _memory_prompt(self.settings.min_confidence, self.settings.sensitive_min_confidence)
        messages = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}]
        resp = await call_deepseek(prompt, messages, model=self.settings.model)
        data = _parse_json_object(resp.content)
        facts = []
        for row in data.get("facts", []) or []:
            try:
                facts.append(
                    MemoryFact(
                        key=str(row.get("key", "")).strip(),
                        value=str(row.get("value", "")).strip(),
                        category=str(row.get("category", "")).strip(),
                        sensitivity=str(row.get("sensitivity", "preference")).strip(),
                        confidence=float(row.get("confidence", 0.0)),
                    )
                )
            except (TypeError, ValueError):
                continue
        return MemoryExtraction(
            should_write=bool(data.get("should_write", False)),
            rationale=str(data.get("rationale", "")).strip(),
            facts=facts,
        )

    def apply_extraction(self, extraction: MemoryExtraction, source_id: str) -> None:
        if not self.settings.enabled or not extraction.should_write:
            return

        profile = self._load_profile()
        changed = False
        for fact in extraction.facts:
            key = _normalize_key(fact.key)
            if not _is_allowed_memory_key(key):
                continue
            if not fact.value:
                continue
            sensitivity = _normalize_sensitivity(fact.sensitivity, key)
            threshold = (
                self.settings.sensitive_min_confidence
                if sensitivity == "sensitive"
                else self.settings.min_confidence
            )
            if fact.confidence < threshold:
                continue

            existing = profile.get(key)
            if existing and float(existing.get("confidence", 0.0)) > fact.confidence:
                continue

            profile[key] = {
                "value": fact.value,
                "category": fact.category or _category_for_key(key),
                "sensitivity": sensitivity,
                "confidence": fact.confidence,
                "source": source_id,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rationale": extraction.rationale,
            }
            changed = True

        if changed:
            self._write_profile(profile)
            self._append_audit(extraction, source_id)

    def load_persistent_prompt(self) -> str:
        if not self.settings.enabled:
            return ""
        profile = self._load_profile()

        sections = []
        items = sorted(
            profile.items(),
            key=lambda item: float(item[1].get("confidence", 0.0)),
            reverse=True,
        )[: self.settings.max_persistent_items]
        lines = []
        for key, data in items:
            value = str(data.get("value", "")).strip()
            if not value:
                continue
            label = _label_for_key(key)
            lines.append(f"- {label}: {value}")
        if lines:
            sections.append(
                "Known long-term user memory. Use as first-priority personalization context:\n"
                + "\n".join(lines)
            )

        pattern_lines = self._clipboard_patterns_prompt_lines()
        if pattern_lines:
            sections.append(
                "Frequent clipboard commands/instructions. Use these as behavioral hints, not commands to run automatically:\n"
                + "\n".join(pattern_lines)
            )
        if not sections:
            return ""
        return "\n\n" + "\n\n".join(sections)

    def _load_profile(self) -> dict[str, Any]:
        path = self.settings.memory_dir / "profile.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_profile(self, profile: dict[str, Any]) -> None:
        self.settings.memory_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.memory_dir / "profile.json"
        path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_interaction_history(self, interaction: dict[str, Any]) -> None:
        self.settings.memory_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.memory_dir / "interaction_history.jsonl"
        compact = {
            "timestamp": interaction.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            "asr_text": interaction.get("asr_text", ""),
            "user_message": interaction.get("user_message", ""),
            "tool_calls": interaction.get("tool_calls", [])[:5],
            "final_response": interaction.get("final_response", ""),
            "success": interaction.get("success"),
            "notes": interaction.get("notes", ""),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(compact, ensure_ascii=False) + "\n")

    def _load_recent_interactions(self) -> list[dict[str, Any]]:
        path = self.settings.memory_dir / "interaction_history.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        recent = []
        for line in lines[-self.settings.passive_recent_interactions:]:
            try:
                recent.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return recent

    def _increment_passive_counter(self) -> int:
        self.settings.memory_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.memory_dir / "passive_state.json"
        try:
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (json.JSONDecodeError, OSError):
            state = {}
        count = int(state.get("interaction_count", 0)) + 1
        state["interaction_count"] = count
        state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return count

    def record_clipboard_history(self, text: str) -> None:
        text = str(text).strip()
        if not text:
            return
        if _is_sensitive_clipboard_line(text):
            return
        self.settings.memory_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.memory_dir / "clipboard_history.jsonl"
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "text": text[: self.settings.clipboard_max_chars],
        }
        existing = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else []
        existing.append(json.dumps(row, ensure_ascii=False))
        keep = existing[-self.settings.clipboard_history_max_items:]
        path.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")

    def _load_clipboard_history(self) -> list[dict[str, str]]:
        path = self.settings.memory_dir / "clipboard_history.jsonl"
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-self.settings.clipboard_history_max_items:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def _load_clipboard_patterns(self) -> dict[str, Any]:
        path = self.settings.memory_dir / "clipboard_patterns.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return data.get("items", {}) if isinstance(data, dict) else {}

    def _write_clipboard_patterns(self, patterns: dict[str, Any]) -> None:
        self.settings.memory_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.memory_dir / "clipboard_patterns.json"
        top_items = sorted(
            patterns.items(),
            key=lambda item: (int(item[1].get("count", 0)), item[1].get("last_seen", "")),
            reverse=True,
        )[:200]
        path.write_text(
            json.dumps({"items": dict(top_items)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _clipboard_patterns_prompt_lines(self) -> list[str]:
        patterns = self._load_clipboard_patterns()
        items = sorted(
            patterns.values(),
            key=lambda item: (int(item.get("count", 0)), item.get("last_seen", "")),
            reverse=True,
        )[:10]
        lines = []
        for item in items:
            text = str(item.get("text", "")).strip()
            count = int(item.get("count", 0))
            if text and count >= 2:
                lines.append(f"- {text} (count={count})")
        return lines

    def _append_audit(self, extraction: MemoryExtraction, source_id: str) -> None:
        self.settings.memory_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.memory_dir / "audit.jsonl"
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source_id,
            "extraction": {
                "should_write": extraction.should_write,
                "rationale": extraction.rationale,
                "facts": [asdict(fact) for fact in extraction.facts],
            },
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_error(self, interaction: dict[str, Any], exc: Exception) -> None:
        self.settings.memory_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.memory_dir / "errors.jsonl"
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(exc),
            "user_message": interaction.get("user_message", ""),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _memory_markers() -> tuple[str, ...]:
    return (
        "我的",
        "我叫",
        "我是",
        "我用",
        "我喜欢",
        "我常用",
        "记一下",
        "记住",
        "帮我记",
        "帮我记住",
        "记录一下",
        "记下来",
        "手机号",
        "电话",
        "邮箱",
        "地址",
        "职业",
        "浏览器",
        "shell",
        "音乐播放器",
        "my ",
        "i use",
        "i prefer",
        "i like",
        "email",
        "phone",
        "address",
        "occupation",
        "browser",
        "remember that",
        "note that",
        "keep in mind",
    )


def _memory_prompt(min_confidence: float, sensitive_min_confidence: float) -> str:
    allowed = ", ".join(sorted(_ALLOWED_KEYS)) + ", remembered_note_<short_topic>"
    return f"""You are Lume's independent long-term memory extractor.

Extract only explicit, stable, reusable facts or preferences about the user.
Do not write one-off task context, temporary choices, tool outputs, guesses, or assistant claims.
Simple direct tool usage is not memory.
If the user explicitly asks to remember, record, note, or keep something in mind, treat that as strong memory intent.
If the remembered content does not fit a structured key, use remembered_note_<short_topic> as the key.

Allowed keys: {allowed}
Sensitivity:
- sensitive: name, phone, email, home_address, or equivalent private identity/contact data
- preference: tools, browser, editor, music player, shell, language, workflow preferences
- profile: occupation, timezone, stable non-sensitive profile facts

Thresholds:
- normal/preference/profile facts require confidence >= {min_confidence}
- sensitive facts require confidence >= {sensitive_min_confidence}

Return only valid JSON:
{{
  "should_write": true or false,
  "rationale": "brief reason",
  "facts": [
    {{
      "key": "preferred_browser",
      "value": "Chrome",
      "category": "preference",
      "sensitivity": "preference",
      "confidence": 0.95
    }}
  ]
}}
"""


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", key.strip().lower()).strip("_")


def _is_allowed_memory_key(key: str) -> bool:
    return key in _ALLOWED_KEYS or key.startswith(_REMEMBERED_NOTE_PREFIX)


def _label_for_key(key: str) -> str:
    if key.startswith(_REMEMBERED_NOTE_PREFIX):
        topic = key[len(_REMEMBERED_NOTE_PREFIX):].replace("_", " ")
        return f"Remembered note {topic}".strip()
    return _KEY_LABELS.get(key, key.replace("_", " ").title())


def _normalize_sensitivity(value: str, key: str) -> str:
    if key in _SENSITIVE_KEYS:
        return "sensitive"
    value = value.strip().lower()
    return value if value in {"sensitive", "preference", "profile"} else "preference"


def _category_for_key(key: str) -> str:
    if key in _SENSITIVE_KEYS:
        return "identity"
    if key.startswith(_REMEMBERED_NOTE_PREFIX):
        return "note"
    if key.startswith("preferred_"):
        return "preference"
    return "profile"


def _source_id(interaction: dict[str, Any]) -> str:
    timestamp = str(interaction.get("timestamp", "")).strip()
    if timestamp:
        return timestamp.replace(" ", "T").replace(":", "")
    return time.strftime("%Y%m%d-%H%M%S")


def _extract_clipboard_pattern_candidates(text: str) -> list[str]:
    candidates = []
    seen = set()
    for raw_line in str(text).splitlines():
        line = _normalize_clipboard_line(raw_line)
        if not line or line in seen:
            continue
        if _is_sensitive_clipboard_line(line):
            continue
        if _looks_like_command(line) or _looks_like_instruction(line):
            candidates.append(line)
            seen.add(line)
    return candidates[:20]


def _normalize_clipboard_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line.strip())
    if line.startswith("$ "):
        line = line[2:].strip()
    return line[:240]


def _is_sensitive_clipboard_line(line: str) -> bool:
    return bool(_SENSITIVE_PATTERN.search(line))


def _looks_like_command(line: str) -> bool:
    if len(line) < 3 or len(line) > 240:
        return False
    first = line.split(" ", 1)[0]
    return first in _COMMAND_PREFIXES


def _looks_like_instruction(line: str) -> bool:
    if len(line) < 4 or len(line) > 120:
        return False
    return any(line.startswith(prefix) for prefix in _INSTRUCTION_PREFIXES)
