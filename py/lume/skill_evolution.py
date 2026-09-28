"""Automatic, conservative skill evolution for Lume.

This module never uses the main execution agent to create skills. It records
completed interactions and asks a separate curator model to decide whether the
trace is worth turning into a persistent skill.
"""

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import SkillEvolutionSettings
from .llm import DEEPSEEK_MODEL, call_deepseek

_LUME_DIR = Path.home() / ".lume"
_DEFAULT_SKILL_DIR = _LUME_DIR / "skills"
_DEFAULT_RUN_DIR = _LUME_DIR / "skill_runs"
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


@dataclass
class SkillEvolutionConfig:
    enabled: bool = False
    model: str = DEEPSEEK_MODEL
    min_confidence: float = 0.92
    max_persistent_skills: int = 12
    skill_dir: Path = _DEFAULT_SKILL_DIR
    run_dir: Path = _DEFAULT_RUN_DIR

    @classmethod
    def from_settings(cls, settings: SkillEvolutionSettings) -> "SkillEvolutionConfig":
        return cls(
            enabled=settings.enabled,
            model=settings.model,
            min_confidence=settings.min_confidence,
            max_persistent_skills=settings.max_persistent_skills,
        )


@dataclass
class SkillReview:
    should_write: bool
    confidence: float
    skill_name: str
    skill_markdown: str
    rationale: str


class SkillEvolutionEngine:
    """Record interactions and conservatively promote reusable traces to skills."""

    def __init__(self, config: SkillEvolutionConfig | None = None):
        self.config = config or SkillEvolutionConfig()

    @staticmethod
    def is_candidate(interaction: dict[str, Any]) -> bool:
        """Hard-filter traces before spending curator tokens."""
        if not interaction.get("final_response"):
            return False
        if interaction.get("success") is False:
            return False

        tool_calls = interaction.get("tool_calls") or []
        if len(tool_calls) <= 1:
            tool = tool_calls[0].get("tool") if tool_calls else ""
            if tool in _SIMPLE_TOOLS:
                return False

        if _has_tool_exploration_signal(tool_calls):
            return True

        combined = "\n".join(
            [
                interaction.get("user_message", ""),
                interaction.get("notes", ""),
                interaction.get("final_response", ""),
                json.dumps(tool_calls, ensure_ascii=False),
            ]
        ).lower()
        if any(marker in combined for marker in _learning_markers()):
            return True

        return False

    @staticmethod
    def should_curate_with_follow_up(
        target: dict[str, Any], follow_up: dict[str, Any]
    ) -> bool:
        """Decide whether a target trace deserves curator review after a follow-up."""
        if not target.get("final_response") or target.get("success") is False:
            return False
        tool_calls = target.get("tool_calls") or []
        if not tool_calls:
            return False
        if len(tool_calls) == 1 and tool_calls[0].get("tool") in _SIMPLE_TOOLS:
            return False
        if SkillEvolutionEngine.is_candidate(target):
            return True
        return _has_success_signal(follow_up)

    def record_interaction(self, interaction: dict[str, Any]) -> Path | None:
        """Persist the raw interaction trace for later audit/debugging."""
        if not self.config.enabled:
            return None
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        run_id = time.strftime("%Y%m%d-%H%M%S")
        path = self.config.run_dir / f"{run_id}.json"
        path.write_text(json.dumps(interaction, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def observe_completed_interaction(self, interaction: dict[str, Any]) -> None:
        """Queue current trace and curate the previous one after a follow-up exists."""
        if not self.config.enabled:
            return

        self.record_interaction(interaction)
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        pending_path = self.config.run_dir / "pending_interaction.json"

        pending = None
        if pending_path.exists():
            try:
                pending = json.loads(pending_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pending = None

        pending_path.write_text(json.dumps(interaction, ensure_ascii=False, indent=2), encoding="utf-8")

        if pending and self.should_curate_with_follow_up(pending, interaction):
            self.curate_with_follow_up_background(pending, interaction)

    async def maybe_curate(self, interaction: dict[str, Any]) -> Path | None:
        """Review a completed interaction and write a skill only when very certain."""
        if not self.config.enabled or not self.is_candidate(interaction):
            return None

        self.record_interaction(interaction)
        review = await self._review_with_curator(interaction)
        return self.write_skill(interaction, review)

    async def curate_with_follow_up(
        self, target: dict[str, Any], follow_up: dict[str, Any]
    ) -> Path | None:
        """Curate a target trace only after seeing the next completed interaction."""
        if not self.config.enabled or not self.should_curate_with_follow_up(target, follow_up):
            return None
        review_trace = {
            "target_interaction": target,
            "next_interaction_for_success_signal": follow_up,
            "curation_rule": (
                "Write only if the next interaction does not continue correcting "
                "the target issue, and it provides acceptance or moves to a new task."
            ),
        }
        review = await self._review_with_curator(review_trace)
        return self.write_skill(target, review)

    def maybe_curate_background(self, interaction: dict[str, Any]) -> None:
        """Run curation asynchronously without blocking the voice assistant."""
        if not self.config.enabled or not self.is_candidate(interaction):
            return

        def _runner():
            try:
                asyncio.run(self.maybe_curate(interaction))
            except Exception as exc:
                self._write_error(interaction, exc)

        import threading

        threading.Thread(target=_runner, daemon=True).start()

    def curate_with_follow_up_background(
        self, target: dict[str, Any], follow_up: dict[str, Any]
    ) -> None:
        if not self.config.enabled or not self.should_curate_with_follow_up(target, follow_up):
            return

        def _runner():
            try:
                asyncio.run(self.curate_with_follow_up(target, follow_up))
            except Exception as exc:
                self._write_error(target, exc)

        import threading

        threading.Thread(target=_runner, daemon=True).start()

    async def _review_with_curator(self, interaction: dict[str, Any]) -> SkillReview:
        prompt = _curator_prompt(self.config.min_confidence)
        messages = [
            {
                "role": "user",
                "content": json.dumps(interaction, ensure_ascii=False, indent=2),
            }
        ]
        resp = await call_deepseek(prompt, messages, model=self.config.model)
        data = _parse_json_object(resp.content)
        return SkillReview(
            should_write=bool(data.get("should_write", False)),
            confidence=float(data.get("confidence", 0.0)),
            skill_name=str(data.get("skill_name", "")).strip(),
            skill_markdown=str(data.get("skill_markdown", "")).strip(),
            rationale=str(data.get("rationale", "")).strip(),
        )

    def write_skill(self, interaction: dict[str, Any], review: SkillReview) -> Path | None:
        if not review.should_write:
            return None
        if review.confidence < self.config.min_confidence:
            return None
        if not review.skill_name or not review.skill_markdown:
            return None

        slug = _slugify(review.skill_name)
        if not slug:
            return None

        path = self.config.skill_dir / slug
        path.mkdir(parents=True, exist_ok=True)
        skill_path = path / "SKILL.md"
        metadata_path = path / "metadata.json"

        skill_path.write_text(review.skill_markdown + "\n", encoding="utf-8")
        metadata = {
            "name": slug,
            "title": review.skill_name,
            "status": "active",
            "confidence": review.confidence,
            "rationale": review.rationale,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "success_count": 1,
            "failure_count": 0,
            "source_user_message": interaction.get("user_message", ""),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self.maintain_all_skills_background(reason=f"new skill written: {slug}")
        return path

    def load_persistent_prompt(self) -> str:
        """Load compact active skill summaries for system-prompt residency."""
        if not self.config.enabled or not self.config.skill_dir.exists():
            return ""

        active: list[tuple[float, str]] = []
        for metadata_path in self.config.skill_dir.glob("*/metadata.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if metadata.get("status") != "active":
                continue
            confidence = float(metadata.get("confidence", 0.0))
            skill_md = metadata_path.with_name("SKILL.md")
            if not skill_md.exists():
                continue
            index_entry = _skill_index_entry(
                skill_md.read_text(encoding="utf-8"),
                metadata,
            )
            active.append((confidence, index_entry))

        active.sort(reverse=True, key=lambda item: item[0])
        selected = [summary for _, summary in active[: self.config.max_persistent_skills]]
        if not selected:
            return ""

        return (
            "\n\nPersistent learned skill index. Treat these as first-priority execution knowledge. "
            "When a user request matches a skill, first call load_skill with the skill name, "
            "then follow the full skill content before using other tools:\n"
            + "\n".join(selected)
        )

    def _write_error(self, interaction: dict[str, Any], exc: Exception) -> None:
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.run_dir / "curator_errors.jsonl"
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(exc),
            "user_message": interaction.get("user_message", ""),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def maintain_all_skills_background(self, reason: str = "skill set changed") -> None:
        """Review the entire skill store after each new write."""
        if not self.config.enabled:
            return

        def _runner():
            try:
                asyncio.run(self.maintain_all_skills(reason=reason))
            except Exception as exc:
                self._write_error({"user_message": f"skill maintenance: {reason}"}, exc)

        import threading

        threading.Thread(target=_runner, daemon=True).start()

    async def maintain_all_skills(self, reason: str = "skill set changed") -> None:
        """Ask the curator to detect duplicates, conflicts, unclear boundaries."""
        if not self.config.enabled or not self.config.skill_dir.exists():
            return

        snapshot = self._skill_snapshot()
        if len(snapshot) < 2:
            return

        prompt = _maintenance_prompt(self.config.min_confidence)
        messages = [
            {
                "role": "user",
                "content": json.dumps(
                    {"reason": reason, "skills": snapshot},
                    ensure_ascii=False,
                    indent=2,
                ),
            }
        ]
        resp = await call_deepseek(prompt, messages, model=self.config.model)
        plan = _parse_json_object(resp.content)
        if float(plan.get("confidence", 0.0)) < self.config.min_confidence:
            return
        self.apply_maintenance_plan(plan)

    def apply_maintenance_plan(self, plan: dict[str, Any]) -> None:
        """Apply conservative curator maintenance actions to local skill metadata/files."""
        actions = plan.get("actions") or []
        for action in actions:
            kind = action.get("action")
            slug = _slugify(str(action.get("skill", "")))
            if not slug:
                continue
            skill_dir = self.config.skill_dir / slug
            metadata_path = skill_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            reason = str(action.get("reason", "")).strip()
            if kind in {"deprecate", "disable", "conflict"}:
                metadata["status"] = {
                    "deprecate": "deprecated",
                    "disable": "disabled",
                    "conflict": "conflict",
                }[kind]
                metadata["maintenance_reason"] = reason
                metadata["last_maintained"] = time.strftime("%Y-%m-%d %H:%M:%S")
                metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            elif kind == "update_metadata":
                allowed = {
                    "title",
                    "confidence",
                    "rationale",
                    "maintenance_reason",
                    "success_count",
                    "failure_count",
                }
                for key, value in (action.get("metadata") or {}).items():
                    if key in allowed:
                        metadata[key] = value
                metadata["last_maintained"] = time.strftime("%Y-%m-%d %H:%M:%S")
                metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            elif kind == "rewrite":
                new_markdown = str(action.get("skill_markdown", "")).strip()
                if not new_markdown:
                    continue
                history_dir = skill_dir / "history"
                history_dir.mkdir(exist_ok=True)
                skill_path = skill_dir / "SKILL.md"
                if skill_path.exists():
                    backup = history_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.md"
                    backup.write_text(skill_path.read_text(encoding="utf-8"), encoding="utf-8")
                skill_path.write_text(new_markdown + "\n", encoding="utf-8")
                metadata["last_maintained"] = time.strftime("%Y-%m-%d %H:%M:%S")
                metadata["maintenance_reason"] = reason
                metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _skill_snapshot(self) -> list[dict[str, Any]]:
        skills: list[dict[str, Any]] = []
        for metadata_path in self.config.skill_dir.glob("*/metadata.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            skill_path = metadata_path.with_name("SKILL.md")
            if not skill_path.exists():
                continue
            skills.append(
                {
                    "name": metadata.get("name", metadata_path.parent.name),
                    "status": metadata.get("status", "active"),
                    "confidence": metadata.get("confidence", 0.0),
                    "title": metadata.get("title", ""),
                    "content": skill_path.read_text(encoding="utf-8")[:4000],
                }
            )
        return skills


def _learning_markers() -> tuple[str, ...]:
    return (
        "error",
        "failed",
        "wrong",
        "retry",
        "not correct",
        "不对",
        "错",
        "失败",
        "还是",
        "没有",
        "不是",
        "重新",
        "纠正",
        "修复",
        "沉淀",
        "保存成skill",
        "保存成 skill",
        "存成skill",
        "存成 skill",
        "没问题",
        "可以了",
        "成功了",
    )


def _success_markers() -> tuple[str, ...]:
    return (
        "可以了",
        "成功了",
        "没问题",
        "对了",
        "这次对",
        "这次成功",
        "有效了",
        "生效了",
        "works",
        "worked",
        "success",
        "correct",
    )


def _has_success_signal(interaction: dict[str, Any]) -> bool:
    text = "\n".join(
        [
            str(interaction.get("asr_text", "")),
            str(interaction.get("user_message", "")),
            str(interaction.get("final_response", "")),
        ]
    ).lower()
    return any(marker in text for marker in _success_markers())


def _has_tool_exploration_signal(tool_calls: list[dict[str, Any]]) -> bool:
    if len(tool_calls) >= 3:
        return True
    if not tool_calls:
        return False
    results = "\n".join(str(call.get("result", "")) for call in tool_calls).lower()
    if any(marker in results for marker in ("error", "exit code", "stderr", "not found", "unable to find")):
        return True
    commands = [json.dumps(call.get("params", {}), ensure_ascii=False) for call in tool_calls]
    return len(tool_calls) >= 2 and len(set(commands)) > 1


def _curator_prompt(min_confidence: float) -> str:
    return f"""You are Lume's independent Skill Curator, not the execution agent.

Decide whether this completed interaction should be turned into a persistent skill.
Be extremely conservative. Write a skill only when all conditions are true:
- The trace shows a reusable execution lesson, not a one-off action.
- There was a failure, wrong direction, retry, or user correction.
- Or the user explicitly asked to save/learn/sediment the reusable method and then confirmed it worked.
- The user does not need to explicitly ask for saving. Repeated attempts followed by later acceptance is enough evidence to review.
- The final approach appears successful.
- The user did not continue correcting the same issue in the available trace.
- It is not a simple direct tool call.
- The lesson would improve future execution quality.

Use user tone only as weak evidence. Prefer not writing when uncertain.
The required confidence threshold is {min_confidence}.

Return only valid JSON:
{{
  "should_write": true or false,
  "confidence": 0.0,
  "skill_name": "short-kebab-case-name",
  "rationale": "brief reason",
  "skill_markdown": "---\\nname: short-kebab-case-name\\ndescription: when to use this skill\\n---\\n\\n# Title\\n\\n## When to Use\\n...\\n\\n## Workflow\\n...\\n\\n## Common Failures\\n...\\n\\n## Verification\\n..."
}}
"""


def _maintenance_prompt(min_confidence: float) -> str:
    return f"""You are Lume's independent Skill Maintainer.

Review the complete local skill set after a new skill was written.
Find only high-confidence maintenance issues:
- duplicate or overlapping skills that should not both stay active
- conflicting instructions
- vague boundaries that make future execution ambiguous
- overly broad skills that should be rewritten with clearer scope
- stale or low-quality skills that should be disabled/deprecated

Be conservative. If unsure, return no actions.
Do not invent new skills. Do not rewrite unless the rewritten skill is clearly better.
Required confidence threshold: {min_confidence}.

Return only valid JSON:
{{
  "confidence": 0.0,
  "rationale": "brief global rationale",
  "actions": [
    {{
      "action": "deprecate|disable|conflict|update_metadata|rewrite",
      "skill": "existing-skill-name",
      "reason": "why this action is safe",
      "metadata": {{}},
      "skill_markdown": "only for rewrite"
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


def _slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return name.strip("-")[:80]


def _compact_skill_summary(markdown: str) -> str:
    lines = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("---"):
            continue
        lines.append(stripped)
        if len("\n".join(lines)) > 900:
            break
    return "\n".join(lines)[:1000]


def _skill_index_entry(markdown: str, metadata: dict[str, Any]) -> str:
    frontmatter = _parse_frontmatter(markdown)
    name = str(frontmatter.get("name") or metadata.get("name") or "").strip()
    description = str(frontmatter.get("description") or metadata.get("rationale") or "").strip()
    confidence = metadata.get("confidence", 0.0)
    if not name:
        name = "unknown-skill"
    if description:
        return f"- {name} (confidence={confidence}): {description}"
    return f"- {name} (confidence={confidence})"


def _parse_frontmatter(markdown: str) -> dict[str, str]:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    data: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data
