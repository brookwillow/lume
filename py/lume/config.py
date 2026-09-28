"""Lume configuration loaded from ~/.lume/config.json."""

import json
from dataclasses import dataclass
from pathlib import Path

from .llm import DEEPSEEK_MODEL

_CONFIG_DIR = Path.home() / ".lume"
_CONFIG_FILE = _CONFIG_DIR / "config.json"


@dataclass
class SkillEvolutionSettings:
    enabled: bool = False
    model: str = DEEPSEEK_MODEL
    min_confidence: float = 0.92
    max_persistent_skills: int = 12


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


@dataclass
class StatsReportSettings:
    enabled: bool = True
    interval_hours: float = 3.0
    recipient_email: str = ""
    report_dir: str = "~/.lume/reports/stats"


@dataclass
class ASRSettings:
    correction_enabled: bool = True
    corrections: dict | None = None
    lexicon_evolution_enabled: bool = True
    model: str = DEEPSEEK_MODEL
    min_confidence: float = 0.96
    min_evidence_count: int = 3
    curation_interval_interactions: int = 30
    curation_interval_hours: float = 6.0
    candidate_ttl_days: int = 30
    lexicon_file: str = "~/.lume/asr_lexicon.json"
    run_dir: str = "~/.lume/asr_lexicon_runs"
    hotword_recall_enabled: bool = True
    hotword_recall_top_k: int = 80
    hotword_recency_days: int = 90
    memory_profile_file: str = "~/.lume/memory/profile.json"
    local_entity_index_file: str = "~/.lume/local_entities.json"
    local_entity_hotword_top_k: int = 20
    local_entity_refresh_hours: float = 6.0


@dataclass
class LumeConfig:
    skill_evolution: SkillEvolutionSettings
    memory: MemorySettings
    stats_report: StatsReportSettings
    asr: ASRSettings
    model: str = DEEPSEEK_MODEL


def load_config(path: Path = _CONFIG_FILE) -> LumeConfig:
    """Load optional user config, falling back to conservative defaults."""
    if not path.exists():
        return LumeConfig(
            model=DEEPSEEK_MODEL,
            skill_evolution=SkillEvolutionSettings(),
            memory=MemorySettings(),
            stats_report=StatsReportSettings(),
            asr=ASRSettings(),
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return LumeConfig(
            model=DEEPSEEK_MODEL,
            skill_evolution=SkillEvolutionSettings(),
            memory=MemorySettings(),
            stats_report=StatsReportSettings(),
            asr=ASRSettings(),
        )

    skill_data = data.get("skill_evolution", {})
    memory_data = data.get("memory", {})
    stats_data = data.get("stats_report", {})
    asr_data = data.get("asr", {})
    return LumeConfig(
        model=DEEPSEEK_MODEL,
        skill_evolution=SkillEvolutionSettings(
            enabled=bool(skill_data.get("enabled", False)),
            model=DEEPSEEK_MODEL,
            min_confidence=float(skill_data.get("min_confidence", 0.92)),
            max_persistent_skills=int(skill_data.get("max_persistent_skills", 12)),
        ),
        memory=MemorySettings(
            enabled=bool(memory_data.get("enabled", False)),
            model=DEEPSEEK_MODEL,
            min_confidence=float(memory_data.get("min_confidence", 0.9)),
            sensitive_min_confidence=float(memory_data.get("sensitive_min_confidence", 0.97)),
            max_persistent_items=int(memory_data.get("max_persistent_items", 20)),
            clipboard_enabled=bool(memory_data.get("clipboard_enabled", False)),
            clipboard_poll_interval_seconds=float(memory_data.get("clipboard_poll_interval_seconds", 2.0)),
            clipboard_max_chars=int(memory_data.get("clipboard_max_chars", 4000)),
            passive_interval_interactions=int(memory_data.get("passive_interval_interactions", 6)),
            passive_recent_interactions=int(memory_data.get("passive_recent_interactions", 12)),
            clipboard_history_max_items=int(memory_data.get("clipboard_history_max_items", 50)),
        ),
        stats_report=StatsReportSettings(
            enabled=bool(stats_data.get("enabled", True)),
            interval_hours=float(stats_data.get("interval_hours", 3.0)),
            recipient_email=str(stats_data.get("recipient_email", "")),
            report_dir=str(stats_data.get("report_dir", "~/.lume/reports/stats")),
        ),
        asr=ASRSettings(
            correction_enabled=bool(asr_data.get("correction_enabled", True)),
            corrections=asr_data.get("corrections") if isinstance(asr_data.get("corrections"), dict) else None,
            lexicon_evolution_enabled=bool(asr_data.get("lexicon_evolution_enabled", True)),
            model=DEEPSEEK_MODEL,
            min_confidence=float(asr_data.get("min_confidence", 0.96)),
            min_evidence_count=int(asr_data.get("min_evidence_count", 3)),
            curation_interval_interactions=int(asr_data.get("curation_interval_interactions", 30)),
            curation_interval_hours=float(asr_data.get("curation_interval_hours", 6.0)),
            candidate_ttl_days=int(asr_data.get("candidate_ttl_days", 30)),
            lexicon_file=str(asr_data.get("lexicon_file", "~/.lume/asr_lexicon.json")),
            run_dir=str(asr_data.get("run_dir", "~/.lume/asr_lexicon_runs")),
            hotword_recall_enabled=bool(asr_data.get("hotword_recall_enabled", True)),
            hotword_recall_top_k=int(asr_data.get("hotword_recall_top_k", 80)),
            hotword_recency_days=int(asr_data.get("hotword_recency_days", 90)),
            memory_profile_file=str(asr_data.get("memory_profile_file", "~/.lume/memory/profile.json")),
            local_entity_index_file=str(asr_data.get("local_entity_index_file", "~/.lume/local_entities.json")),
            local_entity_hotword_top_k=int(asr_data.get("local_entity_hotword_top_k", 20)),
            local_entity_refresh_hours=float(asr_data.get("local_entity_refresh_hours", 6.0)),
        ),
    )
