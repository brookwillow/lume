"""Conservative automatic evolution for ASR correction terms."""

import asyncio
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ASRSettings
from .llm import DEEPSEEK_MODEL, call_deepseek


@dataclass
class ASRLexiconSettings:
    enabled: bool = True
    model: str = DEEPSEEK_MODEL
    min_confidence: float = 0.96
    min_evidence_count: int = 3
    curation_interval_interactions: int = 30
    curation_interval_hours: float = 6.0
    candidate_ttl_days: int = 30
    lexicon_file: Path = Path.home() / ".lume" / "asr_lexicon.json"
    run_dir: Path = Path.home() / ".lume" / "asr_lexicon_runs"
    hotword_recall_enabled: bool = True
    hotword_recall_top_k: int = 80
    hotword_recency_days: int = 90
    memory_profile_file: Path = Path.home() / ".lume" / "memory" / "profile.json"
    local_entity_index_file: Path = Path.home() / ".lume" / "local_entities.json"
    local_entity_hotword_top_k: int = 20
    local_entity_refresh_hours: float = 6.0

    @classmethod
    def from_config(cls, settings: ASRSettings) -> "ASRLexiconSettings":
        return cls(
            enabled=bool(settings.lexicon_evolution_enabled),
            model=settings.model,
            min_confidence=settings.min_confidence,
            min_evidence_count=settings.min_evidence_count,
            curation_interval_interactions=settings.curation_interval_interactions,
            curation_interval_hours=settings.curation_interval_hours,
            candidate_ttl_days=settings.candidate_ttl_days,
            lexicon_file=Path(settings.lexicon_file).expanduser(),
            run_dir=Path(settings.run_dir).expanduser(),
            hotword_recall_enabled=bool(settings.hotword_recall_enabled),
            hotword_recall_top_k=int(settings.hotword_recall_top_k),
            hotword_recency_days=int(settings.hotword_recency_days),
            memory_profile_file=Path(settings.memory_profile_file).expanduser(),
            local_entity_index_file=Path(settings.local_entity_index_file).expanduser(),
            local_entity_hotword_top_k=int(settings.local_entity_hotword_top_k),
            local_entity_refresh_hours=float(settings.local_entity_refresh_hours),
        )


class ASRLexiconEngine:
    _HOTWORD_CACHE_SECONDS = 5.0

    def __init__(self, settings: ASRLexiconSettings | None = None):
        self.settings = settings or ASRLexiconSettings()
        self._hotword_records: list[dict[str, Any]] = []
        self._hotword_snapshots: dict[str, list[dict[str, Any]]] = {}
        self._hotword_cache_expires_at = 0.0

    def observe_completed_interaction(self, interaction: dict[str, Any]) -> None:
        if not self.settings.enabled:
            return
        self.refresh_local_entities_background()
        self._append_interaction(interaction)
        changed = self.capture_candidates(interaction)
        counter = self._increment_counter()
        if not self._curation_due(counter):
            return
        self.curate_background(reason="interaction_interval" if changed else "scheduled")

    def capture_candidates(self, interaction: dict[str, Any]) -> bool:
        candidates = self._load_candidates()
        changed = False
        for variant, canonical in _extract_corrections(interaction):
            key = f"{canonical}\t{variant}"
            row = candidates.get(key, {
                "canonical": canonical,
                "variant": variant,
                "count": 0,
                "evidence": [],
                "first_seen": _now(),
            })
            row["count"] = int(row.get("count", 0)) + 1
            row["last_seen"] = _now()
            evidence = row.setdefault("evidence", [])
            evidence.append(_evidence(interaction))
            row["evidence"] = evidence[-5:]
            candidates[key] = row
            changed = True
        if changed:
            self._write_candidates(self._prune_candidates(candidates))
        return changed

    def curate_background(self, reason: str = "scheduled") -> None:
        if not self.settings.enabled:
            return

        def _runner():
            try:
                asyncio.run(self.curate(reason=reason))
            except Exception as exc:
                self._write_error(exc)

        threading.Thread(target=_runner, daemon=True).start()

    async def curate(self, reason: str = "scheduled") -> Path | None:
        candidates = {
            key: row for key, row in self._load_candidates().items()
            if int(row.get("count", 0)) >= self.settings.min_evidence_count
        }
        recent_interactions = self._load_recent_interactions()
        # Explicit correction pairs are one evidence source, not a gate for
        # task-resolved entities. A successful tool trace can independently
        # establish a canonical contact, place, or product name.
        if not candidates and not any(_has_task_evidence(row) for row in recent_interactions):
            self._save_last_curation()
            return None
        payload = {
            "reason": reason,
            "existing_lexicon": self._load_lexicon(),
            "long_term_memory": self._load_hotword_memory(),
            "candidate_corrections": list(candidates.values()),
            "recent_completed_interactions": recent_interactions,
            "rules": {
                "min_confidence": self.settings.min_confidence,
                "promote_task_resolved_entities_only_when_evidence_is_clear": True,
                "avoid_general_words": True,
                "avoid_conflicts": True,
            },
        }
        prompt = _curator_prompt()
        resp = await call_deepseek(prompt, [{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}], model=self.settings.model)
        data = _parse_json_object(resp.content)
        path = self.apply_review(data)
        self._save_last_curation()
        return path

    def apply_review(self, review: dict[str, Any]) -> Path | None:
        entries = review.get("entries", [])
        if not bool(review.get("should_write", False)) or not isinstance(entries, list):
            return None
        lexicon = self._load_lexicon()
        changed = False
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            confidence = _float(entry.get("confidence"))
            if confidence < self.settings.min_confidence:
                continue
            canonical = _clean_term(str(entry.get("canonical", "")))
            variants = _entry_variants(entry, canonical)
            if not canonical:
                continue
            is_new = canonical not in lexicon
            row = lexicon.setdefault(canonical, {"variants": [], "confidence": confidence, "updated_at": _now()})
            stored_variants = row.setdefault("variants", [])
            for variant in variants:
                if variant not in stored_variants:
                    stored_variants.append(variant)
                    changed = True
            if is_new:
                changed = True
            row["confidence"] = max(_float(row.get("confidence")), confidence)
            row["base_weight"] = max(_float(row.get("base_weight")), _float(entry.get("base_weight", confidence)))
            row["entity_type"] = str(entry.get("entity_type", row.get("entity_type", "other"))).strip() or "other"
            row["contexts"] = _clean_contexts(entry.get("contexts", row.get("contexts", [])))
            row["evidence_count"] = max(int(row.get("evidence_count", 0)), int(entry.get("evidence_count", 0)))
            row["last_seen"] = str(entry.get("last_seen", _now()))
            row["source"] = str(entry.get("source", row.get("source", "curated")))
            row["updated_at"] = _now()
        if not changed:
            return None
        self.settings.lexicon_file.parent.mkdir(parents=True, exist_ok=True)
        self.settings.lexicon_file.write_text(
            json.dumps({"entries": _dedupe_lexicon(lexicon)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._invalidate_hotword_cache()
        return self.settings.lexicon_file

    def recall_hotwords(self, scene: str | None = None, context_terms: list[str] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """Return a bounded list from an in-memory, scene-ranked snapshot."""
        if not self.settings.enabled or not self.settings.hotword_recall_enabled:
            return []
        scene = (scene or "").strip().lower()
        terms = [term.lower() for term in (context_terms or []) if term.strip()]
        self._refresh_hotword_cache_if_needed()
        maximum = max(1, limit or self.settings.hotword_recall_top_k)
        if not terms:
            return [dict(item) for item in self._hotword_snapshots.get(scene, self._hotword_snapshots.get("", []))[:maximum]]
        return self._rank_hotword_records(scene, terms)[:maximum]

    def _refresh_hotword_cache_if_needed(self) -> None:
        if time.monotonic() < self._hotword_cache_expires_at:
            return
        records = self._hotword_records_from_storage()
        self._hotword_records = records
        scenes = ("", "agent", "input", "application_management")
        self._hotword_snapshots = {scene: self._rank_hotword_records(scene, []) for scene in scenes}
        self._hotword_cache_expires_at = time.monotonic() + self._HOTWORD_CACHE_SECONDS

    def _invalidate_hotword_cache(self) -> None:
        self._hotword_cache_expires_at = 0.0

    def _hotword_records_from_storage(self) -> list[dict[str, Any]]:
        records = []
        for canonical, row in self._load_lexicon().items():
            if not isinstance(row, dict) or not canonical.strip() or not _valid_local_entity_hotword(canonical):
                continue
            confidence = _float(row.get("confidence"))
            if confidence <= 0:
                continue
            records.append({
                "text": canonical, "kind": "learned", "confidence": confidence,
                "base_weight": _float(row.get("base_weight")) or confidence,
                "contexts": _clean_contexts(row.get("contexts", [])),
                "last_seen": str(row.get("last_seen", row.get("updated_at", ""))),
                "evidence_count": int(row.get("evidence_count", 0)),
                "entity_type": row.get("entity_type", "other"), "source": row.get("source", "learned_lexicon"),
                "aliases": _entry_variants(row, canonical),
            })
        data = self._load_local_entity_index()
        entities = data.get("entities", []) if isinstance(data, dict) else []
        local_records = []
        for entity in entities if isinstance(entities, list) else []:
            if not isinstance(entity, dict):
                continue
            canonical = str(entity.get("canonical", "")).strip()
            if _valid_local_entity_hotword(canonical):
                local_records.append({
                    "text": canonical, "kind": "local", "base_weight": _float(entity.get("base_weight")),
                    "contexts": _clean_contexts(entity.get("contexts", [])),
                    "entity_type": entity.get("entity_type", "application"),
                    "source": entity.get("source", "local_entity_index"), "aliases": entity.get("aliases", []),
                })
        local_records.sort(key=lambda item: (-item["base_weight"], item["text"]))
        records.extend(local_records[: max(0, self.settings.local_entity_hotword_top_k)])
        return records

    def _rank_hotword_records(self, scene: str, terms: list[str]) -> list[dict[str, Any]]:
        now = time.time()
        results = []
        for record in self._hotword_records:
            contexts = record["contexts"]
            scene_factor = 1.0 if not scene or not contexts else (1.25 if scene in contexts else (0.35 if record["kind"] == "local" else 0.55))
            context_factor = 1.15 if any(term in record["text"].lower() for term in terms) else 1.0
            if record["kind"] == "learned":
                last_seen = _parse_ts(record["last_seen"])
                age_days = max(0.0, (now - last_seen) / 86400) if last_seen else self.settings.hotword_recency_days
                recency = max(0.35, 1.0 - age_days / max(1, self.settings.hotword_recency_days * 2))
                frequency = min(1.0, record["evidence_count"] / 5)
                score = record["confidence"] * record["base_weight"] * scene_factor * recency * (1.0 + 0.15 * frequency) * context_factor
            else:
                score = record["base_weight"] * scene_factor * context_factor
            results.append({"text": record["text"], "weight": round(min(1.0, score), 4), "score": score, "entity_type": record["entity_type"], "source": record["source"], "aliases": record.get("aliases", [])})
        return _merge_hotword_results(results)

    def refresh_local_entities_background(self, reason: str = "scheduled") -> None:
        """Refresh application inventory outside the ASR latency-critical path."""
        if reason == "scheduled" and not self._local_entity_refresh_due():
            return
        def _runner():
            try:
                self.refresh_local_entities(reason=reason)
            except Exception as exc:
                self._write_error(exc)
        threading.Thread(target=_runner, daemon=True).start()

    def refresh_local_entities(self, reason: str = "scheduled") -> Path:
        from .tools import local_application_entities

        payload = {"updated_at": _now(), "reason": reason, "entities": local_application_entities()}
        self.settings.local_entity_index_file.parent.mkdir(parents=True, exist_ok=True)
        self.settings.local_entity_index_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._invalidate_hotword_cache()
        return self.settings.local_entity_index_file

    def _load_local_entity_index(self) -> dict[str, Any]:
        try:
            data = json.loads(self.settings.local_entity_index_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _local_entity_refresh_due(self) -> bool:
        try:
            age_seconds = time.time() - self.settings.local_entity_index_file.stat().st_mtime
        except OSError:
            return True
        return age_seconds >= max(1.0, self.settings.local_entity_refresh_hours) * 3600

    def _curation_due(self, counter: int) -> bool:
        interval = max(1, self.settings.curation_interval_interactions)
        if counter % interval == 0:
            return True
        last = self._load_state().get("last_curation_ts", 0)
        if not last:
            return False
        return time.time() - float(last) >= self.settings.curation_interval_hours * 3600

    def _increment_counter(self) -> int:
        state = self._load_state()
        counter = int(state.get("interaction_count", 0)) + 1
        state["interaction_count"] = counter
        self._write_state(state)
        return counter

    def _save_last_curation(self) -> None:
        state = self._load_state()
        state["last_curation_ts"] = time.time()
        self._write_state(state)

    def _load_candidates(self) -> dict[str, dict]:
        path = self.settings.run_dir / "candidates.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_candidates(self, candidates: dict[str, dict]) -> None:
        self.settings.run_dir.mkdir(parents=True, exist_ok=True)
        (self.settings.run_dir / "candidates.json").write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _prune_candidates(self, candidates: dict[str, dict]) -> dict[str, dict]:
        cutoff = time.time() - self.settings.candidate_ttl_days * 86400
        kept = {}
        for key, row in candidates.items():
            last_seen = _parse_ts(str(row.get("last_seen", "")))
            if last_seen is None or last_seen >= cutoff:
                kept[key] = row
        return kept

    def _load_lexicon(self) -> dict[str, dict]:
        try:
            data = json.loads(self.settings.lexicon_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        entries = data.get("entries", data)
        return entries if isinstance(entries, dict) else {}

    def _load_state(self) -> dict[str, Any]:
        path = self.settings.run_dir / "state.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_state(self, state: dict[str, Any]) -> None:
        self.settings.run_dir.mkdir(parents=True, exist_ok=True)
        (self.settings.run_dir / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_error(self, exc: Exception) -> None:
        self.settings.run_dir.mkdir(parents=True, exist_ok=True)
        (self.settings.run_dir / "last_error.txt").write_text(f"{_now()}\n{type(exc).__name__}: {exc}\n", encoding="utf-8")

    def _append_interaction(self, interaction: dict[str, Any]) -> None:
        self.settings.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.run_dir / "interaction_history.jsonl"
        compact = {"timestamp": interaction.get("timestamp", _now()), "asr_raw_text": interaction.get("asr_raw_text", ""), "asr_text": interaction.get("asr_text", ""), "user_message": interaction.get("user_message", ""), "tool_calls": (interaction.get("tool_calls") or [])[:5], "final_response": interaction.get("final_response", ""), "success": interaction.get("success")}
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(compact, ensure_ascii=False) + "\n")

    def _load_recent_interactions(self) -> list[dict[str, Any]]:
        path = self.settings.run_dir / "interaction_history.jsonl"
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def _load_hotword_memory(self) -> dict[str, Any]:
        """Expose only non-sensitive memory as curator evidence, never directly to ASR."""
        try:
            profile = json.loads(self.settings.memory_profile_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(profile, dict):
            return {}
        return {
            key: value for key, value in profile.items()
            if isinstance(value, dict) and value.get("sensitivity") != "sensitive"
        }


def _extract_corrections(interaction: dict[str, Any]) -> list[tuple[str, str]]:
    text = "\n".join([
        str(interaction.get("asr_raw_text", "")),
        str(interaction.get("asr_text", "")),
        str(interaction.get("user_message", "")),
        str(interaction.get("final_response", "")),
    ])
    patterns = [
        r"不是\s*([A-Za-z0-9_\- ]{2,40}|[\u4e00-\u9fff]{1,12})\s*[，, ]?\s*(?:是|应该是|要写成)\s*([A-Za-z0-9_\- ]{2,40}|[\u4e00-\u9fff]{1,12})",
        r"(?:识别成|识别为|听成)\s*([A-Za-z0-9_\- ]{2,40}|[\u4e00-\u9fff]{1,12})\s*[，, ]?\s*(?:应该是|其实是|改成)\s*([A-Za-z0-9_\- ]{2,40}|[\u4e00-\u9fff]{1,12})",
    ]
    pairs = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            variant = _clean_term(match.group(1))
            canonical = _clean_term(match.group(2))
            if _valid_pair(variant, canonical):
                pairs.append((variant, canonical))
    return pairs


def _valid_pair(variant: str, canonical: str) -> bool:
    if not variant or not canonical or variant == canonical:
        return False
    if len(variant) > 40 or len(canonical) > 40:
        return False
    return True


def _clean_term(value: str) -> str:
    return value.strip(" ，,。！？!?\"'“”‘’")


def _evidence(interaction: dict[str, Any]) -> dict[str, str]:
    return {
        "timestamp": str(interaction.get("timestamp", _now())),
        "asr_raw_text": str(interaction.get("asr_raw_text", ""))[:200],
        "asr_text": str(interaction.get("asr_text", ""))[:200],
        "user_message": str(interaction.get("user_message", ""))[:200],
    }


def _curator_prompt() -> str:
    return """You are a conservative ASR lexicon curator.
Return JSON only:
{
  "should_write": true/false,
  "entries": [
    {"canonical": "望京 SOHO", "variants": ["王晶搜猴"], "entity_type": "place", "contexts": ["navigation"], "confidence": 0.97, "base_weight": 0.97, "evidence_count": 2, "source": "task_execution", "rationale": "..."}
  ],
  "rationale": "..."
}
Promote explicit user corrections, or named entities clearly resolved by successful task execution (for example a navigation destination or contact).
Long-term memory is supplementary evidence only. Never turn sensitive memory into a hotword, and do not promote a memory fact unless it is a named entity useful for speech recognition.
Task evidence may initialize a new term at the confidence justified by its evidence; do not force all new terms to low confidence.
Only use canonical entities that appear in task parameters or other provided evidence. Do not infer, invent, or promote generic natural-language rewrites, one-off uncertainty, secrets, or conflicting mappings.
Use an empty variants list when there is no observed ASR alias."""


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def _dedupe_lexicon(lexicon: dict[str, dict]) -> dict[str, dict]:
    result = {}
    for canonical, row in lexicon.items():
        variants = row.get("variants", []) if isinstance(row, dict) else []
        if not isinstance(variants, list):
            continue
        clean = []
        for variant in variants:
            value = str(variant).strip()
            if value and value != canonical and value not in clean:
                clean.append(value)
        result[canonical] = {**row, "variants": clean}
    return result


def _entry_variants(entry: dict[str, Any], canonical: str) -> list[str]:
    raw = entry.get("variants", entry.get("variant", []))
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return list(dict.fromkeys(value for value in (_clean_term(str(item)) for item in raw) if value and value != canonical))


def _clean_contexts(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(item.strip().lower() for item in map(str, value) if item.strip()))


def _valid_local_entity_hotword(value: str) -> bool:
    if len(value.strip()) < 2 or len(value) > 80:
        return False
    return value.strip().lower() not in {"item", "item one", "item two", "application", "app"}


def _merge_hotword_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate terms while preserving the strongest source and stable order."""
    merged: dict[str, dict[str, Any]] = {}
    for item in results:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        existing = merged.get(text)
        if existing is None or float(item.get("score", 0.0)) > float(existing.get("score", 0.0)):
            merged[text] = item
    return sorted(merged.values(), key=lambda item: (-float(item["score"]), item["text"]))


def _has_task_evidence(interaction: dict[str, Any]) -> bool:
    return bool(interaction.get("tool_calls")) and interaction.get("success") is not False


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _parse_ts(value: str) -> float | None:
    try:
        return time.mktime(time.strptime(value, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return None


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
