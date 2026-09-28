"""Lightweight ASR text correction for common product and developer terms."""

import json
import re
from pathlib import Path

from .config import ASRSettings

DEFAULT_TERM_CORRECTIONS = {
    "skill": ["scale", "skills", "skeel", "skil", "skilled", "斯kill", "思skill", "斯扣", "四扣", "丝扣", "司扣", "斯口", "思口", "死扣", "s扣"],
    "Claude": ["claude", "克劳德", "克劳的", "克劳得", "cloud", "Cloud", "Claud", "克洛德"],
    "Codex": ["codex", "code x", "codeex", "Code X", "扣代克斯", "扣德克斯", "科德克斯"],
    "DeepSeek": ["deepseek", "deep seek", "deep sick", "Deep Sick", "迪普西克", "低普西克"],
    "ChatGPT": ["chatgpt", "chat gpt", "Chat G P T", "恰GPT", "查GPT"],
    "GitHub": ["github", "git hub", "Git Hub"],
    "Python": ["python", "派森", "派送"],
    "JavaScript": ["javascript", "java script", "Java Script", "加瓦script"],
    "AppleScript": ["applescript", "apple script", "Apple Script"],
}


def normalize_asr_text(text: str, settings: ASRSettings | None = None) -> str:
    """Normalize common ASR mistakes for product and developer terms."""
    normalized, _ = normalize_asr_text_with_trace(text, settings)
    return normalized


def normalize_asr_text_with_trace(text: str, settings: ASRSettings | None = None) -> tuple[str, list[dict[str, str | int]]]:
    """Normalize ASR text and return each replacement actually applied."""
    settings = settings or ASRSettings()
    if not getattr(settings, "correction_enabled", True):
        return text, []

    corrections = merged_corrections(
        getattr(settings, "corrections", None),
        Path(str(getattr(settings, "lexicon_file", "~/.lume/asr_lexicon.json"))).expanduser(),
    )
    normalized = text
    applied = []
    for canonical, variants in corrections.items():
        for variant in variants:
            normalized, count = _replace_term_variant_with_count(normalized, variant, canonical)
            if count:
                applied.append({"variant": variant, "canonical": canonical, "count": count})
    return normalized, applied


def normalize_hotword_aliases(text: str, hotwords: list[dict]) -> tuple[str, list[dict[str, str | int]]]:
    """Apply only safe alias corrections from the hotwords used for this turn.

    Exact aliases work for any language. One-edit fuzzy correction is limited
    to unique ASCII tokens of five or more characters (``chrom`` → ``Chrome``)
    so Chinese speech and short product names are never guessed here.
    """
    normalized = text
    applied: list[dict[str, str | int]] = []
    exact_pairs = []
    for hotword in hotwords:
        canonical = str(hotword.get("text", "")).strip()
        aliases = hotword.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        if not canonical or not isinstance(aliases, list):
            continue
        for alias in aliases:
            alias = str(alias).strip()
            if alias and alias != canonical:
                exact_pairs.append((alias, canonical))
    for alias, canonical in exact_pairs:
        normalized, count = _replace_term_variant_with_count(normalized, alias, canonical)
        if count:
            applied.append({"variant": alias, "canonical": canonical, "count": count})

    fuzzy_candidates: dict[str, set[str]] = {}
    complete_phrases = set()
    for hotword in hotwords:
        canonical = str(hotword.get("text", "")).strip()
        aliases = hotword.get("aliases", [])
        values = [canonical, *(aliases if isinstance(aliases, list) else [])]
        for value in values:
            value = str(value).strip()
            if len(re.findall(r"[A-Za-z][A-Za-z0-9_-]*", value)) > 1:
                complete_phrases.add(value.lower())
            words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", str(value))
            if words:
                word = words[-1].lower()
                if len(word) >= 5:
                    fuzzy_candidates.setdefault(word, set()).add(canonical)

    def replace_token(match: re.Match) -> str:
        token = match.group(0)
        lowered = token.lower()
        # Do not replace the last word inside an already-correct multi-word
        # application name ("Google Chrome" must not become "Google Google Chrome").
        if any(phrase in normalized.lower() and re.search(rf"\b{re.escape(lowered)}\b", phrase) for phrase in complete_phrases):
            return token
        matches = {
            canonical for candidate, canonicals in fuzzy_candidates.items()
            if _ascii_edit_distance_one(lowered, candidate)
            for canonical in canonicals
        }
        if len(matches) != 1:
            return token
        canonical = matches.pop()
        if lowered == canonical.lower():
            return token
        applied.append({"variant": token, "canonical": canonical, "count": 1})
        return canonical

    return re.sub(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_-]*(?![A-Za-z0-9_])", replace_token, normalized), applied


def merged_corrections(custom: dict | None, lexicon_file: Path | None = None) -> dict[str, list[str]]:
    merged = {key: list(values) for key, values in DEFAULT_TERM_CORRECTIONS.items()}
    for key, values in _load_persistent_lexicon(lexicon_file).items():
        merged.setdefault(key, [])
        for value in values:
            if value not in merged[key]:
                merged[key].append(value)
    if not custom:
        return merged
    for canonical, variants in custom.items():
        if isinstance(variants, str):
            values = [variants]
        elif isinstance(variants, list):
            values = [str(v) for v in variants if str(v).strip()]
        else:
            continue
        key = str(canonical).strip()
        if not key:
            continue
        merged.setdefault(key, [])
        for value in values:
            if value not in merged[key]:
                merged[key].append(value)
    return merged


def _replace_term_variant(text: str, variant: str, canonical: str) -> str:
    return _replace_term_variant_with_count(text, variant, canonical)[0]


def _replace_term_variant_with_count(text: str, variant: str, canonical: str) -> tuple[str, int]:
    variant = variant.strip()
    if not variant or variant == canonical:
        return text, 0
    if _is_ascii_phrase(variant):
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(variant)}(?![A-Za-z0-9_])", re.IGNORECASE)
        return pattern.subn(canonical, text)
    count = text.count(variant)
    return text.replace(variant, canonical), count


def _is_ascii_phrase(value: str) -> bool:
    return all(ord(ch) < 128 for ch in value)


def _ascii_edit_distance_one(left: str, right: str) -> bool:
    """True for an exact string or a Levenshtein distance of one."""
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) == 1
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    index = 0
    skipped = False
    for char in longer:
        if index < len(shorter) and char == shorter[index]:
            index += 1
        elif skipped:
            return False
        else:
            skipped = True
    return True


def _load_persistent_lexicon(path: Path | None = None) -> dict[str, list[str]]:
    path = path or (Path.home() / ".lume" / "asr_lexicon.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = data.get("entries", data)
    if not isinstance(entries, dict):
        return {}
    result = {}
    for canonical, row in entries.items():
        if isinstance(row, dict):
            variants = row.get("variants", [])
        else:
            variants = row
        if isinstance(variants, str):
            variants = [variants]
        if not isinstance(variants, list):
            continue
        values = [str(v).strip() for v in variants if str(v).strip()]
        if values:
            result[str(canonical).strip()] = values
    return result
