import os
import subprocess
import json
import plistlib
import re
import tempfile
import time
from pathlib import Path

from .llm import analyze_image

BLOCKED_COMMANDS = ["rm -rf /", "mkfs", ":(){", "dd if="]
_SKILL_DIR = Path.home() / ".lume" / "skills"
_MEMORY_DIR = Path.home() / ".lume" / "memory"
_ASR_LEXICON_FILE = Path.home() / ".lume" / "asr_lexicon.json"
_APP_SEARCH_ROOTS = (Path("/Applications"), Path("/System/Applications"), Path.home() / "Applications")
_APP_INDEX_CACHE: tuple[float, list[dict]] = (0.0, [])
_APP_INDEX_TTL_SECONDS = 60.0
_SCREENSHOT_DIR = Path(tempfile.gettempdir()) / "lume-screenshots"
_SCREENSHOT_MAX_AGE_SECONDS = 60 * 60
_MAX_ANALYSIS_IMAGE_EDGE = 2560


def execute_tool(tool_name: str, params: dict) -> str:
    try:
        match tool_name:
            case "read_file":
                return _read_file(params.get("path", params.get("file", "")))
            case "write_file":
                return _write_file(params.get("path", params.get("file", "")), params.get("content", ""))
            case "run_command":
                return _run_command(params.get("command", params.get("cmd", "")))
            case "list_dir":
                return _list_dir(params.get("path", params.get("dir", ".")))
            case "search_files":
                return _search_files(params.get("pattern", params.get("query", "")), params.get("path", "."))
            case "load_skill":
                return _load_skill(params.get("name", params.get("skill", "")))
            case "remember_memory":
                return _remember_memory(params)
            case "read_memory":
                return _read_memory()
            case "open_app":
                return _open_app(params.get("name", params.get("app", "")))
            case "manage_application":
                return _manage_application(params)
            case "open_url":
                return _open_url(params.get("url", params.get("link", "")))
            case "applescript":
                return _applescript(params.get("script", params.get("code", "")))
            case "clipboard":
                return _clipboard(params.get("action", "get"), params.get("text"))
            case "insert_text":
                return _insert_text(params.get("text", params.get("content", "")))
            case "read_notes":
                return _read_notes(int(params.get("limit", params.get("count", 5))))
            case "get_selected_text":
                return _get_selected_text()
            case "replace_selected_text":
                return _replace_selected_text(params.get("text", params.get("content", "")))
            case "replace_before_cursor":
                return _replace_before_cursor(
                    params.get("text", params.get("content", "")),
                    params.get("scope", "line"),
                )
            case "system_info":
                return _system_info()
            case "screenshot":
                return _screenshot(params)
            case "set_volume":
                level = params.get("level", params.get("value", params.get("vol", 50)))
                return _set_volume(int(level))
            case "music_control":
                return _music_control(params.get("action", params.get("cmd", "status")))
            case "spotlight_search":
                return _spotlight_search(params.get("query", params.get("keyword", "")))
            case "notification":
                return _notification(params.get("title", "Lume"), params.get("body", params.get("message", "")))
            case "shortcut":
                return _shortcut(params.get("name", ""), params.get("input"))
            case "lark":
                return _lark(params.get("command", params.get("cmd", "")), params.get("cwd"))
            case "notion":
                return _notion(params.get("command", params.get("cmd", "")))
            case "chrome":
                return _chrome(params.get("command", params.get("cmd", "")))
            case "browser":
                return _browser(params)
            case _:
                return f"Unknown tool: {tool_name}"
    except KeyError as e:
        return f"Error: missing parameter {e}"
    except Exception as e:
        return f"Error: {e}"


def _resolve_path(p: str) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _read_file(file_path: str) -> str:
    p = _resolve_path(file_path)
    if not p.exists():
        return f"File not found: {file_path}"
    return p.read_text(encoding="utf-8", errors="replace")


def _write_file(file_path: str, content: str) -> str:
    p = _resolve_path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"File written: {file_path}"


def _run_command(command: str) -> str:
    if any(b in command for b in BLOCKED_COMMANDS):
        return _tool_status("failed", reason="Command blocked for safety.", executed=False)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd(),
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if result.returncode != 0:
            return _tool_status(
                "failed",
                executed=True,
                returncode=result.returncode,
                output=stdout,
                stderr=stderr,
                reason=f"Command exited with code {result.returncode}",
            )
        if stdout:
            return _tool_status("success", executed=True, returncode=0, output=stdout)
        if _is_unknown_ui_command(command):
            return _tool_status(
                "unknown",
                executed=True,
                returncode=0,
                reason="Command exited with code 0 but produced no observable output.",
            )
        return _tool_status(
            "success",
            executed=True,
            returncode=0,
            output="",
            reason="Command completed with no output.",
        )
    except subprocess.TimeoutExpired:
        return _tool_status("failed", reason="Command timed out (30s)", executed=False)


def _list_dir(dir_path: str) -> str:
    p = _resolve_path(dir_path)
    if not p.exists():
        return f"Directory not found: {dir_path}"
    entries = []
    for item in sorted(p.iterdir()):
        name = item.name + ("/" if item.is_dir() else "")
        entries.append(name)
    return "\n".join(entries)


def _search_files(pattern: str, search_path: str) -> str:
    p = _resolve_path(search_path)
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.{py,js,ts,go,rs,java,c,cpp,h,json,yaml,yml,md,txt}",
             pattern, str(p)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip()
        if not output:
            return "No matches found."
        lines = output.split("\n")
        if len(lines) > 50:
            return "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more)"
        return output
    except subprocess.TimeoutExpired:
        return "Search timed out."


def _load_skill(name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "-" for ch in name.strip().lower()).strip("-")
    if not slug:
        return "Missing skill name."
    path = _SKILL_DIR / slug / "SKILL.md"
    if not path.exists():
        return f"Skill not found: {slug}"
    return path.read_text(encoding="utf-8", errors="replace")


def _remember_memory(params: dict) -> str:
    key = _normalize_memory_key(params.get("key", ""))
    value = str(params.get("value", params.get("text", ""))).strip()
    if not key or not value:
        return "Missing memory key or value."
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = _MEMORY_DIR / "profile.json"
    try:
        profile = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        profile = {}
    profile[key] = {
        "value": value,
        "category": str(params.get("category", "note")),
        "sensitivity": str(params.get("sensitivity", "profile")),
        "confidence": float(params.get("confidence", 1.0)),
        "source": "active_memory_tool",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rationale": str(params.get("rationale", "User explicitly asked Lume to remember this.")),
    }
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return "Memory saved"


def _read_memory() -> str:
    path = _MEMORY_DIR / "profile.json"
    if not path.exists():
        return "(no long-term memory)"
    return path.read_text(encoding="utf-8", errors="replace")


def _normalize_memory_key(key: str) -> str:
    key = str(key).strip().lower()
    key = "".join(ch if ch.isalnum() else "_" for ch in key)
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_")[:120]


# ─── macOS Native Tools ───

def _manage_application(params: dict) -> str:
    """Resolve a local app, then perform a bounded macOS app operation."""
    action = str(params.get("action", "resolve")).strip().lower()
    aliases = {"activate": "focus", "close": "quit", "kill": "force_quit", "remove_from_dock": "dock_remove", "add_to_dock": "dock_add"}
    action = aliases.get(action, action)
    supported = {"resolve", "list", "open", "quit", "minimize", "focus", "force_quit", "dock_add", "dock_remove", "uninstall"}
    if action not in supported:
        return _application_status("failed", reason=f"Unsupported application action: {action}")

    query = str(params.get("query", params.get("name", ""))).strip()
    bundle_id = str(params.get("bundle_id", "")).strip()
    candidates = _find_local_applications(query=query, bundle_id=bundle_id)
    if action == "list":
        return _application_status("success", action=action, candidates=candidates[:50])
    if action == "resolve":
        status = "resolved" if len(candidates) == 1 and candidates[0]["score"] >= 0.82 else ("ambiguous" if candidates else "not_found")
        return _application_status(status, action=action, candidates=candidates[:8], query=query)
    if action != "open" and not bundle_id:
        return _application_status("needs_resolution", action=action, candidates=candidates[:8], query=query,
                                   reason="Resolve the application first, then repeat this action with its bundle_id.")
    if not candidates:
        return _application_status("not_found", action=action, query=query, reason="No installed application matched the supplied name or bundle_id.")
    if len(candidates) != 1:
        return _application_status("needs_confirmation", action=action, candidates=candidates[:8], reason="More than one installed application matches; choose a bundle_id first.")

    app = candidates[0]
    if action in {"force_quit", "uninstall"} and not bool(params.get("confirm", False)):
        return _application_status("needs_confirmation", action=action, app=app, reason=f"{action} changes application state; repeat with confirm=true and this bundle_id.")
    try:
        result = _perform_application_action(action, app)
        return _application_status("success", action=action, app=app, **result)
    except Exception as exc:
        return _application_status("failed", action=action, app=app, reason=str(exc))


def _application_index() -> list[dict]:
    global _APP_INDEX_CACHE
    now = time.time()
    cached_at, cached = _APP_INDEX_CACHE
    if now - cached_at < _APP_INDEX_TTL_SECONDS:
        return cached
    apps = []
    seen_paths = set()
    for root in _APP_SEARCH_ROOTS:
        if not root.exists():
            continue
        for bundle in root.rglob("*.app"):
            # Do not index helper apps embedded inside another application.
            if "Contents" in bundle.parts or bundle.is_symlink() or bundle in seen_paths:
                continue
            seen_paths.add(bundle)
            app = _read_application_bundle(bundle)
            if app:
                apps.append(app)
    apps.sort(key=lambda item: (item["name"].lower(), item["bundle_id"]))
    _APP_INDEX_CACHE = (now, apps)
    return apps


def local_application_entities() -> list[dict]:
    """Return non-sensitive, ASR-ready entities derived from local applications."""
    dock_bundle_ids = _dock_bundle_ids()
    running_executables = _running_application_executables()
    entities = []
    for app in _application_index():
        in_dock = app["bundle_id"] in dock_bundle_ids
        running = app.get("executable", "") in running_executables
        weight = 0.60 + (0.18 if in_dock else 0.0) + (0.22 if running else 0.0)
        entities.append({
            "canonical": app["name"], "aliases": app["aliases"], "entity_type": "application",
            "contexts": ["agent", "application_management"], "base_weight": round(min(0.95, weight), 3),
            "source": "local_application_index", "bundle_id": app["bundle_id"], "path": app["path"],
            "in_dock": in_dock, "running": running, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    return entities


def _dock_bundle_ids() -> set[str]:
    plist_path = Path.home() / "Library" / "Preferences" / "com.apple.dock.plist"
    try:
        with plist_path.open("rb") as file:
            dock = plistlib.load(file)
    except (OSError, plistlib.InvalidFileException):
        return set()
    items = dock.get("persistent-apps", []) if isinstance(dock, dict) else []
    return {bundle_id for bundle_id in (_dock_item_bundle_id(item) for item in items) if bundle_id}


def _running_application_executables() -> set[str]:
    try:
        result = subprocess.run(["ps", "-axo", "comm="], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if result.returncode != 0:
        return set()
    return {Path(line.strip()).name for line in result.stdout.splitlines() if line.strip()}


def _read_application_bundle(bundle: Path) -> dict | None:
    info_path = bundle / "Contents" / "Info.plist"
    try:
        with info_path.open("rb") as file:
            info = plistlib.load(file)
    except (OSError, plistlib.InvalidFileException):
        return None
    bundle_id = str(info.get("CFBundleIdentifier", "")).strip()
    name = str(info.get("CFBundleDisplayName") or info.get("CFBundleName") or bundle.stem).strip()
    localized_names = _localized_bundle_names(bundle, name)
    aliases = list(dict.fromkeys([name, bundle.stem, str(info.get("CFBundleName", "")).strip(), *localized_names]))
    aliases.extend(_lexicon_aliases_for_application(aliases))
    return {
        "name": name,
        "localized_names": localized_names,
        "aliases": list(dict.fromkeys(value for value in aliases if value)),
        "bundle_id": bundle_id,
        "path": str(bundle),
        "version": str(info.get("CFBundleShortVersionString") or info.get("CFBundleVersion") or ""),
        "executable": str(info.get("CFBundleExecutable", "")),
    }


def _localized_bundle_names(bundle: Path, fallback: str) -> list[str]:
    names = [fallback]
    resources = bundle / "Contents" / "Resources"
    if not resources.exists():
        return names
    for strings_path in resources.glob("*.lproj/InfoPlist.strings"):
        try:
            text = strings_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r'CFBundleDisplayName\s*=\s*"([^"]+)"', text)
        if match:
            names.append(match.group(1).strip())
    return list(dict.fromkeys(name for name in names if name))


def _lexicon_aliases_for_application(names: list[str]) -> list[str]:
    """Reuse learned ASR aliases only when their canonical text names this app."""
    try:
        payload = json.loads(_ASR_LEXICON_FILE.read_text(encoding="utf-8"))
        entries = payload.get("entries", payload)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(entries, dict):
        return []
    name_keys = {_application_search_key(name) for name in names}
    aliases = []
    for canonical, row in entries.items():
        canonical_key = _application_search_key(canonical)
        if not canonical_key or not any(canonical_key in name or name in canonical_key for name in name_keys):
            continue
        variants = row.get("variants", []) if isinstance(row, dict) else row
        if isinstance(variants, str):
            variants = [variants]
        if isinstance(variants, list):
            aliases.extend(str(value).strip() for value in variants if str(value).strip())
    return aliases


def _find_local_applications(query: str = "", bundle_id: str = "") -> list[dict]:
    query_key = _application_search_key(query)
    scored = []
    for app in _application_index():
        if bundle_id and app["bundle_id"] != bundle_id:
            continue
        if bundle_id:
            score = 1.0
        elif not query_key:
            score = 1.0
        else:
            fields = [app["bundle_id"], *app["aliases"]]
            keys = [_application_search_key(value) for value in fields]
            if query_key in keys:
                score = 1.0
            elif any(query_key in value or value in query_key for value in keys if value):
                score = 0.82
            else:
                score = max((_character_similarity(query_key, value) for value in keys), default=0.0)
            # Do not turn weak character overlap (for example Safari -> Siri)
            # into an executable app candidate. Product aliases and phonetic
            # variants belong in a curated resolver source, not this fallback.
            if score < 0.72:
                continue
        scored.append({**app, "score": round(score, 3)})
    return sorted(scored, key=lambda item: (-item["score"], item["name"].lower()))


def _application_search_key(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value).lower())


def _character_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    common = sum(min(left.count(ch), right.count(ch)) for ch in set(left))
    return 2 * common / (len(left) + len(right))


def _perform_application_action(action: str, app: dict) -> dict:
    path = app["path"]
    bundle_id = app["bundle_id"]
    if action == "open":
        _run_native(["open", path], "open application")
        return {"verified": True}
    if action in {"quit", "focus"}:
        verb = "quit" if action == "quit" else "activate"
        _run_native(["osascript", "-e", f'tell application id "{_escape_applescript(bundle_id)}" to {verb}'], action)
        return {"verified": False}
    if action == "minimize":
        script = (f'tell application "System Events" to tell first process whose bundle identifier is "{_escape_applescript(bundle_id)}" '
                  'to set value of attribute "AXMinimized" of every window to true')
        _run_native(["osascript", "-e", script], "minimize application")
        return {"verified": False}
    if action == "force_quit":
        executable = app.get("executable")
        if not executable:
            raise RuntimeError("Application has no executable name for force quit.")
        result = subprocess.run(["pkill", "-x", executable], capture_output=True, text=True, timeout=10)
        if result.returncode not in {0, 1}:
            raise RuntimeError(result.stderr.strip() or "Force quit failed.")
        return {"verified": result.returncode == 0, "reason": "No matching process was running." if result.returncode == 1 else ""}
    if action in {"dock_add", "dock_remove"}:
        return _update_dock_item(app, present=action == "dock_add")
    if action == "uninstall":
        script = f'tell application "Finder" to delete POSIX file "{_escape_applescript(path)}"'
        _run_native(["osascript", "-e", script], "move application to Trash")
        return {"verified": not Path(path).exists()}
    raise RuntimeError(f"Unhandled application action: {action}")


def _update_dock_item(app: dict, present: bool) -> dict:
    """Edit the Dock preference plist using bundle IDs, never display names."""
    plist_path = Path.home() / "Library" / "Preferences" / "com.apple.dock.plist"
    try:
        with plist_path.open("rb") as file:
            dock = plistlib.load(file)
    except (OSError, plistlib.InvalidFileException) as exc:
        raise RuntimeError(f"Could not read Dock preferences: {exc}") from exc
    items = dock.get("persistent-apps", [])
    if not isinstance(items, list):
        raise RuntimeError("Dock persistent-apps data is invalid.")
    bundle_id = app["bundle_id"]
    remaining = [item for item in items if _dock_item_bundle_id(item) != bundle_id]
    changed = len(remaining) != len(items)
    if present and not changed:
        remaining.append({"tile-data": {"bundle-identifier": bundle_id, "file-label": app["name"], "file-data": {"_CFURLString": Path(app["path"]).as_uri(), "_CFURLStringType": 15}}, "tile-type": "file-tile"})
        changed = True
    if not changed:
        return {"verified": True, "reason": "Dock was already in the requested state."}
    dock["persistent-apps"] = remaining
    with plist_path.open("wb") as file:
        plistlib.dump(dock, file, fmt=plistlib.FMT_BINARY)
    _run_native(["killall", "Dock"], "reload Dock")
    return {"verified": True}


def _dock_item_bundle_id(item: object) -> str:
    if not isinstance(item, dict):
        return ""
    tile_data = item.get("tile-data", {})
    return str(tile_data.get("bundle-identifier", "")) if isinstance(tile_data, dict) else ""


def _run_native(command: list[str], operation: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"Could not {operation}.")


def _escape_applescript(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _application_status(status: str, **data) -> str:
    return json.dumps({"status": status, **data}, ensure_ascii=False)

def _open_app(name: str) -> str:
    result = subprocess.run(["open", "-a", name], capture_output=True, text=True)
    if result.returncode != 0:
        return f"Failed to open '{name}': {result.stderr.strip()}"
    return f"Opened {name}"


def _open_url(url: str) -> str:
    result = subprocess.run(["open", url], capture_output=True, text=True)
    if result.returncode != 0:
        return f"Failed to open URL: {result.stderr.strip()}"
    return f"Opened {url}"


def _applescript(script: str) -> str:
    """Run AppleScript - the most powerful macOS automation tool."""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            return _tool_status(
                "failed",
                executed=True,
                returncode=result.returncode,
                stderr=result.stderr.strip(),
                reason="AppleScript returned an error.",
            )
        if output:
            return _tool_status("success", executed=True, returncode=0, output=output)
        return _tool_status(
            "unknown",
            executed=True,
            returncode=0,
            reason="AppleScript exited with code 0 but produced no observable output.",
        )
    except subprocess.TimeoutExpired:
        return _tool_status("failed", reason="AppleScript timed out (15s)", executed=False)


def _is_unknown_ui_command(command: str) -> bool:
    lowered = command.lower()
    return (
        "osascript" in lowered
        or "system events" in lowered
        or "keystroke" in lowered
        or "key code" in lowered
        or lowered.strip().startswith("open ")
    )


def _tool_status(
    status: str,
    *,
    executed: bool,
    returncode: int | None = None,
    output: str = "",
    stderr: str = "",
    reason: str = "",
) -> str:
    return json.dumps(
        {
            "status": status,
            "executed": executed,
            "returncode": returncode,
            "output": output,
            "stderr": stderr,
            "reason": reason,
        },
        ensure_ascii=False,
    )


def _clipboard(action: str, text: str = None) -> str:
    if action == "set" and text is not None:
        subprocess.run(["pbcopy"], input=text, text=True)
        return "Clipboard set"
    else:
        result = subprocess.run(["pbpaste"], capture_output=True, text=True)
        content = result.stdout
        if len(content) > 2000:
            return content[:2000] + f"\n... ({len(content)} chars total)"
        return content or "(clipboard empty)"


def _insert_text(text: str) -> str:
    """Insert text at the current cursor using paste, preserving clipboard when possible."""
    if not text:
        return "No text to insert."

    old_clipboard = subprocess.run(
        ["pbpaste"],
        capture_output=True,
        text=True,
    ).stdout

    subprocess.run(["pbcopy"], input=text, text=True)
    paste = subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to keystroke "v" using command down'],
        capture_output=True,
        text=True,
        timeout=5,
    )
    try:
        subprocess.run(["pbcopy"], input=old_clipboard, text=True)
    except Exception:
        pass

    if paste.returncode != 0:
        return f"Insert text error: {paste.stderr.strip()}"
    return "Inserted text"


def _copy_selection() -> tuple[str, str]:
    old_clipboard = subprocess.run(
        ["pbpaste"],
        capture_output=True,
        text=True,
    ).stdout
    subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to keystroke "c" using command down'],
        capture_output=True,
        text=True,
        timeout=5,
    )
    selected = subprocess.run(
        ["pbpaste"],
        capture_output=True,
        text=True,
    ).stdout
    try:
        subprocess.run(["pbcopy"], input=old_clipboard, text=True)
    except Exception:
        pass
    return selected, old_clipboard


def _get_selected_text() -> str:
    selected, _ = _copy_selection()
    return selected or "(no selected text)"


def _replace_selected_text(text: str) -> str:
    if not text:
        return "No text to replace with."
    return _insert_text(text).replace("Inserted text", "Replaced selected text")


def _select_before_cursor(scope: str) -> str:
    if scope == "word":
        script = 'tell application "System Events" to key code 123 using {option down, shift down}'
    else:
        script = 'tell application "System Events" to key code 123 using {command down, shift down}'
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        return f"Select text error: {result.stderr.strip()}"
    return "(ok)"


def _replace_before_cursor(text: str, scope: str = "line") -> str:
    if not text:
        return "No text to replace with."
    selected = _select_before_cursor(scope)
    if selected.startswith("Select text error:"):
        return selected
    result = _insert_text(text)
    if result.startswith("Insert text error:"):
        return result
    return "Replaced text before cursor"


def _read_notes(limit: int = 5) -> str:
    """Read recent Apple Notes bodies without modifying clipboard or app state."""
    limit = max(1, min(20, int(limit)))
    script = f'''
tell application "Notes"
    set outputText to ""
    set noteList to notes
    set noteCount to count of noteList
    set maxItems to {limit}
    if noteCount < maxItems then set maxItems to noteCount
    repeat with i from 1 to maxItems
        set noteBody to body of note i
        set outputText to outputText & noteBody
        if i is not maxItems then set outputText to outputText & linefeed & "---NOTE---" & linefeed
    end repeat
    return outputText
end tell
'''
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return f"Notes error: {result.stderr.strip()}"

    parts = result.stdout.split("---NOTE---")
    formatted = []
    for i, part in enumerate(parts, start=1):
        text = part.strip()
        if text:
            formatted.append(f"[Note {i}]\n{text}")
    return "\n\n".join(formatted) or "(no notes)"


def _system_info() -> str:
    """Get system status: battery, wifi, volume, dark mode, frontmost app."""
    parts = []

    # Battery
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True)
    for line in batt.stdout.split("\n"):
        if "%" in line:
            parts.append(f"Battery: {line.strip()}")
            break

    # WiFi
    wifi = subprocess.run(
        ["networksetup", "-getairportnetwork", "en0"],
        capture_output=True, text=True
    )
    parts.append(wifi.stdout.strip())

    # Volume
    vol = subprocess.run(
        ["osascript", "-e", "output volume of (get volume settings)"],
        capture_output=True, text=True
    )
    muted = subprocess.run(
        ["osascript", "-e", "output muted of (get volume settings)"],
        capture_output=True, text=True
    )
    parts.append(f"Volume: {vol.stdout.strip()}% (muted: {muted.stdout.strip()})")

    # Dark mode
    dm = subprocess.run(
        ["defaults", "read", "-g", "AppleInterfaceStyle"],
        capture_output=True, text=True
    )
    mode = "Dark" if dm.returncode == 0 else "Light"
    parts.append(f"Appearance: {mode}")

    # Frontmost app
    front = subprocess.run(
        ["osascript", "-e", 'tell app "System Events" to get name of first process whose frontmost is true'],
        capture_output=True, text=True
    )
    parts.append(f"Frontmost: {front.stdout.strip()}")

    return "\n".join(parts)


def _set_volume(level: int) -> str:
    level = max(0, min(100, int(level)))
    subprocess.run(["osascript", "-e", f"set volume output volume {level}"])
    return f"Volume set to {level}%"


def _music_control(action: str) -> str:
    """Control Music.app: play, pause, next, prev, status."""
    if action == "status":
        script = '''
tell app "Music"
    if player state is playing then
        set t to name of current track
        set a to artist of current track
        return "Playing: " & t & " - " & a
    else
        return "Paused/Stopped"
    end if
end tell'''
    elif action in ("play", "pause", "next", "prev"):
        cmd = {"play": "play", "pause": "pause", "next": "next track", "prev": "previous track"}[action]
        script = f'tell app "Music" to {cmd}'
    else:
        return f"Unknown music action: {action}. Use: play, pause, next, prev, status"

    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=5)
    if result.returncode != 0:
        return f"Music error: {result.stderr.strip()}"
    return result.stdout.strip() or f"Music: {action}"


def _spotlight_search(query: str) -> str:
    """Search files via Spotlight (mdfind)."""
    query = str(query).strip()
    if not query:
        return "Missing Spotlight query."
    try:
        result = subprocess.run(
            # mdfind has no `-limit` option on macOS. Limit its output after
            # the query instead of turning every search into a usage error.
            ["mdfind", query],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return f"Spotlight error: {result.stderr.strip()}"
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        output = "\n".join(lines[:20])
        if not output:
            return "No files found."
        if len(lines) > 20:
            output += f"\n... ({len(lines) - 20} more)"
        return output
    except subprocess.TimeoutExpired:
        return "Search timed out."


def _notification(title: str, body: str) -> str:
    script = f'display notification "{body}" with title "{title}"'
    subprocess.run(["osascript", "-e", script], capture_output=True)
    return "Notification sent"


def _screenshot(params: dict) -> str:
    """Capture the current display, optionally asking DeepSeek to inspect it."""
    display = str(params.get("display", "main")).strip().lower()
    mode = str(params.get("mode", "image")).strip().lower()
    if display not in {"main", "all"}:
        return _tool_status("failed", reason="display must be 'main' or 'all'.", executed=False)
    if mode not in {"image", "analyze"}:
        return _tool_status("failed", reason="mode must be 'image' or 'analyze'.", executed=False)

    _cleanup_old_screenshots()
    _SCREENSHOT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = _SCREENSHOT_DIR / f"screen-{int(time.time() * 1000)}.png"
    command = ["screencapture", "-x", "-t", "png"]
    if display == "main":
        command.append("-m")
    command.append(str(path))
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return _tool_status("failed", reason="Screenshot timed out.", executed=False)
    if result.returncode != 0 or not path.exists() or path.stat().st_size == 0:
        path.unlink(missing_ok=True)
        return _tool_status("failed", reason=(result.stderr or "macOS did not produce a screenshot.").strip(), executed=True, returncode=result.returncode)

    path.chmod(0o600)
    response = {
        "status": "success",
        "path": str(path),
        "directory": str(path.parent),
        "filename": path.name,
        "display": display,
        "bytes": path.stat().st_size,
        "mode": mode,
    }
    if mode == "image":
        return json.dumps(response, ensure_ascii=False)
    analysis_path, analysis_image = _prepare_screenshot_for_analysis(path)
    response["analysis_image"] = analysis_image
    try:
        response["analysis"] = analyze_image(
            analysis_path,
            str(params.get("prompt") or "Describe the screenshot faithfully in Chinese. Extract visible text, identify the active application and important controls, and do not infer information that is not visible."),
        )
        return json.dumps(response, ensure_ascii=False)
    except Exception as exc:
        return _tool_status("failed", reason=f"Screenshot captured but DeepSeek analysis failed: {exc}", executed=True)
    finally:
        if analysis_path != path:
            analysis_path.unlink(missing_ok=True)


def _prepare_screenshot_for_analysis(path: Path) -> tuple[Path, dict]:
    """Downscale only the image sent to the model; preserve the original PNG."""
    width, height = _image_dimensions(path)
    metadata = {"original_width": width, "original_height": height}
    if not width or not height or max(width, height) <= _MAX_ANALYSIS_IMAGE_EDGE:
        metadata.update({"uploaded_width": width, "uploaded_height": height, "resized": False, "format": "png"})
        return path, metadata

    output = path.with_name(f"{path.stem}-analysis.jpg")
    try:
        result = subprocess.run(
            ["sips", "-s", "format", "jpeg", "-s", "formatOptions", "85", "-Z", str(_MAX_ANALYSIS_IMAGE_EDGE), str(path), "--out", str(output)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        resized_width, resized_height = _image_dimensions(output)
        if result.returncode == 0 and output.exists() and resized_width and resized_height:
            metadata.update({"uploaded_width": resized_width, "uploaded_height": resized_height, "resized": True, "format": "jpeg"})
            return output, metadata
    except (OSError, subprocess.TimeoutExpired):
        pass
    output.unlink(missing_ok=True)
    metadata.update({"uploaded_width": width, "uploaded_height": height, "resized": False, "format": "png", "resize_error": "Using original image because resize failed."})
    return path, metadata


def _image_dimensions(path: Path) -> tuple[int, int]:
    try:
        result = subprocess.run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return 0, 0
        width_match = re.search(r"pixelWidth:\s*(\d+)", result.stdout)
        height_match = re.search(r"pixelHeight:\s*(\d+)", result.stdout)
        return (int(width_match.group(1)), int(height_match.group(1))) if width_match and height_match else (0, 0)
    except (OSError, subprocess.TimeoutExpired):
        return 0, 0


def _cleanup_old_screenshots() -> None:
    if not _SCREENSHOT_DIR.exists():
        return
    cutoff = time.time() - _SCREENSHOT_MAX_AGE_SECONDS
    for path in _SCREENSHOT_DIR.glob("screen-*.png"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def _shortcut(name: str, input_text: str = None) -> str:
    """Run a macOS Shortcut by name."""
    try:
        cmd = ["shortcuts", "run", name]
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return f"Shortcut error: {result.stderr.strip()}"
        return result.stdout.strip() or f"Shortcut '{name}' executed"
    except subprocess.TimeoutExpired:
        return "Shortcut timed out (30s)"


def _lark(command: str, cwd: str | None = None) -> str:
    """Run lark-cli command for Feishu/Lark operations."""
    try:
        result = subprocess.run(
            f"lark-cli {command}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=cwd,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            return f"lark-cli error: {err or output}"
        if len(output) > 3000:
            return output[:3000] + f"\n... (truncated, {len(output)} chars total)"
        return output or "(ok)"
    except subprocess.TimeoutExpired:
        return "lark-cli timed out (30s)"


def _notion(command: str) -> str:
    """Run ntn (Notion CLI) command."""
    try:
        result = subprocess.run(
            f"ntn {command}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "PATH": os.environ.get("PATH", "") + ":/Users/wangjie/.script/ntn-cli"},
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            return f"ntn error: {err or output}"
        if len(output) > 3000:
            return output[:3000] + f"\n... (truncated, {len(output)} chars total)"
        return output or "(ok)"
    except subprocess.TimeoutExpired:
        return "ntn timed out (30s)"


def _chrome(command: str) -> str:
    """Run chrome-cli command to control Google Chrome browser."""
    try:
        result = subprocess.run(
            f"chrome-cli {command}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            return f"chrome-cli error: {err or output}"
        if len(output) > 5000:
            return output[:5000] + f"\n... (truncated, {len(output)} chars total)"
        if output:
            return output
        if str(command).lstrip().startswith("execute "):
            return _tool_status(
                "unknown",
                executed=True,
                returncode=0,
                reason=(
                    "Chrome executed the script but returned no observable value. "
                    "Use JavaScript that returns a concise string; ensure Chrome allows "
                    "JavaScript from Apple Events."
                ),
            )
        return _tool_status(
            "unknown",
            executed=True,
            returncode=0,
            reason="Chrome command completed but returned no observable output.",
        )
    except subprocess.TimeoutExpired:
        return "chrome-cli timed out (15s)"


def _browser(params: dict) -> str:
    """Perform predictable DOM actions through chrome-cli without shell interpolation."""
    action = str(params.get("action", "")).strip().lower()
    selector = str(params.get("selector", "")).strip()
    text = str(params.get("text", "")).strip()
    index = max(0, int(params.get("index", 0)))

    if action == "inspect":
        limit = min(max(1, int(params.get("limit", 20))), 50)
        script = f"""(() => {{ const visible = el => !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length); const items = [...document.querySelectorAll('button,a,input,textarea,select,[role="button"]')].filter(visible).slice(0, {limit}).map((el, index) => ({{index, tag: el.tagName.toLowerCase(), type: el.type || '', text: (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().slice(0, 120), id: el.id || '', name: el.getAttribute('name') || '', ariaLabel: el.getAttribute('aria-label') || ''}})); return JSON.stringify({{status: 'success', action: 'inspect', url: location.href, items}}); }})()"""
    elif action == "click":
        if not selector and not text:
            return _tool_status("failed", reason="browser click requires selector or text.", executed=False)
        script = _browser_click_script(selector, text, index)
    elif action == "fill":
        value = str(params.get("value", params.get("input", "")))
        if not selector:
            return _tool_status("failed", reason="browser fill requires selector.", executed=False)
        script = _browser_fill_script(selector, value)
    elif action == "read":
        max_chars = min(max(1, int(params.get("max_chars", 2000))), 5000)
        script = _browser_read_script(selector or "body", max_chars)
    elif action == "select":
        value = str(params.get("value", ""))
        if not selector or not (value or text):
            return _tool_status("failed", reason="browser select requires selector and value or text.", executed=False)
        script = _browser_select_script(selector, value, text)
    elif action == "scroll":
        direction = str(params.get("direction", "down")).lower()
        amount = min(max(100, int(params.get("amount", 600))), 3000)
        delta = -amount if direction == "up" else amount
        script = f"window.scrollBy({{top: {delta}, behavior: 'smooth'}}); JSON.stringify({{status: 'success', action: 'scroll'}})"
    else:
        return _tool_status("failed", reason="Unsupported browser action. Use inspect, click, fill, read, select, or scroll.", executed=False)
    return _browser_execute(script)


def _browser_execute(script: str) -> str:
    try:
        # chrome-cli itself waits up to 15 seconds for Chrome to launch; leave
        # it a small margin so its actionable diagnostic reaches the user.
        result = subprocess.run(["chrome-cli", "execute", script], capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return _tool_status("failed", reason="browser action timed out (20s).", executed=False)
    output = (result.stdout or "").strip()
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        reason = "chrome-cli execute failed."
        if "Chrome did not start" in output or (not stderr and not output):
            reason = "Chrome is unavailable or has no active tab. Start Google Chrome, then try again."
        return _tool_status("failed", executed=True, returncode=result.returncode, output=output, stderr=stderr, reason=reason)
    if not output:
        return _tool_status(
            "unknown",
            executed=True,
            returncode=0,
            reason=(
                "Browser action returned no observable result. In Google Chrome, enable "
                "View > Developer > Allow JavaScript from Apple Events, then try again."
            ),
        )
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return _tool_status("unknown", executed=True, returncode=0, output=output, reason="Browser returned non-JSON output.")
    if not isinstance(data, dict) or "status" not in data:
        return _tool_status("unknown", executed=True, returncode=0, output=output, reason="Browser returned an invalid result.")
    return json.dumps(data, ensure_ascii=False)


def _browser_click_script(selector: str, text: str, index: int) -> str:
    selector_json, text_json = json.dumps(selector), json.dumps(text)
    return f"""(() => {{ const visible = el => !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length); const selector = {selector_json}, text = {text_json}; let candidates = selector ? [...document.querySelectorAll(selector)] : [...document.querySelectorAll('button,a,input[type="button"],input[type="submit"],[role="button"]')].filter(el => (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().includes(text)); const el = candidates.filter(visible)[{index}]; if (!el) return JSON.stringify({{status: 'not_found', action: 'click', selector, text}}); el.scrollIntoView({{block: 'center', inline: 'center'}}); el.click(); return JSON.stringify({{status: 'success', action: 'click', tag: el.tagName.toLowerCase(), text: (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().slice(0, 120)}}); }})()"""


def _browser_fill_script(selector: str, value: str) -> str:
    selector_json, value_json = json.dumps(selector), json.dumps(value)
    return f"""(() => {{ const el = document.querySelector({selector_json}); if (!el) return JSON.stringify({{status: 'not_found', action: 'fill', selector: {selector_json}}}); if (!['INPUT', 'TEXTAREA'].includes(el.tagName) && !el.isContentEditable) return JSON.stringify({{status: 'failed', action: 'fill', reason: 'Target is not an input field.'}}); const value = {value_json}; if (el.isContentEditable) el.textContent = value; else {{ const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), 'value')?.set; setter ? setter.call(el, value) : el.value = value; }} el.dispatchEvent(new Event('input', {{bubbles: true}})); el.dispatchEvent(new Event('change', {{bubbles: true}})); return JSON.stringify({{status: 'success', action: 'fill', selector: {selector_json}}}); }})()"""


def _browser_read_script(selector: str, max_chars: int) -> str:
    selector_json = json.dumps(selector)
    return f"""(() => {{ const el = document.querySelector({selector_json}); if (!el) return JSON.stringify({{status: 'not_found', action: 'read', selector: {selector_json}}}); return JSON.stringify({{status: 'success', action: 'read', text: (el.innerText || el.textContent || '').trim().slice(0, {max_chars})}}); }})()"""


def _browser_select_script(selector: str, value: str, text: str) -> str:
    selector_json, value_json, text_json = json.dumps(selector), json.dumps(value), json.dumps(text)
    return f"""(() => {{ const el = document.querySelector({selector_json}); if (!el) return JSON.stringify({{status: 'not_found', action: 'select', selector: {selector_json}}}); if (el.tagName !== 'SELECT') return JSON.stringify({{status: 'failed', action: 'select', reason: 'Target is not a select element.'}}); const option = [...el.options].find(o => o.value === {value_json} || o.text.trim() === {text_json}); if (!option) return JSON.stringify({{status: 'not_found', action: 'select', selector: {selector_json}, value: {value_json}, text: {text_json}}}); el.value = option.value; el.dispatchEvent(new Event('input', {{bubbles: true}})); el.dispatchEvent(new Event('change', {{bubbles: true}})); return JSON.stringify({{status: 'success', action: 'select', value: option.value, text: option.text.trim()}}); }})()"""
