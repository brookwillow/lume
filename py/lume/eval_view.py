#!/usr/bin/env python3
"""Lume eval log viewer — analyze interaction logs.

Usage:
    python -m lume.eval_view              # summary of all interactions
    python -m lume.eval_view --last 10    # last 10 interactions
    python -m lume.eval_view --detail     # full detail view
    python -m lume.eval_view --tools      # tool usage stats
    python -m lume.eval_view --errors     # only failed interactions
    python -m lume.eval_view --annotate   # interactively mark success/fail
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_LOG_FILE = Path.home() / ".lume" / "eval_log.jsonl"


def load_logs() -> list[dict]:
    if not _LOG_FILE.exists():
        print(f"No log file found at {_LOG_FILE}")
        sys.exit(1)
    logs = []
    for line in _LOG_FILE.read_text().splitlines():
        if line.strip():
            logs.append(json.loads(line))
    return logs


def show_summary(logs: list[dict]):
    total = len(logs)
    errors = sum(1 for l in logs if l.get("success") is False)
    annotated = sum(1 for l in logs if l.get("success") is not None)
    with_tools = sum(1 for l in logs if l.get("tool_calls"))

    # Timing
    durations = [l["total_duration_ms"] for l in logs if l.get("total_duration_ms")]
    asr_durations = [l["asr_duration_ms"] for l in logs if l.get("asr_duration_ms")]
    tokens = [l["total_tokens"] for l in logs if l.get("total_tokens")]

    print(f"\n{'=' * 50}")
    print(f"  Lume Eval Summary")
    print(f"{'=' * 50}")
    print(f"  Total interactions:  {total}")
    print(f"  With tool calls:     {with_tools} ({with_tools/total*100:.0f}%)" if total else "")
    print(f"  Errors/aborted:      {errors}")
    print(f"  Annotated:           {annotated}/{total}")
    print()

    if durations:
        durations.sort()
        print(f"  Latency (total):")
        print(f"    P50: {durations[len(durations)//2]}ms")
        print(f"    P90: {durations[int(len(durations)*0.9)]}ms")
        print(f"    Max: {durations[-1]}ms")
        print()

    if asr_durations:
        asr_durations.sort()
        print(f"  ASR Latency:")
        print(f"    P50: {asr_durations[len(asr_durations)//2]}ms")
        print(f"    P90: {asr_durations[int(len(asr_durations)*0.9)]}ms")
        print()

    if tokens:
        print(f"  Tokens:")
        print(f"    Avg: {sum(tokens)//len(tokens)}")
        print(f"    Max: {max(tokens)}")
        print()

    # Response length
    resp_lens = [len(l.get("final_response", "")) for l in logs if l.get("final_response")]
    if resp_lens:
        resp_lens.sort()
        print(f"  Response length (chars):")
        print(f"    Avg: {sum(resp_lens)//len(resp_lens)}")
        print(f"    Max: {max(resp_lens)}")
        too_long = sum(1 for r in resp_lens if r > 100)
        print(f"    >100 chars: {too_long} ({too_long/len(resp_lens)*100:.0f}%)")
        print()


def show_tools(logs: list[dict]):
    tool_counter = Counter()
    tool_durations: dict[str, list] = {}
    for l in logs:
        for tc in l.get("tool_calls", []):
            name = tc["tool"]
            tool_counter[name] += 1
            if name not in tool_durations:
                tool_durations[name] = []
            tool_durations[name].append(tc.get("duration_ms", 0))

    print(f"\n{'=' * 50}")
    print(f"  Tool Usage Stats")
    print(f"{'=' * 50}")
    for tool, count in tool_counter.most_common():
        durs = tool_durations[tool]
        avg_dur = sum(durs) // len(durs) if durs else 0
        print(f"  {tool:20s}  calls={count:3d}  avg_latency={avg_dur}ms")
    print()


def show_detail(logs: list[dict], last_n: int = 0):
    if last_n > 0:
        logs = logs[-last_n:]
    for i, l in enumerate(logs):
        status = "✓" if l.get("success") is True else "✗" if l.get("success") is False else "?"
        print(f"\n{'─' * 60}")
        print(f"  [{status}] #{i+1}  {l.get('timestamp', '?')}")
        print(f"  ASR:      \"{l.get('asr_text', '')}\"  [{l.get('asr_language', '')}/{l.get('asr_emotion', '')}]  {l.get('asr_duration_ms', 0)}ms")
        print(f"  Input:    {l.get('user_message', '')}")
        for tc in l.get("tool_calls", []):
            print(f"  Tool:     {tc['tool']}({json.dumps(tc.get('params', {}), ensure_ascii=False)})  → {tc.get('result', '')[:80]}  [{tc.get('duration_ms', 0)}ms]")
        print(f"  Response: {l.get('final_response', '')[:200]}")
        print(f"  Timing:   total={l.get('total_duration_ms', 0)}ms  tokens={l.get('total_tokens', 0)}  cache={l.get('cache_rate', '-')}")
        if l.get("notes"):
            print(f"  Notes:    {l['notes']}")


def show_errors(logs: list[dict]):
    errors = [l for l in logs if l.get("success") is False or (not l.get("final_response") and l.get("notes"))]
    if not errors:
        print("No errors found.")
        return
    print(f"\n{len(errors)} errors found:")
    show_detail(errors)


def annotate(logs: list[dict]):
    """Interactively annotate unannotated interactions."""
    unannotated = [(i, l) for i, l in enumerate(logs) if l.get("success") is None and l.get("final_response")]
    if not unannotated:
        print("All interactions already annotated.")
        return

    print(f"\n{len(unannotated)} interactions to annotate (y=success, n=fail, s=skip, q=quit)\n")
    updated = 0
    for idx, (i, l) in enumerate(unannotated):
        print(f"{'─' * 50}")
        print(f"  [{idx+1}/{len(unannotated)}]  {l.get('timestamp', '')}")
        print(f"  ASR:      \"{l.get('asr_text', '')}\"")
        for tc in l.get("tool_calls", []):
            print(f"  Tool:     {tc['tool']}  → {tc.get('result', '')[:60]}")
        print(f"  Response: {l.get('final_response', '')[:200]}")
        print()

        while True:
            ans = input("  Result [y/n/s/q]: ").strip().lower()
            if ans in ("y", "n", "s", "q"):
                break

        if ans == "q":
            break
        elif ans == "y":
            logs[i]["success"] = True
            updated += 1
        elif ans == "n":
            logs[i]["success"] = False
            note = input("  Note (optional): ").strip()
            if note:
                logs[i]["notes"] = note
            updated += 1

    if updated > 0:
        # Rewrite the entire log file
        with open(_LOG_FILE, "w", encoding="utf-8") as f:
            for l in logs:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")
        print(f"\nUpdated {updated} annotations.")


def main():
    parser = argparse.ArgumentParser(description="Lume eval log viewer")
    parser.add_argument("--last", type=int, default=0, help="Show last N interactions")
    parser.add_argument("--detail", action="store_true", help="Show detailed view")
    parser.add_argument("--tools", action="store_true", help="Show tool usage stats")
    parser.add_argument("--errors", action="store_true", help="Show only errors")
    parser.add_argument("--annotate", action="store_true", help="Interactively annotate")
    args = parser.parse_args()

    logs = load_logs()

    if args.annotate:
        annotate(logs)
    elif args.tools:
        show_tools(logs)
    elif args.errors:
        show_errors(logs)
    elif args.detail or args.last:
        show_detail(logs, args.last)
    else:
        show_summary(logs)


if __name__ == "__main__":
    main()
