"""
Summary script for LoCoMo eval results.

Usage:
    python summary.py <path_to_results.jsonl>
    python summary.py environments/locomo_context/outputs/evals/.../results.jsonl
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def summarize(results_path: str):
    results_path = Path(results_path)
    if results_path.is_dir():
        results_path = results_path / "results.jsonl"

    meta_path = results_path.parent / "metadata.json"

    with open(results_path) as f:
        rows = [json.loads(l) for l in f]

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    rewards = [r["reward"] for r in rows]
    cat_scores = defaultdict(list)
    conv_scores = defaultdict(list)
    for r in rows:
        info = r.get("info", {})
        if isinstance(info, str):
            info = json.loads(info)
        cat = info.get("category", 0)
        conv = info.get("conv_id", "?")
        cat_scores[cat].append(r["reward"])
        conv_scores[conv].append(r["reward"])

    non5 = [r["reward"] for r, s in zip(rows, [s.get("info", {}) for s in rows])
            if (json.loads(s) if isinstance(s, str) else s).get("category", 0) != 5]

    mode = meta.get("env_args", {}).get("mode", "?")
    model = meta.get("model", "?")

    print(f"=== {mode.upper()} | model={model} | {len(rows)} questions ===")
    print(f"Avg F1 (all):      {sum(rewards) / len(rewards):.4f}")
    if non5:
        print(f"Avg F1 (no Cat 5): {sum(non5) / len(non5):.4f}")
    if meta.get("time_ms"):
        print(f"Time: {meta['time_ms'] / 1000:.1f}s")
    if meta.get("usage"):
        print(f"Avg input tokens:  {meta['usage'].get('input_tokens', 0):.0f}")
        print(f"Avg output tokens: {meta['usage'].get('output_tokens', 0):.0f}")
    print()

    print("F1 by category:")
    for cat in sorted(cat_scores.keys()):
        s = cat_scores[cat]
        print(f"  Cat {cat}: {sum(s) / len(s):.3f} ({len(s)} questions)")
    print()

    if len(conv_scores) > 1:
        print("F1 by conversation:")
        for conv in sorted(conv_scores.keys()):
            s = conv_scores[conv]
            print(f"  {conv}: {sum(s) / len(s):.3f} ({len(s)} questions)")
        print()

    # Show compression info if available
    for r in rows:
        edit_log = r.get("_edit_log", [])
        answer = [e for e in edit_log if e.get("step") == "answer"]
        if answer and answer[0].get("full_history_tokens"):
            info = r.get("info", {})
            if isinstance(info, str):
                info = json.loads(info)
            conv = info.get("conv_id", "?")
            a = answer[0]
            print(
                f"  [{conv}] context_tokens={a.get('context_tokens'):,} "
                f"/ full_tokens={a.get('full_history_tokens'):,} "
                f"→ {a.get('compression_pct', 0):.1f}% compression"
            )
            break  # only show one sample for brevity


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summary.py <path_to_results.jsonl_or_dir>")
        sys.exit(1)
    summarize(sys.argv[1])
