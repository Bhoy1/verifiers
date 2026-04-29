"""
Dump a tau2-bench-context task trajectory to a human-readable text file.

Usage:
    python dump_trajectory.py <run_dir> <task_id>
    python dump_trajectory.py <run_dir> all

Example:
    python dump_trajectory.py outputs/evals/tau2-bench-context--anthropic--claude-opus-4.6/052df3e2 0
    python dump_trajectory.py outputs/evals/tau2-bench-context--anthropic--claude-opus-4.6/052df3e2 all
"""

import json
import sys
from pathlib import Path


def format_tool_calls(tool_calls):
    if not tool_calls:
        return ""
    parts = []
    for tc in tool_calls:
        name = tc.get("name", "?")
        args = tc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        args_str = json.dumps(args, indent=2) if isinstance(args, dict) else str(args)
        parts.append(f"  TOOL CALL: {name}\n  ARGS: {args_str}")
    return "\n".join(parts)


def format_message(i, msg):
    role = msg.get("role", "?").upper()
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls")

    lines = [f"━━━ [{i}] {role} ━━━"]

    if content:
        lines.append(content)

    if tool_calls:
        lines.append("")
        lines.append(format_tool_calls(tool_calls))

    return "\n".join(lines)


def dump_task(result, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        # Header
        f.write("=" * 70 + "\n")
        f.write(f"TASK {result['example_id']} | Reward: {result['reward']}\n")
        f.write("=" * 70 + "\n\n")

        # Task info
        info = result.get("info", {})
        if isinstance(info, str):
            info = json.loads(info)
        task_id = info.get("id", "?")
        description = info.get("description", {})
        if isinstance(description, dict):
            purpose = description.get("purpose", "")
            if purpose:
                f.write(f"PURPOSE:\n{purpose}\n\n")
        f.write(f"Task ID: {task_id}\n")
        f.write(f"Turns: {result.get('num_turns', '?')}\n")
        f.write(f"Tool calls: {result.get('num_assistant_tool_calls', '?')}\n")
        f.write(f"Steps: {result.get('num_steps', '?')}\n")
        f.write(f"Errors: {result.get('num_errors', '?')}\n")
        f.write(f"Stop condition: {result.get('stop_condition', '?')}\n")
        f.write("\n")

        # Context metrics (if available)
        metrics = result.get("metrics", {})
        if "prefix_stability_metric" in metrics:
            f.write("CONTEXT METRICS:\n")
            f.write(f"  Prefix stability: {metrics.get('prefix_stability_metric', 0):.3f}\n")
            f.write(f"  Append ratio: {metrics.get('append_ratio_metric', 0):.3f}\n")
            f.write(f"  Context tokens: {metrics.get('context_tokens_metric', 0):.0f}\n")
            f.write(f"  Total edits: {metrics.get('total_edits_metric', 0):.0f}\n")
            f.write("\n")

        # Full trajectory
        f.write("\n" + "=" * 70 + "\n")
        f.write("FULL CONVERSATION (tau2 trajectory)\n")
        f.write("=" * 70 + "\n\n")

        trajectory = result.get("_tau2_trajectory", [])
        if not trajectory:
            f.write("(No _tau2_trajectory — did you re-run with the updated env?)\n\n")
        else:
            for i, msg in enumerate(trajectory):
                f.write(format_message(i, msg) + "\n\n")

        # Edit log
        edit_log = result.get("_edit_log", [])
        if edit_log:
            f.write("\n" + "=" * 70 + "\n")
            f.write("CONTEXT EDIT LOG (per turn)\n")
            f.write("=" * 70 + "\n\n")
            for i, e in enumerate(edit_log):
                f.write(f"━━━ Turn {i} ━━━\n")
                f.write(f"edit_success: {e.get('edit_success')}\n")
                if e.get("edit_error"):
                    f.write(f"edit_error: {e['edit_error']}\n")
                f.write(f"ctx_len: {e.get('context_list_len', 0)}\n")
                f.write(f"ctx_tokens: {e.get('context_tokens', 0)}\n")
                f.write(f"prefix_stability: {e.get('prefix_stability', 0):.3f}\n")
                code = e.get("edit_code") or "(no edit)"
                f.write(f"\nedit_code:\n{code}\n\n")

        # Final context list
        ctx = result.get("_context_list", [])
        if ctx:
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"FINAL CONTEXT LIST ({len(ctx)} items)\n")
            f.write("=" * 70 + "\n\n")
            for i, item in enumerate(ctx):
                f.write(f"  [{i}] {item}\n")

    print(f"Wrote: {output_path}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    task_arg = sys.argv[2]

    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)

    with open(results_path, encoding="utf-8") as f:
        results = [json.loads(l) for l in f]

    out_dir = run_dir / "trajectories"
    out_dir.mkdir(exist_ok=True)

    if task_arg.lower() == "all":
        for r in results:
            out_path = out_dir / f"task_{r['example_id']}_reward_{r['reward']}.txt"
            dump_task(r, out_path)
    else:
        tid = int(task_arg)
        result = next((r for r in results if r["example_id"] == tid), None)
        if result is None:
            print(f"Error: task {tid} not found. Available: {[r['example_id'] for r in results]}")
            sys.exit(1)
        out_path = out_dir / f"task_{tid}_reward_{result['reward']}.txt"
        dump_task(result, out_path)


if __name__ == "__main__":
    main()
