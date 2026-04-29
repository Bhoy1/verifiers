# longmemeval-context

LongMemEval long-term memory benchmark with two modes:

- **baseline** — Full conversation history dumped per question
- **context** — Incremental session-by-session fact extraction into a managed context list

Supports three variants:
- `longmemeval_s_cleaned` (~40 sessions, ~115k tokens) — default
- `longmemeval_m_cleaned` (~500 sessions) — large scale
- `longmemeval_oracle` — oracle retrieval (evidence sessions only)

Scoring: LLM judge (per-question-type prompts matching the paper).

## Usage

```bash
# Context mode (default, small variant)
prime eval run longmemeval-context -m gpt-5.4

# Baseline mode
prime eval run longmemeval-context -m gpt-5.4 --env-args '{"mode": "baseline"}'

# Large variant
prime eval run longmemeval-context -m gpt-5.4 --env-args '{"variant": "longmemeval_m_cleaned"}'
```

## Metrics

Judge accuracy, prefix stability, compression %, append ratio, total edits, per-session context tokens.
