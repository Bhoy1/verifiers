# locomo-context

LoCoMo long-conversation memory benchmark with two modes:

- **baseline** — Full conversation history dumped per question
- **context** — Incremental session-by-session fact extraction into a managed context list, then QA from compressed facts

Dataset: [bhoy/locomo](https://huggingface.co/datasets/bhoy/locomo) (1,986 QAs across 10 conversations).

Scoring: token-level F1 with category-5 unanswerable handling.

## Usage

```bash
# Context mode (default)
prime eval run locomo-context -m openai/gpt-5.4

# Baseline mode
prime eval run locomo-context -m openai/gpt-5.4 --env-args '{"mode": "baseline"}'

# Filter to one conversation
prime eval run locomo-context -m openai/gpt-5.4 --env-args '{"mode": "context", "conv_id": "conv-26"}'
```

## Metrics tracked

F1, prefix stability, compression %, append ratio, total edits, per-session context tokens.
