# tau2-bench-context

τ²-bench with context-managed agent. Instead of full conversation history, the agent maintains a compressed context list via `<edit>` blocks.

Supports all tau2 domains: airline, retail, telecom.

## Usage

```bash
prime eval run tau2-bench-context -m openai/gpt-5.4 --env-args '{"domain": "telecom"}'
```

## Metrics

Task completion reward (from tau2 evaluator), plus context metrics: prefix stability, compression, append ratio, total edits.
