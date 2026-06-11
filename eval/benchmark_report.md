# Barbershop Assistant — Multi-Model Benchmark

Evaluated 3 questions across 2 models.

Categories: pricing, hours, location, services, booking, refusal, language.

## Headline results

| Model | Accuracy | Mean latency | p95 latency | Mean tokens/s |
|---|---|---|---|---|
| `qwen2.5:3b` | 100.0% (3/3) | 62.83s | 142.23s | 0.74 |
| `llama3.2:3b` | 100.0% (3/3) | 96.67s | 179.42s | 0.44 |

## Accuracy by category

| Model | pricing |
|---|---|
| `qwen2.5:3b` | 3/3 |
| `llama3.2:3b` | 3/3 |

## Methodology

- Each model is queried in isolation through the same system prompt.
- Grading is rule-based: a response passes if it contains required tokens and avoids forbidden ones.
- Latency includes Ollama model load. With `keep_alive=10m` we mostly hit warm models.
- Hardware: CPU-only inference (no GPU). Tokens/s will be higher with hardware acceleration.

## How to reproduce

```bash
# All models, full dataset
python eval/run_eval.py

# Subset of items for a quick smoke run
python eval/run_eval.py --limit 3

# Single model
python eval/run_eval.py --models qwen2.5:3b
```

## Raw per-row results

- `eval/results/qwen2.5_3b.csv`
- `eval/results/llama3.2_3b.csv`
