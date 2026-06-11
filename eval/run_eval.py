"""
Multi-model truth-analysis runner.

For each item in `dataset.jsonl`, send the user message through the system
prompt to every model under test. Score:
    - accuracy: matches all `must_contain_any` (case-insensitive) AND no `must_not_contain`
    - latency: total wall-clock seconds end-to-end
    - tokens_per_second: from Ollama metrics

Output:
    - eval/results/<model>.csv: per-row results
    - eval/benchmark_report.md: human-readable comparison
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import statistics

# Allow `python eval/run_eval.py` execution from project root.
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm_client import Message, chat_messages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (PROJECT_ROOT / "app" / "prompts" / "barber_system.md").read_text(encoding="utf-8")
DATASET = PROJECT_ROOT / "eval" / "dataset.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
REPORT_PATH = PROJECT_ROOT / "eval" / "benchmark_report.md"

DEFAULT_MODELS = ["qwen2.5:3b", "llama3.2:3b"]


def load_dataset() -> list[dict]:
    rows = []
    with DATASET.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def grade(response: str, item: dict) -> bool:
    """Pass if response matches any of the required tokens and none of the forbidden ones."""
    text = response.lower()
    if item.get("must_contain_any"):
        needles = [n.lower() for n in item["must_contain_any"]]
        if not any(re.search(re.escape(n) if not n.startswith("regex:") else n[6:], text) for n in needles):
            return False
    for forbidden in item.get("must_not_contain", []):
        if re.search(forbidden.lower(), text):
            return False
    return True


async def evaluate_model(model: str, items: list[dict]) -> dict:
    print(f"\n=== Evaluating {model} on {len(items)} items ===")
    rows: list[dict] = []
    for i, item in enumerate(items, 1):
        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=item["user"]),
        ]
        t0 = time.perf_counter()
        try:
            result = await chat_messages(messages, model=model)
            elapsed = time.perf_counter() - t0
            ok = grade(result.text, item)
            rows.append({
                "id": item["id"],
                "category": item["category"],
                "user": item["user"],
                "response": result.text.replace("\n", " ")[:280],
                "passed": ok,
                "latency_s": round(elapsed, 2),
                "tokens_per_second": round(result.metrics.tokens_per_second, 2),
                "output_tokens": result.metrics.output_tokens,
            })
            mark = "OK" if ok else "FAIL"
            print(f"  [{i:>2}/{len(items)}] {mark} ({elapsed:.1f}s) {item['id']}")
        except Exception as e:
            rows.append({
                "id": item["id"],
                "category": item["category"],
                "user": item["user"],
                "response": f"ERROR: {e}",
                "passed": False,
                "latency_s": -1,
                "tokens_per_second": 0,
                "output_tokens": 0,
            })
            print(f"  [{i:>2}/{len(items)}] ERROR {item['id']}: {e}")

    # Persist per-model CSV
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f"{model.replace(':', '_').replace('/', '_')}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate
    passed = sum(1 for r in rows if r["passed"])
    valid_latencies = [r["latency_s"] for r in rows if r["latency_s"] > 0]
    valid_tps = [r["tokens_per_second"] for r in rows if r["tokens_per_second"] > 0]

    by_category: dict[str, list[bool]] = {}
    for r in rows:
        by_category.setdefault(r["category"], []).append(r["passed"])

    return {
        "model": model,
        "total": len(rows),
        "passed": passed,
        "accuracy": round(passed / len(rows), 4) if rows else 0,
        "mean_latency_s": round(statistics.mean(valid_latencies), 2) if valid_latencies else 0,
        "p95_latency_s": round(statistics.quantiles(valid_latencies, n=20)[-1], 2) if len(valid_latencies) >= 2 else 0,
        "mean_tps": round(statistics.mean(valid_tps), 2) if valid_tps else 0,
        "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in by_category.items()},
        "csv_path": str(csv_path.relative_to(PROJECT_ROOT)),
    }


def render_report(summaries: list[dict], n_items: int) -> str:
    lines = []
    lines.append("# Barbershop Assistant — Multi-Model Benchmark")
    lines.append("")
    lines.append(f"Evaluated {n_items} questions across {len(summaries)} models.")
    lines.append("")
    lines.append("Categories: pricing, hours, location, services, booking, refusal, language.")
    lines.append("")
    lines.append("## Headline results")
    lines.append("")
    lines.append("| Model | Accuracy | Mean latency | p95 latency | Mean tokens/s |")
    lines.append("|---|---|---|---|---|")
    for s in summaries:
        lines.append(f"| `{s['model']}` | {s['accuracy']*100:.1f}% ({s['passed']}/{s['total']}) | {s['mean_latency_s']}s | {s['p95_latency_s']}s | {s['mean_tps']} |")
    lines.append("")
    lines.append("## Accuracy by category")
    lines.append("")
    categories = sorted({c for s in summaries for c in s["by_category"]})
    header = "| Model | " + " | ".join(categories) + " |"
    sep = "|---|" + "|".join("---" for _ in categories) + "|"
    lines.append(header)
    lines.append(sep)
    for s in summaries:
        row = f"| `{s['model']}` | " + " | ".join(s["by_category"].get(c, "-") for c in categories) + " |"
        lines.append(row)
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("- Each model is queried in isolation through the same system prompt.")
    lines.append("- Grading is rule-based: a response passes if it contains required tokens and avoids forbidden ones.")
    lines.append("- Latency includes Ollama model load. With `keep_alive=10m` we mostly hit warm models.")
    lines.append("- Hardware: CPU-only inference (no GPU). Tokens/s will be higher with hardware acceleration.")
    lines.append("")
    lines.append("## How to reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("# All models, full dataset")
    lines.append("python eval/run_eval.py")
    lines.append("")
    lines.append("# Subset of items for a quick smoke run")
    lines.append("python eval/run_eval.py --limit 3")
    lines.append("")
    lines.append("# Single model")
    lines.append("python eval/run_eval.py --models qwen2.5:3b")
    lines.append("```")
    lines.append("")
    lines.append("## Raw per-row results")
    lines.append("")
    for s in summaries:
        lines.append(f"- `{s['csv_path']}`")
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N items (for smoke runs).")
    args = parser.parse_args()

    items = load_dataset()
    if args.limit:
        items = items[: args.limit]

    summaries = []
    for model in args.models:
        summary = await evaluate_model(model, items)
        summaries.append(summary)

    report = render_report(summaries, n_items=len(items))
    REPORT_PATH.write_text(report)
    print(f"\nReport written to {REPORT_PATH}")
    for s in summaries:
        print(f"  {s['model']:<20} accuracy={s['accuracy']*100:.1f}% mean_latency={s['mean_latency_s']}s")


if __name__ == "__main__":
    asyncio.run(main())
