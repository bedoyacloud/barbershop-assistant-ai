"""
Prometheus metrics for the Barbershop Assistant.

What we expose at /metrics:
    - llm_requests_total{model,status}              counter
    - llm_request_duration_seconds{model}           histogram (end-to-end)
    - llm_generation_duration_seconds{model}        histogram (model-only)
    - llm_tokens_total{model,kind="prompt|output"}  counter
    - llm_tokens_per_second{model}                  histogram
    - llm_tool_calls_total{tool,status}             counter
    - app_info{version}                              gauge (build/version)

These map directly to the JD requirement: "monitoring LLM performance,
detect data drift, monitor key metrics".
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

LLM_REQUESTS = Counter(
    "llm_requests_total",
    "Total chat requests served, partitioned by model and status.",
    labelnames=("model", "status"),
)

LLM_REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "End-to-end duration of a /chat request.",
    labelnames=("model",),
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120, 180),
)

LLM_GENERATION_DURATION = Histogram(
    "llm_generation_duration_seconds",
    "Model-side generation duration (eval_duration from Ollama).",
    labelnames=("model",),
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120, 180),
)

LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Tokens processed, partitioned by model and kind (prompt / output).",
    labelnames=("model", "kind"),
)

LLM_TPS = Histogram(
    "llm_tokens_per_second",
    "Observed tokens/s for each generation.",
    labelnames=("model",),
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100),
)

LLM_TOOL_CALLS = Counter(
    "llm_tool_calls_total",
    "Tool invocations issued by the model, partitioned by tool and status.",
    labelnames=("tool", "status"),
)

APP_INFO = Gauge(
    "app_info",
    "Static build info exposed as labels.",
    labelnames=("version",),
)


def record_chat(
    *,
    model: str,
    status: str,
    request_duration_s: float,
    generation_duration_s: float,
    prompt_tokens: int,
    output_tokens: int,
    tokens_per_second: float,
) -> None:
    LLM_REQUESTS.labels(model=model, status=status).inc()
    LLM_REQUEST_DURATION.labels(model=model).observe(request_duration_s)
    if generation_duration_s > 0:
        LLM_GENERATION_DURATION.labels(model=model).observe(generation_duration_s)
    LLM_TOKENS.labels(model=model, kind="prompt").inc(prompt_tokens)
    LLM_TOKENS.labels(model=model, kind="output").inc(output_tokens)
    if tokens_per_second > 0:
        LLM_TPS.labels(model=model).observe(tokens_per_second)


def record_tool_call(tool: str, status: str) -> None:
    LLM_TOOL_CALLS.labels(tool=tool, status=status).inc()


def render_latest() -> tuple[bytes, str]:
    """Return (body, content_type) suitable for the /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
