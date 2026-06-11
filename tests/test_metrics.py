"""Smoke tests for the Prometheus metrics module."""

from __future__ import annotations


def test_render_latest_returns_text():
    from app import metrics

    metrics.record_chat(
        model="qwen2.5:3b",
        status="ok",
        request_duration_s=1.5,
        generation_duration_s=1.2,
        prompt_tokens=42,
        output_tokens=20,
        tokens_per_second=16.7,
    )
    body, ctype = metrics.render_latest()
    text = body.decode()
    assert "llm_requests_total" in text
    assert "llm_tokens_total" in text
    assert ctype.startswith("text/plain")


def test_record_tool_call_increments():
    from app import metrics

    metrics.record_tool_call("book_appointment", "ok")
    body, _ = metrics.render_latest()
    assert "llm_tool_calls_total" in body.decode()
