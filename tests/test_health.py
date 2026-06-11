"""Sanity tests for the FastAPI app that don't need Ollama."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _client():
    from app.main import app

    return TestClient(app)


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_info_exposes_allowed_models():
    r = _client().get("/info")
    assert r.status_code == 200
    body = r.json()
    assert "qwen2.5:3b" in body["allowed_models"]
    assert "llama3.2:3b" in body["allowed_models"]


def test_metrics_endpoint_returns_prometheus_text():
    r = _client().get("/metrics")
    assert r.status_code == 200
    assert "llm_requests_total" in r.text


def test_unknown_model_is_rejected():
    r = _client().post("/chat", json={"messages": [{"role": "user", "content": "hi"}], "model": "evil-model"})
    assert r.status_code == 400
