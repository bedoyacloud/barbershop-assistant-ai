"""
Thin async client for the Ollama HTTP API.

Why this module exists:
    All LLM calls go through here. If we ever swap Ollama for OpenAI, add
    retries, or attach metrics, we change one file instead of twenty.

Public surface:
    - chat(prompt, ...) -> ChatResponse        (single-shot, full response)
    - chat_stream(prompt, ...) -> AsyncIterator (token-by-token streaming)
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Literal

import httpx
from pydantic import BaseModel

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
REQUEST_TIMEOUT_S = 180


class ToolCall(BaseModel):
    """A tool invocation requested by the model."""

    name: str
    arguments: dict


class Message(BaseModel):
    """One turn in a conversation, OpenAI-style schema."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] | None = None


class ChatMetrics(BaseModel):
    """Performance metrics captured for every LLM call."""

    model: str
    prompt_tokens: int
    output_tokens: int
    total_duration_s: float
    eval_duration_s: float
    tokens_per_second: float


class ChatResponse(BaseModel):
    text: str
    tool_calls: list[ToolCall] = []
    metrics: ChatMetrics


def _build_payload(
    prompt: str,
    *,
    model: str,
    system: str | None,
    stream: bool,
    keep_alive: str,
) -> dict:
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "keep_alive": keep_alive,
    }
    if system:
        payload["system"] = system
    return payload


async def chat(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    keep_alive: str = "10m",
) -> ChatResponse:
    """Send a single-turn prompt and return the full response with metrics."""
    payload = _build_payload(
        prompt, model=model, system=system, stream=False, keep_alive=keep_alive
    )

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
        r = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        r.raise_for_status()
        data = r.json()

    eval_count = data["eval_count"]
    eval_duration_s = data["eval_duration"] / 1e9
    tokens_per_second = eval_count / eval_duration_s if eval_duration_s > 0 else 0.0

    return ChatResponse(
        text=data["response"],
        metrics=ChatMetrics(
            model=model,
            prompt_tokens=data.get("prompt_eval_count", 0),
            output_tokens=eval_count,
            total_duration_s=data["total_duration"] / 1e9,
            eval_duration_s=eval_duration_s,
            tokens_per_second=tokens_per_second,
        ),
    )


async def chat_stream(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    keep_alive: str = "10m",
) -> AsyncIterator[str]:
    """Stream the response token by token. Yields text chunks as they arrive."""
    payload = _build_payload(
        prompt, model=model, system=system, stream=True, keep_alive=keep_alive
    )

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
        async with client.stream(
            "POST", f"{OLLAMA_URL}/api/generate", json=payload
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("response"):
                    yield chunk["response"]
                if chunk.get("done"):
                    return


async def chat_messages(
    messages: list[Message],
    *,
    model: str = DEFAULT_MODEL,
    tools: list[dict] | None = None,
    keep_alive: str = "10m",
) -> ChatResponse:
    """Multi-turn chat using Ollama's /api/chat endpoint.

    When `tools` is supplied, the model may respond with `tool_calls` instead
    of (or in addition to) free text. The caller is expected to execute the
    tools and feed results back as `role="tool"` messages on the next turn.
    """
    payload: dict = {
        "model": model,
        "messages": [m.model_dump(exclude_none=True) for m in messages],
        "stream": False,
        "keep_alive": keep_alive,
    }
    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
        r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    msg = data["message"]
    raw_tool_calls = msg.get("tool_calls") or []
    parsed_calls = [
        ToolCall(name=tc["function"]["name"], arguments=tc["function"].get("arguments", {}))
        for tc in raw_tool_calls
    ]

    eval_count = data["eval_count"]
    eval_duration_s = data["eval_duration"] / 1e9
    tokens_per_second = eval_count / eval_duration_s if eval_duration_s > 0 else 0.0

    return ChatResponse(
        text=msg.get("content", "") or "",
        tool_calls=parsed_calls,
        metrics=ChatMetrics(
            model=model,
            prompt_tokens=data.get("prompt_eval_count", 0),
            output_tokens=eval_count,
            total_duration_s=data["total_duration"] / 1e9,
            eval_duration_s=eval_duration_s,
            tokens_per_second=tokens_per_second,
        ),
    )


if __name__ == "__main__":
    import asyncio
    import sys

    async def main() -> None:
        prompt = " ".join(sys.argv[1:]) or "Say hi in one short sentence."
        print(f"PROMPT: {prompt}\n")
        result = await chat(prompt)
        print(f"RESPONSE: {result.text}")
        print(f"\nMETRICS:\n{result.metrics.model_dump_json(indent=2)}")

    asyncio.run(main())
