"""
FastAPI service for the Barbershop Assistant AI.

Design notes:
    - Stateless server. The client (web UI) keeps the conversation history
      and sends the full message list on every turn. This lets us scale
      horizontally with no shared session store.
    - System prompt is loaded once at startup from a versioned markdown
      file (`app/prompts/barber_system.md`). Editing it is a regular code
      change, reviewable in a PR.
    - The model is configurable per request via `?model=...`, which is
      what powers the multi-model benchmark in milestone M11.
"""

from __future__ import annotations

import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.llm_client import DEFAULT_MODEL, ChatMetrics, Message, chat_messages
from app.logging_config import configure_logging
from app.metrics import APP_INFO, record_chat, record_tool_call, render_latest
from app.tools import TOOLS_SCHEMA, execute_tool, init_db, list_appointments

configure_logging()
log = structlog.get_logger()
APP_VERSION = "0.1.0"
APP_INFO.labels(version=APP_VERSION).set(1)

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
WEB_DIR = PROJECT_ROOT / "web"
SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "barber_system.md"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

ALLOWED_MODELS = {"llama3.2:3b", "qwen2.5:3b", "qwen2.5:7b", "llama3.1:8b"}

_WEEKDAYS_ES = {
    "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
    "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6,
}
_WEEKDAYS_EN = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_WEEKDAY_NAME_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
_WEEKDAY_NAME_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}
_MONTHS_ES_INV = {v: k for k, v in _MONTHS_ES.items()}

def _next_weekday(day_idx: int) -> date:
    """Return the next future date for the given weekday index (0=Monday)."""
    today = date.today()
    days_ahead = (day_idx - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7  # "next thursday" when today IS thursday → next week
    return today + timedelta(days=days_ahead)


def _inject_next_weekday_hint(text: str) -> str | None:
    """If the user asks 'what/which is the next <weekday>', return a server hint."""
    text_lower = text.lower()
    next_patterns_es = [
        "próximo", "proximo", "próxima", "proxima",
        "siguiente", "que dia es", "qué día es",
        "cuál es el", "cual es el", "cuando cae", "cuándo cae",
        "de la proxima", "de la próxima",
    ]
    next_patterns_en = ["next", "what day is", "when is the next", "when does"]
    all_weekdays = {**_WEEKDAYS_ES, **_WEEKDAYS_EN}

    is_next_question = any(p in text_lower for p in next_patterns_es + next_patterns_en)
    if not is_next_question:
        return None

    for day_name, day_idx in all_weekdays.items():
        if day_name in text_lower:
            next_date = _next_weekday(day_idx)
            es_name = _WEEKDAY_NAME_ES[day_idx]
            month_es = _MONTHS_ES_INV[next_date.month]
            return (
                f"[SERVIDOR: El próximo {es_name} es el {next_date.day} de {month_es} "
                f"de {next_date.year}. Usa esta fecha exacta en tu respuesta.]"
            )
    return None


def _detect_weekday_date_mismatch(text: str) -> dict | None:
    """Return mismatch info dict if the message contains a weekday+date that don't match."""
    text_lower = text.lower()
    all_weekdays = {**_WEEKDAYS_ES, **_WEEKDAYS_EN}

    for day_name, day_idx in all_weekdays.items():
        if day_name not in text_lower:
            continue
        numbers = re.findall(r"\b([12]?\d)\b", text_lower)
        for num_str in numbers:
            day_num = int(num_str)
            if not 1 <= day_num <= 31:
                continue
            today = date.today()
            for delta in range(0, 366):
                candidate = today + timedelta(days=delta)
                if candidate.day == day_num:
                    actual_idx = candidate.weekday()
                    if actual_idx != day_idx:
                        return {
                            "said": day_name,
                            "actual_es": _WEEKDAY_NAME_ES[actual_idx],
                            "actual_en": _WEEKDAY_NAME_EN[actual_idx],
                            "date": candidate,
                            "month_es": _MONTHS_ES_INV[candidate.month],
                        }
                    break
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Barbershop Assistant AI",
    version="0.1.0",
    description="Local LLM assistant for barbershop bookings, with MLOps observability.",
    lifespan=lifespan,
)

# Permissive CORS for local dev. Tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    model: str | None = None


class ToolCallExecuted(BaseModel):
    name: str
    arguments: dict
    result: dict


class ChatResponseAPI(BaseModel):
    text: str
    tool_calls: list[ToolCallExecuted] = []
    metrics: ChatMetrics


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "default_model": DEFAULT_MODEL}


@app.get("/info")
async def info() -> dict:
    return {
        "default_model": DEFAULT_MODEL,
        "allowed_models": sorted(ALLOWED_MODELS),
        "system_prompt_chars": len(SYSTEM_PROMPT),
    }


@app.get("/appointments")
async def appointments_endpoint() -> list[dict]:
    return list_appointments()


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    body, ctype = render_latest()
    return Response(content=body, media_type=ctype)


@app.post("/chat", response_model=ChatResponseAPI)
async def chat_endpoint(req: ChatRequest) -> ChatResponseAPI:
    request_id = uuid.uuid4().hex[:8]
    structlog.contextvars.bind_contextvars(request_id=request_id)

    t0 = time.perf_counter()
    model = req.model or DEFAULT_MODEL
    if model not in ALLOWED_MODELS:
        log.warning("model_not_allowed", model=model)
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not in the allowed list: {sorted(ALLOWED_MODELS)}",
        )

    # Always pin the system prompt; never trust the client to set it.
    # Inject today's date so the model can validate day-of-week consistency.
    today = date.today()
    system_with_date = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Today is {today.strftime('%A, %B %d, %Y')}. "
        f"When a customer mentions a weekday and a date together (e.g. 'Thursday the 15th'), "
        f"verify they match before booking. If they don't match, point it out and ask for clarification."
    )
    user_msgs = [m for m in req.messages if m.role != "system"]

    last_user_text = next((m.content for m in reversed(user_msgs) if m.role == "user"), "")

    # Inject next-weekday hint so model doesn't have to calculate it.
    weekday_hint = _inject_next_weekday_hint(last_user_text)
    if weekday_hint:
        log.info("weekday_hint_injected", hint=weekday_hint)
        user_msgs = user_msgs[:-1] + [
            Message(role="user", content=f"{last_user_text}\n\n{weekday_hint}")
        ]
        last_user_text = user_msgs[-1].content

    # Server-side date check: if the last user message contains a weekday+date
    # mismatch, return the correction immediately — no LLM call, no hallucination risk.
    mismatch = _detect_weekday_date_mismatch(last_user_text)
    if mismatch:
        log.info("date_mismatch_detected", mismatch=mismatch)
        d = mismatch
        in_spanish = any(w in last_user_text.lower() for w in _WEEKDAYS_ES)
        if in_spanish:
            reply = (
                f"El {d['date'].day} de {d['month_es']} es {d['actual_es']}, no {d['said']}. "
                f"¿Quieres reservar para ese {d['actual_es']} {d['date'].day} de {d['month_es']}, "
                f"o prefieres otra fecha?"
            )
        else:
            reply = (
                f"{d['date'].strftime('%B %d')} is a {d['actual_en']}, not {d['said']}. "
                f"Would you like to book for {d['actual_en']} {d['date'].strftime('%B %d')}, "
                f"or a different date?"
            )
        dummy_metrics = ChatMetrics(
            model=model, prompt_tokens=0, output_tokens=0,
            total_duration_s=0, eval_duration_s=0, tokens_per_second=0,
        )
        return ChatResponseAPI(text=reply, tool_calls=[], metrics=dummy_metrics)

    messages = [Message(role="system", content=system_with_date), *user_msgs]
    log.info("chat_request_received", model=model, turns=len(user_msgs))

    status = "ok"
    try:
        result = await chat_messages(messages, model=model, tools=TOOLS_SCHEMA)

        # Agent loop: if the model called tools, run them and re-prompt for a
        # natural-language confirmation.
        executed: list[ToolCallExecuted] = []
        if result.tool_calls:
            followup_messages = list(messages) + [
                Message(role="assistant", content=result.text, tool_calls=result.tool_calls)
            ]
            tool_rejected: str | None = None
            for call in result.tool_calls:
                tool_result = execute_tool(call.name, call.arguments)
                tool_status = "ok" if tool_result.get("ok") else "error"
                record_tool_call(call.name, tool_status)
                log.info(
                    "tool_executed",
                    tool=call.name,
                    status=tool_status,
                    arguments=call.arguments,
                )
                executed.append(
                    ToolCallExecuted(name=call.name, arguments=call.arguments, result=tool_result)
                )
                if tool_status == "error":
                    tool_rejected = tool_result.get("error", "The booking could not be completed.")
                followup_messages.append(Message(role="tool", content=str(tool_result)))

            # If any tool rejected the request, skip the follow-up LLM call
            # and return the validation error directly — the model cannot override it.
            if tool_rejected:
                result = await chat_messages(
                    followup_messages + [
                        Message(
                            role="user",
                            content=(
                                "The booking failed with this error: "
                                f'"{tool_rejected}". '
                                "Inform the customer clearly and ask them to clarify."
                            ),
                        )
                    ],
                    model=model,
                )
            else:
                result = await chat_messages(followup_messages, model=model, tools=TOOLS_SCHEMA)

        elapsed = time.perf_counter() - t0
        record_chat(
            model=model,
            status=status,
            request_duration_s=elapsed,
            generation_duration_s=result.metrics.eval_duration_s,
            prompt_tokens=result.metrics.prompt_tokens,
            output_tokens=result.metrics.output_tokens,
            tokens_per_second=result.metrics.tokens_per_second,
        )
        log.info(
            "chat_request_completed",
            model=model,
            elapsed_s=round(elapsed, 2),
            tokens_per_second=round(result.metrics.tokens_per_second, 2),
            output_tokens=result.metrics.output_tokens,
            tool_calls=len(executed),
        )
        return ChatResponseAPI(text=result.text, tool_calls=executed, metrics=result.metrics)
    except Exception:
        status = "error"
        elapsed = time.perf_counter() - t0
        record_chat(
            model=model,
            status=status,
            request_duration_s=elapsed,
            generation_duration_s=0,
            prompt_tokens=0,
            output_tokens=0,
            tokens_per_second=0,
        )
        log.exception("chat_request_failed", model=model)
        raise
    finally:
        structlog.contextvars.clear_contextvars()
