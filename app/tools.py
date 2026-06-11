"""
Tools the LLM can invoke.

Each tool has two parts:
    1. A Python function that does the actual work (here, persist to SQLite).
    2. A JSON schema declared in `TOOLS_SCHEMA` so the LLM knows when and how
       to call it. Schema is OpenAI / Ollama compatible.

To add a new tool: write the function, append its schema to `TOOLS_SCHEMA`,
and register it in `TOOL_REGISTRY` at the bottom.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

DB_PATH = Path(os.getenv("APPOINTMENTS_DB", "appointments.sqlite"))


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the appointments table if missing. Safe to call repeatedly."""
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS appointments (
                id TEXT PRIMARY KEY,
                customer_name TEXT NOT NULL,
                service TEXT NOT NULL,
                appointment_at TEXT NOT NULL,
                contact TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'confirmed'
            )
            """
        )
        conn.commit()


def book_appointment(
    customer_name: str,
    service: str,
    appointment_at: str,
    contact: str,
) -> dict:
    """Persist a new appointment and return a confirmation payload.

    Args:
        customer_name: Full name of the customer.
        service: One of the named services from the catalog.
        appointment_at: ISO-8601 datetime string (e.g. "2026-06-13T10:30").
        contact: Phone or WhatsApp identifier.
    """
    # Validate that the ISO date is parseable and not in the past.
    try:
        appt_dt = datetime.fromisoformat(appointment_at)
    except ValueError:
        return {"ok": False, "error": f"Invalid date format: '{appointment_at}'. Use ISO-8601, e.g. '2026-06-19T10:30'."}

    if appt_dt.date() < datetime.now(UTC).date():
        return {"ok": False, "error": f"Cannot book appointments in the past ({appt_dt.date()})."}

    init_db()
    appt_id = uuid.uuid4().hex[:10]
    created_at = datetime.now(UTC).isoformat(timespec="seconds")

    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO appointments (id, customer_name, service, appointment_at, contact, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (appt_id, customer_name, service, appointment_at, contact, created_at),
        )
        conn.commit()

    return {
        "ok": True,
        "appointment_id": appt_id,
        "summary": f"Booked {service} for {customer_name} at {appointment_at}.",
    }


def list_appointments(limit: int = 50) -> list[dict]:
    """Return the most recent appointments."""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM appointments ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# JSON schema sent to the model so it knows the tool's signature.
TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Books a barbershop appointment. Only call when you have ALL four: "
                "customer's full name, service name from the catalog, exact date and time, "
                "and contact (phone or WhatsApp)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {
                        "type": "string",
                        "description": "Customer's full name.",
                    },
                    "service": {
                        "type": "string",
                        "description": "Service name from the catalog (e.g. 'Classic Haircut', 'Fade', 'Haircut + Beard').",
                    },
                    "appointment_at": {
                        "type": "string",
                        "description": "ISO-8601 datetime, e.g. '2026-06-13T10:30'.",
                    },
                    "contact": {
                        "type": "string",
                        "description": "Phone number or WhatsApp handle.",
                    },
                },
                "required": ["customer_name", "service", "appointment_at", "contact"],
            },
        },
    }
]


TOOL_REGISTRY = {
    "book_appointment": book_appointment,
}


def execute_tool(name: str, arguments: dict) -> dict:
    """Look up a tool by name and run it. Returns a JSON-serializable result."""
    if name not in TOOL_REGISTRY:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        return TOOL_REGISTRY[name](**arguments)
    except TypeError as e:
        return {"ok": False, "error": f"Bad arguments for {name}: {e}"}


if __name__ == "__main__":
    # Quick CLI smoke test
    result = book_appointment(
        customer_name="John Doe",
        service="Classic Haircut",
        appointment_at="2026-06-13T11:00",
        contact="+34 600 000 000",
    )
    print(json.dumps(result, indent=2))
    print("\nAll appointments:")
    print(json.dumps(list_appointments(), indent=2))
