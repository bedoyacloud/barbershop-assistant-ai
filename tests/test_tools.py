"""Unit tests for the booking tool. Hits SQLite, never the LLM."""

from __future__ import annotations

import tempfile
from pathlib import Path


def _fresh_db(monkeypatch) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "appts.sqlite"
    monkeypatch.setenv("APPOINTMENTS_DB", str(tmp))
    # Re-import after env var is set so the module picks up the new DB_PATH.
    import importlib

    from app import tools as tools_mod

    importlib.reload(tools_mod)
    return tmp


def test_book_appointment_persists(monkeypatch):
    _fresh_db(monkeypatch)
    from app import tools

    result = tools.book_appointment(
        customer_name="Jane Doe",
        service="Classic Haircut",
        appointment_at="2026-06-13T10:30",
        contact="+34 600 111 222",
    )
    assert result["ok"] is True
    assert "appointment_id" in result
    appts = tools.list_appointments()
    assert len(appts) == 1
    assert appts[0]["customer_name"] == "Jane Doe"
    assert appts[0]["service"] == "Classic Haircut"


def test_execute_tool_unknown_name(monkeypatch):
    _fresh_db(monkeypatch)
    from app import tools

    result = tools.execute_tool("does_not_exist", {})
    assert result["ok"] is False
    assert "Unknown tool" in result["error"]


def test_execute_tool_bad_arguments(monkeypatch):
    _fresh_db(monkeypatch)
    from app import tools

    result = tools.execute_tool("book_appointment", {"customer_name": "X"})
    assert result["ok"] is False
    assert "Bad arguments" in result["error"]


def test_tools_schema_shape():
    from app import tools

    assert isinstance(tools.TOOLS_SCHEMA, list)
    assert tools.TOOLS_SCHEMA[0]["type"] == "function"
    required = tools.TOOLS_SCHEMA[0]["function"]["parameters"]["required"]
    assert set(required) == {"customer_name", "service", "appointment_at", "contact"}
