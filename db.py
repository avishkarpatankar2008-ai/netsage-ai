"""
Plain sqlite3 persistence for NetSage Lite -- no ORM, since a class project
doesn't need one and it keeps the schema visible in one place.

Four tables, one per stage of the workflow:
  cases         -- what the student typed in
  rule_results  -- output of the deterministic checks (simple_rules.py)
  diagnoses     -- validated AI output (ai_diagnosis.py)
  reviews       -- the human accept/edit/reject decision
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "netsage.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    symptom TEXT NOT NULL,
    topology_note TEXT DEFAULT '',
    command_output_json TEXT NOT NULL,  -- [{"source": str, "text": str}, ...]
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rule_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES cases(id),
    rule_id TEXT NOT NULL,
    status TEXT NOT NULL,   -- pass | fail
    finding TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS diagnoses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES cases(id),
    result_json TEXT NOT NULL,  -- full dict returned by ai_diagnosis.diagnose()
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES cases(id),
    diagnosis_id INTEGER REFERENCES diagnoses(id),
    decision TEXT NOT NULL,  -- accept | edit | reject
    note TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def save_case(title: str, symptom: str, topology_note: str, command_output: list[dict]) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO cases (title, symptom, topology_note, command_output_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (title, symptom, topology_note, json.dumps(command_output), _now()),
        )
        return cur.lastrowid


def save_rule_results(case_id: int, results: list[dict]) -> None:
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO rule_results (case_id, rule_id, status, finding, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [(case_id, r["rule_id"], r["status"], r["finding"], _now()) for r in results],
        )


def save_diagnosis(case_id: int, result: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO diagnoses (case_id, result_json, created_at) VALUES (?, ?, ?)",
            (case_id, json.dumps(result), _now()),
        )
        return cur.lastrowid


def save_review(case_id: int, diagnosis_id: int | None, decision: str, note: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO reviews (case_id, diagnosis_id, decision, note, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (case_id, diagnosis_id, decision, note, _now()),
        )


def list_cases() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM cases ORDER BY id DESC").fetchall()


def get_case_bundle(case_id: int) -> dict:
    """Everything about one case: the case itself, its latest rule run,
    latest diagnosis, and latest review -- what the history page needs."""
    with get_conn() as conn:
        case = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        rules = conn.execute(
            "SELECT * FROM rule_results WHERE case_id = ? ORDER BY id", (case_id,)
        ).fetchall()
        diagnosis = conn.execute(
            "SELECT * FROM diagnoses WHERE case_id = ? ORDER BY id DESC LIMIT 1", (case_id,)
        ).fetchone()
        review = conn.execute(
            "SELECT * FROM reviews WHERE case_id = ? ORDER BY id DESC LIMIT 1", (case_id,)
        ).fetchone()
    return {"case": case, "rules": rules, "diagnosis": diagnosis, "review": review}
