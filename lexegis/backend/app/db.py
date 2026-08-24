"""Thin SQLite data-access layer.

Deliberately dependency-free: the schema is small, the queries are simple, and
keeping stdlib sqlite3 means the whole engine runs from a single file on disk
(useful for air-gapped deployments common in this problem domain).
"""
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Iterable

from .config import DB_PATH

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS orgs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    plan TEXT NOT NULL DEFAULT 'free',
    stripe_customer_id TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL REFERENCES orgs(id),
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'owner',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL REFERENCES orgs(id),
    name TEXT NOT NULL,
    prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    revoked_at TEXT
);
CREATE TABLE IF NOT EXISTS matters (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL REFERENCES orgs(id),
    name TEXT NOT NULL,
    forum TEXT,
    jurisdictions TEXT,
    last_analysis TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    matter_id TEXT NOT NULL REFERENCES matters(id),
    filename TEXT NOT NULL,
    media_type TEXT,
    sha256 TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    tier TEXT NOT NULL DEFAULT 'L3',
    quarantined INTEGER NOT NULL DEFAULT 0,
    forensics TEXT,
    triage TEXT,
    extraction TEXT,
    text_path TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS findings (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    matter_id TEXT NOT NULL,
    category TEXT NOT NULL,
    subtype TEXT,
    severity TEXT NOT NULL,
    confidence REAL NOT NULL,
    title TEXT NOT NULL,
    detail TEXT,
    spans TEXT,
    provenance TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS equations (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    matter_id TEXT,
    document_id TEXT,
    source_text TEXT NOT NULL,
    latex TEXT,
    sympy_repr TEXT,
    canonical_key TEXT,
    content_mathml TEXT,
    openmath TEXT,
    omdoc TEXT,
    verification TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    subject TEXT,
    payload TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    entry_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS usage_events (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    period TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_docs_matter ON documents(matter_id);
CREATE INDEX IF NOT EXISTS idx_findings_matter ON findings(matter_id);
CREATE INDEX IF NOT EXISTS idx_eq_canonical ON equations(canonical_key);
CREATE INDEX IF NOT EXISTS idx_ledger_org ON ledger(org_id, seq);
CREATE INDEX IF NOT EXISTS idx_usage ON usage_events(org_id, metric, period);
"""


def connect() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return conn


MIGRATIONS = [
    ("matters", "last_analysis", "ALTER TABLE matters ADD COLUMN last_analysis TEXT"),
]


def init_db() -> None:
    conn = connect()
    conn.executescript(SCHEMA)
    for table, column, statement in MIGRATIONS:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(statement)
    conn.commit()


@contextmanager
def tx():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def query(sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    return connect().execute(sql, tuple(params)).fetchall()


def query_one(sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
    return connect().execute(sql, tuple(params)).fetchone()


def execute(sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
    with tx() as conn:
        return conn.execute(sql, tuple(params))


def row_to_dict(row: sqlite3.Row | None, json_fields: tuple[str, ...] = ()) -> dict | None:
    if row is None:
        return None
    out = dict(row)
    for field in json_fields:
        if out.get(field):
            try:
                out[field] = json.loads(out[field])
            except (TypeError, ValueError):
                pass
    return out
