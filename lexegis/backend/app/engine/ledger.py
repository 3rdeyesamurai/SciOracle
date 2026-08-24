"""Hash-chained, append-only provenance ledger.

Every consequential act of the engine -- a forensic scan, a triage decision, an
equation canonicalisation, a Lean compilation -- is written here. Entries are
chained (entry_hash = H(prev_hash || payload_hash || metadata)) so that any
retroactive edit of the audit trail is detectable by `verify_chain`.
"""
import hashlib
import json
from datetime import datetime, timezone

from ..db import query, query_one, tx

GENESIS = "0" * 64


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append(org_id: str, actor: str, action: str, payload: dict, subject: str | None = None) -> dict:
    body = _canonical(payload)
    payload_hash = _sha(body)
    ts = datetime.now(timezone.utc).isoformat()
    with tx() as conn:
        prev = conn.execute(
            "SELECT entry_hash FROM ledger WHERE org_id = ? ORDER BY seq DESC LIMIT 1", (org_id,)
        ).fetchone()
        prev_hash = prev["entry_hash"] if prev else GENESIS
        entry_hash = _sha("|".join([prev_hash, payload_hash, ts, actor, action, subject or ""]))
        cur = conn.execute(
            """INSERT INTO ledger (org_id, ts, actor, action, subject, payload, payload_hash, prev_hash, entry_hash)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (org_id, ts, actor, action, subject, body, payload_hash, prev_hash, entry_hash),
        )
        seq = cur.lastrowid
    return {"seq": seq, "ts": ts, "action": action, "entry_hash": entry_hash, "prev_hash": prev_hash}


def entries(org_id: str, limit: int = 200, subject: str | None = None) -> list[dict]:
    if subject:
        rows = query(
            "SELECT * FROM ledger WHERE org_id = ? AND subject = ? ORDER BY seq DESC LIMIT ?",
            (org_id, subject, limit),
        )
    else:
        rows = query("SELECT * FROM ledger WHERE org_id = ? ORDER BY seq DESC LIMIT ?", (org_id, limit))
    out = []
    for row in rows:
        item = dict(row)
        try:
            item["payload"] = json.loads(item["payload"])
        except ValueError:
            pass
        out.append(item)
    return out


def verify_chain(org_id: str) -> dict:
    """Recompute the whole chain. Returns the first broken link, if any."""
    rows = query("SELECT * FROM ledger WHERE org_id = ? ORDER BY seq ASC", (org_id,))
    prev_hash = GENESIS
    for row in rows:
        payload_hash = _sha(row["payload"])
        expected = _sha("|".join([prev_hash, payload_hash, row["ts"], row["actor"], row["action"], row["subject"] or ""]))
        if payload_hash != row["payload_hash"]:
            return {"valid": False, "broken_at": row["seq"], "reason": "payload does not match recorded payload_hash"}
        if row["prev_hash"] != prev_hash:
            return {"valid": False, "broken_at": row["seq"], "reason": "prev_hash discontinuity (entry inserted or removed)"}
        if expected != row["entry_hash"]:
            return {"valid": False, "broken_at": row["seq"], "reason": "entry_hash does not match recomputed digest"}
        prev_hash = row["entry_hash"]
    return {"valid": True, "entries": len(rows), "head": prev_hash}


def head(org_id: str) -> str:
    row = query_one("SELECT entry_hash FROM ledger WHERE org_id = ? ORDER BY seq DESC LIMIT 1", (org_id,))
    return row["entry_hash"] if row else GENESIS
