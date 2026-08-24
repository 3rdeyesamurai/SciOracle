"""Plans, entitlements and metering.

Quotas are enforced at the API boundary rather than inside the engine, so that a
plan limit can never change an analytical result -- only whether the analysis is
permitted to run.
"""
from datetime import datetime, timezone

from ..db import execute, new_id, query_one

PLANS: dict[str, dict] = {
    "free": {
        "name": "Evidence Desk",
        "price_usd_month": 0,
        "limits": {"documents_per_month": 25, "matters": 2, "seats": 1,
                   "equations_per_month": 100, "analyses_per_month": 25},
        "features": ["Forensic ingestion & injection triage", "Discrepancy detection",
                     "SymPy verification", "Equation archive (Content MathML / OpenMath / OMDoc)",
                     "Hash-chained audit ledger"],
    },
    "pro": {
        "name": "Counsel",
        "price_usd_month": 690,
        "limits": {"documents_per_month": 2000, "matters": 50, "seats": 10,
                   "equations_per_month": 20000, "analyses_per_month": 2000},
        "features": ["Everything in Evidence Desk", "Cross-document contradiction graph",
                     "Equality-saturation prior-art search", "Wolfram adjudication (bring your own key)",
                     "API access & webhooks", "Chronology export"],
    },
    "enterprise": {
        "name": "Tribunal",
        "price_usd_month": 4200,
        "limits": {"documents_per_month": 100000, "matters": 1000, "seats": 250,
                   "equations_per_month": 1000000, "analyses_per_month": 100000},
        "features": ["Everything in Counsel", "Lean 4 certification of archived claims",
                     "Air-gapped / on-premise deployment", "Ledger export with chain attestation",
                     "SSO, custom retention, dedicated support"],
    },
}


def period(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{now.year:04d}-{now.month:02d}"


def limits_for(plan: str) -> dict:
    return PLANS.get(plan, PLANS["free"])["limits"]


def usage(org_id: str, metric: str, at: str | None = None) -> int:
    row = query_one(
        "SELECT COALESCE(SUM(quantity), 0) AS total FROM usage_events WHERE org_id = ? AND metric = ? AND period = ?",
        (org_id, metric, at or period()))
    return int(row["total"]) if row else 0


def record(org_id: str, metric: str, quantity: int = 1) -> None:
    execute("INSERT INTO usage_events (id, org_id, metric, quantity, period, created_at) VALUES (?,?,?,?,?,?)",
            (new_id("use"), org_id, metric, quantity, period(), datetime.now(timezone.utc).isoformat()))


def check(org_id: str, plan: str, metric: str, requested: int = 1) -> tuple[bool, dict]:
    limit = limits_for(plan).get(metric)
    if limit is None:
        return True, {"metric": metric, "limit": None, "used": 0}
    used = usage(org_id, metric)
    allowed = used + requested <= limit
    return allowed, {"metric": metric, "limit": limit, "used": used, "requested": requested,
                     "remaining": max(0, limit - used)}


def snapshot(org_id: str, plan: str) -> dict:
    current = period()
    return {
        "plan": plan,
        "plan_name": PLANS.get(plan, PLANS["free"])["name"],
        "period": current,
        "limits": limits_for(plan),
        "usage": {metric: usage(org_id, metric, current) for metric in limits_for(plan)},
        "features": PLANS.get(plan, PLANS["free"])["features"],
    }
