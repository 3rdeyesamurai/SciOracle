"""Billing surface.

Stripe is optional. Without credentials the endpoints operate in
self-serve mode: the plan changes locally and every change is written to the
ledger, which is what a pilot deployment needs. With credentials configured the
checkout endpoint returns a real Stripe session.
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from ..config import STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET
from ..db import execute, query_one
from ..engine import ledger
from ..saas import plans
from .deps import Principal, current_principal

router = APIRouter(prefix="/api/v1/billing", tags=["billing"])


class PlanChange(BaseModel):
    plan: str


@router.get("/plans")
def list_plans():
    return {"plans": plans.PLANS, "billing_backend": "stripe" if STRIPE_SECRET_KEY else "self_serve"}


@router.get("/subscription")
def subscription(principal: Principal = Depends(current_principal)):
    return plans.snapshot(principal.org_id, principal.plan)


@router.post("/subscription")
def change_plan(body: PlanChange, principal: Principal = Depends(current_principal)):
    if body.plan not in plans.PLANS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"unknown plan {body.plan!r}")
    if principal.via != "jwt" or principal.role != "owner":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "only an organisation owner may change the plan")
    if STRIPE_SECRET_KEY:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "Stripe is configured; start a checkout session instead of setting the plan directly")
    previous = principal.plan
    execute("UPDATE orgs SET plan = ? WHERE id = ?", (body.plan, principal.org_id))
    ledger.append(principal.org_id, principal.actor, "plan.changed",
                  {"from": previous, "to": body.plan}, subject=principal.org_id)
    return plans.snapshot(principal.org_id, body.plan)


@router.post("/checkout")
def checkout(body: PlanChange, principal: Principal = Depends(current_principal)):
    if body.plan not in plans.PLANS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"unknown plan {body.plan!r}")
    if not STRIPE_SECRET_KEY:
        return {"mode": "self_serve", "detail": "STRIPE_SECRET_KEY unset; call POST /subscription instead",
                "plan": plans.PLANS[body.plan]}
    import httpx
    org = query_one("SELECT * FROM orgs WHERE id = ?", (principal.org_id,))
    response = httpx.post(
        "https://api.stripe.com/v1/checkout/sessions",
        auth=(STRIPE_SECRET_KEY, ""),
        data={
            "mode": "subscription",
            "success_url": "https://app.lexegis.ai/billing?status=success",
            "cancel_url": "https://app.lexegis.ai/billing?status=cancelled",
            "client_reference_id": principal.org_id,
            "line_items[0][price_data][currency]": "usd",
            "line_items[0][price_data][unit_amount]": plans.PLANS[body.plan]["price_usd_month"] * 100,
            "line_items[0][price_data][recurring][interval]": "month",
            "line_items[0][price_data][product_data][name]": f"Lexegis {plans.PLANS[body.plan]['name']}",
            "line_items[0][quantity]": 1,
            "metadata[org_id]": principal.org_id,
            "metadata[plan]": body.plan,
            "metadata[org_name]": org["name"] if org else "",
        },
        timeout=20.0,
    )
    if response.status_code >= 400:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"stripe error: {response.text[:300]}")
    session = response.json()
    return {"mode": "stripe", "checkout_url": session.get("url"), "session_id": session.get("id")}


@router.post("/webhook")
async def webhook(request: Request):
    """Stripe webhook. Signature verification is mandatory when a secret is set."""
    payload = await request.body()
    if STRIPE_WEBHOOK_SECRET:
        import hashlib
        import hmac
        import time
        header = request.headers.get("stripe-signature", "")
        parts = dict(p.split("=", 1) for p in header.split(",") if "=" in p)
        timestamp, signature = parts.get("t"), parts.get("v1")
        if not timestamp or not signature:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "missing stripe-signature header")
        if abs(time.time() - int(timestamp)) > 300:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "stale webhook timestamp")
        expected = hmac.new(STRIPE_WEBHOOK_SECRET.encode(),
                            f"{timestamp}.".encode() + payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid webhook signature")
    import json
    event = json.loads(payload or b"{}")
    obj = event.get("data", {}).get("object", {})
    metadata = obj.get("metadata", {}) or {}
    org_id, plan = metadata.get("org_id"), metadata.get("plan")
    if event.get("type") in ("checkout.session.completed", "customer.subscription.updated") and org_id and plan:
        execute("UPDATE orgs SET plan = ?, stripe_customer_id = ? WHERE id = ?",
                (plan, obj.get("customer"), org_id))
        ledger.append(org_id, "stripe", "plan.changed", {"to": plan, "event": event.get("type")}, subject=org_id)
    elif event.get("type") == "customer.subscription.deleted" and org_id:
        execute("UPDATE orgs SET plan = 'free' WHERE id = ?", (org_id,))
        ledger.append(org_id, "stripe", "plan.changed", {"to": "free", "event": event.get("type")}, subject=org_id)
    return {"received": True}
