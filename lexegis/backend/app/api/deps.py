"""Request-scoped identity and quota dependencies."""
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status

from ..db import query_one
from ..saas import plans
from ..security import auth


@dataclass
class Principal:
    user_id: str | None
    org_id: str
    role: str
    plan: str
    via: str

    @property
    def actor(self) -> str:
        return self.user_id or f"apikey:{self.org_id}"


def current_principal(authorization: str | None = Header(default=None),
                      x_api_key: str | None = Header(default=None)) -> Principal:
    if x_api_key:
        record = auth.resolve_api_key(x_api_key)
        if not record:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")
        org = query_one("SELECT * FROM orgs WHERE id = ?", (record["org_id"],))
        return Principal(None, record["org_id"], "service", org["plan"] if org else "free", "api_key")

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token or X-API-Key")
    claims = auth.decode_token(authorization.split(" ", 1)[1])
    if not claims:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired token")
    org = query_one("SELECT * FROM orgs WHERE id = ?", (claims["org"],))
    if not org:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "organisation no longer exists")
    return Principal(claims["sub"], claims["org"], claims.get("role", "member"), org["plan"], "jwt")


def require_quota(metric: str, quantity: int = 1):
    def dependency(principal: Principal = Depends(current_principal)) -> Principal:
        allowed, detail = plans.check(principal.org_id, principal.plan, metric, quantity)
        if not allowed:
            raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED,
                                {"error": "plan_limit_reached", **detail, "plan": principal.plan})
        return principal
    return dependency


def owned_matter(matter_id: str, principal: Principal) -> dict:
    row = query_one("SELECT * FROM matters WHERE id = ? AND org_id = ?", (matter_id, principal.org_id))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "matter not found")
    return dict(row)
