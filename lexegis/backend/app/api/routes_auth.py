from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from ..db import execute, new_id, query_one
from ..engine import ledger
from ..saas import plans
from ..security import auth
from .deps import Principal, current_principal

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=200)
    organisation: str = Field(min_length=2, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ApiKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


@router.post("/signup", status_code=201)
def signup(body: SignupRequest):
    if query_one("SELECT id FROM users WHERE email = ?", (body.email.lower(),)):
        raise HTTPException(status.HTTP_409_CONFLICT, "an account with that email already exists")
    now = datetime.now(timezone.utc).isoformat()
    org_id, user_id = new_id("org"), new_id("usr")
    execute("INSERT INTO orgs (id, name, plan, created_at) VALUES (?,?,?,?)",
            (org_id, body.organisation, "free", now))
    execute("INSERT INTO users (id, org_id, email, password_hash, role, created_at) VALUES (?,?,?,?,?,?)",
            (user_id, org_id, body.email.lower(), auth.hash_password(body.password), "owner", now))
    ledger.append(org_id, user_id, "org.created", {"organisation": body.organisation, "owner": body.email.lower()},
                  subject=org_id)
    token = auth.issue_token(user_id, org_id, "owner")
    return {**token, "org_id": org_id, "user_id": user_id, "plan": "free"}


@router.post("/login")
def login(body: LoginRequest):
    row = query_one("SELECT * FROM users WHERE email = ?", (body.email.lower(),))
    if not row or not auth.verify_password(body.password, row["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
    org = query_one("SELECT * FROM orgs WHERE id = ?", (row["org_id"],))
    token = auth.issue_token(row["id"], row["org_id"], row["role"])
    return {**token, "org_id": row["org_id"], "user_id": row["id"], "plan": org["plan"]}


@router.get("/me")
def me(principal: Principal = Depends(current_principal)):
    org = query_one("SELECT * FROM orgs WHERE id = ?", (principal.org_id,))
    user = query_one("SELECT id, email, role, created_at FROM users WHERE id = ?", (principal.user_id,)) \
        if principal.user_id else None
    return {
        "org": dict(org), "user": dict(user) if user else None, "via": principal.via,
        "entitlements": plans.snapshot(principal.org_id, principal.plan),
    }


@router.get("/api-keys")
def list_keys(principal: Principal = Depends(current_principal)):
    return {"keys": auth.list_api_keys(principal.org_id)}


@router.post("/api-keys", status_code=201)
def create_key(body: ApiKeyRequest, principal: Principal = Depends(current_principal)):
    if principal.via != "jwt":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "API keys may not mint further API keys")
    key = auth.create_api_key(principal.org_id, body.name)
    ledger.append(principal.org_id, principal.actor, "apikey.created",
                  {"key_id": key["id"], "name": body.name, "prefix": key["prefix"]}, subject=key["id"])
    return key


@router.delete("/api-keys/{key_id}")
def revoke_key(key_id: str, principal: Principal = Depends(current_principal)):
    if not auth.revoke_api_key(principal.org_id, key_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "key not found or already revoked")
    ledger.append(principal.org_id, principal.actor, "apikey.revoked", {"key_id": key_id}, subject=key_id)
    return {"revoked": key_id}
