"""Tenant authentication: password hashing, JWTs, API keys.

Passwords use PBKDF2-HMAC-SHA256 with a per-user salt (stdlib only, no native
build step -- this service is meant to be deployable inside an air-gapped
review environment). API keys are stored only as digests; the plaintext is
returned exactly once, at creation.
"""
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from . import jwt_hs256 as jwt

from ..config import JWT_ALGO, JWT_SECRET, JWT_TTL_SECONDS
from ..db import execute, new_id, query, query_one

PBKDF2_ROUNDS = 240_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, rounds, salt_hex, digest_hex = encoded.split("$")
        if algo != "pbkdf2_sha256":
            return False
        expected = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(rounds))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(expected.hex(), digest_hex)


def issue_token(user_id: str, org_id: str, role: str) -> dict:
    now = datetime.now(timezone.utc)
    expires = now + timedelta(seconds=JWT_TTL_SECONDS)
    token = jwt.encode({"sub": user_id, "org": org_id, "role": role,
                        "iat": int(now.timestamp()), "exp": int(expires.timestamp())},
                       JWT_SECRET, algorithm=JWT_ALGO)
    return {"access_token": token, "token_type": "bearer", "expires_at": expires.isoformat()}


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.InvalidToken:
        return None


def create_api_key(org_id: str, name: str) -> dict:
    secret = secrets.token_urlsafe(32)
    key_id = new_id("key")
    plaintext = f"lxg_{key_id.split('_')[1][:8]}_{secret}"
    prefix = plaintext[:16]
    execute("INSERT INTO api_keys (id, org_id, name, prefix, key_hash, created_at) VALUES (?,?,?,?,?,?)",
            (key_id, org_id, name, prefix, hashlib.sha256(plaintext.encode()).hexdigest(),
             datetime.now(timezone.utc).isoformat()))
    return {"id": key_id, "name": name, "prefix": prefix, "api_key": plaintext,
            "warning": "stored as a digest; this is the only time the plaintext is shown"}


def resolve_api_key(plaintext: str) -> dict | None:
    row = query_one("SELECT * FROM api_keys WHERE key_hash = ? AND revoked_at IS NULL",
                    (hashlib.sha256(plaintext.encode()).hexdigest(),))
    return dict(row) if row else None


def list_api_keys(org_id: str) -> list[dict]:
    return [dict(r) for r in query(
        "SELECT id, name, prefix, created_at, revoked_at FROM api_keys WHERE org_id = ? ORDER BY created_at DESC",
        (org_id,))]


def revoke_api_key(org_id: str, key_id: str) -> bool:
    cur = execute("UPDATE api_keys SET revoked_at = ? WHERE org_id = ? AND id = ? AND revoked_at IS NULL",
                  (datetime.now(timezone.utc).isoformat(), org_id, key_id))
    return cur.rowcount > 0
