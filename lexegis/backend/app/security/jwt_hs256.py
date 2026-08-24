"""Minimal HS256 JSON Web Tokens over the standard library.

Rationale: the deployment target for this service includes air-gapped review
rooms where the native crypto wheels are a liability. HS256 is HMAC-SHA256 --
`hmac` and `hashlib` are all it needs, and the verification path is constant
time. Only HS256 is accepted; the `alg` header is checked against it explicitly
so that an attacker cannot downgrade a token to `none`.
"""
import base64
import hmac
import json
import time
from hashlib import sha256


class InvalidToken(Exception):
    pass


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64u_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def encode(payload: dict, secret: str, algorithm: str = "HS256") -> str:
    if algorithm != "HS256":
        raise ValueError("only HS256 is supported")
    header = _b64u(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64u(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    signing_input = f"{header}.{body}".encode()
    signature = hmac.new(secret.encode(), signing_input, sha256).digest()
    return f"{header}.{body}.{_b64u(signature)}"


def decode(token: str, secret: str, algorithms: list[str] | None = None, leeway: int = 0) -> dict:
    algorithms = algorithms or ["HS256"]
    if "HS256" not in algorithms:
        raise InvalidToken("unsupported algorithm set")
    try:
        header_b64, body_b64, signature_b64 = token.split(".")
        header = json.loads(_b64u_decode(header_b64))
        payload = json.loads(_b64u_decode(body_b64))
    except (ValueError, UnicodeDecodeError) as exc:
        raise InvalidToken(f"malformed token: {exc}") from exc
    if header.get("alg") != "HS256":
        raise InvalidToken(f"unexpected algorithm {header.get('alg')!r}")
    expected = hmac.new(secret.encode(), f"{header_b64}.{body_b64}".encode(), sha256).digest()
    if not hmac.compare_digest(expected, _b64u_decode(signature_b64)):
        raise InvalidToken("signature mismatch")
    now = time.time()
    if "exp" in payload and now > float(payload["exp"]) + leeway:
        raise InvalidToken("token expired")
    if "nbf" in payload and now + leeway < float(payload["nbf"]):
        raise InvalidToken("token not yet valid")
    return payload
