"""Runtime configuration for the Lexegis engine.

Every external dependency (Lean 4, Wolfram, Stripe, an LLM provider) is optional.
When a dependency is absent the corresponding subsystem degrades to a
deterministic local implementation and reports that fact in its output, rather
than silently pretending the check ran.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("LEXEGIS_DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path(os.environ.get("LEXEGIS_DB", DATA_DIR / "lexegis.db"))
BLOB_DIR = Path(os.environ.get("LEXEGIS_BLOBS", DATA_DIR / "blobs"))
BLOB_DIR.mkdir(parents=True, exist_ok=True)

JWT_SECRET = os.environ.get("LEXEGIS_JWT_SECRET", "dev-secret-change-me")
JWT_ALGO = "HS256"
JWT_TTL_SECONDS = int(os.environ.get("LEXEGIS_JWT_TTL", 60 * 60 * 12))

# Optional System 2 backends.
LEAN_BIN = os.environ.get("LEXEGIS_LEAN_BIN", "lean")
LEAN_TIMEOUT = int(os.environ.get("LEXEGIS_LEAN_TIMEOUT", 60))
WOLFRAM_APP_ID = os.environ.get("WOLFRAM_APP_ID", "")
WOLFRAM_MCP_URL = os.environ.get("WOLFRAM_MCP_URL", "")

# Optional billing.
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

MAX_UPLOAD_BYTES = int(os.environ.get("LEXEGIS_MAX_UPLOAD", 32 * 1024 * 1024))
