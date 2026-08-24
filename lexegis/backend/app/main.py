"""Lexegis — neuro-symbolic discrepancy validation and equation archiving."""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import routes_auth, routes_billing, routes_core, routes_demo, routes_math
from .config import WOLFRAM_APP_ID, WOLFRAM_MCP_URL
from .db import init_db
from .engine.mathx.verify import lean_verify
from .saas.plans import PLANS

DESCRIPTION = """
A dual-process engine for international lawfare.

* **System 1 (heuristic)** — forensic ingestion, injection triage, semantic extraction with spans.
* **System 2 (deliberative)** — equality saturation over e-graphs, SymPy/Wolfram/Lean 4 adjudication.
* **Provenance** — every decision is written to a hash-chained ledger that can be re-verified.
"""

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Lexegis Engine", version="1.0.0", description=DESCRIPTION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    return response


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


app.include_router(routes_auth.router)
app.include_router(routes_core.router)
app.include_router(routes_math.router)
app.include_router(routes_billing.router)
app.include_router(routes_demo.router)


@app.get("/api/v1/health", tags=["meta"])
def health():
    import sympy as sp

    lean = lean_verify(sp.Eq(sp.Symbol("x"), sp.Symbol("x")))
    return {
        "status": "ok",
        "engine": "lexegis",
        "version": app.version,
        "system2_backends": {
            "sympy": {"available": True, "version": sp.__version__},
            "lean4": {"available": lean.get("available", False), "detail": lean.get("detail")},
            "wolfram": {"available": bool(WOLFRAM_APP_ID or WOLFRAM_MCP_URL),
                        "transport": "mcp" if WOLFRAM_MCP_URL else ("api" if WOLFRAM_APP_ID else None)},
        },
        "plans": list(PLANS),
    }


@app.get("/", tags=["meta"])
def root():
    return {"service": "Lexegis Engine", "docs": "/docs", "health": "/api/v1/health"}
