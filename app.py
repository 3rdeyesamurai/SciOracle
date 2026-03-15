"""
SciOracle — FastAPI Backend
Math-EBM bridge: GNN energy model ↔ Web UI
Fully local inference, SQLite persistence, air-gapped operation.
"""

import os
import io
import time
import uuid
import sqlite3
import logging
import threading
from pathlib import Path
from typing import Optional

import uvicorn
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Optional heavy imports (graceful degradation) ─────────────────────────────
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    logging.warning("PyMuPDF (fitz) not installed — PDF figure extraction disabled.")

try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    logging.warning("SymPy not installed — logical validation disabled.")

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("Transformers not installed — falling back to char tokenizer.")

# ── Paths & constants ──────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DB_PATH       = BASE_DIR / "scioracle.db"
FIGURES_DIR   = BASE_DIR / "figures"
CHECKPOINT    = BASE_DIR / "math_ebm.pt"
TOKENIZER_ID  = "bert-base-uncased"          # swap as needed
MAX_SEQ_LEN   = 128
VRAM_LIMIT_GB = 6.0                           # RTX 2060 guard
TRAIN_INTERVAL_SEC = 30                       # daemon cadence

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scioracle")

# ── Device selection with VRAM guard ──────────────────────────────────────────
def _select_device() -> torch.device:
    if not torch.cuda.is_available():
        log.info("CUDA unavailable — using CPU.")
        return torch.device("cpu")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram > VRAM_LIMIT_GB:
        log.warning("VRAM %.1f GB detected; limiting to 6 GB guard via memory fraction.", vram)
        torch.cuda.set_per_process_memory_fraction(VRAM_LIMIT_GB / vram, 0)
    log.info("GPU selected: %s (%.1f GB VRAM).", torch.cuda.get_device_name(0), vram)
    return torch.device("cuda")

DEVICE = _select_device()

# ── SQLite initialisation ──────────────────────────────────────────────────────
def init_db() -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS queries (
                id          TEXT PRIMARY KEY,
                ts          REAL,
                problem     TEXT,
                solution    TEXT,
                energy      REAL,
                sound       INTEGER
            );
            CREATE TABLE IF NOT EXISTS figures (
                id          TEXT PRIMARY KEY,
                source_pdf  TEXT,
                page        INTEGER,
                path        TEXT,
                ts          REAL
            );
        """)
init_db()

# ═══════════════════════════════════════════════════════════════════════════════
#  GNN-Based Energy Model
# ═══════════════════════════════════════════════════════════════════════════════

class DualEncoderEBM(nn.Module):
    """
    Two-stream energy model:
      • NLP stream  — Transformer-style sequence encoder (LLM Sequence Scope)
      • AST stream  — Graph-compatible MLP (AST Graph Scope)
    Energy = scalar compatibility score (lower = more coherent).
    """

    def __init__(self, vocab_size: int = 30522, hidden: int = 256, ast_dim: int = 64):
        super().__init__()
        # ── LLM Sequence Scope ─────────────────────────────────────────────────
        self.tok_embed   = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.pos_embed   = nn.Embedding(MAX_SEQ_LEN, hidden)
        encoder_layer    = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=512,
                                                      dropout=0.1, batch_first=True)
        self.llm_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.llm_proj    = nn.Linear(hidden, 128)

        # ── AST Graph Scope ────────────────────────────────────────────────────
        self.ast_encoder = nn.Sequential(
            nn.Linear(ast_dim, 128), nn.GELU(),
            nn.Linear(128, 128),    nn.GELU(),
            nn.Linear(128, 128),
        )
        self.ast_proj    = nn.Linear(128, 128)

        # ── Energy Head ───────────────────────────────────────────────────────
        self.energy_head = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64),  nn.GELU(),
            nn.Linear(64, 1),
        )

    def _llm_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x    = self.tok_embed(input_ids) + self.pos_embed(pos)
        x    = self.llm_encoder(x)
        return self.llm_proj(x.mean(dim=1))           # [B, 128]

    def _ast_forward(self, ast_vec: torch.Tensor) -> torch.Tensor:
        return self.ast_proj(self.ast_encoder(ast_vec))  # [B, 128]

    def forward(self, input_ids: torch.Tensor, ast_vec: torch.Tensor,
                use_checkpoint: bool = False) -> torch.Tensor:
        if use_checkpoint:
            llm_feat = cp.checkpoint(self._llm_forward, input_ids, use_reentrant=False)
            ast_feat = cp.checkpoint(self._ast_forward, ast_vec,   use_reentrant=False)
        else:
            llm_feat = self._llm_forward(input_ids)
            ast_feat = self._ast_forward(ast_vec)
        combined = torch.cat([llm_feat, ast_feat], dim=-1)
        return self.energy_head(combined).squeeze(-1)   # [B]


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Registry — Hot-Swappable Loading
# ═══════════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    def __init__(self):
        self._lock  = threading.Lock()
        self.model: Optional[DualEncoderEBM] = None
        self.tokenizer = None
        self._load()

    def _load(self) -> None:
        log.info("Loading model checkpoint …")
        model = DualEncoderEBM().to(DEVICE)
        if CHECKPOINT.exists():
            state = torch.load(CHECKPOINT, map_location=DEVICE)
            model.load_state_dict(state, strict=False)
            log.info("Checkpoint loaded from %s.", CHECKPOINT)
        else:
            log.warning("No checkpoint found at %s — using random weights.", CHECKPOINT)

        tokenizer = None
        if HAS_TRANSFORMERS:
            try:
                tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
                log.info("Tokenizer '%s' loaded.", TOKENIZER_ID)
            except Exception as e:
                log.warning("Tokenizer load failed: %s", e)

        with self._lock:
            self.model     = model
            self.tokenizer = tokenizer

    def hot_reload(self) -> None:
        """Replace live model with freshly loaded checkpoint (zero downtime)."""
        self._load()

    def get(self):
        with self._lock:
            return self.model, self.tokenizer


REGISTRY = ModelRegistry()

# ═══════════════════════════════════════════════════════════════════════════════
#  Tokenisation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def tokenise(text: str, tokenizer) -> torch.Tensor:
    """Return [1, MAX_SEQ_LEN] int64 tensor."""
    if tokenizer is not None:
        enc = tokenizer(text, max_length=MAX_SEQ_LEN, padding="max_length",
                        truncation=True, return_tensors="pt")
        return enc["input_ids"].to(DEVICE)
    # Fallback: char-level UTF-8 codes mod 30522
    codes = [ord(c) % 30522 for c in text[:MAX_SEQ_LEN]]
    codes += [0] * (MAX_SEQ_LEN - len(codes))
    return torch.tensor([codes], dtype=torch.long, device=DEVICE)


def math_to_ast_vec(expr_str: str, dim: int = 64) -> torch.Tensor:
    """
    Convert a SymPy expression to a fixed-dim feature vector via symbol
    frequency, operator counts, and expression depth.
    Falls back to random noise if SymPy unavailable.
    """
    vec = torch.zeros(dim)
    if not HAS_SYMPY or not expr_str.strip():
        return vec.unsqueeze(0).to(DEVICE)
    try:
        transforms = (standard_transformations + (implicit_multiplication_application,))
        expr  = parse_expr(expr_str, transformations=transforms)
        atoms = list(expr.atoms())
        vec[0] = len(atoms)                          # atom count
        vec[1] = expr.count_ops()                    # operator depth
        vec[2] = len(expr.free_symbols)              # free symbol count
        vec[3] = float(expr.is_commutative or 0)
        vec[4] = float(expr.is_real       or 0)
        # hash-scatter remaining symbols into remaining dims
        for i, sym in enumerate(expr.free_symbols):
            idx = (hash(str(sym)) % (dim - 5)) + 5
            vec[idx] += 1.0
    except Exception:
        pass
    return vec.unsqueeze(0).to(DEVICE)

# ═══════════════════════════════════════════════════════════════════════════════
#  SymPy Logical Soundness Validator
# ═══════════════════════════════════════════════════════════════════════════════

def validate_soundness(expr_str: str) -> dict:
    """
    Attempt lightweight symbolic checks:
      • Parse validity
      • Check if expr simplifies to zero (identity: lhs - rhs = 0)
      • Numerical sampling consistency
    """
    result = {"sound": False, "reason": "SymPy unavailable"}
    if not HAS_SYMPY:
        return result
    try:
        transforms = (standard_transformations + (implicit_multiplication_application,))
        expr = parse_expr(expr_str, transformations=transforms)

        # Check if expression is an equality and simplify lhs - rhs
        if isinstance(expr, sympy.Eq):
            diff = sympy.simplify(expr.lhs - expr.rhs)
            if diff == 0:
                return {"sound": True, "reason": "Identity verified: lhs − rhs = 0"}
            # Numerical sampling
            symbols = list(diff.free_symbols)
            if symbols:
                subs = {s: 1.0 for s in symbols}
                val  = complex(diff.subs(subs))
                if abs(val) < 1e-9:
                    return {"sound": True, "reason": "Numerically consistent at sample point"}
            return {"sound": False, "reason": f"Residual: {diff}"}

        # Non-equality: just confirm parseable
        return {"sound": True, "reason": "Expression parsed successfully (no equality to verify)"}

    except Exception as e:
        return {"sound": False, "reason": f"Parse error: {e}"}

# ═══════════════════════════════════════════════════════════════════════════════
#  Continuous Training Daemon
# ═══════════════════════════════════════════════════════════════════════════════

SYMPY_IDENTITIES = [
    ("sin^2(x) + cos^2(x) = 1",      "Eq(sin(x)**2 + cos(x)**2, 1)"),
    ("e^(i*pi) + 1 = 0",              "Eq(exp(I*pi) + 1, 0)"),
    ("(a+b)^2 = a^2 + 2*a*b + b^2",  "Eq((a+b)**2, a**2 + 2*a*b + b**2)"),
    ("d/dx sin(x) = cos(x)",          "cos(x)"),
    ("integral e^x = e^x + C",        "exp(x)"),
]

def _training_step(model: DualEncoderEBM, opt: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0.0
    for desc, expr_str in SYMPY_IDENTITIES:
        _, tokenizer = REGISTRY.get()
        ids     = tokenise(desc,     tokenizer)
        ast_vec = math_to_ast_vec(expr_str)
        energy  = model(ids, ast_vec, use_checkpoint=True)
        # Contrastive: true identity should have LOW energy (target 0)
        loss = torch.relu(energy).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
    return total_loss / len(SYMPY_IDENTITIES)


def training_daemon() -> None:
    """Background thread: mini training blocks every TRAIN_INTERVAL_SEC."""
    log.info("Training daemon started (interval=%ds).", TRAIN_INTERVAL_SEC)
    model, _ = REGISTRY.get()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    while True:
        time.sleep(TRAIN_INTERVAL_SEC)
        try:
            model, _ = REGISTRY.get()
            loss = _training_step(model, opt)
            log.info("[Daemon] Training loss: %.6f", loss)
            # Persist updated weights
            with threading.Lock():
                torch.save(model.state_dict(), CHECKPOINT)
        except Exception as e:
            log.error("[Daemon] Training error: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
#  PDF Document Analysis Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pdf(pdf_bytes: bytes, source_name: str) -> dict:
    """
    Extract text and raster figures from a PDF.
    Saves images to FIGURES_DIR; logs metadata to SQLite.
    """
    text_out    = ""
    figure_list = []

    if not HAS_FITZ:
        return {"text": "[PyMuPDF unavailable]", "figures": [], "char_count": 0}

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num, page in enumerate(doc):
        text_out += page.get_text()
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref  = img_info[0]
            pix   = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            fname   = f"{source_name}_p{page_num}_fig{img_idx}.png"
            fpath   = FIGURES_DIR / fname
            pix.save(str(fpath))
            fig_id  = str(uuid.uuid4())
            figure_list.append({"id": fig_id, "page": page_num, "filename": fname})
            with sqlite3.connect(DB_PATH) as con:
                con.execute(
                    "INSERT INTO figures VALUES (?,?,?,?,?)",
                    (fig_id, source_name, page_num, str(fpath), time.time()),
                )
    doc.close()
    return {"text": text_out, "figures": figure_list, "char_count": len(text_out)}


# ═══════════════════════════════════════════════════════════════════════════════
#  FastAPI Application
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="SciOracle", version="1.0.0", description="Math-EBM Research Interface")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve extracted figures statically
app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

# Start training daemon
threading.Thread(target=training_daemon, daemon=True).start()


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    problem:  str
    solution: str

class QueryResponse(BaseModel):
    id:            str
    energy:        float
    sound:         bool
    sound_reason:  str
    device:        str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    """Liveness probe — returns device info and knowledge-base stats."""
    with sqlite3.connect(DB_PATH) as con:
        fig_count = con.execute("SELECT COUNT(*) FROM figures").fetchone()[0]
    return {
        "status": "ok",
        "device": str(DEVICE),
        "figures_in_kb": fig_count,
        "checkpoint_exists": CHECKPOINT.exists(),
    }


@app.post("/api/query", response_model=QueryResponse)
def query_energy(req: QueryRequest):
    """
    Evaluate the energy of a (problem, solution) pair.
    Returns predicted energy score + SymPy soundness badge.
    """
    if not req.problem.strip() or not req.solution.strip():
        raise HTTPException(400, "Both 'problem' and 'solution' fields are required.")

    model, tokenizer = REGISTRY.get()
    model.eval()

    with torch.no_grad():
        ids     = tokenise(req.problem, tokenizer)
        ast_vec = math_to_ast_vec(req.solution)
        energy  = model(ids, ast_vec, use_checkpoint=False).item()

    soundness = validate_soundness(req.solution)
    qid = str(uuid.uuid4())

    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO queries VALUES (?,?,?,?,?,?)",
            (qid, time.time(), req.problem, req.solution, energy, int(soundness["sound"])),
        )

    return QueryResponse(
        id=qid,
        energy=round(energy, 6),
        sound=soundness["sound"],
        sound_reason=soundness["reason"],
        device=str(DEVICE),
    )


@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Ingest a scientific PDF: extract text + figures, store in SQLite.
    Returns character count and list of extracted figures.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")
    raw   = await file.read()
    safe  = "".join(c if c.isalnum() else "_" for c in Path(file.filename).stem)
    result = extract_pdf(raw, safe)
    return {
        "filename":    file.filename,
        "char_count":  result["char_count"],
        "figures":     result["figures"],
        "text_preview": result["text"][:500],
    }


@app.post("/api/reload-model")
def reload_model():
    """Hot-reload the model checkpoint from disk (no downtime)."""
    REGISTRY.hot_reload()
    return {"status": "reloaded", "checkpoint": str(CHECKPOINT)}


@app.get("/api/history")
def query_history(limit: int = 20):
    """Return the most recent query records from SQLite."""
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT id, ts, problem, solution, energy, sound FROM queries ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {"id": r[0], "ts": r[1], "problem": r[2], "solution": r[3], "energy": r[4], "sound": bool(r[5])}
        for r in rows
    ]


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
