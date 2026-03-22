import os
import sqlite3
import threading
import time
import asyncio
import importlib.util
import yaml
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import fitz  # PyMuPDF
import torch

# Assuming the model is in the parent directory, we add it to the path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from acp_bridge import acp_router
from ebm_math_discovery import (
    MathEBM, LLMSeqTokenizer, ASTGraphTokenizer, load_checkpoint,
    train_ebm, init_db, log_to_db, evaluate_energy, sympy_to_nl_str, nl_to_sympy_str,
    declare_theorem_if_sound, infer_physics_application, retrieve_analogical_conjectures
)
from state_manager import SciOracleStateManager
import sympy as sp

app = FastAPI(title="SciOracle Math EBM Platform")
MULTIPART_AVAILABLE = importlib.util.find_spec("multipart") is not None

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold hot-swappable model components
app.include_router(acp_router, prefix="/acp")
MODEL = None
LLM_TOKENIZER = None
AST_TOKENIZER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'math_ebm.pt'))
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'math_knowledge.db'))
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))
STATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'state.json'))
DISCOVERIES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'discoveries'))

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DISCOVERIES_DIR, exist_ok=True)
app.mount("/discoveries", StaticFiles(directory=DISCOVERIES_DIR), name="discoveries")


def load_scaling_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def resolve_runtime_training_profile(config):
    scaling = config.get("scaling", {}) if isinstance(config, dict) else {}
    profile_name = str(scaling.get("profile", "high")).lower()
    use_cpu_if_no_cuda = bool(scaling.get("fallback_to_cpu", True))
    return {
        "profile": profile_name,
        "overrides": scaling.get("training_overrides", {}),
    }, use_cpu_if_no_cuda

# Database upgrade for figures
def update_db_schema():
    conn = init_db(DB_PATH)
    conn.close()

update_db_schema()

def reload_model():
    """Reloads the model from disk (Hot-Swapping)"""
    global MODEL, LLM_TOKENIZER, AST_TOKENIZER
    if os.path.exists(MODEL_PATH):
        try:
            MODEL, LLM_TOKENIZER, AST_TOKENIZER = load_checkpoint(MODEL_PATH, DEVICE)
            print("Model successfully reloaded into memory.")
        except Exception as e:
            print(f"Failed to reload model: {e}")
    else:
        print("Model file not found. Waiting for training loop...")

# Initial Model Load
reload_model()

# Background Training Thread
class TrainingWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        
    def run(self):
        print("Starting continuous training daemon...")
        config = load_scaling_config()
        profile, fallback_to_cpu = resolve_runtime_training_profile(config)
        while True:
            try:
                # Load training state to resume
                resume_state = None
                if os.path.exists(MODEL_PATH):
                    try:
                        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
                        resume_state = ckpt.get("training_state", None)
                    except:
                        pass
                
                train_ebm(save_path=MODEL_PATH, use_cpu=fallback_to_cpu, compute_profile=profile, force_retrain=True, resume_from_state=resume_state)
                
                # Signal hot reload
                reload_model()
                
                # Sleep briefly if we finish everything, otherwise loop picks up next epoch/phase
                time.sleep(10)
            except Exception as e:
                print(f"Training loop error: {e}")
                time.sleep(30)

# Start background worker
worker = TrainingWorker()
worker.start()

# API Endpoints
class QueryRequest(BaseModel):
    problem: str
    solution: str

class ChatRequest(BaseModel):
    message: str
    proposed_solution: str | None = None

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "figures_count": len(os.listdir(FIGURES_DIR))
    }

@app.post("/api/query")
def evaluate_query(req: QueryRequest):
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model is currently training and not yet loaded.")
        
    p_text = req.problem
    s_text = req.solution
    p_math_guess = nl_to_sympy_str(p_text)
    s_math = nl_to_sympy_str(s_text)
    
    is_sound = False
    try:
        is_sound = (sp.simplify(sp.sympify(p_math_guess) - sp.sympify(s_math)) == 0)
    except:
        pass
        
    db_conn = sqlite3.connect(DB_PATH)
    energy = evaluate_energy(MODEL, LLM_TOKENIZER, AST_TOKENIZER, p_text, s_math, DEVICE)
    
    if isinstance(energy, str):
        return {"error": energy}
        
    log_to_db(db_conn, p_text, p_math_guess, s_text, s_math, energy, is_sound)
    declaration = declare_theorem_if_sound(db_conn, p_text, p_math_guess, s_text, s_math, energy, is_sound)
    
    return {
        "problem_nl": p_text,
        "problem_math": p_math_guess,
        "solution_math": s_math,
        "energy": energy,
        "is_sound": is_sound,
        "discovery_declaration": declaration
    }

@app.post("/api/chainlink/verify")
def chainlink_oracle_verify(req: QueryRequest):
    """
    Fast execution endpoint prioritized for 9s Chainlink DON timeouts.
    Used by Smart Contracts to cryptographically mint tokens on-chain.
    Bypasses heavier SQLite discovery persistence.
    """
    if not MODEL:
        raise HTTPException(status_code=503, detail="Mathematical EBM Model Offline")
        
    p_math_guess = nl_to_sympy_str(req.problem)
    s_math = nl_to_sympy_str(req.solution)
    
    is_sound = False
    try:
        is_sound = (sp.simplify(sp.sympify(p_math_guess) - sp.sympify(s_math)) == 0)
    except Exception:
        pass
        
    energy = evaluate_energy(MODEL, LLM_TOKENIZER, AST_TOKENIZER, req.problem, s_math, DEVICE)
    
    if isinstance(energy, str):
        return {"error": energy}
    
    return {
        "energy": float(energy),
        "is_sound": bool(is_sound),
        "problem_math": p_math_guess,
        "solution_math": s_math
    }
    
@app.post("/api/chat")
def chat_research(req: ChatRequest):
    """Conversational context endpoint that stores context and suggests symbolic seeds."""
    manager = SciOracleStateManager(STATE_PATH)
    state = manager.read_state()
    history = state.get("conversation_context", [])
    history.append({"role": "user", "content": req.message})
    history = history[-20:]

    domain, attribution = infer_physics_application(req.message, req.message, req.proposed_solution or "")

    conn = sqlite3.connect(DB_PATH)
    analogs = retrieve_analogical_conjectures(conn, physics_domain=domain if domain != "symbolic_algebra" else None, limit=5)
    conn.close()

    manager.update_state({
        "conversation_context": history,
        "target_physics_domain": domain if domain != "symbolic_algebra" else None,
    })

    response = {
        "message": req.message,
        "inferred_domain": domain,
        "attribution_method": attribution,
        "analogical_candidates": analogs,
        "context_window": len(history),
    }

    if req.proposed_solution and MODEL is not None:
        query = QueryRequest(problem=req.message, solution=req.proposed_solution)
        response["evaluation"] = evaluate_query(query)

    return response

@app.get("/api/research/graph")
def research_graph_summary(limit: int = 200):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT id, timestamp, energy, is_sound, theorem_status, physics_domain
        FROM math_discoveries
        ORDER BY id DESC
        LIMIT ?
        ''',
        (int(limit),),
    )
    rows = cursor.fetchall()
    conn.close()

    timeline = [
        {
            "id": row[0],
            "timestamp": row[1],
            "energy": row[2],
            "is_sound": bool(row[3]),
            "theorem_status": row[4],
            "physics_domain": row[5] or "unassigned",
        }
        for row in reversed(rows)
    ]

    domain_counts = {}
    for point in timeline:
        domain_counts[point["physics_domain"]] = domain_counts.get(point["physics_domain"], 0) + 1

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT src_label, dst_label
        FROM conjecture_graph_edges
        ORDER BY id DESC
        LIMIT ?
    ''', (int(limit) * 4,))
    edges = cursor.fetchall()
    conn.close()
    degree = {}
    for src, dst in edges:
        degree[src] = degree.get(src, 0) + 1
        degree[dst] = degree.get(dst, 0) + 1
    top_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "points": timeline,
        "domain_counts": domain_counts,
        "graph_centrality": [{"node": n, "degree": d} for n, d in top_nodes],
        "communities": [{"label": dom, "size": c} for dom, c in domain_counts.items()],
        "size": len(timeline),
    }

@app.get("/api/research/lineage")
def research_lineage(limit: int = 200):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT parent_signature, child_signature, relation_type, created_at
        FROM conjecture_lineage
        ORDER BY id DESC
        LIMIT ?
        ''',
        (int(limit),),
    )
    rows = cursor.fetchall()
    conn.close()
    return {
        "edges": [
            {
                "parent_signature": r[0],
                "child_signature": r[1],
                "relation_type": r[2],
                "created_at": r[3],
            }
            for r in rows
        ],
        "size": len(rows),
    }

if MULTIPART_AVAILABLE:
    @app.post("/api/analyze")
    async def analyze_document(file: UploadFile = File(...)):
        """Parses a PDF, extracting text for EBM logic and saving figures."""
        if not MODEL:
            raise HTTPException(status_code=503, detail="Model not loaded.")
            
        contents = await file.read()
        
        # Write to temp file for PyMuPDF
        temp_path = f"target_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)
            
        extracted_text = []
        extracted_images = []
        
        # Parse PDF using PyMuPDF (fitz)
        try:
            doc = fitz.open(temp_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    extracted_text.append(text.replace("\n", " ").strip())
                    
                # Extract images (Graphical Figure Database mapping)
                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_name = f"figure_{file.filename}_p{page_num}_i{img_index}.{image_ext}"
                    image_path = os.path.join(FIGURES_DIR, image_name)
                    
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    extracted_images.append(image_name)
                    
        except Exception as e:
            if os.path.exists(temp_path): os.remove(temp_path)
            raise HTTPException(status_code=400, detail=str(e))
            
        if os.path.exists(temp_path): os.remove(temp_path)

        # For simplicity, we just evaluate the first few sentences as "problems" conceptually
        # Realistically, you would chunk this and send it through a proper LLM query agent
        context_blob = " ".join(extracted_text)[:500] 
        
        return {
            "filename": file.filename,
            "characters_extracted": sum(len(t) for t in extracted_text),
            "figures_extracted": len(extracted_images),
            "figure_names": extracted_images,
            "context_preview": context_blob
        }
else:
    print("python-multipart not installed; /api/analyze upload endpoint disabled.")

# --- PoD Cryptocurrency Dashboard Endpoints ---

@app.get("/api/wallet")
def get_wallet_info():
    """Mock wallet info returning stats based on the local miner."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Assuming the local node mines as "SciOracle_Local_Miner"
    miner_addr = "SciOracle_Local_Miner"
    cursor.execute("SELECT COUNT(*) FROM blocks WHERE miner_address = ?", (miner_addr,))
    balance = cursor.fetchone()[0]
    
    cursor.execute("SELECT index_id, timestamp, hash, energy_score FROM blocks WHERE miner_address = ? ORDER BY index_id DESC LIMIT 10", (miner_addr,))
    history = [{"index": r[0], "timestamp": r[1], "hash": r[2], "reward": 1.0, "energy": r[3]} for r in cursor.fetchall()]
    conn.close()
    
    return {
        "address": miner_addr,
        "balance": balance,
        "symbol": "POD",
        "recent_transactions": history
    }

@app.get("/api/network")
def get_network_stats():
    """Returns network connectivity and PoD difficulty."""
    conn = sqlite3.connect(DB_PATH)
    from backend.blockchain import DiscoveryLedger
    ledger = DiscoveryLedger(conn)
    current_difficulty = ledger.get_current_difficulty()
    
    # Calculate Hash Rate (Blocks in last hour vs expected)
    now = time.time()
    one_hour_ago = now - 3600
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM blocks WHERE timestamp > ?", (one_hour_ago,))
    blocks_per_hour = cursor.fetchone()[0]
    conn.close()
    
    return {
        "active_peers": 3,  # Mocked for visualization unless parsing active P2P connections
        "hash_rate": f"{blocks_per_hour} Blocks/hr",
        "dynamic_difficulty": current_difficulty,
        "uptime": "99.99%"
    }

@app.get("/api/explorer/blocks")
def get_recent_blocks(limit: int = 15):
    """The Visual Block Explorer data feed."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # We join blocks with math_discoveries to extract the AST image_path if it exists.
    # Blocks store the message dict as JSON.
    try:
        cursor.execute("SELECT index_id, timestamp, data_json, prev_hash, hash, energy_score, miner_address FROM blocks ORDER BY index_id DESC LIMIT ?", (limit,))
        blocks_data = cursor.fetchall()
        
        blocks_feed = []
        for r in blocks_data:
            data_dict = json.loads(r[2]) if r[2] else {}
            signature = data_dict.get("conjecture_signature", "")
            
            # Fetch image_path for this signature if missing from data_dict
            image_path = data_dict.get("image_path")
            if not image_path and signature:
                cursor.execute("SELECT image_path FROM math_discoveries WHERE conjecture_signature = ?", (signature,))
                row = cursor.fetchone()
                if row and row[0]:
                    image_path = row[0]
            
            blocks_feed.append({
                "height": r[0],
                "timestamp": r[1],
                "hash": r[4],
                "prev_hash": r[3],
                "miner": r[6],
                "energy": r[5],
                "problem_math": data_dict.get("problem_math", ""),
                "solution_math": data_dict.get("solution_math", ""),
                "ast_chart": f"/{image_path}" if image_path else None
            })
    except Exception as e:
        print(f"Error fetching blocks: {e}")
        blocks_feed = []
    finally:
        conn.close()
        
    return {"blocks": blocks_feed}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
