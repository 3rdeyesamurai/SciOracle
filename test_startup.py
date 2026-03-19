import sys, os, traceback
sys.path.insert(0, '.')

# Test the exact app.py startup sequence step by step
print("Step 1: core imports...", flush=True)
import json, sqlite3, threading, time, asyncio, yaml
import torch
print(f"  torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)

print("Step 2: fastapi imports...", flush=True)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import fitz
print("  fastapi+fitz OK", flush=True)

print("Step 3: local modules...", flush=True)
from acp_bridge import acp_router
print("  acp_bridge OK", flush=True)
from state_manager import SciOracleStateManager
print("  state_manager OK", flush=True)

print("Step 4: ebm_math_discovery imports...", flush=True)
from ebm_math_discovery import (
    MathEBM, LLMSeqTokenizer, ASTGraphTokenizer, load_checkpoint,
    train_ebm, init_db, log_to_db, evaluate_energy, sympy_to_nl_str, nl_to_sympy_str,
    declare_theorem_if_sound, infer_physics_application, retrieve_analogical_conjectures
)
print("  ebm_math_discovery OK", flush=True)

print("Step 5: sympy...", flush=True)
import sympy as sp
print("  sympy OK", flush=True)

print("Step 6: init_db...", flush=True)
DB_PATH = os.path.abspath("math_knowledge.db")
conn = init_db(DB_PATH)
conn.close()
print("  init_db OK", flush=True)

print("Step 7: load_checkpoint...", flush=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.abspath("math_ebm.pt")
model, llm_tok, ast_tok = load_checkpoint(MODEL_PATH, DEVICE)
print(f"  Model loaded OK on {DEVICE}", flush=True)

print("\n✅ ALL STARTUP STEPS PASSED — backend/app.py should start correctly.", flush=True)
