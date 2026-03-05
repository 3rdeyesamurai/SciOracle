import os
import sqlite3
import threading
import time
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import fitz  # PyMuPDF
import torch

# Assuming the model is in the parent directory, we add it to the path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ebm_math_discovery import (
    MathEBM, LLMSeqTokenizer, ASTGraphTokenizer, load_checkpoint,
    train_ebm, init_db, log_to_db, evaluate_energy, sympy_to_nl_str, nl_to_sympy_str
)
import sympy as sp

app = FastAPI(title="SciOracle Math EBM Platform")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold hot-swappable model components
MODEL = None
LLM_TOKENIZER = None
AST_TOKENIZER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'math_ebm.pt'))
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'math_knowledge.db'))
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))

os.makedirs(FIGURES_DIR, exist_ok=True)

# Database upgrade for figures
def update_db_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT image_path FROM math_discoveries LIMIT 1")
    except sqlite3.OperationalError:
        print("Upgrading database schema to support figures...")
        cursor.execute("ALTER TABLE math_discoveries ADD COLUMN image_path TEXT")
        conn.commit()
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
        while True:
            try:
                # Runs a mini-training block then halts
                # Use CPU=False typically, but we will pass cpu according to hardware
                train_ebm(save_path=MODEL_PATH, use_cpu=False) 
                
                # Signal hot reload
                reload_model()
                
                # Sleep briefly
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
    
    return {
        "problem_nl": p_text,
        "problem_math": p_math_guess,
        "solution_math": s_math,
        "energy": energy,
        "is_sound": is_sound
    }

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
