import sys, os, traceback
sys.path.insert(0, '.')

import torch
from ebm_math_discovery import load_checkpoint

device = torch.device("cpu")
try:
    model, llm_tok, ast_tok = load_checkpoint("math_ebm.pt", device)
    print("SUCCESS")
except Exception as e:
    print("ERROR:", e)
    traceback.print_exc()
