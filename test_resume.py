import torch
import os
from ebm_math_discovery import (
    MathEBM, LLMSeqTokenizer, ASTGraphTokenizer, 
    save_checkpoint, load_checkpoint
)
import torch.nn as nn

def test_resume_logic():
    print("Testing Resume Logic Persistence...")
    path = "test_resume.pt"
    llm = LLMSeqTokenizer()
    ast = ASTGraphTokenizer()
    model = MathEBM(llm.vocab_size, ast.vocab_size)
    
    # Mock state
    target_state = {"phase": "Cold Start (Arithmetic)", "epoch": 42, "global_step": 1337}
    
    print(f"Saving state: {target_state}")
    save_checkpoint(model, llm, ast, path=path, training_state=target_state)
    
    # Load back
    print("Loading back with return_state=True...")
    _, _, _, loaded_state, _ = load_checkpoint(path, device="cpu", return_state=True)
    
    print(f"Loaded state: {loaded_state}")
    assert loaded_state["epoch"] == 42
    assert loaded_state["phase"] == "Cold Start (Arithmetic)"
    assert loaded_state["global_step"] == 1337
    print("Persistence check: PASSED ✓")
    
    if os.path.exists(path):
        os.remove(path)

if __name__ == "__main__":
    try:
        test_resume_logic()
        print("\nALL RESUME TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
