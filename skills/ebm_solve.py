import os
import sys
import torch
import subprocess
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state_manager import SciOracleStateManager
from ebm_math_discovery import evaluate_energy, load_checkpoint

def ensure_discovery_dir():
    os.makedirs("discoveries", exist_ok=True)

def generate_visual_proof(conjecture):
    """Triggers Matplotlib/Manim subprocess if Energy is extremely stable."""
    script_path = os.path.join(os.path.dirname(__file__), "visualize_proof.py")
    
    # Create simple matplotlib script iteratively
    if not os.path.exists(script_path):
        with open(script_path, "w") as f:
            f.write('''import matplotlib.pyplot as plt
import sys
import os
os.makedirs("discoveries", exist_ok=True)
formula = sys.argv[1]
plt.figure(figsize=(10, 4))
plt.text(0.5, 0.5, f"${formula}$", fontsize=20, ha='center', va='center')
plt.axis('off')
plt.title(f"Visual Proof generated")
plt.savefig("discoveries/latest_proof.png")
''')
    
    subprocess.Popen([sys.executable, script_path, conjecture])
    print("Visual proof spawned.")

def execute():
    """
    OpenClaw Skill for EBM Minimization.
    Executes on GPU VRAM. Evaluates the problem/solution energy.
    """
    ensure_discovery_dir()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manager = SciOracleStateManager(os.path.join(base_dir, "state.json"))
    state = manager.read_state()
    
    conjecture = state.get("current_conjecture", "")
    
    if state.get("validation_status") != "passed" or not conjecture:
        return "EBM Solved skipped: Validation not passed or conjecture missing."
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(base_dir, "math_ebm.pt")
    
    if not os.path.exists(model_path):
        return "Failed: math_ebm.pt checkpoint missing. Train EBM first."
        
    try:
        model, llm_tokenizer, ast_tokenizer = load_checkpoint(model_path, device)
        
        # Assuming conjecture is "LHS = RHS". Energy mapping between problem & correct node trees.
        parts = conjecture.split("=")
        if len(parts) == 2:
            p_nl = parts[0].strip()
            s_math = parts[1].strip()
            
            energy = evaluate_energy(model, llm_tokenizer, ast_tokenizer, p_nl, s_math, device)
            
            if isinstance(energy, str):
                return f"EBM Evaluation returned error string: {energy}"
                
            manager.update_state({"ebm_energy": energy})
            
            # Save matrix to binary dump
            with open("discoveries/stability_matrix.bin", "wb") as f:
                pickle.dump({"energy": energy, "formula": conjecture}, f)
                
            if energy < 0.05:
                # Trigger discovery visualization output
                generate_visual_proof(conjecture)
                manager.update_state({"discovery_visualized": True})
                
            return f"EBM evaluated Energy: {energy:.4f}"
            
    except Exception as e:
        return f"EBM solve error: {str(e)}"

if __name__ == "__main__":
    print(execute())
