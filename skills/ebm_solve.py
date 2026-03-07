import os
import sys
import torch
import subprocess
import pickle
import sympy as sp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state_manager import SciOracleStateManager
from ebm_math_discovery import evaluate_energy, load_checkpoint, init_db, log_to_db, declare_theorem_if_sound, store_proof_attempt

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

            db_conn = init_db(os.path.join(base_dir, "math_knowledge.db"))
            try:
                is_sound = (sp.simplify(sp.sympify(p_nl) - sp.sympify(s_math)) == 0)
            except Exception:
                is_sound = False
            proof_status = state.get("proof_status", "unverified")
            counterexample_trace = state.get("counterexample_trace")
            log_to_db(db_conn, p_nl, p_nl, s_math, s_math, energy, is_sound, proof_status=proof_status)
            declaration = declare_theorem_if_sound(
                db_conn, p_nl, p_nl, s_math, s_math, energy, is_sound, proof_status=proof_status
            )
            if declaration:
                seed_signature = state.get("seed_signature")
                if seed_signature:
                    c = db_conn.cursor()
                    c.execute(
                        "INSERT INTO conjecture_lineage (parent_signature, child_signature, relation_type) VALUES (?, ?, ?)",
                        (seed_signature, declaration.get("signature"), "analogical_mutation"),
                    )
                store_proof_attempt(
                    db_conn,
                    declaration.get("signature"),
                    declaration.get("theorem_status", "unknown"),
                    declaration.get("confidence", 0.0),
                    proof_status,
                    {"repeatability": declaration.get("repeatability", 0.0), "class": declaration.get("stability_class", "unknown")},
                    counterexample_trace,
                )
                db_conn.commit()
                
            manager.update_state({
                "ebm_energy": energy,
                "latest_discovery": declaration,
                "law_declared": bool(declaration and declaration.get("law_declaration")),
            })
            
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
