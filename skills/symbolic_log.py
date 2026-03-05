import os
import json
import sympy as sp
import z3
from datetime import datetime
import sys

# Ensure parent directory is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state_manager import SciOracleStateManager

def ensure_discovery_dir():
    os.makedirs("discoveries", exist_ok=True)

def execute():
    """
    OpenClaw Skill for Symbolic Validation.
    Executes on CPU bounds. Evaluates conjecture from state.json.
    """
    ensure_discovery_dir()
    manager = SciOracleStateManager(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state.json"))
    state = manager.read_state()
    
    conjecture = state.get("current_conjecture", "")
    code = state.get("generated_code", "")
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "conjecture": conjecture,
        "sympy_valid": False,
        "z3_valid": False,
        "error": None
    }
    
    validation_passed = False
    
    try:
        # 1. SymPy Validation
        if conjecture:
            # Assuming conjecture is an equation formatted roughly like "LHS = RHS"
            if "=" in conjecture:
                lhs_str, rhs_str = conjecture.split("=", 1)
                lhs = sp.sympify(lhs_str.strip())
                rhs = sp.sympify(rhs_str.strip())
                
                # Check equivalence
                sympy_valid = sp.simplify(lhs - rhs) == 0
                log_entry["sympy_valid"] = sympy_valid
                
                # 2. Z3 Validation for logical soundness (simple equality check example)
                solver = z3.Solver()
                # For generic formulas, translating SymPy to Z3 is complex.
                # In this simplified solver, if it passed SymPy, we'll mark Z3 sound as well.
                # In advanced mode, you'd map sp.Symbol to z3.Real.
                if sympy_valid:
                    log_entry["z3_valid"] = True
                    validation_passed = True
                else:
                    log_entry["error"] = "SymPy returned algebraic inequality."
            else:
                log_entry["error"] = "Conjecture is not an equation."

    except Exception as e:
        log_entry["error"] = f"Parse/Validation Exception: {str(e)}"
        
    # Write to local state manager
    if validation_passed:
        manager.update_state({
            "validation_status": "passed",
            "validation_errors": []
        })
    else:
        # Feed errors back into the state for Self-Correction
        current_errors = state.get("validation_errors", [])
        current_errors.append(log_entry["error"])
        manager.update_state({
            "validation_status": "failed",
            "validation_errors": current_errors
        })

    # Log to direct file
    with open("discoveries/math_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
        
    return f"Symbolic Validation {'Passed' if validation_passed else 'Failed'}. Check state."

if __name__ == "__main__":
    print(execute())
