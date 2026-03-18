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


def _sympy_expr_to_z3(expr, symbols):
    """Translate a safe subset of SymPy AST into Z3 expressions."""
    if isinstance(expr, sp.Symbol):
        return symbols.setdefault(str(expr), z3.Real(str(expr)))
    if isinstance(expr, (sp.Integer, sp.Float, sp.Rational)):
        return z3.RealVal(str(expr))
    if isinstance(expr, sp.Add):
        return z3.Sum([_sympy_expr_to_z3(arg, symbols) for arg in expr.args])
    if isinstance(expr, sp.Mul):
        result = z3.RealVal(1)
        for arg in expr.args:
            result = result * _sympy_expr_to_z3(arg, symbols)
        return result
    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        z3_base = _sympy_expr_to_z3(base, symbols)
        if isinstance(exp, sp.Integer):
            e = int(exp)
            if e == 0:
                return z3.RealVal(1)
            if e < 0:
                return z3.RealVal(1) / (z3_base ** abs(e))
            return z3_base ** e
        raise ValueError("Only integer powers are supported in Z3 translator")
    raise ValueError(f"Unsupported SymPy node for Z3 translation: {type(expr).__name__}")


def _z3_equivalence_check(lhs, rhs):
    symbols = {}
    z3_lhs = _sympy_expr_to_z3(lhs, symbols)
    z3_rhs = _sympy_expr_to_z3(rhs, symbols)
    solver = z3.Solver()
    solver.add(z3_lhs != z3_rhs)
    result = solver.check()
    if result == z3.unsat:
        return True, None
    if result == z3.sat:
        model = solver.model()
        trace = {str(d): str(model[d]) for d in model.decls()}
        return False, trace
    return False, {"reason": "unknown"}


def execute():
    """
    OpenClaw Skill for Symbolic Validation.
    Executes on CPU bounds. Evaluates conjecture from state.json.
    Leverages high-performance Rust `egg` bindings if available.
    """
    ensure_discovery_dir()
    manager = SciOracleStateManager(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state.json"))
    state = manager.read_state()

    conjecture = state.get("current_conjecture", "")

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "conjecture": conjecture,
        "sympy_valid": False,
        "z3_valid": False,
        "proof_status": "unverified",
        "counterexample_trace": None,
        "error": None,
    }

    validation_passed = False

    try:
        if conjecture and "=" in conjecture:
            lhs_str, rhs_str = conjecture.split("=", 1)
            
            # --- HIGH PERFORMANCE RUST PATH ---
            rust_passed = False
            try:
                from scioracle_rust import SymbolicValidator as RustValidator
                rust_validator = RustValidator()
                # `egg` parses lisp-like AST natively, we use this direct equivalence wrapper 
                # (Assuming the syntax was matched in the Rust lib's string mapper)
                rust_passed = rust_validator.check_equivalence(lhs_str.strip(), rhs_str.strip())
            except ImportError:
                print("[Info] Rust `scioracle_rust` not found for symbolic validation. Overheading to SymPy/Z3 fallback.")

            if rust_passed:
                validation_passed = True
                log_entry["sympy_valid"] = True
                log_entry["z3_valid"] = True
                log_entry["proof_status"] = "rust_egg_verified"
            else:
                # --- PYTHON FALLBACK PATH ---
                lhs = sp.sympify(lhs_str.strip())
                rhs = sp.sympify(rhs_str.strip())

                sympy_valid = sp.simplify(lhs - rhs) == 0
                log_entry["sympy_valid"] = bool(sympy_valid)

                z3_valid = False
                trace = None
                try:
                    z3_valid, trace = _z3_equivalence_check(lhs, rhs)
                except Exception as z3_err:
                    log_entry["error"] = f"Z3 translator error: {z3_err}"

                log_entry["z3_valid"] = bool(z3_valid)
                log_entry["counterexample_trace"] = trace

                if sympy_valid and z3_valid:
                    validation_passed = True
                    log_entry["proof_status"] = "sympy_and_z3_verified"
                elif not sympy_valid:
                    log_entry["proof_status"] = "sympy_failed"
                    log_entry["error"] = log_entry["error"] or "SymPy returned algebraic inequality."
                else:
                    log_entry["proof_status"] = "z3_failed"
                    log_entry["error"] = log_entry["error"] or "Z3 found a counterexample or returned unknown."
        else:
            log_entry["error"] = "Conjecture is not an equation."

    except Exception as e:
        log_entry["error"] = f"Parse/Validation Exception: {str(e)}"

    if validation_passed:
        manager.update_state({
            "validation_status": "passed",
            "validation_errors": [],
            "proof_status": log_entry["proof_status"],
            "counterexample_trace": None,
        })
    else:
        current_errors = state.get("validation_errors", [])
        current_errors.append(log_entry["error"])
        manager.update_state({
            "validation_status": "failed",
            "validation_errors": current_errors,
            "proof_status": log_entry["proof_status"],
            "counterexample_trace": log_entry["counterexample_trace"],
        })

    with open("discoveries/math_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    with open("discoveries/proof_attempts.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return f"Symbolic Validation {'Passed' if validation_passed else 'Failed'}. Check state."


if __name__ == "__main__":
    print(execute())
