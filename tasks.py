"""
OpenClaw Task Definitions for SciOracle
"""

def generate_conjecture_task():
    return {
        "name": "Generate Code and Math Conjecture",
        "description": "Utilize Oracle_Coder to produce Python code and mathematical equations based on prior knowledge.",
        "expected_output": "Updates state.json with current_conjecture and generated_code."
    }

def validate_symbolic_task():
    return {
        "name": "Symbolic Validation",
        "description": "Utilize Symbolic_Validator to log logical validity using SymPy and Z3. Executes skills/symbolic_log.py.",
        "expected_output": "Updates state.json validation_status to passed or failed."
    }

def ebm_minimization_task():
    return {
        "name": "Energy-Based Minimization",
        "description": "Minimizes the energy of symbolic expressions to find the most stable mathematical discovery. Executes skills/ebm_solve.py.",
        "expected_output": "Outputs to discoveries/stability_matrix.bin and triggers visual proof if Energy < 0.05."
    }

def get_all_scioracle_tasks():
    return [
        generate_conjecture_task(),
        validate_symbolic_task(),
        ebm_minimization_task()
    ]
