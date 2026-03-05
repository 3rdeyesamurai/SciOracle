import multiprocessing
import time
import os
import sys
import yaml
from state_manager import SciOracleStateManager
from skills.symbolic_log import execute as symbolic_execute
from skills.ebm_solve import execute as ebm_execute

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_oracle_coder(state_manager: SciOracleStateManager):
    """
    Simulates the OpenClaw Oracle_Coder Sub-Agent.
    Running offloaded to system RAM (qwen2.5-coder:32b).
    """
    print("[Oracle_Coder] Initialized on System RAM threads.")
    while True:
        state = state_manager.read_state()
        status = state.get("validation_status")
        
        if status == "pending" and not state.get("current_conjecture"):
            # Initial generation
            print("[Oracle_Coder] Generating initial mathematical conjecture...")
            time.sleep(2) # Simulate LLM inference
            state_manager.update_state({
                "current_conjecture": "x**2 + 2*x + 1 = (x + 1)**2",
                "generated_code": "def example(): return True",
                "validation_status": "validating",
                "iteration_count": state.get("iteration_count", 0) + 1
            })
            
        elif status == "failed":
            # Self-Correction Loop
            errors = state.get("validation_errors", [])
            print(f"[Oracle_Coder] Self-Correcting based on error logs: {errors[-1]}")
            time.sleep(2) # Simulate LLM correcting code
            # Assume it corrects the equation in the next step
            state_manager.update_state({
                "current_conjecture": "x**2 - 1 = (x - 1)*(x + 1)",
                "generated_code": "def corrected(): return True",
                "validation_status": "validating",
                "validation_errors": [], # clear active errors for the retry
                "iteration_count": state.get("iteration_count", 0) + 1
            })
            
        time.sleep(1)

def run_symbolic_validator(state_manager: SciOracleStateManager):
    """
    Simulates the OpenClaw Symbolic_Validator Sub-Agent.
    Must run on CPU-bound threads (i7-9750H) to prevent GPU crashes if Coder is active.
    """
    print("[Symbolic_Validator] Initialized on CPU-bound threads.")
    while True:
        state = state_manager.read_state()
        if state.get("validation_status") == "validating":
            print(f"[Symbolic_Validator] Validating conjecture: {state.get('current_conjecture')}")
            
            # Execute the Skill
            result_msg = symbolic_execute()
            print(f"[Symbolic_Validator] {result_msg}")
            
        time.sleep(1)

def run_ebm_solver(state_manager: SciOracleStateManager, vram_cap: int):
    """
    Simulates the OpenClaw EBM_Solver Sub-Agent.
    Runs on VRAM.
    """
    print(f"[EBM_Solver] Initialized with VRAM Gate Cap: {vram_cap}GB")
    while True:
        state = state_manager.read_state()
        if state.get("validation_status") == "passed" and not state.get("discovery_visualized"):
            # Acquire hardware lock
            state_manager.update_state({"hardware_locks": {"gpu_in_use": True}})
            
            print("[EBM_Solver] Executing EBM Minimization on VRAM...")
            result_msg = ebm_execute()
            print(f"[EBM_Solver] {result_msg}")
            
            # Release hardware lock
            state_manager.update_state({"hardware_locks": {"gpu_in_use": False}})
            
            # Reset for continuous discovery after visualization (or loop termination for demo)
            if state.get("discovery_visualized"):
                print("[Master] Discovery visually proven and completed. Rebooting cycle...")
                time.sleep(5)
                # Restart cycle
                state_manager.update_state({
                    "current_conjecture": None,
                    "validation_status": "pending",
                    "discovery_visualized": False
                })
                
        time.sleep(2)

def main():
    print("Initializing SciOracle Master Loop...")
    config = load_config()
    vram_cap = config.get("system", {}).get("vram_gate", {}).get("cap_gb", 6)
    
    # Initialize State-First Protocol
    state_manager = SciOracleStateManager(config.get("system", {}).get("memory", {}).get("file", "state.json"))
    
    # Check if this is a test run
    if "--test-run" in sys.argv:
        print("[Test] Verified config, state_manager, and module imports successfully.")
        sys.exit(0)

    # Spawn Sub-Agents across 12 threads using Multiprocessing
    # Oracle on RAM, Validator on CPU, Solver on GPU
    
    p_coder = multiprocessing.Process(target=run_oracle_coder, args=(state_manager,))
    p_validator = multiprocessing.Process(target=run_symbolic_validator, args=(state_manager,))
    p_solver = multiprocessing.Process(target=run_ebm_solver, args=(state_manager, vram_cap))
    
    try:
        p_coder.start()
        p_validator.start()
        p_solver.start()
        
        p_coder.join()
        p_validator.join()
        p_solver.join()
    except KeyboardInterrupt:
        print("Shutting down SciOracle Master Loop.")
        p_coder.terminate()
        p_validator.terminate()
        p_solver.terminate()

if __name__ == "__main__":
    main()
