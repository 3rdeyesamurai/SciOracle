import multiprocessing
import time
import os
import sys
import yaml
import random
from state_manager import SciOracleStateManager
from skills.symbolic_log import execute as symbolic_execute
from skills.ebm_solve import execute as ebm_execute
from openclaw_interface import OpenClawBridge
from ebm_math_discovery import init_db, retrieve_analogical_conjectures

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def agent_loop_delay(config, key, default_value):
    return float(config.get("scaling", {}).get("agent_loop_delays", {}).get(key, default_value))

def run_oracle_coder(state_manager: SciOracleStateManager, bridge: OpenClawBridge, config):
    """
    Simulates the OpenClaw Oracle_Coder Sub-Agent.
    Running offloaded to system RAM (qwen2.5-coder:32b).
    """
    print("[Oracle_Coder] Initialized on System RAM threads.")
    sleep_s = agent_loop_delay(config, "oracle", 1.0)
    db = init_db("math_knowledge.db")

    def propose_conjecture(state):
        domain = state.get("target_physics_domain")
        analogs = retrieve_analogical_conjectures(db, physics_domain=domain, limit=3)
        seeds = [
            "x**2 + 2*x + 1 = (x + 1)**2",
            "x**2 - 1 = (x - 1)*(x + 1)",
            "m*a = F",
            "V/R = I",
        ]
        if analogs:
            pick = random.choice(analogs)
            mutated = f"{pick['problem_math']} = {pick['solution_math']}"
            return mutated, pick.get("signature")
        return random.choice(seeds), None

    while True:
        state = state_manager.read_state()
        status = state.get("validation_status")
        
        if status == "pending" and not state.get("current_conjecture"):
            # Initial generation
            print("[Oracle_Coder] Generating initial mathematical conjecture...")
            time.sleep(2) # Simulate LLM inference
            conjecture, seed_signature = propose_conjecture(state)
            state_manager.update_state({
                "current_conjecture": conjecture,
                "seed_signature": seed_signature,
                "generated_code": "def example(): return True",
                "validation_status": "validating",
                "iteration_count": state.get("iteration_count", 0) + 1
            })
            bridge.push_state(state_manager.read_state(), source="oracle_coder")
            
        elif status == "failed":
            # Self-Correction Loop
            errors = state.get("validation_errors", [])
            print(f"[Oracle_Coder] Self-Correcting based on error logs: {errors[-1]}")
            time.sleep(2) # Simulate LLM correcting code
            conjecture, seed_signature = propose_conjecture(state)
            state_manager.update_state({
                "current_conjecture": conjecture,
                "seed_signature": seed_signature,
                "generated_code": "def corrected(): return True",
                "validation_status": "validating",
                "validation_errors": [], # clear active errors for the retry
                "iteration_count": state.get("iteration_count", 0) + 1
            })
            bridge.push_state(state_manager.read_state(), source="oracle_coder")
            
        time.sleep(sleep_s)

def run_symbolic_validator(state_manager: SciOracleStateManager, bridge: OpenClawBridge, config):
    """
    Simulates the OpenClaw Symbolic_Validator Sub-Agent.
    Must run on CPU-bound threads (i7-9750H) to prevent GPU crashes if Coder is active.
    """
    print("[Symbolic_Validator] Initialized on CPU-bound threads.")
    sleep_s = agent_loop_delay(config, "validator", 1.0)
    while True:
        state = state_manager.read_state()
        if state.get("validation_status") == "validating":
            print(f"[Symbolic_Validator] Validating conjecture: {state.get('current_conjecture')}")
            
            # Execute the Skill
            result_msg = symbolic_execute()
            print(f"[Symbolic_Validator] {result_msg}")
            bridge.push_state(state_manager.read_state(), source="symbolic_validator")
            
        time.sleep(sleep_s)

def run_ebm_solver(state_manager: SciOracleStateManager, bridge: OpenClawBridge, config, vram_cap: int):
    """
    Simulates the OpenClaw EBM_Solver Sub-Agent.
    Runs on VRAM.
    """
    print(f"[EBM_Solver] Initialized with VRAM Gate Cap: {vram_cap}GB")
    sleep_s = agent_loop_delay(config, "ebm_solver", 2.0)
    while True:
        state = state_manager.read_state()
        if state.get("validation_status") == "passed" and not state.get("discovery_visualized"):
            # Acquire hardware lock
            state_manager.update_state({"hardware_locks": {"gpu_in_use": True}})
            
            print("[EBM_Solver] Executing EBM Minimization on VRAM...")
            result_msg = ebm_execute()
            print(f"[EBM_Solver] {result_msg}")
            bridge.push_state(state_manager.read_state(), source="ebm_solver")
            
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
                
        time.sleep(sleep_s)


def run_openclaw_sync(state_manager: SciOracleStateManager, bridge: OpenClawBridge, config):
    """Synchronize state and ingest OpenClaw commands when integration is enabled."""
    if not bridge.is_active():
        print("[OpenClaw_Bridge] Disabled. Running local-only mode.")
        return

    print(f"[OpenClaw_Bridge] Connected target: {bridge.base_url}")
    sleep_s = agent_loop_delay(config, "openclaw_sync", 2.0)
    while True:
        state = state_manager.read_state()
        hb = bridge.send_heartbeat(state)
        if not hb.get("ok", True):
            print(f"[OpenClaw_Bridge] Heartbeat warning: {hb.get('error')}")

        commands = bridge.fetch_commands()
        for command in commands:
            if bridge.apply_command(state_manager, command):
                print(f"[OpenClaw_Bridge] Applied command: {command.get('type')}")
                bridge.push_state(state_manager.read_state(), source="openclaw_command")

        time.sleep(sleep_s)

def main():
    print("Initializing SciOracle Master Loop...")
    config = load_config()
    vram_cap = config.get("system", {}).get("vram_gate", {}).get("cap_gb", 6)
    
    # Initialize State-First Protocol
    state_manager = SciOracleStateManager(config.get("system", {}).get("memory", {}).get("file", "state.json"))
    bridge = OpenClawBridge(config.get("openclaw", {}))
    
    # Check if this is a test run
    if "--test-run" in sys.argv:
        print("[Test] Verified config, state_manager, and module imports successfully.")
        sys.exit(0)

    # Spawn Sub-Agents across 12 threads using Multiprocessing
    # Oracle on RAM, Validator on CPU, Solver on GPU
    
    p_bridge = multiprocessing.Process(target=run_openclaw_sync, args=(state_manager, bridge, config))
    p_coder = multiprocessing.Process(target=run_oracle_coder, args=(state_manager, bridge, config))
    p_validator = multiprocessing.Process(target=run_symbolic_validator, args=(state_manager, bridge, config))
    p_solver = multiprocessing.Process(target=run_ebm_solver, args=(state_manager, bridge, config, vram_cap))
    
    try:
        p_bridge.start()
        p_coder.start()
        p_validator.start()
        p_solver.start()
        
        p_bridge.join()
        p_coder.join()
        p_validator.join()
        p_solver.join()
    except KeyboardInterrupt:
        print("Shutting down SciOracle Master Loop.")
        p_bridge.terminate()
        p_coder.terminate()
        p_validator.terminate()
        p_solver.terminate()

if __name__ == "__main__":
    main()
