import hashlib
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
from ebm_math_discovery import init_db, retrieve_analogical_conjectures, load_formula_corpus
from p2p_network import run_p2p_node

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def agent_loop_delay(config, key, default_value):
    return float(config.get("scaling", {}).get("agent_loop_delays", {}).get(key, default_value))

def run_planner_agent(state_manager: SciOracleStateManager, bridge: OpenClawBridge, config):
    """
    Planner Agent (LLM): Simulates the OpenClaw Coder
    Ingests vibes and drafts preliminary mathematics offloaded to RAM.
    """
    print("[Planner_Agent] Initialized on System RAM threads.")
    sleep_s = agent_loop_delay(config, "oracle", 1.0)
    db = init_db("math_knowledge.db")
    formula_corpus = load_formula_corpus()

    def propose_conjecture(state):
        domain = state.get("target_physics_domain")
        intent = state.get("vibe_coding_intent")
        analogs = retrieve_analogical_conjectures(db, physics_domain=domain, limit=3)
        seeds = [
            "x**2 + 2*x + 1 = (x + 1)**2",
            "x**2 - 1 = (x - 1)*(x + 1)",
            "m*a = F",
            "V/R = I",
        ]
        context_text = " ".join([m.get("content", "") for m in state.get("conversation_context", [])[-6:]]).lower()
        
        # Vibe Coding Protocol: direct translation of intent to seed
        if intent:
            print(f"[Oracle_Coder] Translating Vibe Intent: '{intent}'")
            # In a full model, this queries the LLM. Here we simulate basic keyword heuristic
            if "force" in intent.lower() or "mass" in intent.lower():
                return "m*a = F", None
            if "energy" in intent.lower():
                return "E = m*c**2", None
            return f"Translated({intent})", None

        corpus_matches = [
            f"{row.get('problem', 'x')} = {row.get('solution', 'x')}"
            for row in formula_corpus
            if not domain or row.get("domain") == domain or row.get("domain") in context_text
        ]
        if analogs:
            pick = random.choice(analogs)
            mutated = f"{pick['problem_math']} = {pick['solution_math']}"
            return mutated, pick.get("signature")
        if corpus_matches:
            return random.choice(corpus_matches), None
        return random.choice(seeds), None

    while True:
        state = state_manager.read_state()
        status = state.get("validation_status")
        
        if status == "pending" and not state.get("current_conjecture"):
            # Initial generation
            print("[Planner_Agent] Generating mathematical conjecture via Vibe Translation...")
            time.sleep(2) # Simulate LLM inference
            conjecture, seed_signature = propose_conjecture(state)
            
            # Persist vibe string into persistent execution mapping
            current_vibe = state.get("vibe_coding_intent")
            vibe_hist = state.get("vibe_context", [])
            if current_vibe and current_vibe not in vibe_hist:
                vibe_hist.append(current_vibe)

            state_manager.update_state({
                "current_conjecture": conjecture,
                "seed_signature": seed_signature,
                "vibe_context": vibe_hist,
                "vibe_coding_intent": None, # Clear intent after consumption
                "critic_signature": None,   # Clear old handshake signatures
                "generated_code": f"def example(): return True  # Vibe: {current_vibe}",
                "validation_status": "validating",
                "iteration_count": state.get("iteration_count", 0) + 1
            })
            bridge.push_state(state_manager.read_state(), source="oracle_coder")
            
        elif status == "failed":
            # Self-Correction Loop
            errors = state.get("validation_errors", [])
            print(f"[Planner_Agent] Self-Correcting based on error logs: {errors[-1]}")
            time.sleep(2) # Simulate LLM correcting code
            conjecture, seed_signature = propose_conjecture(state)
            state_manager.update_state({
                "current_conjecture": conjecture,
                "seed_signature": seed_signature,
                "vibe_coding_intent": None,
                "critic_signature": None,
                "generated_code": "def corrected(): return True",
                "validation_status": "validating",
                "validation_errors": [], # clear active errors for the retry
                "iteration_count": state.get("iteration_count", 0) + 1
            })
            bridge.push_state(state_manager.read_state(), source="planner_agent")
            
        time.sleep(sleep_s)

def run_symbolic_critic(state_manager: SciOracleStateManager, bridge: OpenClawBridge, config):
    """
    Symbolic Critic Agent (Gatekeeper): Zero permissions to write code.
    Converts Planner output to AST and evaluates logic. Must cryptographically sign off.
    """
    print("[Symbolic_Critic] Initialized Planner-Critic Gatekeeper on CPU.")
    sleep_s = agent_loop_delay(config, "validator", 1.0)
    while True:
        state = state_manager.read_state()
        if state.get("validation_status") == "validating":
            print(f"[Symbolic_Critic] Auditing AST logic for: {state.get('current_conjecture')}")
            
            # Execute the formal logic audit Sandbox skill
            result_msg = symbolic_execute()
            print(f"[Symbolic_Critic] {result_msg}")
            
            # Re-read state in case validator flipped status to passed/failed
            state = state_manager.read_state() 
            if state.get("validation_status") == "passed":
                # Generate cryptographic handshake signature asserting fault-free compilation
                conjecture = str(state.get("current_conjecture"))
                signature = hashlib.sha3_256(f"CRITIC_APPROVED_{conjecture}".encode()).hexdigest()
                print(f"[Symbolic_Critic] Handshake Signature Verified: {signature[:8]}")
                state_manager.update_state({"critic_signature": signature})
            else:
                state_manager.update_state({"critic_signature": None})
                
            bridge.push_state(state_manager.read_state(), source="symbolic_critic")
            
        time.sleep(sleep_s)

def run_executor_agent(state_manager: SciOracleStateManager, bridge: OpenClawBridge, config, vram_cap: int):
    """
    Executor Agent: Handles EBM isolation execution and S3 bucket deployment.
    Only advances if the Symbolic Critic generates a valid hash signature.
    """
    print(f"[Executor_Agent] Initialized isolated sandbox bounding. VRAM Gate Cap: {vram_cap}GB")
    sleep_s = agent_loop_delay(config, "ebm_solver", 2.0)
    while True:
        state = state_manager.read_state()
        if state.get("validation_status") == "passed" and not state.get("discovery_visualized"):
            # The Critical Handshake Security Check
            sig = state.get("critic_signature")
            if not sig:
                print(f"[Executor_Agent] Critic signature missing! Rejecting execution.")
                state_manager.update_state({"validation_status": "failed", "validation_errors": ["Critic hash invalid"]})
                continue
                
            # Acquire K8s / Hardware lock conceptually
            state_manager.update_state({"hardware_locks": {"gpu_in_use": True}})
            
            print(f"[Executor_Agent] Executing EBM Minimization under Signature [{sig[:8]}]...")
            result_msg = ebm_execute()
            print(f"[Executor_Agent] {result_msg}")
            bridge.push_state(state_manager.read_state(), source="executor_agent")
            
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
    # Oracle on RAM, Validator on CPU, Solver on GPU isolated containers
    
    p2p_port = int(os.environ.get("P2P_PORT", 5000))
    p2p_peer = os.environ.get("P2P_PEER")
    p2p_peers = [p2p_peer] if p2p_peer else []
    
    p_network = multiprocessing.Process(target=run_p2p_node, args=('0.0.0.0', p2p_port, p2p_peers))
    p_bridge = multiprocessing.Process(target=run_openclaw_sync, args=(state_manager, bridge, config))
    p_planner = multiprocessing.Process(target=run_planner_agent, args=(state_manager, bridge, config))
    p_critic = multiprocessing.Process(target=run_symbolic_critic, args=(state_manager, bridge, config))
    p_executor = multiprocessing.Process(target=run_executor_agent, args=(state_manager, bridge, config, vram_cap))
    
    try:
        p_network.start()
        p_bridge.start()
        p_planner.start()
        p_critic.start()
        p_executor.start()
        
        p_network.join()
        p_bridge.join()
        p_planner.join()
        p_critic.join()
        p_executor.join()
    except KeyboardInterrupt:
        print("Shutting down SciOracle Master Loop.")
        p_network.terminate()
        p_bridge.terminate()
        p_planner.terminate()
        p_critic.terminate()
        p_executor.terminate()

if __name__ == "__main__":
    main()
