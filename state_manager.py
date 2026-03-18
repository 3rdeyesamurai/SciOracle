import json
import os
import threading

class SciOracleStateManager:
    """
    Python wrapper mapping to the High-Performance Rust State Manager.
    State-First Protocol Implementation bypassing standard memory.
    Immortalizing derivation lineage through Vector DB and Git commits.
    """
    def __init__(self, state_file="scioracle.oracle"):
        self.state_file = state_file
        try:
            from scioracle_rust import SciOracleStateManager as RustStateManager
            self._rust_mgr = RustStateManager(self.state_file)
        except ImportError:
            print("[Warning] scioracle_rust module missing. Falling back to pure Python state (not recommended for production).")
            self._rust_mgr = None
            if not os.path.exists(self.state_file):
                self._init_fallback_state()

    def _init_fallback_state(self):
        initial_state = {
            "current_conjecture": None,
            "generated_code": None,
            "validation_status": "pending",
            "validation_errors": [],
            "conversation_context": [],
            "vibe_coding_intent": None, 
            "vibe_context": [],         
            "target_physics_domain": None,
            "proof_status": "unverified",
            "counterexample_trace": None,
            "ebm_energy": None,
            "critic_signature": None,    
            "latest_discovery": None,
            "law_declared": False,
            "discovery_visualized": False,
            "iteration_count": 0,
            "hardware_locks": {
                "gpu_in_use": False
            }
        }
        self.write_state(initial_state)

    def read_state(self):
        """State Read - must be called at the beginning of every agent action."""
        if self._rust_mgr:
            return json.loads(self._rust_mgr.read_state())
            
        # Fallback
        if not os.path.exists(self.state_file):
            self._init_fallback_state()
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def write_state(self, state_data):
        """State Write - Output immutable .oracle format payload."""
        if self._rust_mgr:
            self._rust_mgr.update_state(json.dumps(state_data))
        else:
            temp_file = self.state_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=4)
            os.replace(temp_file, self.state_file)
            
        # Trigger All-Seeing Mind sync in background
        threading.Thread(target=self._sync_all_seeing_mind, args=(state_data,), daemon=True).start()
        
    def _sync_all_seeing_mind(self, state_data):
        # Intentionally left empty as a placeholder for later scaling
        pass

    def update_state(self, updates):
        """Convenience method to update specific fields."""
        if self._rust_mgr:
            self._rust_mgr.update_state(json.dumps(updates))
        else:
            state = self.read_state()
            state.update(updates)
            self.write_state(state)
