import json
import os

class SciOracleStateManager:
    """
    State-First Protocol Implementation.
    Bypasses standard memory in favor of a file-based state.json.
    Ensures minimal "Contextual Overload" on the 32GB RAM system.
    """
    def __init__(self, state_file="state.json"):
        self.state_file = state_file
        if not os.path.exists(self.state_file):
            self._init_state()

    def _init_state(self):
        initial_state = {
            "current_conjecture": None,
            "generated_code": None,
            "validation_status": "pending",
            "validation_errors": [],
            "conversation_context": [],
            "target_physics_domain": None,
            "proof_status": "unverified",
            "counterexample_trace": None,
            "ebm_energy": None,
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
        if not os.path.exists(self.state_file):
            self._init_state()
            
        # Optional: Add file locking for multiprocessing safety if required on Windows.
        # Since this is Windows, fcntl is not available, using simple read.
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def write_state(self, state_data):
        """State Write - must be called at the end of every agent action."""
        # Atomic write pattern avoids partial reads by other processes
        temp_file = self.state_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(state_data, f, indent=4)
        
        # Replace atomically
        os.replace(temp_file, self.state_file)
        
    def update_state(self, updates):
        """Convenience method to update specific fields."""
        state = self.read_state()
        state.update(updates)
        self.write_state(state)
