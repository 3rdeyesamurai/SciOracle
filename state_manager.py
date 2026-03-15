import json
import os
import threading

class SciOracleStateManager:
    """
    State-First Protocol Implementation.
    Bypasses standard memory in favor of a file-based .oracle state.
    Immortalizing derivation lineage through Vector DB and Git commits.
    """
    def __init__(self, state_file="scioracle.oracle"):
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
            "vibe_coding_intent": None, # Storing raw natural language input vibe
            "vibe_context": [],         # Persistent context of translated vibes
            "target_physics_domain": None,
            "proof_status": "unverified",
            "counterexample_trace": None,
            "ebm_energy": None,
            "critic_signature": None,    # Symbolic Critic's Handshake signature
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
            
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def write_state(self, state_data):
        """State Write - Output immutable .oracle format payload."""
        temp_file = self.state_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(state_data, f, indent=4)
        
        # Replace atomically
        os.replace(temp_file, self.state_file)
        
        # Trigger All-Seeing Mind sync in background
        threading.Thread(target=self._sync_all_seeing_mind, args=(state_data,), daemon=True).start()
        
    def _sync_all_seeing_mind(self, state_data):
        """
        Background process that pushes .oracle diffs to private GitHub repos 
        and updates semantic vector databases (Pinecone/Milvus) for analogical RAG context.
        """
        # (Mock implementation simulating SaaS transition scale)
        # print("[All-Seeing Mind] Synchronizing .oracle state to Tenant Git repository + Vector Database...")
        pass

    def update_state(self, updates):
        """Convenience method to update specific fields."""
        state = self.read_state()
        state.update(updates)
        self.write_state(state)
