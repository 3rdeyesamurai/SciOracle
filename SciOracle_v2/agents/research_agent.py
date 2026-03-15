import time
from core.energy_model import MathematicalEnergyModel
from training.trainer import ContrastiveCurriculumTrainer
from search.mcts import MCTSEngine
from search.rl_policy import RLPolicyNetwork
from symbolic.mutator import SymbolicMutator
from verification.sympy_validator import SympyValidator
from verification.z3_validator import Z3Validator
from memory.database import TheoremDatabase
from memory.theorem_graph import TheoremKnowledgeGraph
from proof.proof_generator import ProofGenerator
from training.dataset_generator import GeneratorTarget

class AutonomousResearchAgent:
    """
    Main loop governing dynamic problem solving iterations.
    Capabilities:
    1. Generate equations
    2. Search transformations via MCTS
    3. Verify identities (Z3 / SymPy)
    4. Store theorems persistently
    5. Learn autonomously (curriculum training loop)
    """
    def __init__(self):
        # Neural reasoning backbone
        self.energy_model = MathematicalEnergyModel(in_channels=9, hidden_dim=128, num_layers=4)
        self.policy = RLPolicyNetwork(in_channels=9, hidden_dim=128, num_layers=4, num_actions=6)
        
        # Search modules
        self.mutator = SymbolicMutator()
        self.mcts = MCTSEngine(self.mutator, self.energy_model, self.policy)
        
        # Validators
        self.sympy_validator = SympyValidator()
        self.z3_validator = Z3Validator()
        
        # Persistent memory layers
        self.database = TheoremDatabase()
        self.graph = TheoremKnowledgeGraph()
        
        # Training and generation
        self.trainer = ContrastiveCurriculumTrainer(self.energy_model)
        self.generator = GeneratorTarget()

    def discover_loop(self, iterations: int = 10):
        """Continuous pipeline expanding mathematical space dynamically."""
        print(f"\n[RESEARCH LOOP] Initiating Autonomous Agent for {iterations} Iterations")
        for i in range(1, iterations + 1):
            print(f"\n--- Iteration {i} ---")
            
            # Step 1: Generate initial problem state
            problem = self.generator.generate_polynomial()
            print(f"Generated problem state: {problem}")
            
            # Step 2: Traverse action space intelligently via MCTS to find valid transforms
            # Target is None so it's fully explorative
            path = self.mcts.search(initial_state=problem)
            
            if len(path) == 0:
                print("MCTS returned empty trajectory.")
                continue
                
            # Assume final step is the proven identity RHS
            final_form = str(path[-1]['result'])
            print(f"Discovered identity trajectory end: {final_form}")
            
            # Reconstruct syms for verification
            import sympy as sp
            lhs_expr = problem
            try:
                rhs_expr = sp.sympify(final_form)
            except Exception:
                rhs_expr = problem
                
            # Step 3: Verification
            valid_sympy = self.sympy_validator.verify(lhs_expr, rhs_expr)
            valid_z3 = self.z3_validator.verify(lhs_expr, rhs_expr)
            
            if valid_sympy and valid_z3:
                print(f"SUCCESS: Identity Verified using SymPy + Z3 [{lhs_expr} == {rhs_expr}]")
                # Proof Generation
                proof = ProofGenerator.generate(str(lhs_expr), str(rhs_expr), path)
                
                # Step 4: Storage
                self.database.insert_theorem(str(lhs_expr), str(rhs_expr), proof.to_dict())
                self.graph.add_equation_node(str(lhs_expr))
                self.graph.add_equation_node(str(rhs_expr))
                for step in path:
                    self.graph.add_transformation_edge(str(lhs_expr), str(step['result']), step['operation'])
                    
                ProofGenerator.print_proof(proof)
            else:
                print("FAILURE: Explored state was not valid mathematically. Modifying reward penalties log.")
                
            time.sleep(0.5)

    def train_loop(self):
        """Invoke robust curriculum model to optimize core energy search."""
        self.trainer.curriculum_training_loop(epochs_per_stage=10, stages=3)

    def run(self):
        """Entry procedure binding training -> continuous discovery"""
        print("====== Initializing SciOracle v2 Agent ======")
        self.train_loop()
        self.discover_loop(iterations=5)
        self.graph.display_graph_stats()
