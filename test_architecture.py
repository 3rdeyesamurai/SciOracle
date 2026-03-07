import torch
from agentic_model_architecture import SciOracleDigitalMind

def test_architecture():
    print("=== Testing SciOracle Digital Mind Architecture ===")
    
    # Initialize the Mind
    d_model = 256
    mind = SciOracleDigitalMind(d_model=d_model)
    
    # Print the model structure briefly
    print(f"[Init] SciOracle Digital Mind initialized with {d_model}-dimensional manifold.")
    
    # Dummy problem (e.g., text encoding of "Find a law for gravitational attraction")
    dummy_problem_data = torch.randn(768) # Simulating a BERT/transformer text embedding
    
    print("\n[Start] Engaging Crystallization Loop (Observe -> Hypothesize -> Verify)")
    print("        Using 3-Scale Hierarchical Search (Energy-Guided Langevin Dynamics)")
    
    # Run loop
    best_hypothesis, lowest_energy = mind.crystallization_loop(dummy_problem_data, steps=20)
    
    print("\n[Result] Crystallization Complete.")
    print(f"         Final Best Hypothesis Energy Score: {lowest_energy:.4f}")
    print(f"         Hypothesis Vector Magnitude: {torch.norm(best_hypothesis).item():.4f}")
    
    # Check if Active State was saved
    if mind.knowledge.working.active_state is not None:
        print(f"         Memory Working Cache matches best hypothesis: {torch.equal(mind.knowledge.working.active_state, best_hypothesis)}")
        
    print("\n=== All modules executed successfully! ===")

if __name__ == "__main__":
    test_architecture()
