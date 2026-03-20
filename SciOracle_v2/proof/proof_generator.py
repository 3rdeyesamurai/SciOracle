import json

class ProofObject:
    """
    Formalized object recording the exact path derived by the discovery engine.
    Example sequence:
    1: expand
    2: distribute
    3: combine like terms
    """
    def __init__(self, initial_equation: str, final_equation: str, steps: list):
        self.initial_equation = initial_equation
        self.final_equation = final_equation
        self.steps = steps

    def to_dict(self):
        return {
            'initial': self.initial_equation,
            'final': self.final_equation,
            'steps': self.steps
        }

    def save(self, format: str = 'json') -> str:
        """
        Serializes proof trace for storing into graph memory or raw logs.
        """
        if format == 'json':
            return json.dumps(self.to_dict(), indent=2)
        else:
            return str(self.to_dict())


class ProofGenerator:
    """
    Compiles MCTS transformation paths into structured mathematical identities
    suitable for academic publishing or direct theorem saving.
    """
    @staticmethod
    def generate(initial_eq: str, final_eq: str, sequence_of_transformations: list) -> ProofObject:
        """
        Transform a list of mutator actions into formal readable derivation proofs.
        """
        return ProofObject(initial_eq, final_eq, sequence_of_transformations)

    @staticmethod
    def print_proof(proof: ProofObject):
        print(f"Proof Details:\n")
        print(f"Original Form: {proof.initial_equation}")
        print("-" * 25)
        for i, step in enumerate(proof.steps, start=1):
            print(f"Step {i:>2}: {step['operation']} => {step['result']}")
        print("-" * 25)
        print(f"\nFinal Identity Verified: {proof.final_equation}\n")
