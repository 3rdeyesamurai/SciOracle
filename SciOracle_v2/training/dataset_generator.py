import sympy as sp
import random
import json
from symbolic.mutator import SymbolicMutator

class GeneratorTarget:
    """
    Curriculum Learning Data Synthesizer generating random initial expressions.
    Stage 1 arithmetic
    Stage 2 linear expressions
    Stage 3 polynomials
    Stage 4 factorization
    Stage 5 trigonometry
    Stage 6 calculus
    """
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.mutator = SymbolicMutator()
        
    def generate_arithmetic(self) -> sp.Expr:
        """Arithmetic scalar identities."""
        left = random.randint(1, 100)
        right = random.randint(1, 100)
        op = random.choice(['+', '-', '*', '/'])
        if op == '+': return left + right
        if op == '-': return left - right
        if op == '*': return left * right
        if op == '/': return sp.Rational(left, right)

    def generate_linear(self) -> sp.Expr:
        """Linear algebraic manipulations ax + b"""
        x = sp.Symbol('x')
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        return a * x + b

    def generate_polynomial(self) -> sp.Expr:
        """Higher order polynomials and factorizations."""
        x = sp.Symbol('x')
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        c = random.randint(-10, 10)
        d = random.randint(-10, 10)
        # Random initial form e.g. (ax + c)(bx + d) -> to be mutated
        return (a * x + c) * (b * x + d)

    def generate_trigonometry(self) -> sp.Expr:
        """Trigonometric generation."""
        x = sp.Symbol('x')
        return sp.sin(x)**2 + sp.cos(x)**2

    def generate_calculus(self) -> sp.Expr:
        """Derivatives and basic integrals."""
        x = sp.Symbol('x')
        a = random.randint(1, 10)
        return sp.Derivative(a * x**3, x)

    def synthesize_dataset(self, stage: int, num_examples: int) -> list:
        """
        Creates automated transformations mapping input -> expansion/simplification.
        """
        dataset = []
        for _ in range(num_examples):
            if stage == 1:
                base = self.generate_arithmetic()
            elif stage == 2:
                base = self.generate_linear()
            elif stage <= 4:
                base = self.generate_polynomial()
            elif stage == 5:
                base = self.generate_trigonometry()
            else:
                base = self.generate_calculus()
                
            # Execute mutator step for solution extraction
            transformed, action = self.mutator.mutate(base)
            dataset.append({
                'problem': str(base),
                'solution': str(transformed),
                'action_taken': action
            })
            
        return dataset

    def build_file(self, stage: int, size: int, filename: str):
        """Build and dump scalable dataset locally."""
        data = self.synthesize_dataset(stage, size)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
