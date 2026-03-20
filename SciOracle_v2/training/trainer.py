import torch
import torch.nn as nn
from core.energy_model import MathematicalEnergyModel
from torch.optim import Adam
import sympy as sp
from symbolic.ast_parser import sympy_to_graph
from torch_geometric.data import Data, Batch

class ContrastiveCurriculumTrainer:
    """
    Trains the core Neural Reasoning Layer through curriculum learning stages.
    """
    def __init__(self, model: MathematicalEnergyModel, lr: float = 0.001, margin: float = 1.0):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.margin = margin
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def compute_loss(self, problem_batch: Batch, solution_batch: Batch, negative_batch: Batch) -> torch.Tensor:
        """
        Loss function: L = max(0, margin + E_positive - E_negative)
        Positive examples possess low energy (true identities).
        Negative examples possess high energy.
        """
        # E_positive
        e_pos = self.model(problem_batch, solution_batch)
        
        # E_negative
        e_neg = self.model(problem_batch, negative_batch)
        
        loss = torch.clamp(self.margin + e_pos - e_neg, min=0.0)
        return loss.mean()
        
    def train_step(self, problems: list[sp.Expr], solutions: list[sp.Expr], negatives: list[sp.Expr]) -> float:
        """Single backpropagation step mapping list of sympy expressions to PyG Batches."""
        self.model.train()
        self.optimizer.zero_grad()
        
        data_problems = []
        data_solutions = []
        data_negatives = []
        
        for p, s, n in zip(problems, solutions, negatives):
            p_x, p_ei = sympy_to_graph(p)
            s_x, s_ei = sympy_to_graph(s)
            n_x, n_ei = sympy_to_graph(n)
            
            data_problems.append(Data(x=p_x.to(self.device), edge_index=p_ei.to(self.device)))
            data_solutions.append(Data(x=s_x.to(self.device), edge_index=s_ei.to(self.device)))
            data_negatives.append(Data(x=n_x.to(self.device), edge_index=n_ei.to(self.device)))
            
        prob_batch = Batch.from_data_list(data_problems)
        sol_batch = Batch.from_data_list(data_solutions)
        neg_batch = Batch.from_data_list(data_negatives)
        
        loss = self.compute_loss(prob_batch, sol_batch, neg_batch)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def curriculum_training_loop(self, epochs_per_stage: int = 50, stages: int = 6):
        """
        Progresses training through increasing mathematical complexity:
        1: arithmetic
        2: linear expressions
        3: polynomials
        4: factorization
        5: trigonometry
        6: calculus
        """
        import random
        print("Starting Neural Agent Training Loop...")
        for stage in range(1, stages + 1):
            print(f"--- Entering Curriculum Stage {stage} ---")
            
            # Simple mock loop. In production, connect Dataset Synthesizer here
            for epoch in range(1, epochs_per_stage + 1):
                # Generates random dummy states for illustration
                # You'd load actual paired ASTs here
                x = sp.Symbol('x')
                p = [x + random.randint(1, 5) for _ in range(8)]
                s = [sp.expand(val * 2) for val in p]
                n = [val + 100 for val in p] # Trivial negatives

                loss = self.train_step(p, s, n)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs_per_stage} Loss: {loss:.4f}")
