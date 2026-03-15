import torch
import torch.nn as nn
from core.graph_encoder import ASTGraphEncoder
import sympy as sp
from symbolic.ast_parser import sympy_to_graph
from torch_geometric.data import Data, Batch

class MathematicalEnergyModel(nn.Module):
    """
    Energy-Based Model defining the energy function for mathematical equivalence:
    E(problem, solution) = || f(problem) - g(solution) ||^2
    """
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int):
        super(MathematicalEnergyModel, self).__init__()
        # For symmetric equivalence analysis, we use a shared AST graph encoder
        self.encoder = ASTGraphEncoder(in_channels, hidden_dim, num_layers)
        
    def forward(self, lhs_batch: Batch, rhs_batch: Batch) -> torch.Tensor:
        """
        Compute energy for batched parsed AST graphs.
        Valid mathematical identities possess low energy values.
        
        Args:
            lhs_batch: Torch geometric Batch for Left Hand Side expressions
            rhs_batch: Torch geometric Batch for Right Hand Side expressions
        Returns:
            Energy vector for batch elements
        """
        f_lhs = self.encoder(lhs_batch.x, lhs_batch.edge_index, lhs_batch.batch)
        g_rhs = self.encoder(rhs_batch.x, rhs_batch.edge_index, rhs_batch.batch)
        
        # Euclidean distance squared (L2 squared)
        energy = torch.sum((f_lhs - g_rhs) ** 2, dim=1)
        return energy
        
    def energy(self, lhs: sp.Expr, rhs: sp.Expr, device: torch.device = torch.device('cpu')) -> float:
        """
        Compute scalar energy for an individual symbolic mathematical identity using SymPy terms.
        """
        self.eval()
        with torch.no_grad():
            x_lhs, edge_idx_lhs = sympy_to_graph(lhs)
            x_rhs, edge_idx_rhs = sympy_to_graph(rhs)
            
            data_lhs = Data(x=x_lhs.to(device), edge_index=edge_idx_lhs.to(device))
            data_rhs = Data(x=x_rhs.to(device), edge_index=edge_idx_rhs.to(device))
            
            # Explicitly wrapping inside a batch
            batch_lhs = Batch.from_data_list([data_lhs])
            batch_rhs = Batch.from_data_list([data_rhs])
            
            energy_val = self.forward(batch_lhs, batch_rhs)
            return energy_val.item()
