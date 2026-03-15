import torch
import torch.nn as nn
import sympy as sp
from core.graph_encoder import ASTGraphEncoder
from symbolic.ast_parser import sympy_to_graph
from torch_geometric.data import Data, Batch

class RLPolicyNetwork(nn.Module):
    """
    Reinforcement Learning Policy determining Transformation probabilities.
    π(operation | equation_embedding)
    Uses identical encoder for shared latent representation but maps to
    number of available transformations (mutations). 
    """
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int, num_actions: int):
        super(RLPolicyNetwork, self).__init__()
        # Shared or decoupled graphical representation engine
        self.encoder = ASTGraphEncoder(in_channels, hidden_dim, num_layers)
        
        # Policy head mapping equation embeddings to transformation probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
            nn.Softmax(dim=-1)
        )
        
        # Optional value head mapping generic worth of current state
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, eq_batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets probability distribution over actions and state value.
        """
        enc = self.encoder(eq_batch.x, eq_batch.edge_index, eq_batch.batch)
        action_probs = self.policy_head(enc)
        state_value = self.value_head(enc)
        return action_probs, state_value

    def get_action_probs(self, expr: sp.Expr, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Get probabilities for an individual symbolic state.
        """
        self.eval()
        with torch.no_grad():
            x, edge_idx = sympy_to_graph(expr)
            data = Data(x=x.to(device), edge_index=edge_idx.to(device))
            batch = Batch.from_data_list([data])
            probs, _ = self.forward(batch)
            return probs.squeeze(0)
