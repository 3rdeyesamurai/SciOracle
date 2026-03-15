import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class ASTGraphEncoder(nn.Module):
    """
    Graph Neural Network that encodes Abstract Syntax Trees (ASTs)
    of mathematical expressions into dense embeddings.
    """
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int):
        super(ASTGraphEncoder, self).__init__()
        self.num_layers = num_layers
        
        # GCN layers to process the AST node embeddings
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        # Final projection layer to embeddings space
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for encoding a batch of AST graphs.
        
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            batch (Tensor, optional): Batch assignments for multi-graph processing.
        Returns:
            Tensor: Graph-level embedding vectors.
        """
        if batch is None:
            # Assuming a single graph if batch is not provided
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            
        # Global mean pooling to achieve a single vector per expression
        x = global_mean_pool(x, batch)
        
        # Additional projection layer
        x = self.fc(x)
        return x
