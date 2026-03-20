import sympy as sp
import torch
from typing import Dict, Tuple, List, Any

# Map mathematical operations to node types
NODE_TYPES = {
    'Add': 0,
    'Mul': 1,
    'Pow': 2,
    'Symbol': 3,
    'Integer': 4,
    'Float': 5,
    'Rational': 6,
    'Function': 7,
    'Constant': 8
}

def get_node_type(expr: sp.Expr) -> int:
    """Identify SymPy expression node type for embedding."""
    if hasattr(expr, 'is_Add') and expr.is_Add: return NODE_TYPES['Add']
    if hasattr(expr, 'is_Mul') and expr.is_Mul: return NODE_TYPES['Mul']
    if hasattr(expr, 'is_Pow') and expr.is_Pow: return NODE_TYPES['Pow']
    if hasattr(expr, 'is_Symbol') and expr.is_Symbol: return NODE_TYPES['Symbol']
    if getattr(expr, 'is_Integer', False): return NODE_TYPES['Integer']
    if getattr(expr, 'is_Float', False): return NODE_TYPES['Float']
    if getattr(expr, 'is_Rational', False): return NODE_TYPES['Rational']
    if getattr(expr, 'is_Function', False) or isinstance(expr, sp.Function): return NODE_TYPES['Function']
    return NODE_TYPES['Constant']

def sympy_to_graph(expr: sp.Expr) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parses a sympy expression into graph representation for PyTorch Geometric AST structures.
    Edges represent Parent -> child operator relationships.
    """
    nodes_list = []
    edges_list = []
    
    def traverse(node: sp.Expr, parent_idx: int = -1):
        idx = len(nodes_list)
        # Node feature is a one-hot vector indicating formula node type
        node_type = get_node_type(node)
        features = [1.0 if i == node_type else 0.0 for i in range(len(NODE_TYPES))]
        nodes_list.append(features)
        
        if parent_idx != -1:
            edges_list.append([parent_idx, idx])
            # Adding reverse edge for undirected feature learning
            edges_list.append([idx, parent_idx])
            
        for child in node.args:
            traverse(child, idx)

    traverse(expr)
    
    if not nodes_list:
        nodes_list.append([0.0] * len(NODE_TYPES))
    
    nodes_tensor = torch.tensor(nodes_list, dtype=torch.float32)
    
    if not edges_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()

    return nodes_tensor, edge_index
