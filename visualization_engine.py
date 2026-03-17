import os
import sympy as sp
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def generate_ast_chart(s_math: str, attn_weights: np.ndarray, output_path: str):
    """
    Generates a high-resolution PNG of the SymPy AST, where node colors are 
    mapped to the GCN's internal node importance (cross-attention weights).
    """
    try:
        expr = sp.sympify(s_math, evaluate=False)
    except Exception as e:
        print(f"Visualization error: Could not sympify {s_math}: {e}")
        return

    nodes = []
    edges = []
    
    # We must traverse the tree in the exact same DFS order as ASTGraphTokenizer 
    # so the node IDs align perfectly with the attention weight array.
    def traverse(node):
        node_id = len(nodes)
        
        if isinstance(node, sp.Symbol) or isinstance(node, sp.Integer) or isinstance(node, sp.Rational) or isinstance(node, sp.Float):
            nodes.append(str(node))
        else:
            op = node.__class__.__name__
            nodes.append(op)
            for arg in node.args:
                child_id = traverse(arg)
                edges.append((node_id, child_id))  # Directed edge for visualization
        return node_id
        
    traverse(expr)
    
    G = nx.DiGraph()
    for i, label in enumerate(nodes):
        # We only take attention weights for the nodes we actually generated. 
        # Pad tokens are safely ignored.
        weight = float(attn_weights[i]) if i < len(attn_weights) else 0.0
        G.add_node(i, label=label, weight=weight)
        
    for u, v in edges:
        G.add_edge(u, v)

    # Extract colors
    node_colors = [G.nodes[n]['weight'] for n in G.nodes()]
    labels = nx.get_node_attributes(G, 'label')

    plt.figure(figsize=(10, 8), dpi=300)
    
    # Attempt to use a tree layout, fallback to kamada_kawai
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    except ImportError:
        pos = nx.kamada_kawai_layout(G)

    # Draw nodes and map colors
    # Hotter color (red/yellow) = higher attention weight
    nodes_drawn = nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors, 
        cmap=plt.cm.coolwarm, 
        node_size=2000, 
        alpha=0.9,
        edgecolors='black'
    )
    
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='gray')
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold")

    plt.colorbar(nodes_drawn, label="GCN Cross-Attention Weight")
    plt.title(f"SciOracle Proof of Discovery AST\nFormula: {s_math}")
    plt.axis("off")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"   -> [Visualization] AST GCN chart saved to {output_path}")
