import networkx as nx
import sympy as sp
import matplotlib.pyplot as plt

class TheoremKnowledgeGraph:
    """
    All discoveries must be stored in a structured knowledge graph representation.
    Nodes = equations,
    Edges = symbolic transformations or proof steps.
    Using NetworkX graph implementation.
    """
    def __init__(self):
        # We model this system as a directed graph representing mathematical derivations
        self.graph = nx.DiGraph()

    def add_equation_node(self, equation: str, complexity_score: float = 1.0):
        """Register a unique equation in our graph space."""
        if not self.graph.has_node(equation):
            self.graph.add_node(equation, complexity=complexity_score, visits=1)
        else:
            self.graph.nodes[equation]['visits'] += 1

    def add_transformation_edge(self, source_eq: str, target_eq: str, operation: str, energy_cost: float = 1.0):
        """Connect equations mathematically proven to be reachable."""
        self.add_equation_node(source_eq)
        self.add_equation_node(target_eq)
        
        if not self.graph.has_edge(source_eq, target_eq):
            self.graph.add_edge(source_eq, target_eq,
                                transformation=operation,
                                weight=energy_cost,
                                frequency=1)
        else:
            self.graph[source_eq][target_eq]['frequency'] += 1

    def query_derivation_path(self, eq1: str, eq2: str) -> list:
        """
        Attempts to compute a recorded shortest path linking equations using
        their transformation cost. Returns path if exists.
        """
        try:
            return nx.shortest_path(self.graph, source=eq1, target=eq2, weight='weight')
        except nx.NetworkXNoPath:
            return []
            
    def display_graph_stats(self):
        """Reports total knowledge capacity derived."""
        print(f"Graph Intelligence Base:")
        print(f"  Nodes (Unique Equations): {self.graph.number_of_nodes()}")
        print(f"  Edges (Transformations):  {self.graph.number_of_edges()}")
