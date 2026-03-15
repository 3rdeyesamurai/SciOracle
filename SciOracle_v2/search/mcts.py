import math
import random
import sympy as sp
import torch
from symbolic.mutator import SymbolicMutator
from core.energy_model import MathematicalEnergyModel

class MCTSNode:
    """
    Monte Carlo Tree Search structural Node denoting a symbolic equation state.
    """
    def __init__(self, state: sp.Expr, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Transformation applied to reach this exact state
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        # Tracks prior probabilities determined during policy phase
        self.prior_probs = {}

    @property
    def value(self) -> float:
        """Mean Node Score combine combinations of:
        Energy score + Validity + Novelty"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class MCTSEngine:
    """
    Equation discovery guided by MCTS.
    Capable of intelligent expansion instead of naive combinations.
    """
    def __init__(self, mutator: SymbolicMutator, energy_model: MathematicalEnergyModel,
                 policy_network=None, num_simulations=100, c_puct=1.5, max_depth=10):
        self.mutator = mutator
        self.energy_model = energy_model
        self.policy_network = policy_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.target_state = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def search(self, initial_state: sp.Expr, target_state: sp.Expr = None) -> list:
        """
        Execute MCTS iterations to find optimal paths or valid symbolic equivalences.
        If target_state is designated, behaves as proof search toward equivalence.
        If target_state None, behaves as exploration matching generalized rules.
        """
        self.target_state = target_state
        root = MCTSNode(initial_state)

        for _ in range(self.num_simulations):
            node = self._select(root)
            reward = self._simulate(node)
            self._backpropagate(node, reward)

        best_path = self._extract_best_path(root)
        return best_path
        
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse downwards prioritizing high UCB scores."""
        current = node
        depth = 0
        while len(current.children) > 0 and depth < self.max_depth:
            best_action, best_child = max(
                current.children.items(),
                key=lambda item: self._ucb_score(current, item[1])
            )
            current = best_child
            depth += 1
            
        # If not terminal (here terminal could just be maximum depth or verified)
        if depth < self.max_depth:
            self._expand(current)
            if len(current.children) > 0:
                actions = list(current.children.keys())
                current = current.children[random.choice(actions)]
                
        return current

    def _expand(self, node: MCTSNode):
        """Create children transformations mapping out algebraic possibilities."""
        actions = self.mutator.get_available_actions(node.state)
        # Assuming RL policy provides probability distributions:
        # In a fully connected model we'd map policy distributions here
        for action in actions:
            new_expr = self.mutator.apply(node.state, action)
            if action not in node.children:
                child_node = MCTSNode(state=new_expr, parent=node, action=action)
                # Assign default prior probability evenly in absence of strict RL evaluation
                child_node.prior_probs[action] = 1.0 / len(actions)
                node.children[action] = child_node

    def _simulate(self, node: MCTSNode) -> float:
        """
        Rollout / Evaluation: Obtain node score relying critically on MathematicalEnergyModel.
        Also computes mathematical equivalence check via SymPy heuristic.
        +1 valid identity
        -1 high energy
        """
        # Node score relies on:
        # 1. Energy
        if self.target_state is not None:
            energy_val = self.energy_model.energy(node.state, self.target_state, self.device)
            # lower energy is better
            reward = math.exp(-energy_val) 
            
            # 2. Validity (Simple Check during search)
            if sp.simplify(node.state - self.target_state) == 0:
                reward += 10.0 # High Reward for perfect equivalence
        else:
            # Exploration reward relies entirely on discovering Novel states
            reward = random.uniform(0.1, 1.0)
            
        return reward

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Push acquired reward up through the search topology."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += reward
            current = current.parent

    def _ucb_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        """Calculate UCB score determining exploitation versus exploration balance."""
        exploration_term = self.c_puct * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        return child.value + exploration_term

    def _extract_best_path(self, root: MCTSNode) -> list:
        """Generates trajectory dict of most visited nodes acting as optimal proof steps."""
        path = []
        current = root
        
        while len(current.children) > 0:
            # The most robust path relies on visit count, not just point estimation
            best_action, best_child = max(
                current.children.items(),
                key=lambda item: item[1].visit_count
            )
            path.append({
                'operation': best_action,
                'result': str(best_child.state)
            })
            current = best_child
            
        return path
