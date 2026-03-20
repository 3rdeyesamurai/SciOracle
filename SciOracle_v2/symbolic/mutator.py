import sympy as sp
import random

class SymbolicMutator:
    """
    Replaces naive equation search with symbolic transformations.
    Performs valid mathematical actions to modify equations during search.
    """
    def __init__(self):
        self.operations = [
            'expand',
            'factor',
            'simplify',
            'substitute',
            'differentiate',
            'integrate'
        ]

    def get_available_actions(self, expr: sp.Expr) -> list:
        """Returns logically applicable transformation names for an expression."""
        # Simple heuristic, typically all operations are somewhat applicable symbolically
        return self.operations

    def apply(self, expr: sp.Expr, operation: str) -> sp.Expr:
        """
        Applies a named symbolic transformation.
        """
        try:
            if operation == 'expand':
                return sp.expand(expr)
            elif operation == 'factor':
                return sp.factor(expr)
            elif operation == 'simplify':
                return sp.simplify(expr)
            elif operation == 'substitute':
                # Random substitution of a variable with a constant or simple expr
                free_vars = list(expr.free_symbols)
                if free_vars:
                    # Toy substitution: v -> v+1
                    var = random.choice(free_vars)
                    return expr.subs(var, var + 1)
                return expr
            elif operation == 'differentiate':
                free_vars = list(expr.free_symbols)
                if free_vars:
                    var = random.choice(free_vars)
                    return sp.diff(expr, var)
                return expr
            elif operation == 'integrate':
                free_vars = list(expr.free_symbols)
                if free_vars:
                    var = random.choice(free_vars)
                    # Use unevaluated or evaluated depending on speed needs
                    return sp.integrate(expr, var)
                return expr
            else:
                return expr
        except Exception as e:
            # When sympy transformations fail (e.g. timeout on difficult integrals)
            # Just return original expression
            return expr

    def mutate(self, expr: sp.Expr) -> tuple[sp.Expr, str]:
        """Randomly mutate an expression, returning the transformed expr and action."""
        actions = self.get_available_actions(expr)
        action = random.choice(actions)
        new_expr = self.apply(expr, action)
        return new_expr, action
