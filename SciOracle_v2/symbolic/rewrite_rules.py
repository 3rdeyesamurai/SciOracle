import sympy as sp

class RewriteLibrary:
    """
    Standard predefined mathematical relationships acting as explicit priors.
    Mainly supplementary, while the Engine should learn new ones autonomously.
    """
    def __init__(self):
        self._rules = {}

    def add_rule(self, name: str, lhs: sp.Expr, rhs: sp.Expr):
        """
        Adds a generic formal rewrite rule if proven.
        """
        self._rules[name] = (lhs, rhs)

    def apply_rule(self, expr: sp.Expr, rule_name: str) -> sp.Expr:
        """
        Substitute a matching LHS pattern with RHS in an expression.
        """
        if rule_name not in self._rules:
            raise KeyError(f"Rule {rule_name} not found in library.")
        
        lhs, rhs = self._rules[rule_name]
        # Basic matching using sympy
        result = expr.replace(lhs, rhs)
        return result

    def get_all_rules(self):
        return self._rules
