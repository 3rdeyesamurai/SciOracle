"""E-graphs and equality saturation for equation canonicalisation.

Purpose in the lawfare setting: an adversary rewrites a formula so it *looks*
novel while denoting the same function. Term rewriting alone is order-dependent
(the phase-ordering problem) -- apply distributivity first and you may never
reach the form that exposes the match. An e-graph applies every rule to every
equivalent form at once, so congruence is decided rather than searched for.

Terms are plain nested tuples:
    ("+", ("var", "x"), ("num", "2"))
Patterns use ("?", name) for pattern variables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import count

Term = tuple
COMMUTATIVE = {"+", "*"}
ASSOCIATIVE = {"+", "*"}


def var(name: str) -> Term:
    return ("var", name)


def num(value) -> Term:
    return ("num", str(Fraction(value)))


def is_leaf(term: Term) -> bool:
    return term[0] in ("var", "num")


def term_str(term: Term) -> str:
    if is_leaf(term):
        return f"{term[0]}:{term[1]}"
    args = [term_str(a) for a in term[1:]]
    if term[0] in COMMUTATIVE:
        args.sort()
    return f"({term[0]} {' '.join(args)})"


@dataclass
class EGraph:
    parents: dict[int, int] = field(default_factory=dict)
    classes: dict[int, set] = field(default_factory=dict)
    hashcons: dict[tuple, int] = field(default_factory=dict)
    _ids: count = field(default_factory=lambda: count(0))
    pending: list = field(default_factory=list)

    # ---- union-find -----------------------------------------------------
    def find(self, eid: int) -> int:
        root = eid
        while self.parents[root] != root:
            root = self.parents[root]
        while self.parents[eid] != root:  # path compression
            self.parents[eid], eid = root, self.parents[eid]
        return root

    def _new_class(self) -> int:
        eid = next(self._ids)
        self.parents[eid] = eid
        self.classes[eid] = set()
        return eid

    def _canonical(self, enode: tuple) -> tuple:
        op, args = enode[0], enode[1:]
        return (op,) + tuple(self.find(a) if isinstance(a, int) else a for a in args)

    # ---- construction ---------------------------------------------------
    def add_enode(self, enode: tuple) -> int:
        enode = self._canonical(enode)
        if enode in self.hashcons:
            return self.find(self.hashcons[enode])
        eid = self._new_class()
        self.hashcons[enode] = eid
        self.classes[eid].add(enode)
        return eid

    def add(self, term: Term) -> int:
        if is_leaf(term):
            return self.add_enode(term)
        return self.add_enode((term[0],) + tuple(self.add(child) for child in term[1:]))

    def merge(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if len(self.classes[ra]) < len(self.classes[rb]):
            ra, rb = rb, ra
        self.parents[rb] = ra
        self.classes[ra] |= self.classes.pop(rb)
        self.pending.append(ra)
        return True

    def rebuild(self) -> None:
        """Restore congruence: if two e-nodes become identical after a merge,
        their classes denote the same term and must themselves be merged."""
        while self.pending:
            todo, self.pending = self.pending, []
            for eid in {self.find(x) for x in todo}:
                seen: dict[tuple, int] = {}
                for enode in list(self.classes.get(eid, ())):
                    canon = self._canonical(enode)
                    if canon != enode:
                        self.classes[eid].discard(enode)
                        self.classes[eid].add(canon)
                    prior = seen.get(canon)
                    if prior is not None:
                        self.merge(prior, eid)
                    else:
                        seen[canon] = eid
                        self.hashcons[canon] = eid

    # ---- e-matching -----------------------------------------------------
    def match(self, pattern: Term, eid: int, subst: dict | None = None) -> list[dict]:
        subst = dict(subst or {})
        if pattern[0] == "?":
            name = pattern[1]
            bound = subst.get(name)
            if bound is not None:
                return [subst] if self.find(bound) == self.find(eid) else []
            subst[name] = self.find(eid)
            return [subst]
        results = []
        for enode in list(self.classes.get(self.find(eid), ())):
            if enode[0] != pattern[0] or len(enode) != len(pattern):
                continue
            if is_leaf(pattern):
                if enode[1] == pattern[1]:
                    results.append(dict(subst))
                continue
            partial = [dict(subst)]
            for sub_pattern, child in zip(pattern[1:], enode[1:]):
                nxt = []
                for s in partial:
                    nxt.extend(self.match(sub_pattern, child, s))
                partial = nxt
                if not partial:
                    break
            results.extend(partial)
        return results

    def search(self, pattern: Term) -> list[tuple[int, dict]]:
        out = []
        for eid in list(self.classes.keys()):
            if self.find(eid) != eid:
                continue
            for subst in self.match(pattern, eid):
                out.append((eid, subst))
        return out

    def instantiate(self, pattern: Term, subst: dict) -> int:
        if pattern[0] == "?":
            return self.find(subst[pattern[1]])
        if is_leaf(pattern):
            return self.add_enode(pattern)
        return self.add_enode((pattern[0],) + tuple(self.instantiate(p, subst) for p in pattern[1:]))

    # ---- extraction -----------------------------------------------------
    def extract(self, eid: int) -> Term:
        """Smallest-AST extraction, with commutative arguments ordered so that
        two congruent expressions serialise identically."""
        cost: dict[int, tuple[int, Term]] = {}
        for _ in range(len(self.classes) + 8):
            changed = False
            for cid in list(self.classes.keys()):
                if self.find(cid) != cid:
                    continue
                best = cost.get(cid)
                for enode in self.classes[cid]:
                    if is_leaf(enode):
                        cand = (1, enode)
                    else:
                        kids = [cost.get(self.find(a)) for a in enode[1:]]
                        if any(k is None for k in kids):
                            continue
                        args = [k[1] for k in kids]
                        if enode[0] in COMMUTATIVE:
                            args.sort(key=term_str)
                        cand = (1 + sum(k[0] for k in kids), (enode[0],) + tuple(args))
                    if best is None or cand[0] < best[0] or (cand[0] == best[0] and term_str(cand[1]) < term_str(best[1])):
                        best, changed = cand, True
                if best is not None:
                    cost[cid] = best
            if not changed:
                break
        root = self.find(eid)
        if root not in cost:
            raise ValueError("no extractable term for e-class")
        return cost[root][1]


def _fold(term_op: str, a: Fraction, b: Fraction) -> Fraction | None:
    try:
        if term_op == "+":
            return a + b
        if term_op == "*":
            return a * b
        if term_op == "^" and b.denominator == 1 and -8 <= b <= 8:
            if a == 0 and b < 0:
                return None
            return a ** int(b)
    except (ZeroDivisionError, OverflowError):
        return None
    return None


def _numeric(graph: "EGraph", *names: str):
    """Side condition: every named pattern variable is bound to a literal.

    Exponent arithmetic on symbolic exponents is a well-known source of
    non-termination in equality saturation, so those rules fire only when the
    exponents are concrete.
    """
    def guard(g: "EGraph", subst: dict) -> bool:
        return all(any(n[0] == "num" for n in g.classes.get(g.find(subst[name]), ())) for name in names)
    return guard


RULES: list[tuple] = [
    ("comm-add", ("+", ("?", "a"), ("?", "b")), ("+", ("?", "b"), ("?", "a"))),
    ("comm-mul", ("*", ("?", "a"), ("?", "b")), ("*", ("?", "b"), ("?", "a"))),
    ("assoc-add", ("+", ("+", ("?", "a"), ("?", "b")), ("?", "c")), ("+", ("?", "a"), ("+", ("?", "b"), ("?", "c")))),
    ("assoc-add-r", ("+", ("?", "a"), ("+", ("?", "b"), ("?", "c"))), ("+", ("+", ("?", "a"), ("?", "b")), ("?", "c"))),
    ("assoc-mul", ("*", ("*", ("?", "a"), ("?", "b")), ("?", "c")), ("*", ("?", "a"), ("*", ("?", "b"), ("?", "c")))),
    ("assoc-mul-r", ("*", ("?", "a"), ("*", ("?", "b"), ("?", "c"))), ("*", ("*", ("?", "a"), ("?", "b")), ("?", "c"))),
    ("distribute", ("*", ("?", "a"), ("+", ("?", "b"), ("?", "c"))),
     ("+", ("*", ("?", "a"), ("?", "b")), ("*", ("?", "a"), ("?", "c")))),
    ("factor", ("+", ("*", ("?", "a"), ("?", "b")), ("*", ("?", "a"), ("?", "c"))),
     ("*", ("?", "a"), ("+", ("?", "b"), ("?", "c")))),
    ("add-zero", ("+", ("?", "a"), ("num", "0")), ("?", "a")),
    ("mul-one", ("*", ("?", "a"), ("num", "1")), ("?", "a")),
    ("mul-zero", ("*", ("?", "a"), ("num", "0")), ("num", "0")),
    ("pow-one", ("^", ("?", "a"), ("num", "1")), ("?", "a")),
    ("pow-zero", ("^", ("?", "a"), ("num", "0")), ("num", "1")),
    ("square", ("*", ("?", "a"), ("?", "a")), ("^", ("?", "a"), ("num", "2"))),
    ("square-r", ("^", ("?", "a"), ("num", "2")), ("*", ("?", "a"), ("?", "a"))),
    ("pow-mul", ("*", ("^", ("?", "a"), ("?", "m")), ("^", ("?", "a"), ("?", "n"))),
     ("^", ("?", "a"), ("+", ("?", "m"), ("?", "n"))), _numeric(None, "m", "n")),
    ("pow-pow", ("^", ("^", ("?", "a"), ("?", "m")), ("?", "n")),
     ("^", ("?", "a"), ("*", ("?", "m"), ("?", "n"))), _numeric(None, "m", "n")),
    ("double", ("+", ("?", "a"), ("?", "a")), ("*", ("num", "2"), ("?", "a"))),
    ("neg-neg", ("*", ("num", "-1"), ("*", ("num", "-1"), ("?", "a"))), ("?", "a")),
]


def saturate(graph: EGraph, roots: list[int], max_iters: int = 14, node_budget: int = 6000) -> dict:
    """Run every rule to fixpoint (or budget). Returns a saturation report."""
    applied: dict[str, int] = {}
    iterations = 0
    saturated = False
    for iterations in range(1, max_iters + 1):
        matches = []
        for rule in RULES:
            name, lhs, rhs = rule[0], rule[1], rule[2]
            guard = rule[3] if len(rule) > 3 else None
            for eid, subst in graph.search(lhs):
                if guard is not None and not guard(graph, subst):
                    continue
                matches.append((name, rhs, eid, subst))
        changed = False
        over_budget = False
        for name, rhs, eid, subst in matches:
            if len(graph.hashcons) > node_budget:
                over_budget = True
                break
            new_id = graph.instantiate(rhs, subst)
            if graph.merge(eid, new_id):
                applied[name] = applied.get(name, 0) + 1
                changed = True
        # constant folding: arithmetic on literals collapses immediately
        for cid in list(graph.classes.keys()):
            if graph.find(cid) != cid:
                continue
            for enode in list(graph.classes[cid]):
                if is_leaf(enode) or len(enode) != 3:
                    continue
                kids = []
                for child in enode[1:]:
                    lit = next((n for n in graph.classes[graph.find(child)] if n[0] == "num"), None)
                    kids.append(Fraction(lit[1]) if lit else None)
                if all(k is not None for k in kids):
                    folded = _fold(enode[0], kids[0], kids[1])
                    if folded is not None and graph.merge(cid, graph.add_enode(("num", str(folded)))):
                        applied["const-fold"] = applied.get("const-fold", 0) + 1
                        changed = True
        graph.rebuild()
        if not changed:
            saturated = True
            break
        if over_budget or len(graph.hashcons) > node_budget:
            break
    return {
        "iterations": iterations,
        "saturated": saturated,
        "eclasses": len({graph.find(c) for c in graph.classes}),
        "enodes": len(graph.hashcons),
        "rules_applied": applied,
        "root_classes": [graph.find(r) for r in roots],
    }


def canonicalize(term: Term, **kwargs) -> tuple[str, dict]:
    """Canonical serialisation of one term: the archive's equivalence index key."""
    graph = EGraph()
    root = graph.add(term)
    report = saturate(graph, [root], **kwargs)
    canonical = graph.extract(root)
    return term_str(canonical), report


def equivalent(a: Term, b: Term, **kwargs) -> dict:
    """Decide congruence of two terms inside a single saturated e-graph."""
    graph = EGraph()
    ra, rb = graph.add(a), graph.add(b)
    report = saturate(graph, [ra, rb], **kwargs)
    same = graph.find(ra) == graph.find(rb)
    return {
        "equivalent": same,
        "method": "equality_saturation",
        "saturated": report["saturated"],
        "iterations": report["iterations"],
        "enodes": report["enodes"],
        "rules_applied": report["rules_applied"],
        "canonical_form": term_str(graph.extract(ra)) if same else None,
    }
