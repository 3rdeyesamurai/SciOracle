"""Locate mathematical claims in legal prose and lift them into SymPy.

Sources in this domain are messy: LaTeX fragments in expert reports, PDF text
layers where superscripts have collapsed, and patent claims written half in
words. Everything recovered keeps its character span so a tribunal can be shown
the sentence the formula was taken from.
"""
import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor, implicit_multiplication_application, standard_transformations,
)

TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

LATEX_BLOCKS = [
    re.compile(r"\\begin\{(equation\*?|align\*?|displaymath|math)\}(.+?)\\end\{\1\}", re.S),
    re.compile(r"\\\[(.+?)\\\]", re.S),
    re.compile(r"\$\$(.+?)\$\$", re.S),
    re.compile(r"(?<!\$)\$([^$\n]{3,300})\$(?!\$)"),
    re.compile(r"\\\((.+?)\\\)", re.S),
]

MATH_SANITY = re.compile(r"[A-Za-z0-9]")
MATH_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_+-*/^().,√·× ")
KNOWN_WORDS = {
    "sin", "cos", "tan", "sec", "csc", "cot", "log", "ln", "exp", "sqrt", "abs", "max", "min",
    "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan", "sum", "prod", "int", "diff",
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega", "pi",
    "Derivative", "Integral",
}
RELATION_RE = re.compile(r"(?<![<>=!])=(?!=)")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

LATEX_REPLACEMENTS = [
    (re.compile(r"\\left|\\right"), ""),
    (re.compile(r"\\cdot|\\times"), "*"),
    (re.compile(r"\\frac\s*\{\s*[d\\partial]+\s*\}\s*\{\s*[d\\partial]+\s*([A-Za-z][A-Za-z0-9_]*)\s*\}\s*([A-Za-z][A-Za-z0-9_]*)"),
     r"Derivative(\2, \1)"),
    (re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}"), r"((\1)/(\2))"),
    (re.compile(r"\\sqrt\s*\{([^{}]+)\}"), r"sqrt(\1)"),
    (re.compile(r"\\(sin|cos|tan|log|ln|exp|sinh|cosh|tanh)\b"), r"\1"),
    (re.compile(r"\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|pi|rho|sigma|tau|phi|omega)\b"), r"\1"),
    (re.compile(r"\\mathrm\{([^{}]*)\}|\\text\{([^{}]*)\}|\\mathbf\{([^{}]*)\}"), r"\1\2\3"),
    (re.compile(r"[{}]"), ""),
    (re.compile(r"\\,|\;|\\!|\\quad|\\qquad"), " "),
    (re.compile(r"\^\s*\{?([\-\d]+)\}?"), r"**\1"),
]


def _latex_to_text(latex: str) -> str:
    out = latex.strip()
    for pattern, repl in LATEX_REPLACEMENTS:
        out = pattern.sub(repl, out)
    out = out.replace("^", "**").replace("\\", "")
    return re.sub(r"\s+", " ", out).strip()


def _local_dict(text: str) -> dict:
    """Identifiers immediately followed by "(" are function applications, not
    implicit products -- H(s) is a transfer function, never H times s."""
    locals_: dict = {}
    applied = {m.group(1) for m in re.finditer(r"([A-Za-z][A-Za-z0-9_]*)\s*\(", text)}
    for m in TOKEN_RE.finditer(text):
        name = m.group(0)
        if name in KNOWN_WORDS or name.lower() in KNOWN_WORDS:
            continue
        # Shadow SymPy builtins (N, S, O, I, E, Q...) so that a claim's variable
        # named N stays a variable rather than the numeric-evaluation function.
        locals_[name] = sp.Function(name) if name in applied else sp.Symbol(name)
    return locals_


def parse_expression(text: str):
    """Parse a candidate string into a SymPy object (Eq/Relational/Expr)."""
    cleaned = text.strip().rstrip(".;,").replace("×", "*").replace("·", "*").replace("−", "-").replace("^", "**")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not MATH_SANITY.search(cleaned):
        raise ValueError("no mathematical content")
    relations = [("<=", sp.Le), (">=", sp.Ge), ("!=", sp.Ne), ("=", sp.Eq), ("<", sp.Lt), (">", sp.Gt)]
    for token, ctor in relations:
        # split on the first *relational* occurrence only
        parts = re.split(rf"(?<![<>=!]){re.escape(token)}(?![=])", cleaned, maxsplit=1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            local = _local_dict(cleaned)
            lhs = sp.parse_expr(parts[0].strip(), transformations=TRANSFORMS, local_dict=local, evaluate=False)
            rhs = sp.parse_expr(parts[1].strip(), transformations=TRANSFORMS, local_dict=local, evaluate=False)
            return ctor(lhs, rhs, evaluate=False)
    return sp.parse_expr(cleaned, transformations=TRANSFORMS, local_dict=_local_dict(cleaned), evaluate=False)


def sympy_to_term(expr):
    """SymPy -> e-graph term. Subtraction/division are normalised away so that
    congruence is decided over a single ring representation."""
    from fractions import Fraction

    if isinstance(expr, sp.Symbol):
        return ("var", expr.name)
    if isinstance(expr, sp.Integer):
        return ("num", str(Fraction(int(expr))))
    if isinstance(expr, sp.Rational):
        return ("num", str(Fraction(int(expr.p), int(expr.q))))
    if isinstance(expr, sp.Float):
        return ("num", str(Fraction(float(expr)).limit_denominator(10**9)))
    if isinstance(expr, sp.Add):
        args = [sympy_to_term(a) for a in expr.args]
    elif isinstance(expr, sp.Mul):
        args = [sympy_to_term(a) for a in expr.args]
    elif isinstance(expr, sp.Pow):
        return ("^", sympy_to_term(expr.base), sympy_to_term(expr.exp))
    elif isinstance(expr, sp.Function):
        return (f"fn:{type(expr).__name__}",) + tuple(sympy_to_term(a) for a in expr.args)
    else:
        return ("var", str(expr))
    op = "+" if isinstance(expr, sp.Add) else "*"
    folded = args[0]
    for nxt in args[1:]:
        folded = (op, folded, nxt)
    return folded


def _prose_token(token: str) -> bool:
    """A token of three or more letters that is not a known mathematical name is
    prose. Identifiers inside formulae are one or two characters, or come from
    the greek/function vocabulary; "per", "and", "the" are not variables."""
    return len(token) > 2 and token not in KNOWN_WORDS and token.lower() not in KNOWN_WORDS


STOPWORDS = {"by", "of", "the", "is", "be", "as", "to", "in", "at", "on", "and", "or", "if",
             "that", "then", "for", "with", "where", "let", "we", "it", "a", "an", "so", "thus"}


def _is_boundary(token: str) -> bool:
    # Single letters are variables, never prose, whatever they spell.
    return len(token) > 1 and (_prose_token(token) or token.lower() in STOPWORDS)


def _trim_prose(fragment: str, from_right: bool) -> str:
    """Drop the prose that abuts a formula. Scanning inward from the outer edge,
    the first non-mathematical token marks the boundary of the expression."""
    tokens = list(TOKEN_RE.finditer(fragment))
    if from_right:
        for m in tokens:
            if _is_boundary(m.group(0)):
                return fragment[:m.start()]
        return fragment
    for m in reversed(tokens):
        if _is_boundary(m.group(0)):
            return fragment[m.end():]
    return fragment


def _balance(fragment: str) -> str:
    """Drop trailing characters that leave parentheses unbalanced."""
    depth = 0
    cut = len(fragment)
    for i, ch in enumerate(fragment):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return fragment[:i]
    while depth > 0 and cut > 0:
        cut = fragment.rfind("(", 0, cut)
        if cut < 0:
            return fragment
        depth -= 1
    return fragment if depth == 0 else fragment[:cut]


def _expand_around(text: str, eq_pos: int) -> tuple[int, str] | None:
    """Grow a maximal math-only window outward from a relation sign."""
    start = eq_pos
    while start > 0 and text[start - 1] in MATH_CHARS and text[start - 1] != "\n":
        start -= 1
    end = eq_pos + 1
    while end < len(text) and text[end] in MATH_CHARS and text[end] != "\n":
        end += 1

    lhs = _balance(_trim_prose(text[start:eq_pos], from_right=False)).strip(" ,.;")
    rhs = _balance(_trim_prose(text[eq_pos + 1:end], from_right=True)).strip(" ,.;")
    if not lhs or not rhs or not MATH_SANITY.search(lhs) or not MATH_SANITY.search(rhs):
        return None
    if len(lhs) > 160 or len(rhs) > 200:
        return None
    offset = text.find(lhs, start, eq_pos)
    if offset < 0:
        offset = start
    return offset, f"{lhs} = {rhs}"


def find_candidates(text: str, doc_id: str = "") -> list[dict]:
    """Return every distinct equation-shaped span, parsed where possible."""
    found: list[dict] = []
    seen: set[str] = set()

    def push(raw: str, start: int, end: int, notation: str):
        source = raw.strip()
        if len(source) < 3 or source in seen:
            return
        seen.add(source)
        entry = {
            "source_text": source,
            "notation": notation,
            "span": {"doc_id": doc_id, "start": start, "end": end, "text": source[:400]},
            "context": text[max(0, start - 200):start].replace("\n", " ").strip()[-200:],
            "parsed": False,
            "sympy": None,
            "error": None,
            "latex": None,
        }
        payload = _latex_to_text(source) if notation == "latex" else source
        try:
            expr = parse_expression(payload)
            entry.update(parsed=True, sympy=sp.srepr(expr), pretty=str(expr))
            entry["_expr"] = expr
            try:
                entry["latex"] = sp.latex(expr)
            except Exception:
                entry["latex"] = None
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
        found.append(entry)

    consumed: list[tuple[int, int]] = []
    for pattern in LATEX_BLOCKS:
        for m in pattern.finditer(text):
            body = m.group(m.lastindex or 0)
            for line in re.split(r"\\\\|\n", body):
                if line.strip():
                    push(line, m.start(), m.end(), "latex")
            consumed.append((m.start(), m.end()))

    def inside_consumed(pos: int) -> bool:
        return any(s <= pos < e for s, e in consumed)

    for m in RELATION_RE.finditer(text):
        if inside_consumed(m.start()):
            continue
        window = _expand_around(text, m.start())
        if window is None:
            continue
        start, candidate = window
        push(candidate, start, m.start() + len(candidate) - candidate.index("="), "plain")

    return found
