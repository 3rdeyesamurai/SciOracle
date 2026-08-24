"""Semantic mathematical markup: Strict Content MathML, OpenMath, OMDoc.

A formula lifted out of a PDF is presentational -- it encodes layout, not
meaning, which is exactly where an adversary hides an ambiguity. The archive
therefore stores only *semantic* encodings, with every operator bound to an
external OpenMath content dictionary so that two jurisdictions reading the same
archived claim resolve the same denotation.
"""
from xml.sax.saxutils import escape

import sympy as sp

# operator -> (content dictionary, symbol name)
CD_MAP: dict[str, tuple[str, str]] = {
    "plus": ("arith1", "plus"),
    "times": ("arith1", "times"),
    "minus": ("arith1", "minus"),
    "divide": ("arith1", "divide"),
    "power": ("arith1", "power"),
    "root": ("arith1", "root"),
    "abs": ("arith1", "abs"),
    "unary_minus": ("arith1", "unary_minus"),
    "eq": ("relation1", "eq"),
    "neq": ("relation1", "neq"),
    "lt": ("relation1", "lt"),
    "gt": ("relation1", "gt"),
    "leq": ("relation1", "leq"),
    "geq": ("relation1", "geq"),
    "sin": ("transc1", "sin"), "cos": ("transc1", "cos"), "tan": ("transc1", "tan"),
    "exp": ("transc1", "exp"), "log": ("transc1", "ln"), "ln": ("transc1", "ln"),
    "diff": ("calculus1", "diff"), "int": ("calculus1", "int"), "defint": ("calculus1", "defint"),
    "sum": ("arith1", "sum"), "product": ("arith1", "product"), "limit": ("limit1", "limit"),
    "lambda": ("fns1", "lambda"),
    "pi": ("nums1", "pi"), "e": ("nums1", "e"), "infinity": ("nums1", "infinity"), "i": ("nums1", "i"),
}

SYMPY_TO_OP = {
    sp.Add: "plus", sp.Mul: "times", sp.Pow: "power",
    sp.Equality: "eq", sp.Unequality: "neq",
    sp.StrictLessThan: "lt", sp.StrictGreaterThan: "gt",
    sp.LessThan: "leq", sp.GreaterThan: "geq",
}

CONSTANTS = {sp.pi: "pi", sp.E: "e", sp.oo: "infinity", sp.I: "i"}


LOCAL = "__local__:"


def _csymbol(op: str) -> str:
    if op.startswith(LOCAL):
        # Not in any published CD: encoded as a document-local identifier so the
        # archive never asserts a denotation it cannot resolve.
        return f'<ci type="function">{escape(op[len(LOCAL):])}</ci>'
    cd, name = CD_MAP.get(op, ("arith1", op))
    return f'<csymbol cd="{cd}">{escape(name)}</csymbol>'


def _oms(op: str) -> str:
    if op.startswith(LOCAL):
        return f'<OMV name="{escape(op[len(LOCAL):])}"/>'
    cd, name = CD_MAP.get(op, ("arith1", op))
    return f'<OMS cd="{cd}" name="{escape(name)}"/>'


def _walk(expr, emit_apply, emit_symbol, emit_var, emit_num) -> str:
    """Shared traversal: the two encodings differ only in their emitters."""
    if expr in CONSTANTS:
        return emit_symbol(CONSTANTS[expr])
    if isinstance(expr, sp.Symbol):
        return emit_var(expr.name)
    if isinstance(expr, sp.Rational) and not isinstance(expr, sp.Integer):
        return emit_apply("divide", [emit_num(expr.p), emit_num(expr.q)])
    if isinstance(expr, (sp.Integer, sp.Float)):
        return emit_num(expr)
    if isinstance(expr, sp.Derivative):
        body = _walk(expr.expr, emit_apply, emit_symbol, emit_var, emit_num)
        return emit_apply("diff", [body] + [emit_var(v.name) for v, _ in expr.variable_count if isinstance(v, sp.Symbol)])
    if isinstance(expr, sp.Integral):
        body = _walk(expr.function, emit_apply, emit_symbol, emit_var, emit_num)
        return emit_apply("int", [body] + [emit_var(v.name) for v in expr.variables if isinstance(v, sp.Symbol)])
    for cls, op in SYMPY_TO_OP.items():
        if isinstance(expr, cls):
            args = [_walk(a, emit_apply, emit_symbol, emit_var, emit_num) for a in expr.args]
            # Strict encodings are binary: n-ary operators fold left.
            if op in ("plus", "times") and len(args) > 2:
                folded = args[0]
                for nxt in args[1:]:
                    folded = emit_apply(op, [folded, nxt])
                return folded
            return emit_apply(op, args)
    if isinstance(expr, sp.Function):
        op = type(expr).__name__.lower()
        if op not in CD_MAP:
            op = LOCAL + type(expr).__name__
        return emit_apply(op, [_walk(a, emit_apply, emit_symbol, emit_var, emit_num) for a in expr.args])
    return emit_var(str(expr))


def to_content_mathml(expr) -> str:
    """Strict Content MathML: an XML encoding of an OpenMath object."""
    body = _walk(
        expr,
        lambda op, args: f"<apply>{_csymbol(op)}{''.join(args)}</apply>",
        lambda name: _csymbol(name),
        lambda name: f"<ci>{escape(name)}</ci>",
        lambda value: f"<cn>{value}</cn>",
    )
    return f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">{body}</math>'


def to_openmath(expr) -> str:
    body = _walk(
        expr,
        lambda op, args: f"<OMA>{_oms(op)}{''.join(args)}</OMA>",
        lambda name: _oms(name),
        lambda name: f'<OMV name="{escape(name)}"/>',
        lambda value: (f"<OMI>{value}</OMI>" if float(value).is_integer() else f'<OMF dec="{value}"/>'),
    )
    return f'<OMOBJ xmlns="http://www.openmath.org/OpenMath" version="2.0">{body}</OMOBJ>'


def to_omdoc(*, theory: str, statement_id: str, statement_type: str, expr, source: str,
             metadata: dict | None = None) -> str:
    """OMDoc wrapper carrying all three levels: theory, statement, object."""
    metadata = metadata or {}
    meta_xml = "".join(
        f'<dc:{escape(str(k))}>{escape(str(v))}</dc:{escape(str(k))}>' for k, v in metadata.items()
        if str(k).isidentifier()
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<omdoc xmlns="http://omdoc.org/ns" xmlns:dc="http://purl.org/dc/elements/1.1/" '
        f'version="1.6" xml:id="{escape(statement_id)}-omdoc">\n'
        f'  <metadata>{meta_xml}</metadata>\n'
        f'  <theory xml:id="{escape(theory)}">\n'
        f'    <{escape(statement_type)} xml:id="{escape(statement_id)}">\n'
        f'      <metadata><dc:source>{escape(source[:2000])}</dc:source></metadata>\n'
        f'      <CMP><p>{escape(source[:2000])}</p></CMP>\n'
        f'      <FMP>{to_content_mathml(expr)}</FMP>\n'
        f'    </{escape(statement_type)}>\n'
        '  </theory>\n'
        '</omdoc>\n'
    )


def to_presentation_mathml(expr) -> str:
    """Retained only for display; never used as the archived representation."""
    try:
        return sp.mathml(expr, printer="presentation")
    except Exception:
        return f"<mtext>{escape(str(expr))}</mtext>"
