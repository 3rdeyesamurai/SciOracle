"""The equation archive: canonicalise, encode, verify, and match against prior art."""
import json
from datetime import datetime, timezone

import sympy as sp

from ...db import execute, new_id, query, query_one
from . import egraph, mathml
from .extract import find_candidates, parse_expression, sympy_to_term
from .verify import verify


def _canonical_terms(expr) -> list[tuple]:
    """E-graph terms for the claim.

    For a relation we canonicalise the residual (lhs - rhs), because that is the
    object whose vanishing the claim asserts; two claims that assert the same
    thing in rearranged form share a residual up to ring equivalence.
    """
    if isinstance(expr, sp.Equality):
        target = sp.expand(expr.lhs - expr.rhs)
    else:
        target = sp.expand(expr)
    terms = [sympy_to_term(target)]
    try:
        factored = sp.factor(target)
        if factored != target:
            terms.append(sympy_to_term(factored))
    except Exception:
        pass
    return terms


def canonical_key(expr) -> tuple[str, dict]:
    terms = _canonical_terms(expr)
    key, report = egraph.canonicalize(terms[0], node_budget=4000)
    return key, report


def congruent(expr_a, expr_b, **kwargs) -> dict:
    """Decide whether two claims denote the same relation, via equality saturation."""
    return egraph.equivalent(_canonical_terms(expr_a)[0], _canonical_terms(expr_b)[0], **kwargs)


def analyse(source_text: str, *, theory: str = "lexegis_archive", statement_id: str | None = None,
            use_lean: bool = True, use_wolfram: bool = True) -> dict:
    """Full System 2 treatment of a single equation string."""
    statement_id = statement_id or f"stmt_{abs(hash(source_text)) % 10**10}"
    expr = parse_expression(source_text)
    key, saturation = canonical_key(expr)
    verification = verify(expr, use_lean=use_lean, use_wolfram=use_wolfram, theorem_name=statement_id)
    return {
        "source_text": source_text,
        "sympy_repr": sp.srepr(expr),
        "pretty": str(expr),
        "latex": sp.latex(expr),
        "canonical_key": key,
        "saturation": saturation,
        "content_mathml": mathml.to_content_mathml(expr),
        "openmath": mathml.to_openmath(expr),
        "presentation_mathml": mathml.to_presentation_mathml(expr),
        "omdoc": mathml.to_omdoc(theory=theory, statement_id=statement_id, statement_type="assertion",
                                 expr=expr, source=source_text,
                                 metadata={"date": datetime.now(timezone.utc).date().isoformat(),
                                           "type": "archived-claim"}),
        "verification": verification,
        "free_symbols": sorted(str(s) for s in getattr(expr, "free_symbols", set())),
    }


def store(org_id: str, analysis: dict, *, matter_id: str | None = None, document_id: str | None = None) -> str:
    eq_id = new_id("eq")
    execute(
        """INSERT INTO equations (id, org_id, matter_id, document_id, source_text, latex, sympy_repr,
                                  canonical_key, content_mathml, openmath, omdoc, verification, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (eq_id, org_id, matter_id, document_id, analysis["source_text"], analysis.get("latex"),
         analysis["sympy_repr"], analysis["canonical_key"], analysis["content_mathml"],
         analysis["openmath"], analysis["omdoc"], json.dumps(analysis["verification"]),
         datetime.now(timezone.utc).isoformat()),
    )
    return eq_id


def prior_art_search(org_id: str, expr, *, exclude_equation_id: str | None = None, limit: int = 40) -> list[dict]:
    """Two-stage retrieval: canonical-key bucket first (cheap, exact), then
    equality saturation against the remaining archive (expensive, decisive)."""
    key, _ = canonical_key(expr)
    rows = query(
        "SELECT * FROM equations WHERE org_id = ? ORDER BY created_at DESC LIMIT ?", (org_id, limit * 4))
    matches = []
    for row in rows:
        if exclude_equation_id and row["id"] == exclude_equation_id:
            continue
        if row["canonical_key"] == key:
            matches.append({"equation_id": row["id"], "source_text": row["source_text"],
                            "match": "canonical_key", "equivalent": True, "detail": "identical canonical form",
                            "created_at": row["created_at"], "document_id": row["document_id"]})
            continue
        try:
            other = sp.sympify(row["sympy_repr"])
        except Exception:
            continue
        result = congruent(expr, other, node_budget=2500, max_iters=8)
        if result["equivalent"]:
            matches.append({"equation_id": row["id"], "source_text": row["source_text"],
                            "match": "equality_saturation", "equivalent": True,
                            "detail": f"congruent after {result['iterations']} saturation rounds "
                                      f"({result['enodes']} e-nodes)",
                            "created_at": row["created_at"], "document_id": row["document_id"]})
        if len(matches) >= limit:
            break
    return matches


def harvest_document(org_id: str, doc_id: str, matter_id: str, text: str, *,
                     use_lean: bool = False, use_wolfram: bool = False, cap: int = 40) -> dict:
    """Extract, archive and adjudicate every equation in one document."""
    candidates = find_candidates(text, doc_id)
    stored, skipped = [], []
    for candidate in candidates[:cap]:
        if not candidate["parsed"]:
            skipped.append({"source_text": candidate["source_text"], "reason": candidate["error"],
                            "span": candidate["span"]})
            continue
        try:
            analysis = analyse(candidate["source_text"], use_lean=use_lean, use_wolfram=use_wolfram)
        except Exception as exc:
            skipped.append({"source_text": candidate["source_text"], "reason": f"{type(exc).__name__}: {exc}"[:200],
                            "span": candidate["span"]})
            continue
        expr = sp.sympify(analysis["sympy_repr"])
        prior = prior_art_search(org_id, expr)
        eq_id = store(org_id, analysis, matter_id=matter_id, document_id=doc_id)
        stored.append({"equation_id": eq_id, "span": candidate["span"], "notation": candidate["notation"],
                       "context": candidate["context"], "analysis": analysis, "prior_art": prior})
    return {"document_id": doc_id, "candidates": len(candidates), "archived": stored, "unparsed": skipped}


def get(org_id: str, equation_id: str) -> dict | None:
    row = query_one("SELECT * FROM equations WHERE org_id = ? AND id = ?", (org_id, equation_id))
    if row is None:
        return None
    item = dict(row)
    if item.get("verification"):
        item["verification"] = json.loads(item["verification"])
    return item
