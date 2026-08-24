import sympy as sp
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..db import query, row_to_dict
from ..engine import ledger
from ..engine.mathx import archive, extract
from ..saas import plans
from .deps import Principal, current_principal, require_quota

router = APIRouter(prefix="/api/v1/math", tags=["mathematics"])


class EquationRequest(BaseModel):
    expression: str = Field(min_length=1, max_length=4000)
    use_lean: bool = False
    use_wolfram: bool = False
    archive: bool = False
    matter_id: str | None = None


class ComparisonRequest(BaseModel):
    left: str = Field(min_length=1, max_length=4000)
    right: str = Field(min_length=1, max_length=4000)


class ExtractionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1_000_000)


def _parse(expression: str):
    try:
        return extract.parse_expression(expression)
    except Exception as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                            f"could not formalise expression: {type(exc).__name__}: {exc}") from exc


@router.post("/analyse")
def analyse_equation(body: EquationRequest,
                     principal: Principal = Depends(require_quota("equations_per_month"))):
    """Canonicalise, encode and adjudicate a single claim."""
    _parse(body.expression)
    result = archive.analyse(body.expression, use_lean=body.use_lean, use_wolfram=body.use_wolfram)
    expr = sp.sympify(result["sympy_repr"])
    result["prior_art"] = archive.prior_art_search(principal.org_id, expr)
    if body.archive:
        equation_id = archive.store(principal.org_id, result, matter_id=body.matter_id)
        result["equation_id"] = equation_id
        ledger.append(principal.org_id, principal.actor, "equation.archived",
                      {"equation_id": equation_id, "canonical_key": result["canonical_key"],
                       "verdict": result["verification"]["verdict"],
                       "prior_art": [m["equation_id"] for m in result["prior_art"]]}, subject=equation_id)
    plans.record(principal.org_id, "equations_per_month")
    return result


@router.post("/compare")
def compare(body: ComparisonRequest, principal: Principal = Depends(current_principal)):
    """Decide congruence of two claims by equality saturation over an e-graph."""
    left, right = _parse(body.left), _parse(body.right)
    result = archive.congruent(left, right)
    left_key, _ = archive.canonical_key(left)
    right_key, _ = archive.canonical_key(right)
    return {
        "left": {"input": body.left, "pretty": str(left), "canonical_key": left_key,
                 "content_mathml": archive.mathml.to_content_mathml(left)},
        "right": {"input": body.right, "pretty": str(right), "canonical_key": right_key,
                  "content_mathml": archive.mathml.to_content_mathml(right)},
        "result": result,
        "interpretation": ("The two formulations denote the same relation; an algebraic restatement "
                           "does not establish novelty." if result["equivalent"] else
                           "No congruence was derived. Note that failure to saturate is not a proof of "
                           "inequivalence -- see `saturated`."),
    }


@router.post("/extract")
def extract_equations(body: ExtractionRequest, principal: Principal = Depends(current_principal)):
    candidates = extract.find_candidates(body.text, "inline")
    for candidate in candidates:
        candidate.pop("_expr", None)
    return {"candidates": candidates,
            "parsed": sum(1 for c in candidates if c["parsed"]),
            "unparsed": sum(1 for c in candidates if not c["parsed"])}


@router.get("/archive")
def list_archive(limit: int = 100, principal: Principal = Depends(current_principal)):
    rows = query(
        """SELECT id, matter_id, document_id, source_text, latex, canonical_key, verification, created_at
           FROM equations WHERE org_id = ? ORDER BY created_at DESC LIMIT ?""",
        (principal.org_id, min(limit, 500)))
    return {"equations": [row_to_dict(r, ("verification",)) for r in rows]}


@router.get("/archive/{equation_id}")
def get_equation(equation_id: str, principal: Principal = Depends(current_principal)):
    record = archive.get(principal.org_id, equation_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "equation not found")
    return record


@router.get("/archive/{equation_id}/omdoc")
def get_omdoc(equation_id: str, principal: Principal = Depends(current_principal)):
    from fastapi.responses import Response
    record = archive.get(principal.org_id, equation_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "equation not found")
    return Response(content=record["omdoc"], media_type="application/omdoc+xml")
