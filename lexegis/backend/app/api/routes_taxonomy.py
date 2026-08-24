"""The semantic law matrix, exposed."""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..engine import taxonomy
from .deps import Principal, current_principal

router = APIRouter(prefix="/api/v1/taxonomy", tags=["taxonomy"])


class AssessRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1_000_000)
    jurisdictions: list[str] = Field(default_factory=list)


@router.get("")
def full_matrix(principal: Principal = Depends(current_principal)):
    """The whole computed matrix: rows, primitives, coverage, similarity, divergence."""
    return taxonomy.matrix()


@router.get("/classes")
def list_classes(family: str | None = None, principal: Principal = Depends(current_principal)):
    classes = taxonomy.matrix()["classes"]
    if family:
        classes = [c for c in classes if c["family"].lower() == family.lower()]
    return {"classes": [{k: v for k, v in c.items() if k != "weights"} | {"weights": c["weights"]}
                        for c in classes]}


@router.get("/classes/{class_id}")
def get_class(class_id: str, principal: Principal = Depends(current_principal)):
    record = taxonomy.classification(class_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "classification not found")
    divergence = next((d for d in taxonomy.matrix()["divergence"] if d["class"] == class_id), None)
    return {"classification": record, "divergence": divergence}


@router.get("/jurisdictions/{jurisdiction_id}")
def get_jurisdiction(jurisdiction_id: str, principal: Principal = Depends(current_principal)):
    record = taxonomy.jurisdiction(jurisdiction_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "jurisdiction not found")
    return record


@router.get("/compare")
def compare(a: str = Query(min_length=2), b: str = Query(min_length=2),
            principal: Principal = Depends(current_principal)):
    """Emphasis distance between two regimes and what drives it."""
    result = taxonomy.compare_jurisdictions(a.upper(), b.upper())
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown jurisdiction identifier")
    return result


@router.post("/assess")
def assess(body: AssessRequest, principal: Principal = Depends(current_principal)):
    """Read a document against the matrix: what it engages, what it omits, where forums diverge."""
    return taxonomy.assess(body.text, [j.upper() for j in body.jurisdictions])
