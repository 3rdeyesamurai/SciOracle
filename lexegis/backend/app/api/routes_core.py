import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from ..config import MAX_UPLOAD_BYTES
from ..db import execute, new_id, query, query_one, row_to_dict
from ..engine import alignment, ledger, pipeline
from ..saas import plans
from .deps import Principal, current_principal, owned_matter, require_quota

router = APIRouter(prefix="/api/v1", tags=["matters"])

DOC_TYPES = ("commercial", "ip_licence", "treaty")


class MatterCreate(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    forum: str | None = Field(default=None, max_length=120)
    jurisdictions: list[str] = Field(default_factory=list)


class AnalysisRequest(BaseModel):
    doc_type: str = "commercial"
    use_lean: bool = False
    use_wolfram: bool = False


class TextIngest(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=4_000_000)


class AlignmentRequest(BaseModel):
    text: str
    predicted: list[dict]
    ground_truth: list[dict]
    overlap: float = 0.3


@router.post("/matters", status_code=201)
def create_matter(body: MatterCreate, principal: Principal = Depends(require_quota("matters"))):
    matter_id = new_id("mat")
    execute("INSERT INTO matters (id, org_id, name, forum, jurisdictions, created_at) VALUES (?,?,?,?,?,?)",
            (matter_id, principal.org_id, body.name, body.forum, json.dumps(body.jurisdictions),
             datetime.now(timezone.utc).isoformat()))
    plans.record(principal.org_id, "matters")
    ledger.append(principal.org_id, principal.actor, "matter.created",
                  {"matter_id": matter_id, "name": body.name, "forum": body.forum,
                   "jurisdictions": body.jurisdictions}, subject=matter_id)
    return {"id": matter_id, "name": body.name, "forum": body.forum, "jurisdictions": body.jurisdictions}


@router.get("/matters")
def list_matters(principal: Principal = Depends(current_principal)):
    rows = query("SELECT * FROM matters WHERE org_id = ? ORDER BY created_at DESC", (principal.org_id,))
    out = []
    for row in rows:
        matter = row_to_dict(row, ("jurisdictions",))
        counts = query_one(
            "SELECT COUNT(*) AS docs FROM documents WHERE matter_id = ?", (row["id"],))
        findings = query(
            "SELECT severity, COUNT(*) AS n FROM findings WHERE matter_id = ? GROUP BY severity", (row["id"],))
        matter["document_count"] = counts["docs"]
        matter["finding_counts"] = {r["severity"]: r["n"] for r in findings}
        out.append(matter)
    return {"matters": out}


@router.get("/matters/{matter_id}")
def get_matter(matter_id: str, principal: Principal = Depends(current_principal)):
    matter = owned_matter(matter_id, principal)
    matter["jurisdictions"] = json.loads(matter["jurisdictions"] or "[]")
    documents = [row_to_dict(r) for r in query(
        """SELECT id, filename, sha256, byte_size, tier, quarantined, created_at
           FROM documents WHERE matter_id = ? ORDER BY created_at""", (matter_id,))]
    for doc in documents:
        doc["quarantined"] = bool(doc["quarantined"])
    findings = [row_to_dict(r, ("spans", "provenance")) for r in query(
        "SELECT * FROM findings WHERE matter_id = ? ORDER BY created_at", (matter_id,))]
    equations = [row_to_dict(r, ("verification",)) for r in query(
        """SELECT id, document_id, source_text, latex, canonical_key, verification, created_at
           FROM equations WHERE matter_id = ? ORDER BY created_at""", (matter_id,))]
    analysis = json.loads(matter.pop("last_analysis")) if matter.get("last_analysis") else None
    return {"matter": matter, "documents": documents, "findings": findings, "equations": equations,
            "analysis": analysis}


@router.delete("/matters/{matter_id}")
def delete_matter(matter_id: str, principal: Principal = Depends(current_principal)):
    owned_matter(matter_id, principal)
    for table in ("findings", "equations", "documents"):
        execute(f"DELETE FROM {table} WHERE matter_id = ? AND org_id = ?", (matter_id, principal.org_id))
    execute("DELETE FROM matters WHERE id = ? AND org_id = ?", (matter_id, principal.org_id))
    ledger.append(principal.org_id, principal.actor, "matter.deleted", {"matter_id": matter_id}, subject=matter_id)
    return {"deleted": matter_id}


@router.post("/matters/{matter_id}/documents", status_code=201)
async def upload_document(matter_id: str, file: UploadFile = File(...),
                          principal: Principal = Depends(require_quota("documents_per_month"))):
    owned_matter(matter_id, principal)
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"file exceeds {MAX_UPLOAD_BYTES} bytes")
    result = pipeline.ingest_document(principal.org_id, matter_id, file.filename or "upload.bin", data,
                                      file.content_type, actor=principal.actor)
    plans.record(principal.org_id, "documents_per_month")
    return result


@router.post("/matters/{matter_id}/documents/text", status_code=201)
def ingest_text(matter_id: str, body: TextIngest,
                principal: Principal = Depends(require_quota("documents_per_month"))):
    owned_matter(matter_id, principal)
    result = pipeline.ingest_document(principal.org_id, matter_id, body.filename,
                                      body.text.encode("utf-8"), "text/plain", actor=principal.actor)
    plans.record(principal.org_id, "documents_per_month")
    return result


@router.get("/documents/{document_id}")
def get_document(document_id: str, principal: Principal = Depends(current_principal)):
    dossier = pipeline.document_dossier(principal.org_id, document_id)
    if dossier is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    return dossier


@router.post("/matters/{matter_id}/analyse")
def analyse(matter_id: str, body: AnalysisRequest,
            principal: Principal = Depends(require_quota("analyses_per_month"))):
    owned_matter(matter_id, principal)
    if body.doc_type not in DOC_TYPES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"doc_type must be one of {DOC_TYPES}")
    result = pipeline.analyse_matter(principal.org_id, matter_id, doc_type=body.doc_type,
                                     actor=principal.actor, use_lean=body.use_lean,
                                     use_wolfram=body.use_wolfram)
    plans.record(principal.org_id, "analyses_per_month")
    plans.record(principal.org_id, "equations_per_month", len(result["equations"]))
    return result


@router.get("/matters/{matter_id}/findings")
def list_findings(matter_id: str, severity: str | None = None, category: str | None = None,
                  principal: Principal = Depends(current_principal)):
    owned_matter(matter_id, principal)
    sql = "SELECT * FROM findings WHERE matter_id = ? AND org_id = ?"
    params: list = [matter_id, principal.org_id]
    if severity:
        sql += " AND severity = ?"
        params.append(severity)
    if category:
        sql += " AND category = ?"
        params.append(category)
    return {"findings": [row_to_dict(r, ("spans", "provenance")) for r in query(sql + " ORDER BY created_at", params)]}


@router.patch("/findings/{finding_id}")
def update_finding(finding_id: str, status_value: str = Query(alias="status"),
                   principal: Principal = Depends(current_principal)):
    if status_value not in ("open", "accepted", "dismissed", "escalated"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid status")
    cur = execute("UPDATE findings SET status = ? WHERE id = ? AND org_id = ?",
                  (status_value, finding_id, principal.org_id))
    if cur.rowcount == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "finding not found")
    ledger.append(principal.org_id, principal.actor, "finding.triaged",
                  {"finding_id": finding_id, "status": status_value}, subject=finding_id)
    return {"finding_id": finding_id, "status": status_value}


@router.get("/ledger")
def read_ledger(limit: int = 200, subject: str | None = None,
                principal: Principal = Depends(current_principal)):
    return {"entries": ledger.entries(principal.org_id, limit=limit, subject=subject),
            "head": ledger.head(principal.org_id)}


@router.get("/ledger/verify")
def verify_ledger(principal: Principal = Depends(current_principal)):
    return ledger.verify_chain(principal.org_id)


@router.post("/eval/alignment")
def eval_alignment(body: AlignmentRequest, principal: Principal = Depends(current_principal)):
    """Score predicted discrepancy spans against ground truth (RAG validation loop)."""
    return alignment.score(body.text, body.predicted, body.ground_truth, body.overlap)
