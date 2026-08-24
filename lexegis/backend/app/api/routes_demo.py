from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from ..db import execute, new_id
from ..engine import ledger, pipeline
from ..engine.samples import CORPUS
from ..saas import plans
from .deps import Principal, current_principal

router = APIRouter(prefix="/api/v1/demo", tags=["demo"])


@router.post("/seed")
def seed(principal: Principal = Depends(current_principal)):
    """Create the demonstration matter and run a full analysis over it."""
    matter_id = new_id("mat")
    execute("INSERT INTO matters (id, org_id, name, forum, jurisdictions, created_at) VALUES (?,?,?,?,?,?)",
            (matter_id, principal.org_id, "Helios / Meridian — licence dispute & EP 4 118 992",
             "UNCITRAL (Geneva) / EPO opposition", '["CH", "GB", "EP"]',
             datetime.now(timezone.utc).isoformat()))
    ledger.append(principal.org_id, principal.actor, "matter.created",
                  {"matter_id": matter_id, "name": "demonstration corpus", "seeded": True}, subject=matter_id)

    ingested = []
    for document in CORPUS:
        ingested.append(pipeline.ingest_document(
            principal.org_id, matter_id, document["filename"], document["text"].encode("utf-8"),
            "text/plain", actor=principal.actor))

    analysis = pipeline.analyse_matter(principal.org_id, matter_id, doc_type="ip_licence",
                                       actor=principal.actor)
    plans.record(principal.org_id, "matters")
    plans.record(principal.org_id, "documents_per_month", len(CORPUS))
    plans.record(principal.org_id, "analyses_per_month")
    return {"matter_id": matter_id, "ingested": ingested, "analysis": analysis}
