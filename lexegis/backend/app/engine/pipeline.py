"""End-to-end orchestration: ingest -> forensics -> triage -> System 1 -> System 2.

The ordering is a security property, not a convenience: nothing is parsed for
meaning before it has been fingerprinted, and nothing reaches a reasoning context
before triage has ruled on it.
"""
import json
from datetime import datetime, timezone

import sympy as sp

from ..config import BLOB_DIR
from ..db import execute, new_id, query, query_one
from ..security import forensics as forensics_mod
from ..security import injection as injection_mod
from ..security import triage as triage_mod
from . import discrepancy, knowledge_graph, ledger
from . import system1
from . import taxonomy
from .mathx import archive as math_archive


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ingest_document(org_id: str, matter_id: str, filename: str, data: bytes,
                    media_type: str | None = None, actor: str = "system") -> dict:
    """Stage 1-3. Returns the stored document row plus its security dossier."""
    scan = forensics_mod.scan(data, filename, media_type)
    report, text = scan["report"], scan["text"]
    injection_report = injection_mod.scan(text)
    decision = triage_mod.triage(text, report, injection_report)

    doc_id = new_id("doc")
    text_path = BLOB_DIR / f"{doc_id}.txt"
    text_path.write_text(text, encoding="utf-8")
    (BLOB_DIR / f"{doc_id}.bin").write_bytes(data)

    extraction = system1.extract(doc_id, text)

    execute(
        """INSERT INTO documents (id, org_id, matter_id, filename, media_type, sha256, byte_size, tier,
                                  quarantined, forensics, triage, extraction, text_path, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (doc_id, org_id, matter_id, filename, media_type, report["digest"]["sha256"],
         report["digest"]["byte_size"], decision.tier, int(decision.quarantined),
         json.dumps({**report, "injection": injection_report}), json.dumps(decision.as_dict()),
         json.dumps(extraction.as_dict()), str(text_path), _now()),
    )

    ledger.append(org_id, actor, "document.ingested", {
        "document_id": doc_id, "filename": filename, "sha256": report["digest"]["sha256"],
        "byte_size": report["digest"]["byte_size"], "forensic_severity": report["max_severity"],
        "forensic_findings": [f["code"] for f in report["findings"]],
        "injection_verdict": injection_report["verdict"], "injection_score": injection_report["score"],
        "triage_tier": decision.tier, "quarantined": decision.quarantined, "reasons": decision.reasons,
    }, subject=doc_id)

    return {
        "document_id": doc_id, "filename": filename, "sha256": report["digest"]["sha256"],
        "tier": decision.tier, "quarantined": decision.quarantined,
        "forensics": report, "injection": injection_report, "triage": decision.as_dict(),
        "characters": len(text), "clauses": len(extraction.clauses),
    }


def _load_documents(org_id: str, matter_id: str) -> list[dict]:
    rows = query("SELECT * FROM documents WHERE org_id = ? AND matter_id = ? ORDER BY created_at",
                 (org_id, matter_id))
    docs = []
    for row in rows:
        payload = json.loads(row["extraction"]) if row["extraction"] else {}
        extraction = system1.Extraction(
            doc_id=row["id"], clauses=payload.get("clauses", []), parties=payload.get("parties", []),
            defined_terms=payload.get("defined_terms", []), obligations=payload.get("obligations", []),
            dates=payload.get("dates", []), facts=payload.get("facts", []))
        try:
            text = open(row["text_path"], encoding="utf-8").read()
        except OSError:
            text = ""
        docs.append({
            "id": row["id"], "title": row["filename"], "text": text, "extraction": extraction,
            "tier": row["tier"], "quarantined": bool(row["quarantined"]),
            "forensics": json.loads(row["forensics"]) if row["forensics"] else {},
            "triage": json.loads(row["triage"]) if row["triage"] else {},
        })
    return docs


def analyse_matter(org_id: str, matter_id: str, *, doc_type: str = "commercial", actor: str = "system",
                   use_lean: bool = False, use_wolfram: bool = False, persist: bool = True,
                   jurisdictions: list[str] | None = None) -> dict:
    """Stage 4-6: discrepancy detection, equation archiving, knowledge graph."""
    docs = _load_documents(org_id, matter_id)
    if not docs:
        return {"matter_id": matter_id, "documents": 0, "findings": [], "chronology": [],
                "graph": {"nodes": [], "edges": [], "stats": {}}, "equations": []}

    doc_titles = {doc["id"]: doc["title"] for doc in docs}
    findings: list[dict] = []

    for doc in docs:
        findings += discrepancy.detect_epistemic(doc["id"], doc["title"], doc["forensics"],
                                                 doc["forensics"].get("injection", {}), doc["triage"])
        # A quarantined document still contributes structured facts and forensic
        # findings; what it does not do is contribute prose to a reasoning context.
        findings += discrepancy.detect_internal_contradictions(doc["id"], doc["title"], doc["extraction"])
        if not doc["quarantined"]:
            findings += discrepancy.detect_ambiguity(doc["id"], doc["title"], doc["text"], doc["extraction"])
            findings += discrepancy.detect_omission(doc["id"], doc["title"], doc["text"], doc_type)
            if jurisdictions:
                findings += discrepancy.detect_regulatory(doc["id"], doc["title"], doc["text"],
                                                          jurisdictions, taxonomy)

    findings += discrepancy.detect_cross_document([d for d in docs if not d["quarantined"]])

    events = system1.chronology([doc["extraction"] for doc in docs], doc_titles)
    findings += discrepancy.detect_chronology_conflicts(events, {d["id"]: d["extraction"] for d in docs})

    harvests = []
    for doc in docs:
        harvest = math_archive.harvest_document(org_id, doc["id"], matter_id, doc["text"],
                                                use_lean=use_lean, use_wolfram=use_wolfram)
        harvests.append(harvest)
        ledger.append(org_id, actor, "equations.harvested", {
            "document_id": doc["id"], "candidates": harvest["candidates"],
            "archived": [item["equation_id"] for item in harvest["archived"]],
            "canonical_keys": [item["analysis"]["canonical_key"] for item in harvest["archived"]],
            "verdicts": [item["analysis"]["verification"]["verdict"] for item in harvest["archived"]],
        }, subject=doc["id"])
    findings += discrepancy.detect_math(harvests, doc_titles)
    findings += discrepancy.detect_definition_conflicts(harvests, doc_titles, math_archive.congruent)

    findings = discrepancy.rank(findings)
    graph = knowledge_graph.build(docs, findings, harvests, events)
    result = _assemble(matter_id, docs, findings, events, graph, harvests)

    if persist:
        execute("DELETE FROM findings WHERE org_id = ? AND matter_id = ?", (org_id, matter_id))
        for finding in findings:
            execute(
                """INSERT INTO findings (id, org_id, matter_id, category, subtype, severity, confidence,
                                         title, detail, spans, provenance, status, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (new_id("fnd"), org_id, matter_id, finding["category"], finding["subtype"], finding["severity"],
                 finding["confidence"], finding["title"], finding["detail"], json.dumps(finding["spans"]),
                 json.dumps(finding["provenance"]), "open", _now()),
            )
        execute("UPDATE matters SET last_analysis = ? WHERE id = ? AND org_id = ?",
                (json.dumps(result), matter_id, org_id))
        ledger.append(org_id, actor, "matter.analysed", {
            "matter_id": matter_id, "documents": len(docs), "findings": len(findings),
            "jurisdictions": jurisdictions or [],
            "severity_counts": _severity_counts(findings),
            "equations_archived": sum(len(h["archived"]) for h in harvests),
            "graph": graph["stats"],
        }, subject=matter_id)

    result["ledger_head"] = ledger.head(org_id)
    return result


def _assemble(matter_id, docs, findings, events, graph, harvests) -> dict:
    return {
        "matter_id": matter_id,
        "documents": len(docs),
        "document_index": [{"id": d["id"], "title": d["title"], "tier": d["tier"],
                            "quarantined": d["quarantined"]} for d in docs],
        "findings": findings,
        "severity_counts": _severity_counts(findings),
        "chronology": events,
        "graph": graph,
        "equations": [
            {"equation_id": item["equation_id"], "document_id": harvest["document_id"],
             "source_text": item["analysis"]["source_text"], "latex": item["analysis"]["latex"],
             "canonical_key": item["analysis"]["canonical_key"],
             "verdict": item["analysis"]["verification"]["verdict"],
             "authority": item["analysis"]["verification"]["authority"],
             "backends": item["analysis"]["verification"]["backends_available"],
             "prior_art": item["prior_art"], "span": item["span"]}
            for harvest in harvests for item in harvest["archived"]
        ],
        "unparsed_equations": [s for harvest in harvests for s in harvest["unparsed"]],
    }


def _severity_counts(findings: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding["severity"]] = counts.get(finding["severity"], 0) + 1
    return counts


def document_dossier(org_id: str, document_id: str) -> dict | None:
    row = query_one("SELECT * FROM documents WHERE org_id = ? AND id = ?", (org_id, document_id))
    if row is None:
        return None
    item = dict(row)
    for field in ("forensics", "triage", "extraction"):
        if item.get(field):
            item[field] = json.loads(item[field])
    try:
        item["text"] = open(item["text_path"], encoding="utf-8").read()
    except OSError:
        item["text"] = ""
    item["quarantined"] = bool(item["quarantined"])
    item["ledger"] = ledger.entries(org_id, subject=document_id)
    return item
