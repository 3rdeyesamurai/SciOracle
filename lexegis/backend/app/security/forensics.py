"""Pre-semantic forensic pass over every ingested artefact.

Nothing here is semantic: the file is treated as a hostile byte string first.
We fingerprint it, mine container metadata, and look for the structural traces
left by fabrication and manipulation (incremental-update chains in PDFs,
producer/creator mismatch, impossible timestamps, invisible render modes).
"""
import hashlib
import io
import re
from datetime import datetime, timezone

try:  # PyMuPDF is optional; text/plain evidence still works without it.
    import pymupdf as fitz
except Exception:  # pragma: no cover - exercised only in minimal installs
    try:
        import fitz
    except Exception:
        fitz = None

PDF_DATE_RE = re.compile(r"D:(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?")
KNOWN_GENERATORS = ("word", "acrobat", "latex", "pdftex", "libreoffice", "chrome", "quartz", "indesign")
SYNTHETIC_HINTS = ("gpt", "llm", "claude", "gemini", "stable diffusion", "midjourney", "synthesized", "ai-generated")


def _parse_pdf_date(value: str | None) -> str | None:
    if not value:
        return None
    m = PDF_DATE_RE.search(value)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    try:
        return datetime(y, mo or 1, d or 1, h, mi, s, tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None


def digest(data: bytes) -> dict:
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "sha512_prefix": hashlib.sha512(data).hexdigest()[:32],
        "md5_legacy": hashlib.md5(data).hexdigest(),
        "byte_size": len(data),
    }


def _pdf_structure(data: bytes) -> dict:
    """Structural facts recoverable without parsing: how many times was this saved?"""
    eofs = data.count(b"%%EOF")
    startxrefs = data.count(b"startxref")
    return {
        "eof_markers": eofs,
        "startxref_markers": startxrefs,
        # A clean single-pass export has exactly one of each. Additional pairs are
        # incremental updates -- legitimate for signed workflows, but each one is a
        # post-hoc edit that a tribunal is entitled to see enumerated.
        "incremental_updates": max(0, eofs - 1),
        "has_javascript": b"/JavaScript" in data or b"/JS" in data,
        "has_embedded_files": b"/EmbeddedFile" in data,
        "has_acroform": b"/AcroForm" in data,
        "has_signature_dict": b"/Sig" in data or b"/ByteRange" in data,
    }


def _pdf_metadata(data: bytes) -> tuple[dict, list[dict], str]:
    findings: list[dict] = []
    if fitz is None:
        return {}, [{"code": "parser_unavailable", "severity": "info",
                     "detail": "PyMuPDF not installed; metadata analysis skipped"}], ""
    try:
        doc = fitz.open(stream=io.BytesIO(data), filetype="pdf")
    except Exception as exc:
        return {}, [{"code": "container_unreadable", "severity": "high", "detail": str(exc)}], ""

    meta = dict(doc.metadata or {})
    created = _parse_pdf_date(meta.get("creationDate"))
    modified = _parse_pdf_date(meta.get("modDate"))
    meta["creationDate_iso"] = created
    meta["modDate_iso"] = modified
    meta["page_count"] = doc.page_count
    meta["is_encrypted"] = bool(doc.is_encrypted)

    if created and modified and modified < created:
        findings.append({
            "code": "timestamp_inversion", "severity": "high",
            "detail": f"modification date {modified} precedes creation date {created}",
        })
    now = datetime.now(timezone.utc).isoformat()
    for label, value in (("creation", created), ("modification", modified)):
        if value and value > now:
            findings.append({"code": "future_timestamp", "severity": "high",
                             "detail": f"{label} date {value} is in the future"})

    producer = (meta.get("producer") or "").lower()
    creator = (meta.get("creator") or "").lower()
    if producer and creator:
        p_known = any(g in producer for g in KNOWN_GENERATORS)
        c_known = any(g in creator for g in KNOWN_GENERATORS)
        if p_known and c_known and not any(g in producer and g in creator for g in KNOWN_GENERATORS):
            findings.append({"code": "generator_mismatch", "severity": "medium",
                             "detail": f"creator={creator!r} but producer={producer!r}; document was re-rendered"})
    for field, value in (("producer", producer), ("creator", creator), ("title", (meta.get("title") or "").lower())):
        for hint in SYNTHETIC_HINTS:
            if hint in value:
                findings.append({"code": "synthetic_generator_signature", "severity": "high",
                                 "detail": f"{field} metadata names a generative system: {value!r}"})
                break

    text_parts, invisible_chars, fonts = [], 0, set()
    for page in doc:
        text_parts.append(page.get_text())
        try:
            info = page.get_text("dict")
        except Exception:
            continue
        for block in info.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    fonts.add(span.get("font", ""))
                    # Render mode 3 = invisible; size 0 = unrenderable. Both are the
                    # classic carriers for text a human reviewer will never see.
                    if span.get("size", 1) < 0.5:
                        invisible_chars += len(span.get("text", ""))
    if invisible_chars:
        findings.append({"code": "invisible_text_layer", "severity": "critical",
                         "detail": f"{invisible_chars} characters rendered at sub-visible size"})
    meta["font_count"] = len(fonts)
    if len(fonts) > 12:
        findings.append({"code": "font_heterogeneity", "severity": "low",
                         "detail": f"{len(fonts)} distinct fonts; consistent with splicing of multiple sources"})
    doc.close()
    return meta, findings, "\n".join(text_parts)


def scan(data: bytes, filename: str, media_type: str | None = None) -> dict:
    """Full forensic report plus the extracted text layer (empty for opaque blobs)."""
    report = {
        "filename": filename,
        "media_type": media_type,
        "digest": digest(data),
        "container": {},
        "metadata": {},
        "findings": [],
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }
    text = ""
    if data[:5] == b"%PDF-":
        report["container"] = _pdf_structure(data)
        meta, findings, text = _pdf_metadata(data)
        report["metadata"] = meta
        report["findings"].extend(findings)
        if report["container"]["incremental_updates"] > 0 and not report["container"]["has_signature_dict"]:
            report["findings"].append({
                "code": "unsigned_incremental_update", "severity": "medium",
                "detail": f"{report['container']['incremental_updates']} post-hoc save(s) with no signature dictionary",
            })
        if report["container"]["has_javascript"]:
            report["findings"].append({"code": "active_content", "severity": "high",
                                       "detail": "PDF carries JavaScript; treated as active code, not evidence"})
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = data.decode("latin-1")
            except Exception:
                text = ""
                report["findings"].append({"code": "opaque_binary", "severity": "info",
                                           "detail": "no recoverable text layer"})

    zero_width = sum(text.count(c) for c in ("​", "‌", "‍", "⁠", "﻿"))
    if zero_width > 8:
        report["findings"].append({"code": "zero_width_payload", "severity": "high",
                                   "detail": f"{zero_width} zero-width characters; possible steganographic channel"})
    homoglyphs = len(re.findall(r"[А-я]", text)) if re.search(r"[A-Za-z]", text) else 0
    if homoglyphs and homoglyphs < max(20, len(text) * 0.002):
        report["findings"].append({"code": "homoglyph_substitution", "severity": "medium",
                                   "detail": f"{homoglyphs} Cyrillic glyphs sprinkled through Latin text"})

    order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
    report["max_severity"] = max((f["severity"] for f in report["findings"]), key=lambda s: order[s], default="none")
    report["integrity_score"] = round(max(0.0, 1.0 - sum(order[f["severity"]] for f in report["findings"]) / 12), 3)
    return {"report": report, "text": text}
