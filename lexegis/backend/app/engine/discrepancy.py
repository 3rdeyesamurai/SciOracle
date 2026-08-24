"""Multi-document discrepancy detection.

Taxonomy follows the contradiction categories used in contract-auditing
benchmarks: ambiguity, omission and direct contradiction, split across in-text
(one instrument) and cross-document (an evidentiary corpus) scope, plus the
mathematical and epistemic categories this engine adds.

Every finding carries the spans it was derived from. A finding with no span is a
bug, not an opinion.
"""
import re
from collections import defaultdict

VAGUE_TERMS = {
    "reasonable efforts": "no benchmark defines the required level of effort",
    "best efforts": "unqualified standard; construed differently across jurisdictions",
    "as soon as practicable": "no outer time limit; unenforceable as a deadline",
    "from time to time": "open-ended; permits unilateral variation",
    "material adverse": "threshold undefined; the classic arbitration battleground",
    "substantially": "degree unquantified",
    "including but not limited to": "enumeration is non-exhaustive; scope indeterminate",
    "and/or": "disjunction/conjunction ambiguity",
    "appropriate": "standard-setter unidentified",
    "promptly": "no measurable period",
    "customary": "reference class unstated",
    "satisfactory to": "subjective acceptance criterion",
}

REQUIRED_CLAUSES = {
    "commercial": {
        "governing law": r"govern(?:ed|ing)\s+by",
        "dispute resolution": r"arbitrat|jurisdiction of the courts|dispute resolution",
        "termination": r"terminat",
        "limitation of liability": r"liab",
        "confidentiality": r"confidential",
        "force majeure": r"force majeure",
        "notices": r"notice",
        "assignment": r"assign",
    },
    "ip_licence": {
        "governing law": r"govern(?:ed|ing)\s+by",
        "dispute resolution": r"arbitrat|jurisdiction",
        "grant of rights": r"grants?\s+(?:to\s+)?\w+\s+(?:a\s+)?(?:non-?exclusive|exclusive|licen[cs]e)",
        "royalty": r"royalt",
        "field of use": r"field of use",
        "improvements": r"improvement",
        "termination": r"terminat",
        "warranty of title": r"warrant",
    },
    "treaty": {
        "entry into force": r"enter(?:s|ed)?\s+into\s+force",
        "dispute settlement": r"dispute settlement|arbitrat|international court",
        "withdrawal": r"withdraw|denounc",
        "amendment": r"amend",
        "verification": r"verif|inspect|monitor",
        "reservations": r"reservation",
    },
}

ATTRIBUTE_LABELS = {
    "governing_law": "governing law",
    "arbitration_seat": "seat of arbitration",
    "arbitration_rules": "arbitral rules",
    "notice_period_days": "notice period",
    "liability_cap": "cap on liability",
}

SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}


def _finding(category, subtype, severity, confidence, title, detail, spans, provenance=None) -> dict:
    return {
        "category": category,
        "subtype": subtype,
        "severity": severity,
        "confidence": round(confidence, 3),
        "title": title,
        "detail": detail,
        "spans": spans,
        "provenance": provenance or [],
    }


def _normalise_predicate(text: str) -> str:
    words = re.findall(r"[a-z]{3,}", text.lower())
    stop = {"the", "any", "all", "such", "with", "from", "into", "that", "this", "shall", "not", "and", "for"}
    return " ".join(w for w in words if w not in stop)[:80]


def detect_ambiguity(doc_id: str, title: str, text: str, extraction) -> list[dict]:
    findings = []
    for term, why in VAGUE_TERMS.items():
        for m in re.finditer(re.escape(term), text, re.I):
            context = text[max(0, m.start() - 120):m.end() + 120].replace("\n", " ").strip()
            findings.append(_finding(
                "in_text", "ambiguity", "low" if term in ("promptly", "customary", "appropriate") else "medium",
                0.62, f"Ambiguous standard: “{m.group(0)}”",
                f"{why}. Occurs in {title}.",
                [{"doc_id": doc_id, "start": m.start(), "end": m.end(), "text": context[:300]}],
            ))
    # A defined term that is used but never defined is ambiguity by construction.
    defined = {d["term"].lower() for d in extraction.defined_terms}
    used = defaultdict(list)
    for m in re.finditer(r"[\"“]([A-Z][A-Za-z0-9 \-]{2,40})[\"”]", text):
        used[m.group(1).lower()].append(m)
    for term, matches in used.items():
        if term not in defined and len(matches) >= 2:
            m = matches[0]
            findings.append(_finding(
                "in_text", "ambiguity", "medium", 0.7,
                f"Quoted term “{matches[0].group(1)}” used {len(matches)} times but never defined",
                "The instrument treats the term as defined without supplying a definition clause.",
                [{"doc_id": doc_id, "start": m.start(), "end": m.end(), "text": m.group(0)}],
            ))
    return findings


INSTRUMENT_MARKERS = (r"\bthis agreement\b", r"\bthe parties\b", r"\bhereby agrees?\b", r"\bside letter\b",
                      r"\bthis deed\b", r"\bwitnesseth\b", r"\bin witness whereof\b", r"\bthis convention\b",
                      r"\bthe contracting parties\b", r"\bthis protocol\b")


def looks_like_instrument(text: str) -> bool:
    """Only instruments have required provisions.

    Running a contract checklist over an expert report or a correspondence
    bundle produces a wall of omissions that are not defects, which buries the
    findings that are.
    """
    markers = sum(1 for pattern in INSTRUMENT_MARKERS if re.search(pattern, text, re.I))
    obligations = len(re.findall(r"\bshall\b", text, re.I))
    return markers >= 1 and obligations >= 3


def detect_omission(doc_id: str, title: str, text: str, doc_type: str = "commercial") -> list[dict]:
    if not looks_like_instrument(text):
        return []
    checklist = REQUIRED_CLAUSES.get(doc_type, REQUIRED_CLAUSES["commercial"])
    findings = []
    for label, pattern in checklist.items():
        if not re.search(pattern, text, re.I):
            findings.append(_finding(
                "in_text", "omission", "high" if label in ("governing law", "dispute resolution") else "medium",
                0.8, f"Missing required provision: {label}",
                f"No clause in {title} addresses {label}; required for instruments of type “{doc_type}”.",
                [{"doc_id": doc_id, "start": 0, "end": min(len(text), 200), "text": text[:200]}],
            ))
    return findings


def detect_internal_contradictions(doc_id: str, title: str, extraction) -> list[dict]:
    findings = []
    by_key = defaultdict(list)
    for fact in extraction.facts:
        if fact["kind"] == "attribute":
            by_key[fact["key"]].append(fact)
    for key, facts in by_key.items():
        values = {}
        for fact in facts:
            values.setdefault(str(fact["normalized"]).lower(), []).append(fact)
        if len(values) > 1:
            spans = [f["span"] | {"clause": f["clause"], "value": f["value"]}
                     for group in values.values() for f in group[:2]]
            findings.append(_finding(
                "in_text", "direct_contradiction", "high", 0.85,
                f"Conflicting {ATTRIBUTE_LABELS.get(key, key)} within {title}",
                "The same instrument states " + " and ".join(
                    f"“{group[0]['value']}” (clause {group[0]['clause']})" for group in values.values()) + ".",
                spans,
            ))

    # Opposed deontic modality over the same subject/predicate.
    grouped = defaultdict(list)
    for obligation in extraction.obligations:
        grouped[(obligation["subject"].lower(), _normalise_predicate(obligation["predicate"]))].append(obligation)
    for (subject, predicate), group in grouped.items():
        polarities = {o["polarity"] for o in group}
        if len(polarities) > 1 and predicate:
            findings.append(_finding(
                "in_text", "direct_contradiction", "critical", 0.8,
                f"Opposed obligations for “{group[0]['subject']}”",
                f"The instrument both requires and prohibits: {predicate!r} "
                f"(clauses {', '.join(sorted({str(o['clause']) for o in group}))}).",
                [o["span"] | {"clause": o["clause"], "modal": o["modal"]} for o in group[:4]],
            ))

    # Conflicting definitions of the same term.
    by_term = defaultdict(list)
    for definition in extraction.defined_terms:
        if definition["definition"]:
            by_term[definition["term"].lower()].append(definition)
    for term, defs in by_term.items():
        normalised = {re.sub(r"\W+", " ", d["definition"].lower()).strip() for d in defs}
        if len(normalised) > 1:
            findings.append(_finding(
                "in_text", "direct_contradiction", "high", 0.75,
                f"Term “{defs[0]['term']}” defined {len(normalised)} different ways",
                " | ".join(d["definition"][:120] for d in defs[:3]),
                [d["span"] | {"clause": d["clause"]} for d in defs[:4]],
            ))
    return findings


def detect_cross_document(docs: list[dict]) -> list[dict]:
    """docs: [{id, title, extraction}] -- conflicts between instruments."""
    findings = []
    by_key = defaultdict(list)
    for doc in docs:
        for fact in doc["extraction"].facts:
            if fact["kind"] == "attribute":
                by_key[fact["key"]].append((doc, fact))
    for key, pairs in by_key.items():
        values = defaultdict(list)
        for doc, fact in pairs:
            values[str(fact["normalized"]).lower()].append((doc, fact))
        if len(values) > 1 and len({doc["id"] for doc, _ in pairs}) > 1:
            spans, descriptions = [], []
            for group in values.values():
                doc, fact = group[0]
                spans.append(fact["span"] | {"document": doc["title"], "clause": fact["clause"],
                                             "value": fact["value"]})
                descriptions.append(f"“{fact['value']}” in {doc['title']} (clause {fact['clause']})")
            findings.append(_finding(
                "cross_document", "direct_contradiction", "critical", 0.88,
                f"Corpus states {len(values)} different values for {ATTRIBUTE_LABELS.get(key, key)}",
                "; ".join(descriptions) + ". Instruments in one transaction must agree on this term.",
                spans,
            ))

    # The same term defined differently in different instruments.
    term_defs = defaultdict(list)
    for doc in docs:
        for definition in doc["extraction"].defined_terms:
            if definition["definition"]:
                term_defs[definition["term"].lower()].append((doc, definition))
    for term, pairs in term_defs.items():
        variants = {re.sub(r"\W+", " ", d["definition"].lower()).strip(): (doc, d) for doc, d in pairs}
        if len(variants) > 1 and len({doc["id"] for doc, _ in pairs}) > 1:
            findings.append(_finding(
                "cross_document", "direct_contradiction", "high", 0.8,
                f"“{pairs[0][1]['term']}” carries inconsistent definitions across the corpus",
                " | ".join(f"{doc['title']}: {d['definition'][:110]}" for doc, d in list(variants.values())[:3]),
                [d["span"] | {"document": doc["title"]} for doc, d in list(variants.values())[:4]],
            ))
    return findings


def detect_chronology_conflicts(events: list[dict], extractions_by_doc: dict) -> list[dict]:
    """Dates attached to the same named event that do not agree."""
    findings = []
    anchors = ("effective date", "commencement date", "closing date", "priority date",
               "filing date", "termination date", "entry into force", "signature date")
    buckets = defaultdict(list)
    for event in events:
        lowered = event["assertion"].lower()
        raw_pos = lowered.find(event["raw"].lower()) if event.get("raw") else -1
        for anchor in anchors:
            anchor_pos = lowered.rfind(anchor, 0, raw_pos if raw_pos > 0 else None)
            # The anchor must actually govern this date, not merely share a window.
            if anchor_pos >= 0 and (raw_pos < 0 or raw_pos - anchor_pos <= 70):
                buckets[anchor].append(event)
    for anchor, group in buckets.items():
        dates = {e["date"] for e in group}
        if len(dates) > 1:
            findings.append(_finding(
                "cross_document", "direct_contradiction", "critical", 0.9,
                f"“{anchor.title()}” asserted as {len(dates)} different dates",
                "; ".join(f"{e['date']} in {e['document']} (clause {e['clause']})" for e in group[:5]),
                [e["span"] | {"document": e["document"], "date": e["date"]} for e in group[:5]],
            ))

    priority = [e for e in events if "priority date" in e["assertion"].lower()]
    filing = [e for e in events if "filing date" in e["assertion"].lower() or "filed on" in e["assertion"].lower()]
    for p in priority:
        for f in filing:
            if p["date"] > f["date"]:
                findings.append(_finding(
                    "cross_document", "direct_contradiction", "critical", 0.85,
                    "Priority date postdates the filing date",
                    f"Priority claimed at {p['date']} ({p['document']}) but filing recorded at "
                    f"{f['date']} ({f['document']}); the priority claim is chronologically impossible.",
                    [p["span"] | {"document": p["document"]}, f["span"] | {"document": f["document"]}],
                ))
    return findings


def detect_math(harvests: list[dict], doc_titles: dict[str, str]) -> list[dict]:
    findings = []
    for harvest in harvests:
        doc_id = harvest["document_id"]
        title = doc_titles.get(doc_id, doc_id)
        for item in harvest["archived"]:
            analysis, verification = item["analysis"], item["analysis"]["verification"]
            if verification["verdict"] == "refuted":
                sym = next(b for b in verification["backends"] if b["backend"] == "sympy")
                counter = sym.get("counterexample", {})
                findings.append(_finding(
                    "mathematical", "false_claim", "critical", 0.95,
                    f"Mathematical claim is false: {analysis['source_text'][:80]}",
                    f"{sym['detail']} Residual: {sym.get('residual')}. "
                    + (f"Counterexample: {counter.get('assignment')} gives "
                       f"{counter.get('lhs_value')} ≠ {counter.get('rhs_value')}." if counter else ""),
                    [item["span"] | {"document": title}],
                    provenance=[{"equation_id": item["equation_id"], "backend": "sympy",
                                 "canonical_key": analysis["canonical_key"]}],
                ))
            elif verification["verdict"] == "conditional":
                findings.append(_finding(
                    "mathematical", "overbroad_claim", "medium", 0.7,
                    f"Claim asserted generally but holds only conditionally: {analysis['source_text'][:70]}",
                    next(b for b in verification["backends"] if b["backend"] == "sympy")["detail"],
                    [item["span"] | {"document": title}],
                    provenance=[{"equation_id": item["equation_id"]}],
                ))
            if verification.get("disagreement"):
                findings.append(_finding(
                    "mathematical", "backend_disagreement", "high", 0.6,
                    f"Verification backends disagree on: {analysis['source_text'][:70]}",
                    verification["disagreement"], [item["span"] | {"document": title}],
                    provenance=[{"equation_id": item["equation_id"]}],
                ))
            for match in item["prior_art"]:
                findings.append(_finding(
                    "mathematical", "prior_art_congruence", "high", 0.85,
                    f"Equation is mathematically identical to archived prior art",
                    f"“{analysis['source_text'][:80]}” is congruent to archived “{match['source_text'][:80]}” "
                    f"({match['detail']}). Algebraic restatement does not create novelty.",
                    [item["span"] | {"document": title}],
                    provenance=[{"equation_id": item["equation_id"], "matched_equation_id": match["equation_id"],
                                 "method": match["match"]}],
                ))
        for skipped in harvest["unparsed"][:10]:
            findings.append(_finding(
                "mathematical", "unparsed_expression", "low", 0.5,
                f"Equation-shaped text could not be formalised: {skipped['source_text'][:60]}",
                f"{skipped['reason']}. Archived unadjudicated; a human should confirm the notation.",
                [skipped["span"] | {"document": title}],
            ))
    return findings


def detect_definition_conflicts(harvests: list[dict], doc_titles: dict[str, str], congruent) -> list[dict]:
    """The same symbol defined two incompatible ways across the corpus.

    This is the mathematical analogue of a defined-term conflict, and the most
    consequential one in patent lawfare: if a report defines the gain as (a+b)^2
    in one paragraph and a^2+b^2 in another, at most one of them supports the
    distinction being drawn over the prior art. Congruent restatements are not
    conflicts, which is why the comparison runs through equality saturation
    rather than string comparison.
    """
    import sympy as sp

    definitions = defaultdict(list)
    for harvest in harvests:
        for item in harvest["archived"]:
            verification = item["analysis"]["verification"]
            sym = next((b for b in verification["backends"] if b["backend"] == "sympy"), {})
            if sym.get("status") != "definitional":
                continue
            definitions[sym["defines"]].append({
                "item": item, "document_id": harvest["document_id"],
                "definiens": sym["definiens"], "expr": sp.sympify(item["analysis"]["sympy_repr"]),
            })

    findings = []
    for symbol, group in definitions.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                left, right = group[i], group[j]
                if left["definiens"] == right["definiens"]:
                    continue
                try:
                    result = congruent(left["expr"], right["expr"], node_budget=2500, max_iters=8)
                except Exception:
                    continue
                if result["equivalent"]:
                    continue
                findings.append(_finding(
                    "mathematical", "conflicting_definition", "critical", 0.9,
                    f"Symbol “{symbol}” is defined two incompatible ways",
                    f"{symbol} = {left['definiens']} in {doc_titles.get(left['document_id'], left['document_id'])} "
                    f"but {symbol} = {right['definiens']} in "
                    f"{doc_titles.get(right['document_id'], right['document_id'])}. "
                    "Equality saturation derived no congruence between the two definientia, so at most one "
                    "of them can support the conclusions drawn from it.",
                    [left["item"]["span"] | {"document": doc_titles.get(left["document_id"])},
                     right["item"]["span"] | {"document": doc_titles.get(right["document_id"])}],
                    provenance=[{"equation_id": left["item"]["equation_id"]},
                                {"equation_id": right["item"]["equation_id"]},
                                {"method": "equality_saturation", "saturated": result["saturated"]}],
                ))
    return findings


def detect_regulatory(doc_id: str, title: str, text: str, jurisdiction_ids: list[str],
                      taxonomy) -> list[dict]:
    """Findings from the semantic law matrix.

    Two kinds. A missing provision is an omission with a named source: this
    document engages a body of law, and the provision that body expects is not
    in the text. A divergence is a cross-border problem rather than a drafting
    one: the forums selected for this matter do not agree about the same
    classification, so one set of terms cannot satisfy all of them.
    """
    assessment = taxonomy.assess(text, jurisdiction_ids)
    engaged = {hit["class"]: hit for hit in assessment["engaged"]}
    findings = []

    def cue_span(class_id: str) -> dict:
        hit = engaged.get(class_id)
        cue = hit["cues"][0] if hit and hit["cues"] else None
        if not cue:
            return {"doc_id": doc_id, "start": 0, "end": 0, "text": title}
        start, end = cue["start"], cue["end"]
        return {"doc_id": doc_id, "start": start, "end": end, "document": title,
                "text": text[max(0, start - 90):end + 90].replace("\n", " ").strip()}

    for item in assessment["missing_provisions"]:
        findings.append(_finding(
            "statutory", "missing_provision",
            "high" if (item.get("density") or 0) >= 28 else "medium", 0.72,
            f"{item['class_label']}: no provision on {item['provision']}",
            f"The document engages {item['class_label'].lower()} but contains nothing addressing "
            f"{item['provision']}. Instruments to check: {', '.join(item['instruments']) or 'see matrix'}. "
            f"{item['verify'] or ''}".strip(),
            [cue_span(item["class"])],
            provenance=[{"matrix_class": item["class"], "expects": item["provision"],
                         "matrix_generated": assessment["matrix_generated"]}],
        ))

    for item in assessment["divergence"]:
        coverage = ", ".join(f"{jid} {level}" for jid, level in item["coverage"].items())
        findings.append(_finding(
            "statutory", "cross_border_divergence",
            "high" if item["risk"] >= 18 else "medium", 0.68,
            f"Forum divergence on {item['label'].lower()}",
            f"Coverage across the selected forums differs by {item['spread']} points ({coverage}); "
            f"{item['strictest']} is strictest, {item['loosest']} the loosest. Weighted by the normative "
            f"load of this classification the exposure scores {item['risk']}"
            + (f", ranking {item['global_risk_rank']} of 32 for divergence field-wide"
               if item["global_risk_rank"] else "")
            + f". {item['verify']}",
            [cue_span(item["class"])],
            provenance=[{"matrix_class": item["class"], "spread": item["spread"],
                         "instruments": item["instruments"]}],
        ))
    return findings


def detect_epistemic(doc_id: str, title: str, forensics: dict, injection: dict, triage: dict) -> list[dict]:
    findings = []
    for item in forensics.get("findings", []):
        findings.append(_finding(
            "epistemic", f"forensic:{item['code']}", item["severity"], 0.9,
            f"Forensic anomaly in {title}: {item['code'].replace('_', ' ')}",
            item["detail"], [{"doc_id": doc_id, "start": 0, "end": 0, "text": title}],
            provenance=[{"sha256": forensics.get("digest", {}).get("sha256")}],
        ))
    if injection.get("verdict") != "clean":
        top = injection["signals"][0] if injection.get("signals") else {}
        findings.append(_finding(
            "epistemic", "prompt_injection", "critical" if injection["verdict"] == "hostile" else "high",
            min(0.99, 0.5 + injection.get("score", 0) / 2),
            f"Prompt-injection payload detected in {title}",
            f"Verdict {injection['verdict']} (score {injection['score']}). "
            f"Signals: {', '.join(injection.get('distinct_signals', []))}. "
            f"Highest-weighted match: “{top.get('match', '')[:120]}”. "
            + ("Document held at tier L3 and excluded from the reasoning context."
               if triage.get("quarantined") else "Document promoted with neutralised quotation only."),
            [{"doc_id": doc_id, "start": top.get("offset", 0), "end": top.get("end", 0),
              "text": top.get("context", "")[:300]}],
        ))
    return findings


def rank(findings: list[dict]) -> list[dict]:
    return sorted(findings, key=lambda f: (-SEVERITY_ORDER.get(f["severity"], 0), -f["confidence"], f["title"]))
