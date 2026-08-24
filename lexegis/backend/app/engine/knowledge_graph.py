"""Matter-scoped knowledge graph.

Nodes: documents, parties, defined terms, obligations, dates, equations,
findings. Edges record how a conclusion was reached, which is what makes the
graph an audit object rather than a visualisation.
"""
from collections import defaultdict


def build(docs: list[dict], findings: list[dict], harvests: list[dict], events: list[dict]) -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def node(node_id: str, kind: str, label: str, **attrs) -> str:
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "kind": kind, "label": label[:160], **attrs}
        return node_id

    def edge(source: str, target: str, relation: str, **attrs) -> None:
        edges.append({"source": source, "target": target, "relation": relation, **attrs})

    for doc in docs:
        doc_node = node(doc["id"], "document", doc["title"], tier=doc.get("tier"),
                        quarantined=doc.get("quarantined", False))
        extraction = doc["extraction"]
        for party in extraction.parties:
            party_node = node(f"party:{party['name'].lower()}", "party", party["name"])
            edge(doc_node, party_node, "names_party", span=party["span"])
        for definition in extraction.defined_terms:
            term_node = node(f"term:{definition['term'].lower()}", "defined_term", definition["term"])
            edge(doc_node, term_node, "defines", clause=definition["clause"], span=definition["span"])
        for i, obligation in enumerate(extraction.obligations[:200]):
            ob_node = node(f"{doc['id']}:ob{i}", "obligation",
                           f"{obligation['subject']} {obligation['modal']} {obligation['predicate'][:60]}",
                           deontic=obligation["deontic"], polarity=obligation["polarity"])
            edge(doc_node, ob_node, "imposes", clause=obligation["clause"])
            subject_node = f"party:{obligation['subject'].lower()}"
            if subject_node in nodes:
                edge(ob_node, subject_node, "binds")
        for fact in extraction.facts:
            if fact["kind"] == "attribute":
                attr_node = node(f"attr:{fact['key']}:{str(fact['normalized']).lower()}", "attribute",
                                 f"{fact['key']} = {fact['value']}", key=fact["key"])
                edge(doc_node, attr_node, "asserts", clause=fact["clause"], span=fact["span"])

    for event in events:
        event_node = node(f"event:{event['date']}:{hash(event['assertion']) % 10**6}", "event",
                          f"{event['date']} — {event['assertion'][:80]}", date=event["date"])
        edge(event["document_id"], event_node, "evidences", span=event["span"])

    for harvest in harvests:
        for item in harvest["archived"]:
            eq_node = node(item["equation_id"], "equation", item["analysis"]["source_text"][:120],
                           verdict=item["analysis"]["verification"]["verdict"],
                           canonical_key=item["analysis"]["canonical_key"])
            edge(harvest["document_id"], eq_node, "postulates", span=item["span"])
            for match in item["prior_art"]:
                edge(eq_node, match["equation_id"], "congruent_to", method=match["match"])

    for i, finding in enumerate(findings):
        finding_node = node(f"finding:{i}", "finding", finding["title"], severity=finding["severity"],
                            subtype=finding["subtype"], category=finding["category"])
        for span in finding["spans"]:
            if span.get("doc_id") in nodes:
                edge(finding_node, span["doc_id"], "derived_from", span=span)
        for item in finding.get("provenance", []):
            if item.get("equation_id") in nodes:
                edge(finding_node, item["equation_id"], "derived_from")

    degrees = defaultdict(int)
    for e in edges:
        degrees[e["source"]] += 1
        degrees[e["target"]] += 1
    for node_id, count in degrees.items():
        if node_id in nodes:
            nodes[node_id]["degree"] = count

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "by_kind": {kind: sum(1 for n in nodes.values() if n["kind"] == kind)
                        for kind in {n["kind"] for n in nodes.values()}},
        },
    }
