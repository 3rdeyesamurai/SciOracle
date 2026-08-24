"""Semantic law matrix: integrity of the computed artefact and its use on documents."""
import pytest

from app.engine import discrepancy, taxonomy

TERMS = """SUBSCRIPTION TERMS
Your subscription will renew automatically every month until cancelled.
We collect personal data including your payment method, and cookies are used for
interest-based advertising. Prices exclude applicable fees.
Buy now, pay later is available in instalments.
This Agreement shall be governed by the laws of Switzerland.
"""


def test_matrix_is_well_formed():
    report = taxonomy.matrix()
    assert len(report["classes"]) == 32
    assert len(report["jurisdictions"]) == 12
    assert len(report["primitives"]) == 16
    for cls in report["classes"]:
        assert set(cls["weights"]) == set(report["primitiveIds"])
        assert set(cls["coverage"]) == set(report["jurisdictionIds"])
        assert all(0 <= v <= 3 for v in cls["weights"].values())
        assert all(0 <= v <= 3 for v in cls["coverage"].values())
        assert cls["instruments"], f"{cls['id']} cites no instrument"
        assert cls["verify"], f"{cls['id']} has no verification note"
        assert cls["density"] == sum(cls["weights"].values())


def test_similarity_matrix_is_a_proper_cosine_matrix():
    similarity = taxonomy.matrix()["matrices"]["similarity"]
    assert len(similarity) == 32
    for i, row in enumerate(similarity):
        assert row[i] == pytest.approx(1.0, abs=0.001)
        for j, value in enumerate(row):
            assert -1.001 <= value <= 1.001
            assert value == pytest.approx(similarity[j][i], abs=0.001)


def test_classifications_cluster_with_their_neighbours():
    index = {c["id"]: c for c in taxonomy.matrix()["classes"]}
    # withdrawal and distance selling are the same normative animal
    assert index["withdrawal_rights"]["neighbours"][0]["class"] == "distance_selling"
    # conformity of digital content sits next to goods with digital elements
    assert index["digital_content_conformity"]["neighbours"][0]["class"] == "goods_digital_elements"


def test_divergence_ranking_is_ordered_and_bounded():
    divergence = taxonomy.matrix()["divergence"]
    assert len(divergence) == 32
    risks = [d["risk"] for d in divergence]
    assert risks == sorted(risks, reverse=True)
    # the most universal rule in the field should be the least divergent
    assert divergence[-1]["class"] == "unfair_commercial_practices"


def test_jurisdiction_signatures_are_distinct():
    report = taxonomy.matrix()
    signatures = {j["id"]: tuple(j["signature"]["over"]) for j in report["jurisdictions"]}
    # the emphasis statistic must actually discriminate: raw exposure cosine did not
    assert len(set(signatures.values())) >= 9
    distance = report["matrices"]["jurisdictionDistance"]
    off_diagonal = [distance[a][b] for a in range(12) for b in range(12) if a != b]
    assert max(off_diagonal) > 1.0


def test_classify_finds_engaged_bodies_of_law_with_spans():
    engaged = {hit["class"]: hit for hit in taxonomy.classify(TERMS)}
    assert "subscriptions_autorenewal" in engaged
    assert "data_protection" in engaged
    assert "credit_bnpl" in engaged
    for hit in engaged.values():
        for cue in hit["cues"]:
            assert TERMS[cue["start"]:cue["end"]]


def test_missing_provisions_are_reported_against_named_instruments():
    missing = taxonomy.missing_provisions(TERMS, ["subscriptions_autorenewal", "credit_bnpl"])
    provisions = {m["provision"] for m in missing}
    assert "cancellation mechanism" in provisions
    assert "cost of credit disclosure" in provisions
    assert all(m["instruments"] for m in missing)


def test_present_provision_is_not_reported_missing():
    text = TERMS + "\nYou may cancel at any time from your account settings."
    provisions = {m["provision"] for m in taxonomy.missing_provisions(text, ["subscriptions_autorenewal"])}
    assert "cancellation mechanism" not in provisions


def test_divergence_exposure_names_strictest_and_loosest_forum():
    exposure = taxonomy.divergence_exposure(["data_protection", "withdrawal_rights"], ["EU", "US_FED"])
    by_class = {d["class"]: d for d in exposure}
    assert by_class["withdrawal_rights"]["strictest"] == "EU"
    assert by_class["withdrawal_rights"]["loosest"] == "US_FED"
    assert by_class["withdrawal_rights"]["spread"] == 3


def test_single_jurisdiction_raises_no_divergence():
    assert taxonomy.assess(TERMS, ["EU"])["divergence"] == []


def test_compare_jurisdictions_reports_emphasis_deltas():
    result = taxonomy.compare_jurisdictions("EU", "US_FED")
    assert result["distance"] > 1.0
    drivers = {d["primitive"] for d in result["primitiveDeltas"][:4]}
    # the US federal regime leans on prohibition and enforcement; the EU on structure
    assert {"prohibition", "enforcement"} & drivers
    assert any(g["class"] == "withdrawal_rights" for g in result["coverageGaps"])


def test_regulatory_detector_emits_span_anchored_findings():
    findings = discrepancy.detect_regulatory("d1", "terms.txt", TERMS, ["EU", "US_FED"], taxonomy)
    subtypes = {f["subtype"] for f in findings}
    assert "missing_provision" in subtypes
    assert "cross_border_divergence" in subtypes
    for finding in findings:
        assert finding["category"] == "statutory"
        assert finding["spans"] and finding["provenance"]
        span = finding["spans"][0]
        assert span["doc_id"] == "d1"


def test_matrix_endpoints(client, account):
    matrix = client.get("/api/v1/taxonomy", headers=account["headers"]).json()
    assert len(matrix["classes"]) == 32

    one = client.get("/api/v1/taxonomy/classes/dark_patterns", headers=account["headers"]).json()
    assert one["classification"]["label"].startswith("Manipulative")
    assert one["divergence"]["risk"] > 0

    assert client.get("/api/v1/taxonomy/classes/nope", headers=account["headers"]).status_code == 404

    compare = client.get("/api/v1/taxonomy/compare?a=eu&b=us_fed", headers=account["headers"]).json()
    assert compare["distance"] > 1.0

    assessed = client.post("/api/v1/taxonomy/assess",
                           json={"text": TERMS, "jurisdictions": ["EU", "US_FED", "SG"]},
                           headers=account["headers"]).json()
    assert assessed["engaged"] and assessed["missing_provisions"] and assessed["divergence"]


def test_analysis_accepts_jurisdictions_and_produces_statutory_findings(client, account):
    client.post("/api/v1/billing/subscription", json={"plan": "pro"}, headers=account["headers"])
    matter = client.post("/api/v1/matters", json={"name": "Consumer terms review"},
                         headers=account["headers"]).json()
    client.post(f"/api/v1/matters/{matter['id']}/documents/text",
                json={"filename": "terms.txt", "text": TERMS}, headers=account["headers"])
    result = client.post(f"/api/v1/matters/{matter['id']}/analyse",
                         json={"doc_type": "commercial", "jurisdictions": ["EU", "US_FED"]},
                         headers=account["headers"]).json()
    statutory = [f for f in result["findings"] if f["category"] == "statutory"]
    assert statutory, "jurisdiction-aware analysis should raise statutory findings"
    assert any(f["subtype"] == "cross_border_divergence" for f in statutory)
