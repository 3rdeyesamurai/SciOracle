"""API and tenancy tests, plus a full pass over the demonstration corpus."""


def test_health_reports_backend_availability(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["system2_backends"]["sympy"]["available"] is True
    assert "lean4" in body["system2_backends"]


def test_signup_login_and_me(client, account):
    login = client.post("/api/v1/auth/login",
                        json={"email": account["email"], "password": "a-very-long-passphrase"})
    assert login.status_code == 200
    me = client.get("/api/v1/auth/me", headers=account["headers"]).json()
    assert me["org"]["plan"] == "free"
    assert me["entitlements"]["limits"]["matters"] == 2


def test_authentication_is_required(client):
    assert client.get("/api/v1/matters").status_code == 401
    assert client.get("/api/v1/matters", headers={"Authorization": "Bearer nonsense"}).status_code == 401


def test_tenant_isolation(client, account):
    matter = client.post("/api/v1/matters", json={"name": "Confidential matter"},
                         headers=account["headers"]).json()
    other = client.post("/api/v1/auth/signup", json={
        "email": "outsider@example.com", "password": "another-long-passphrase",
        "organisation": "Other Firm"}).json()
    response = client.get(f"/api/v1/matters/{matter['id']}",
                          headers={"Authorization": f"Bearer {other['access_token']}"})
    assert response.status_code == 404


def test_api_key_lifecycle(client, account):
    created = client.post("/api/v1/auth/api-keys", json={"name": "ci"}, headers=account["headers"])
    assert created.status_code == 201
    key = created.json()["api_key"]
    assert client.get("/api/v1/matters", headers={"X-API-Key": key}).status_code == 200
    # a service key may not mint further keys
    assert client.post("/api/v1/auth/api-keys", json={"name": "escalation"},
                       headers={"X-API-Key": key}).status_code == 403
    key_id = created.json()["id"]
    assert client.delete(f"/api/v1/auth/api-keys/{key_id}", headers=account["headers"]).status_code == 200
    assert client.get("/api/v1/matters", headers={"X-API-Key": key}).status_code == 401


def test_plan_quota_blocks_beyond_limit(client, account):
    for i in range(2):
        assert client.post("/api/v1/matters", json={"name": f"matter {i}"},
                           headers=account["headers"]).status_code == 201
    blocked = client.post("/api/v1/matters", json={"name": "third"}, headers=account["headers"])
    assert blocked.status_code == 402
    upgrade = client.post("/api/v1/billing/subscription", json={"plan": "pro"}, headers=account["headers"])
    assert upgrade.status_code == 200
    assert client.post("/api/v1/matters", json={"name": "third"},
                       headers=account["headers"]).status_code == 201


def test_demo_corpus_end_to_end(client, account):
    client.post("/api/v1/billing/subscription", json={"plan": "pro"}, headers=account["headers"])
    seeded = client.post("/api/v1/demo/seed", headers=account["headers"])
    assert seeded.status_code == 200, seeded.text
    body = seeded.json()
    analysis = body["analysis"]

    titles = [f["title"] for f in analysis["findings"]]
    subtypes = {f["subtype"] for f in analysis["findings"]}

    # the hostile exhibit is quarantined, not reasoned over
    hostile = [d for d in body["ingested"] if "exhibit_c" in d["filename"]][0]
    assert hostile["quarantined"] is True
    assert hostile["tier"] == "L3"
    assert "prompt_injection" in subtypes

    # cross-instrument conflicts
    assert any("governing law" in t for t in titles)
    assert any("Effective Date" in t or "effective date" in t.lower() for t in titles)

    # mathematics
    assert any("Priority date postdates" in t for t in titles)
    assert "prior_art_congruence" in subtypes or any("prior art" in t.lower() for t in titles)
    assert analysis["equations"], "equations should be archived"
    assert any(e["verdict"] in ("refuted", "verified", "definitional") for e in analysis["equations"])

    # graph and ledger
    assert analysis["graph"]["stats"]["node_count"] > 10
    assert client.get("/api/v1/ledger/verify", headers=account["headers"]).json()["valid"] is True


def test_math_endpoints(client, account):
    compare = client.post("/api/v1/math/compare",
                          json={"left": "y = (a+b)^2", "right": "y = a^2 + 2*a*b + b^2"},
                          headers=account["headers"]).json()
    assert compare["result"]["equivalent"] is True

    analysis = client.post("/api/v1/math/analyse",
                           json={"expression": "(a+b)^2 = a^2 + b^2", "archive": True},
                           headers=account["headers"]).json()
    assert analysis["verification"]["verdict"] == "refuted"
    assert "csymbol" in analysis["content_mathml"]
    assert analysis["omdoc"].startswith("<?xml")

    omdoc = client.get(f"/api/v1/math/archive/{analysis['equation_id']}/omdoc", headers=account["headers"])
    assert omdoc.headers["content-type"].startswith("application/omdoc+xml")

    bad = client.post("/api/v1/math/analyse", json={"expression": "))("}, headers=account["headers"])
    assert bad.status_code == 422


def test_alignment_endpoint(client, account):
    text = "The seat shall be Geneva. The seat shall be London. The parties shall cooperate."
    result = client.post("/api/v1/eval/alignment", json={
        "text": text,
        "predicted": [{"start": 0, "end": 25}],
        "ground_truth": [{"start": 2, "end": 24}],
    }, headers=account["headers"]).json()
    assert result["true_positives"] == 1
    assert result["f1"] == 1.0


def test_ingested_text_document_dossier(client, account):
    matter = client.post("/api/v1/matters", json={"name": "Dossier"}, headers=account["headers"]).json()
    ingested = client.post(f"/api/v1/matters/{matter['id']}/documents/text", json={
        "filename": "note.txt",
        "text": "1. Term. This Agreement shall be governed by the laws of Switzerland for three years.",
    }, headers=account["headers"]).json()
    dossier = client.get(f"/api/v1/documents/{ingested['document_id']}", headers=account["headers"]).json()
    assert dossier["tier"] == "L2"
    assert dossier["forensics"]["digest"]["sha256"] == ingested["sha256"]
    assert dossier["ledger"][0]["action"] == "document.ingested"
