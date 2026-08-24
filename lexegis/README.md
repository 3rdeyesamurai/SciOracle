# Lexegis — neuro-symbolic discrepancy validation & equation archiving

A deployable SaaS implementation of the dual-process legal-engine architecture: a fast heuristic
layer reads international paperwork, and a deterministic symbolic layer decides what is actually
true. Findings are span-anchored, mathematical claims are archived in semantic markup and formally
adjudicated, and every decision is written to a hash-chained ledger that can be re-verified.

```
lexegis/
├── backend/        FastAPI engine + multi-tenant SaaS layer (Python, stdlib-heavy)
├── frontend/       React (Vite) application — matters, findings, equation lab, ledger
├── law-matrix/     Semantic law matrix — 32 classifications of digital consumer law, computed in Wolfram
└── docker-compose.yml
```

## Run it

```bash
cd lexegis
cp .env.example .env && sed -i "s/change-me.*/$(openssl rand -hex 32)/" .env
docker compose up --build          # web app on http://localhost:8080
```

Local development without Docker:

```bash
cd backend && pip install -r requirements.txt
LEXEGIS_JWT_SECRET=dev uvicorn app.main:app --reload      # engine + OpenAPI at /docs
cd ../frontend && npm install && npm run dev              # http://localhost:5173
```

Sign up, then press **Load demonstration matter** — it ingests a four-document corpus seeded with
the defect classes the engine exists to find, and runs a full analysis.

## What actually happens to a document

| Stage | Module | What it does |
|---|---|---|
| 1 · Forensics | `app/security/forensics.py` | SHA-256/512 fingerprints, PDF incremental-update chains, signature dictionaries, timestamp inversions, invisible render layers, zero-width and homoglyph channels. |
| 2 · Injection scan | `app/security/injection.py` | Directive lexicon, role/frame forgery, exfiltration patterns, base64 carriers that decode to directives, and a salience ratio standing in for attention-head anomaly tracking. |
| 3 · Triage | `app/security/triage.py` | Default-deny promotion L3 → L2. Hostile documents stay at L3; **nothing** is ever promoted to L1, the operator-only command tier. The gate summariser is extractive, never generative. |
| 4 · System 1 | `app/engine/system1.py` | Clauses, parties, defined terms, obligations with deontic polarity, dates, and normalised attributes — every fact carries the character span it came from. |
| 5 · Discrepancies | `app/engine/discrepancy.py` | Ambiguity, omission and direct contradiction, in-text and cross-document, plus impossible-chronology and mathematical categories. |
| 6 · Equations | `app/engine/mathx/` | Extraction (LaTeX + plain), Strict Content MathML / OpenMath / OMDoc encoding, canonicalisation by equality saturation, adjudication by SymPy / Wolfram / Lean 4. |
| 6b · Statutory | `app/engine/taxonomy.py` | The semantic law matrix applied to a document: which bodies of digital consumer law its own language engages, which provisions they expect that are absent, and where the selected forums diverge about them. |
| 7 · Provenance | `app/engine/ledger.py` | Append-only hash chain; `GET /api/v1/ledger/verify` recomputes it and names the first broken link. |

### The e-graph

`app/engine/mathx/egraph.py` is a real e-graph: union-find over e-classes, hash-consed e-nodes,
congruence closure on rebuild, e-matching with pattern variables, guarded rewrite rules (exponent
arithmetic fires only on literal exponents, which is what keeps saturation terminating), constant
folding, and smallest-AST extraction with commutative arguments ordered so congruent expressions
serialise identically.

Retrieval against the archive is two-stage: the canonical key buckets candidates cheaply, then
equality saturation decides congruence. So `R = 0.07·N + 0.03·N` and `R = 0.1·N` are matched, and a
claim restated as `a² + 2ab + b²` is matched to `(a + b)²`.

### The verification tiers

`verify()` runs every available backend and reconciles them:

* **Lean 4** — the obligation is generated always and compiled when a toolchain is on `PATH`. A claim
  is `certified` only if it compiles with zero `sorry`.
* **Wolfram** — over `WOLFRAM_MCP_URL` or the public API when `WOLFRAM_APP_ID` is set.
* **SymPy** — always available. Decides `identity`, `refuted` (with an explicit counterexample),
  `conditional`, or `definitional`.

An unavailable prover is reported as unavailable. The engine never upgrades a claim to verified
because a prover was missing, and it reports backend disagreement rather than silently picking one.

### The semantic law matrix

`law-matrix/` holds a computable representation of digital consumer law: 32 classifications scored
against 16 normative primitives, and 12 jurisdictions scored against the classifications. Wolfram
Language computes the cosine geometry, clustering, SVD embedding, primitive statistics and
jurisdictional emphasis; `build_matrix.py` re-derives every shared quantity and refuses to publish if
the two disagree. The engine consumes the result, so an analysis run with `jurisdictions` set reports
missing provisions cited to instruments and cross-border divergence on the classifications the
document actually engages. See [`law-matrix/README.md`](law-matrix/README.md).

## API

`/docs` carries the full OpenAPI schema. The endpoints that matter:

```
POST /api/v1/auth/signup | login | api-keys
POST /api/v1/matters                        create a matter
POST /api/v1/matters/{id}/documents         upload (multipart) — forensics + triage run here
POST /api/v1/matters/{id}/analyse           full analysis: findings, chronology, graph, equations
GET  /api/v1/documents/{id}                 forensic + triage dossier with its ledger entries
POST /api/v1/math/analyse                   canonicalise, encode and adjudicate one claim
POST /api/v1/math/compare                   decide congruence of two claims
POST /api/v1/math/extract                   pull equations out of prose
GET  /api/v1/math/archive/{id}/omdoc        OMDoc export
GET  /api/v1/ledger/verify                  recompute the hash chain
POST /api/v1/eval/alignment                 score predicted spans against ground truth
GET  /api/v1/taxonomy                       the full semantic law matrix
POST /api/v1/taxonomy/assess                read a document against the matrix
GET  /api/v1/taxonomy/compare?a=EU&b=US_FED emphasis distance between two forums
```

Authenticate with `Authorization: Bearer <jwt>` or `X-API-Key: lxg_…`. API keys are stored as
digests and may not mint further keys.

## Multi-tenancy and plans

Organisations are isolated tenants; every query is scoped by `org_id` and cross-tenant reads return
404. Plan limits (`app/saas/plans.py`) are enforced at the API boundary only — a quota can stop an
analysis from running, but it can never change an analytical result. Billing runs self-serve until
`STRIPE_SECRET_KEY` is set, at which point checkout sessions and signature-verified webhooks take over.

## Tests

```bash
cd backend && python -m pytest -q      # 48 tests
```

They cover the security posture (hostile documents quarantined, clean documents promoted no further
than L2), extraction spans addressing real bytes, each discrepancy category, e-graph congruence and
non-congruence, counterexample refutation, well-formed semantic markup, alignment scoring, ledger
tamper detection, tenant isolation, quota enforcement, the integrity of the computed law matrix and
its findings, and a full pass over the demonstration corpus.

## Limits worth stating

* Extraction is deterministic and pattern-based. It is precise about what it finds and silent about
  phrasings it does not cover; it is a floor, not a ceiling, and an LLM enrichment layer would sit
  above it without replacing the span guarantees.
* Failure to derive congruence is not a proof of inequivalence — the saturation report says whether
  a fixpoint was reached or the budget ran out, and the UI repeats that caveat.
* Findings are analytical output for review by qualified counsel. This is not legal advice.
