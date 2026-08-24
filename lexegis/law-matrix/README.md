# Semantic Law Matrix — digital consumer law

A computable representation of the field. Thirty-two classifications of digital consumer law are
scored against sixteen normative primitives, and twelve jurisdictions are scored against the
classifications. With the field as a matrix, questions that were matters of opinion become
computations: which regimes argue alike, what a jurisdiction systematically over-weights, and where
a single set of terms cannot be lawful everywhere.

```
smat (32 × 16)   what a classification normatively demands
cvg  (12 × 32)   how strongly a jurisdiction occupies it
cvg · sn         normative exposure → ÷ coverage → z-score = emphasis
```

| File | What it is |
|---|---|
| `SemanticLawMatrix.wl` | Source of truth: the data and the Wolfram Language analysis. |
| `wolfram-kernel-output.json` | What the Wolfram 15.0.1 kernel actually returned. |
| `build_matrix.py` | Re-derives every shared quantity in pure Python, cross-checks it against the kernel, and writes the artefact. |
| `semantic-law-matrix.json` | The published matrix, consumed by the engine and the atlas. |
| `atlas.html` | Self-contained interactive atlas (matrix, similarity map, forum emphasis, divergence ranking). |

## Reproduce

```bash
python3 build_matrix.py          # cross-check against the kernel and rewrite the JSON
python3 build_matrix.py --check  # cross-check only
```

`wolframscript` is not installed in this environment: the analysis was executed in a Wolfram 15.0.1
kernel with the data supplied inline, and the kernel's output stored verbatim. With a local
toolchain, `wolframscript -file SemanticLawMatrix.wl` regenerates it directly. Either way
`build_matrix.py` is the gate — it fails loudly if the two representations disagree by more than
0.003 on any shared quantity, so a transcription slip cannot reach the product.

## What the computation produces

* **Similarity and neighbours** — cosine geometry over the primitive vectors. Withdrawal rights and
  distance selling come out at 0.975; conformity of digital content and goods with digital elements
  at 0.967. Those are the pairs a drafter should read together.
* **Clusters** — agglomerative clustering over cosine distance, six groups.
* **Map** — SVD embedding; the first two singular vectors hold 50.7% of the variance, so read
  proximity rather than coordinates.
* **Primitive statistics** — reach and normalised entropy per primitive. Transparency and
  enforcement are structural (reach 79, entropy 0.99); interoperability is a specialism (reach 20,
  entropy 0.69).
* **Emphasis** — what a forum leans on per unit of coverage, z-scored across the twelve. The US
  federal regime over-weights prohibition, enforcement and vulnerable users while under-weighting
  exit rights; the EU over-weights interoperability, ex-ante structure and security; California
  over-weights consent, transparency and minimisation.
* **Divergence risk** — coverage spread × density. Gatekeeper obligations (11.6), content-moderation
  rights (9.1) and dark patterns (8.6) top the list; misleading and aggressive practices (2.4) is
  the nearest thing to a universal rule.

### A correction worth recording

The first jurisdiction statistic was cosine similarity over raw exposure sums. It returned 0.997 to
1.000 for every pair — every forum sums the same thirty-two rows, so they all point the same way.
The metric measured nothing. Emphasis (exposure per unit of coverage, standardised per primitive) is
what discriminates, and the widest pair under it is UK ↔ US federal at 1.714.

## How the engine uses it

`backend/app/engine/taxonomy.py` loads the matrix and applies it to documents:

```
POST /api/v1/taxonomy/assess     {text, jurisdictions[]}
GET  /api/v1/taxonomy/compare?a=EU&b=US_FED
GET  /api/v1/taxonomy/classes/{id}
POST /api/v1/matters/{id}/analyse   {jurisdictions: ["EU","US_FED"]}
```

A document's own language decides which classifications it engages (cue terms, with spans). From
there the engine reports two things a general discrepancy sweep cannot: **missing provisions** — a
body of law is engaged and the provision it expects is absent, cited to the instruments behind the
row — and **cross-border divergence** — the selected forums do not agree about a classification the
document actually engages, so one set of terms cannot satisfy them all.

## Provenance and limits

Scores are an analyst scaffold current to 2026-08, prepared from the instruments cited on each row.
They are a research instrument, not a source of law, and no score is a substitute for reading the
instrument. Every row carries a `verify` note naming what to re-check — a phase-in date, a
designation, a national transposition — because those details decide cases and they move faster than
any matrix. Coverage scores are judgements of regulatory intensity, not counts of anything, and the
jurisdiction set is a sample rather than the world.
