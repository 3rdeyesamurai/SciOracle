#!/usr/bin/env python3
"""Assemble semantic-law-matrix.json from the Wolfram source of truth.

The Wolfram kernel is the computational authority: `SemanticLawMatrix.wl` holds
the data and the analysis, and `wolfram-kernel-output.json` holds what the
kernel actually returned. This script re-derives every quantity it can in pure
Python and asserts agreement with the kernel before writing the artefact, so a
transcription slip between the two representations fails loudly instead of
propagating into the product.

Two quantities are kernel-only because they have no cheap pure-Python
equivalent: the agglomerative cluster assignment and the SVD embedding.

    python3 build_matrix.py            # verify and write
    python3 build_matrix.py --check    # verify only
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "SemanticLawMatrix.wl"
KERNEL = HERE / "wolfram-kernel-output.json"
TARGET = HERE / "semantic-law-matrix.json"
TOLERANCE = 0.003

PRIMITIVE_IDS = ["consent", "transparency", "fairness", "exit", "redress", "liability",
                 "enforcement", "extraterritorial", "vulnerability", "minimisation",
                 "interoperability", "auditability", "timeliness", "security",
                 "prohibition", "exante"]
JURISDICTION_IDS = ["EU", "UK", "US_FED", "US_CA", "CAN", "BR", "IN", "CN", "JP", "KR", "AU", "SG"]

CLASS_RE = re.compile(
    r'classRow\["(?P<id>[a-z_]+)",\s*"(?P<label>[^"]+)",\s*"(?P<family>[^"]+)",\s*'
    r'\{(?P<weights>[0-9,\s]+)\},\s*\{(?P<coverage>[0-9,\s]+)\},\s*'
    r'\{(?P<instruments>[^}]*)\},\s*"(?P<verify>[^"]+)"\]', re.S)
PRIMITIVE_RE = re.compile(
    r'<\|"id" -> "(?P<id>[a-z]+)",\s*"label" -> "(?P<label>[^"]+)",\s*"gloss" -> "(?P<gloss>[^"]+)"\|>')
JURISDICTION_RE = re.compile(
    r'<\|"id" -> "(?P<id>[A-Z_]+)",\s*"label" -> "(?P<label>[^"]+)",\s*"family" -> "(?P<family>[^"]+)"\|>')


# ------------------------------------------------------------------ parsing
def parse_source(text: str) -> dict:
    primitives = [m.groupdict() for m in PRIMITIVE_RE.finditer(text)]
    jurisdictions = [m.groupdict() for m in JURISDICTION_RE.finditer(text)]
    classes = []
    for m in CLASS_RE.finditer(text):
        row = m.groupdict()
        weights = [int(x) for x in row["weights"].replace("\n", "").split(",")]
        coverage = [int(x) for x in row["coverage"].replace("\n", "").split(",")]
        instruments = re.findall(r'"([^"]+)"', row["instruments"])
        if len(weights) != len(PRIMITIVE_IDS) or len(coverage) != len(JURISDICTION_IDS):
            raise SystemExit(f"{row['id']}: expected {len(PRIMITIVE_IDS)} weights and "
                             f"{len(JURISDICTION_IDS)} coverage entries, got "
                             f"{len(weights)} and {len(coverage)}")
        classes.append({
            "id": row["id"], "label": row["label"], "family": row["family"],
            "weights": dict(zip(PRIMITIVE_IDS, weights)),
            "coverage": dict(zip(JURISDICTION_IDS, coverage)),
            "instruments": instruments, "verify": row["verify"].replace("\\\n", " "),
        })
    if len(primitives) != len(PRIMITIVE_IDS):
        raise SystemExit(f"parsed {len(primitives)} primitives, expected {len(PRIMITIVE_IDS)}")
    if len(jurisdictions) != len(JURISDICTION_IDS):
        raise SystemExit(f"parsed {len(jurisdictions)} jurisdictions, expected {len(JURISDICTION_IDS)}")
    return {"primitives": primitives, "jurisdictions": jurisdictions, "classes": classes}


# -------------------------------------------------------------- arithmetic
def normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else [float(x) for x in vector]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(column) for column in zip(*matrix)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    """Sample standard deviation, matching Wolfram's StandardDeviation."""
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def correlation(a: list[float], b: list[float]) -> float:
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else 0.0


def compute(data: dict) -> dict:
    classes = data["classes"]
    S = [[c["weights"][p] for p in PRIMITIVE_IDS] for c in classes]
    C = [[c["coverage"][j] for c in classes] for j in JURISDICTION_IDS]
    Sn = [normalise(row) for row in S]

    similarity = [[round(dot(Sn[i], Sn[j]), 3) for j in range(len(S))] for i in range(len(S))]
    density = [sum(row) for row in S]
    reach = [sum(column) for column in transpose(S)]

    # Shannon entropy of each primitive's distribution across classifications,
    # normalised to [0,1]: 1 means the primitive is spread evenly over the whole
    # field (structural), low means it is concentrated in a few specialisms.
    entropy = []
    for column in transpose(S):
        total = sum(column)
        shares = [v / total for v in column if v > 0]
        entropy.append(round(-sum(p * math.log(p) for p in shares) / math.log(len(column)), 3))

    primitive_correlation = [[round(correlation(a, b), 3) for b in transpose(S)] for a in transpose(S)]

    # Jurisdictional emphasis: the normative profile per unit of coverage, so a
    # large regime and a small one are compared by what they emphasise rather
    # than by how much law they have. Raw exposure sums cannot do this — every
    # jurisdiction ends up pointing the same direction.
    exposure_raw = [[sum(C[j][i] * Sn[i][k] for i in range(len(S)))
                     for k in range(len(PRIMITIVE_IDS))] for j in range(len(JURISDICTION_IDS))]
    exposure = [[round(v, 3) for v in row] for row in exposure_raw]
    coverage_score = [sum(row) for row in C]
    # emphasis is derived from the unrounded exposure: rounding first shifts the
    # z-scores by enough to break agreement with the kernel at three decimals
    emphasis = [[exposure_raw[j][k] / coverage_score[j] for k in range(len(PRIMITIVE_IDS))]
                for j in range(len(JURISDICTION_IDS))]
    emphasis_z = []
    columns = transpose(emphasis)
    z_columns = []
    for column in columns:
        m, s = mean(column), stdev(column)
        z_columns.append([(v - m) / s for v in column])
    emphasis_z = [[round(v, 3) for v in row] for row in transpose(z_columns)]

    jurisdiction_distance = [[round(1 - dot(normalise(emphasis_z[a]), normalise(emphasis_z[b])), 3)
                              for b in range(len(JURISDICTION_IDS))]
                             for a in range(len(JURISDICTION_IDS))]

    signature = {}
    for j, jid in enumerate(JURISDICTION_IDS):
        ranked = sorted(range(len(PRIMITIVE_IDS)), key=lambda k: -emphasis_z[j][k])
        signature[jid] = {
            "over": {PRIMITIVE_IDS[k]: round(emphasis_z[j][k], 2) for k in ranked[:3]},
            "under": {PRIMITIVE_IDS[k]: round(emphasis_z[j][k], 2) for k in ranked[-3:][::-1]},
        }

    gaps = {}
    for j, jid in enumerate(JURISDICTION_IDS):
        weak = [{"class": classes[i]["id"], "label": classes[i]["label"],
                 "coverage": C[j][i], "density": density[i]}
                for i in range(len(classes)) if C[j][i] <= 1]
        gaps[jid] = sorted(weak, key=lambda g: -g["density"])[:6]

    divergence = sorted(
        [{"class": classes[i]["id"], "label": classes[i]["label"],
          "mean": round(mean([C[j][i] for j in range(len(JURISDICTION_IDS))]), 2),
          "sd": round(stdev([C[j][i] for j in range(len(JURISDICTION_IDS))]), 2),
          "density": density[i],
          "risk": round(stdev([C[j][i] for j in range(len(JURISDICTION_IDS))]) * density[i] / 3, 2)}
         for i in range(len(classes))], key=lambda d: -d["risk"])

    neighbours = {}
    for i, cls in enumerate(classes):
        ranked = sorted((j for j in range(len(classes)) if j != i), key=lambda j: -similarity[i][j])
        neighbours[cls["id"]] = [{"class": classes[j]["id"], "label": classes[j]["label"],
                                  "similarity": similarity[i][j]} for j in ranked[:4]]

    return {
        "S": S, "C": C, "similarity": similarity, "density": density, "reach": reach,
        "entropy": entropy, "primitiveCorrelation": primitive_correlation,
        "exposure": exposure, "coverageScore": coverage_score,
        "emphasis": [[round(v, 4) for v in row] for row in emphasis],
        "emphasisZ": emphasis_z, "jurisdictionDistance": jurisdiction_distance,
        "signature": signature, "gaps": gaps, "divergence": divergence, "neighbours": neighbours,
    }


# ------------------------------------------------------------- validation
def close(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE


def validate(computed: dict, kernel: dict, classes: list[dict]) -> list[str]:
    failures: list[str] = []

    def check(label: str, ours, theirs):
        if isinstance(theirs, list):
            for i, (x, y) in enumerate(zip(ours, theirs)):
                if isinstance(y, list):
                    for k, (xx, yy) in enumerate(zip(x, y)):
                        if not close(xx, yy):
                            failures.append(f"{label}[{i}][{k}]: python {xx} vs kernel {yy}")
                elif not close(x, y):
                    failures.append(f"{label}[{i}]: python {x} vs kernel {y}")
        elif not close(ours, theirs):
            failures.append(f"{label}: python {ours} vs kernel {theirs}")

    check("density", computed["density"], kernel["density"])
    check("reach", computed["reach"], kernel["reach"])
    check("entropy", computed["entropy"], kernel["entropy"])
    check("primitiveCorrelation", computed["primitiveCorrelation"], kernel["primitiveCorrelation"])
    check("emphasisZ", computed["emphasisZ"], kernel["emphasisZ"])
    check("jurisdictionDistance", computed["jurisdictionDistance"], kernel["jurisdictionDistance"])
    check("coverageScore", computed["coverageScore"],
          [kernel["coverageScore"][j] for j in JURISDICTION_IDS])

    for index, jid in enumerate(JURISDICTION_IDS):
        ours = {g["class"] for g in computed["gaps"][jid]}
        theirs = {classes[i - 1]["id"] for i, _, _ in kernel["gaps"][jid]}
        if ours != theirs:
            failures.append(f"gaps[{jid}]: python {sorted(ours)} vs kernel {sorted(theirs)}")

    kernel_risk = {d["class"]: d["risk"] for d in kernel["divergence"]}
    for item in computed["divergence"]:
        if not close(item["risk"], kernel_risk[item["class"]]):
            failures.append(f"divergence[{item['class']}]: python {item['risk']} "
                            f"vs kernel {kernel_risk[item['class']]}")

    for i, j, value in kernel["similaritySpotChecks"]:
        ours = computed["similarity"][i - 1][j - 1]
        if not close(ours, value):
            failures.append(f"similarity[{i}][{j}]: python {ours} vs kernel {value}")

    for jid, sig in kernel["signature"].items():
        for bucket in ("over", "under"):
            if list(computed["signature"][jid][bucket]) != list(sig[bucket]):
                failures.append(f"signature[{jid}][{bucket}]: python "
                                f"{list(computed['signature'][jid][bucket])} vs kernel {list(sig[bucket])}")
    return failures


# ------------------------------------------------------------------ output
def build() -> dict:
    data = parse_source(SOURCE.read_text())
    kernel = json.loads(KERNEL.read_text())
    computed = compute(data)

    failures = validate(computed, kernel, data["classes"])
    if failures:
        print(f"cross-check FAILED against the Wolfram kernel ({len(failures)} mismatches):")
        for line in failures[:20]:
            print("  " + line)
        raise SystemExit(1)
    print(f"cross-check passed: python re-derivation agrees with {kernel['engine']} "
          f"to {TOLERANCE} on every shared quantity")

    classes = []
    for i, cls in enumerate(data["classes"]):
        classes.append({
            **cls,
            "density": computed["density"][i],
            "cluster": kernel["cluster"][i],
            "embedding": kernel["embedding"][i],
            "neighbours": computed["neighbours"][cls["id"]],
        })

    return {
        "title": "Semantic Law Matrix — digital consumer law",
        "generated": kernel["computed"],
        "engine": kernel["engine"],
        "method": "Classification vectors over normative primitives; cosine geometry, agglomerative "
                  "clustering and SVD embedding computed in Wolfram Language, re-derived in Python "
                  "and cross-checked before publication.",
        "disclaimer": "Analyst scaffold current to 2026-08. A research instrument, not legal advice; "
                      "re-check each row against primary sources before relying on it.",
        "scales": {"weights": {"0": "absent", "1": "peripheral", "2": "substantial", "3": "defining"},
                   "coverage": {"0": "none", "1": "partial or sectoral", "2": "substantial",
                                "3": "comprehensive"}},
        "primitiveIds": PRIMITIVE_IDS,
        "jurisdictionIds": JURISDICTION_IDS,
        "primitives": [{**p, "reach": computed["reach"][k], "entropy": computed["entropy"][k]}
                       for k, p in enumerate(data["primitives"])],
        "jurisdictions": [{**j, "coverageScore": computed["coverageScore"][k],
                           "signature": computed["signature"][j["id"]],
                           "gaps": computed["gaps"][j["id"]]}
                          for k, j in enumerate(data["jurisdictions"])],
        "classes": classes,
        "matrices": {"S": computed["S"], "C": computed["C"], "similarity": computed["similarity"],
                     "primitiveCorrelation": computed["primitiveCorrelation"],
                     "exposure": computed["exposure"], "emphasisZ": computed["emphasisZ"],
                     "jurisdictionDistance": computed["jurisdictionDistance"]},
        "divergence": computed["divergence"],
        "widestPairs": kernel["widestPairs"],
        "embeddingVariance": kernel["embeddingVariance"],
        "singularValues": kernel["singularValues"],
    }


if __name__ == "__main__":
    report = build()
    if "--check" in sys.argv:
        print("check only; nothing written")
    else:
        TARGET.write_text(json.dumps(report, indent=1) + "\n")
        print(f"wrote {TARGET.relative_to(HERE.parent.parent)} "
              f"({TARGET.stat().st_size // 1024} KB, {len(report['classes'])} classifications, "
              f"{len(report['jurisdictions'])} jurisdictions)")
