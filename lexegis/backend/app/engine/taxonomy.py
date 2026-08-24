"""The semantic law matrix, as an engine resource.

Loads the Wolfram-computed matrix (lexegis/law-matrix/semantic-law-matrix.json)
and puts it to work on documents:

  * classify   -- which bodies of digital consumer law a document implicates,
                  from cue terms found in its own text, with spans;
  * checklist  -- which provisions those classifications expect to see, and
                  which are missing;
  * divergence -- where the selected jurisdictions disagree most about the
                  classifications this document actually engages, which is the
                  cross-border exposure a single set of terms cannot satisfy.

The matrix supplies the geometry; this module supplies the reading of text. The
cue lexicon is deliberately narrow: a cue fires on language that appears in the
instrument itself, not on topic guesses.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

MATRIX_PATH = Path(__file__).resolve().parents[3] / "law-matrix" / "semantic-law-matrix.json"

# classification -> cue terms (does the document engage this body of law?) and
# expected provisions (what that body of law looks for in the text).
LEXICON: dict[str, dict] = {
    "data_protection": {
        "cues": [r"personal data", r"data subject", r"privacy policy", r"processing of .{0,20}data",
                 r"\bGDPR\b", r"data controller", r"personal information"],
        "expects": [("lawful basis for processing", r"legitimate interest|consent|lawful basis|contractual necessity"),
                    ("data subject rights", r"right to (access|erasure|rectification|object)|data subject rights"),
                    ("retention period", r"retention period|retained for|we (will )?(keep|retain)")],
    },
    "eprivacy_tracking": {
        "cues": [r"cookies?\b", r"tracking technolog", r"pixel", r"local storage"],
        "expects": [("prior consent for non-essential cookies", r"consent|opt[- ]?in"),
                    ("means of withdrawal", r"withdraw|reject all|manage (cookie|preferences)")],
    },
    "profiling_adtech": {
        "cues": [r"profiling", r"behavioural advertis|behavioral advertis", r"targeted ads?",
                 r"interest[- ]based advertising"],
        "expects": [("right to object to profiling", r"object to (profiling|direct marketing)|opt[- ]?out")],
    },
    "portability_interop": {
        "cues": [r"data portability", r"export your data", r"interoperab"],
        "expects": [("machine-readable export", r"machine[- ]readable|structured, commonly used")],
    },
    "automated_decisions": {
        "cues": [r"automated decision", r"algorithmic decision", r"artificial intelligence",
                 r"machine learning", r"\bAI\b system"],
        "expects": [("human review of automated decisions", r"human (review|intervention|oversight)"),
                    ("logic disclosure", r"logic involved|how (the|our) (system|model) (works|decides)")],
    },
    "unfair_terms": {
        "cues": [r"limitation of liability", r"exclusion of liability", r"unilaterally (vary|amend|change)",
                 r"we may (change|modify) these terms"],
        "expects": [("notice before unilateral variation", r"notice|notify you"),
                    ("right to terminate on variation", r"terminate|cancel")],
    },
    "distance_selling": {
        "cues": [r"place an order", r"online purchase", r"checkout", r"distance contract"],
        "expects": [("trader identity and contact details", r"registered (office|address)|contact (us|details)|company (number|registration)"),
                    ("total price before order", r"total price|inclusive of (all )?(taxes|VAT)"),
                    ("order confirmation", r"confirmation (email|of your order)")],
    },
    "withdrawal_rights": {
        "cues": [r"right of withdrawal", r"cooling[- ]off", r"cancel(lation)? (period|right)", r"return the goods"],
        "expects": [("withdrawal period stated", r"\b(7|14|15|30) (calendar |business )?days"),
                    ("model withdrawal instructions", r"withdrawal form|how to (cancel|withdraw)"),
                    ("consequences of withdrawal", r"refund")],
    },
    "subscriptions_autorenewal": {
        "cues": [r"auto[- ]?renew", r"subscription (will|shall) (be )?renew", r"recurring (payment|charge|billing)",
                 r"free trial"],
        "expects": [("renewal terms before purchase", r"renew(s|al)? (automatically|every|annually|monthly)"),
                    ("cancellation mechanism", r"cancel (at any time|your subscription|online)"),
                    ("pre-renewal reminder", r"remind|notice before (renewal|each renewal)")],
    },
    "pricing_transparency": {
        "cues": [r"\bfees?\b", r"service charge", r"additional charges", r"price"],
        "expects": [("all mandatory fees in the headline price", r"inclusive of|total (price|amount)|no hidden"),
                    ("taxes and delivery stated", r"tax|VAT|delivery (charge|cost)|shipping")],
    },
    "personalised_pricing": {
        "cues": [r"personalised price|personalized price", r"dynamic pricing", r"price .{0,20}based on your"],
        "expects": [("disclosure that the price is personalised", r"personalis|personaliz")],
    },
    "digital_content_conformity": {
        "cues": [r"digital content", r"digital service", r"software licence|software license", r"the app\b"],
        "expects": [("conformity and functionality description", r"functionalit|compatib|interoperab"),
                    ("remedy for non-conformity", r"repair|replace|price reduction|refund")],
    },
    "goods_digital_elements": {
        "cues": [r"goods with digital elements", r"connected (device|product)", r"smart (device|home)"],
        "expects": [("update duty for the expected lifetime", r"updates? (will|shall) be provided|support period")],
    },
    "updates_repair": {
        "cues": [r"security updates?", r"firmware", r"end of support", r"spare parts", r"right to repair"],
        "expects": [("support period stated", r"support (period|until)|for (at least )?\d+ (years?|months?)")],
    },
    "product_liability_digital": {
        "cues": [r"defect", r"product liability", r"damage caused by"],
        "expects": [("no exclusion of liability for death or personal injury",
                     r"nothing in (these|this) .{0,40}(excludes?|limits?) .{0,60}(death|personal injury)")],
    },
    "iot_security": {
        "cues": [r"default password", r"vulnerability disclosure", r"connected device security", r"IoT"],
        "expects": [("vulnerability reporting contact", r"report a vulnerability|security\.txt|security contact")],
    },
    "platform_transparency": {
        "cues": [r"marketplace", r"third[- ]party sellers?", r"ranking", r"search results"],
        "expects": [("main ranking parameters", r"ranking (parameters|criteria)|how (we|results are) (rank|ordered)"),
                    ("trader or non-trader status of sellers", r"trader|business seller|private seller")],
    },
    "intermediary_liability": {
        "cues": [r"user[- ]generated content", r"we host", r"notice and takedown", r"report (illegal )?content"],
        "expects": [("notice mechanism", r"report|notify us|flag"),
                    ("statement of reasons on removal", r"reasons?|explain")],
    },
    "gatekeeper_obligations": {
        "cues": [r"gatekeeper", r"core platform service", r"designated undertaking"],
        "expects": [("no self-preferencing", r"self[- ]preferenc|equal (treatment|terms)")],
    },
    "content_moderation_rights": {
        "cues": [r"suspend (your )?account", r"remove (your )?content", r"community (guidelines|standards)",
                 r"moderation"],
        "expects": [("statement of reasons", r"reasons?|why (we|your)"),
                    ("internal appeal route", r"appeal|challenge (the|our) decision|dispute")],
    },
    "online_safety_illegal": {
        "cues": [r"illegal content", r"harmful content", r"child sexual", r"terrorist content"],
        "expects": [("reporting route for illegal content", r"report|notify")],
    },
    "unfair_commercial_practices": {
        "cues": [r"limited time offer", r"only \d+ left", r"guarantee", r"risk[- ]free"],
        "expects": [("substantiation of claims", r"terms (and conditions )?apply|subject to|evidence")],
    },
    "dark_patterns": {
        "cues": [r"pre[- ]?(ticked|checked)", r"by continuing you agree", r"countdown", r"only .{0,12}left in stock"],
        "expects": [("symmetric accept and reject", r"reject all|decline|no thanks")],
    },
    "fake_reviews_influencers": {
        "cues": [r"reviews?", r"testimonial", r"star rating", r"sponsored (post|content)"],
        "expects": [("how reviews are verified", r"verified (purchase|review)|we (check|verify)")],
    },
    "direct_marketing_spam": {
        "cues": [r"marketing (emails?|communications?)", r"newsletter", r"promotional (emails?|messages?)",
                 r"text messages?"],
        "expects": [("consent to marketing", r"consent|opt[- ]?in|subscribe"),
                    ("unsubscribe mechanism", r"unsubscribe|opt[- ]?out|stop receiving")],
    },
    "minors_protection": {
        "cues": [r"under (13|16|18)", r"children", r"minors?", r"age (verification|assurance)", r"parental consent"],
        "expects": [("age threshold stated", r"under (13|14|15|16|17|18)|at least \d+ years"),
                    ("parental consent route", r"parent|guardian")],
    },
    "accessibility": {
        "cues": [r"accessibility", r"WCAG", r"screen reader"],
        "expects": [("accessibility statement", r"accessibility statement|conformance")],
    },
    "payments_chargebacks": {
        "cues": [r"payment (method|card)", r"chargeback", r"unauthorised transaction|unauthorized transaction",
                 r"strong customer authentication"],
        "expects": [("unauthorised transaction procedure", r"unauthoris|unauthoriz|dispute (a )?(charge|transaction)"),
                    ("refund timing", r"within \d+ (business |working )?days")],
    },
    "crypto_consumer": {
        "cues": [r"crypto[- ]?asset", r"digital token", r"stablecoin", r"wallet", r"virtual currency"],
        "expects": [("risk warning", r"risk warning|you may lose|value can (go down|fluctuate)"),
                    ("custody arrangements", r"custody|segregat|held on your behalf")],
    },
    "credit_bnpl": {
        "cues": [r"buy now,? pay later", r"instal?ments?", r"credit agreement", r"\bAPR\b", r"deferred payment"],
        "expects": [("cost of credit disclosure", r"\bAPR\b|total (amount|cost) (payable|of credit)|interest rate"),
                    ("late payment consequences", r"late (fee|payment)|missed payment|default")],
    },
    "redress_odr": {
        "cues": [r"dispute resolution", r"complaints? (procedure|handling)", r"arbitration", r"class action"],
        "expects": [("complaint route", r"complain|contact us"),
                    ("ADR body or court named", r"ADR|ombuds|court|arbitral")],
    },
    "cross_border_enforcement": {
        "cues": [r"governing law", r"jurisdiction", r"choice of law"],
        "expects": [("consumer's home-forum rights preserved",
                     r"mandatory (provisions|rules)|does not (affect|deprive) .{0,50}rights|your local law")],
    },
}


@lru_cache(maxsize=1)
def matrix() -> dict:
    """The full computed matrix. Raises if the artefact is missing."""
    if not MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"semantic law matrix not found at {MATRIX_PATH}; run law-matrix/build_matrix.py")
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _index() -> dict:
    report = matrix()
    return {
        "classes": {c["id"]: c for c in report["classes"]},
        "jurisdictions": {j["id"]: j for j in report["jurisdictions"]},
        "primitives": {p["id"]: p for p in report["primitives"]},
        "divergence": {d["class"]: d for d in report["divergence"]},
        "classOrder": [c["id"] for c in report["classes"]],
        "jurisdictionOrder": report["jurisdictionIds"],
        "primitiveOrder": report["primitiveIds"],
    }


def classification(class_id: str) -> dict | None:
    return _index()["classes"].get(class_id)


def jurisdiction(jurisdiction_id: str) -> dict | None:
    return _index()["jurisdictions"].get(jurisdiction_id)


def classify(text: str, *, threshold: int = 1) -> list[dict]:
    """Which classifications the document engages, with the cues that fired."""
    hits = []
    for class_id, entry in LEXICON.items():
        cues = []
        for pattern in entry["cues"]:
            for match in re.finditer(pattern, text, re.I):
                cues.append({"cue": pattern, "match": match.group(0)[:80], "start": match.start(),
                             "end": match.end()})
                break  # one span per cue is enough to evidence engagement
        if len(cues) >= threshold:
            record = classification(class_id) or {"id": class_id, "label": class_id}
            hits.append({
                "class": class_id, "label": record.get("label", class_id),
                "family": record.get("family"), "density": record.get("density"),
                "cues": cues, "strength": len(cues),
            })
    return sorted(hits, key=lambda h: (-h["strength"], -(h["density"] or 0)))


def missing_provisions(text: str, class_ids: list[str]) -> list[dict]:
    """Provisions a classification expects that the text does not contain."""
    missing = []
    for class_id in class_ids:
        entry = LEXICON.get(class_id)
        if not entry:
            continue
        record = classification(class_id) or {}
        for label, pattern in entry["expects"]:
            if not re.search(pattern, text, re.I):
                missing.append({
                    "class": class_id, "class_label": record.get("label", class_id),
                    "provision": label, "pattern": pattern,
                    "density": record.get("density"),
                    "instruments": record.get("instruments", [])[:3],
                    "verify": record.get("verify"),
                })
    return missing


def divergence_exposure(class_ids: list[str], jurisdiction_ids: list[str]) -> list[dict]:
    """Where the chosen forums disagree about the law this document engages.

    Reported as the spread of coverage across the selected jurisdictions for each
    engaged classification, weighted by how much normative load the class carries.
    A high spread means one set of terms cannot be compliant everywhere.
    """
    index = _index()
    out = []
    for class_id in class_ids:
        record = index["classes"].get(class_id)
        if not record:
            continue
        coverage = {j: record["coverage"][j] for j in jurisdiction_ids if j in record["coverage"]}
        if len(coverage) < 2:
            continue
        spread = max(coverage.values()) - min(coverage.values())
        if spread == 0:
            continue
        strictest = max(coverage, key=lambda j: coverage[j])
        loosest = min(coverage, key=lambda j: coverage[j])
        out.append({
            "class": class_id, "label": record["label"], "coverage": coverage, "spread": spread,
            "density": record["density"],
            "risk": round(spread * record["density"] / 3, 2),
            "strictest": strictest, "loosest": loosest,
            "global_risk_rank": next((i + 1 for i, d in enumerate(matrix()["divergence"])
                                      if d["class"] == class_id), None),
            "instruments": record["instruments"][:4],
            "verify": record["verify"],
        })
    return sorted(out, key=lambda d: -d["risk"])


def compare_jurisdictions(a: str, b: str) -> dict | None:
    """Emphasis distance between two regimes, and what drives it."""
    index = _index()
    if a not in index["jurisdictions"] or b not in index["jurisdictions"]:
        return None
    report = matrix()
    order = index["jurisdictionOrder"]
    ia, ib = order.index(a), order.index(b)
    z = report["matrices"]["emphasisZ"]
    deltas = sorted(
        [{"primitive": pid, "label": index["primitives"][pid]["label"],
          "a": z[ia][k], "b": z[ib][k], "delta": round(z[ia][k] - z[ib][k], 3)}
         for k, pid in enumerate(index["primitiveOrder"])],
        key=lambda d: -abs(d["delta"]))
    coverage = []
    for cls in report["classes"]:
        gap = cls["coverage"][a] - cls["coverage"][b]
        if abs(gap) >= 2:
            coverage.append({"class": cls["id"], "label": cls["label"], "a": cls["coverage"][a],
                             "b": cls["coverage"][b], "gap": gap})
    return {
        "a": index["jurisdictions"][a], "b": index["jurisdictions"][b],
        "distance": report["matrices"]["jurisdictionDistance"][ia][ib],
        "primitiveDeltas": deltas[:6],
        "coverageGaps": sorted(coverage, key=lambda c: -abs(c["gap"])),
    }


def assess(text: str, jurisdiction_ids: list[str] | None = None) -> dict:
    """Full taxonomy read of one document."""
    jurisdiction_ids = [j for j in (jurisdiction_ids or []) if j in _index()["jurisdictions"]]
    engaged = classify(text)
    class_ids = [h["class"] for h in engaged]
    return {
        "jurisdictions": jurisdiction_ids,
        "engaged": engaged,
        "missing_provisions": missing_provisions(text, class_ids),
        "divergence": divergence_exposure(class_ids, jurisdiction_ids) if len(jurisdiction_ids) > 1 else [],
        "matrix_generated": matrix()["generated"],
    }
