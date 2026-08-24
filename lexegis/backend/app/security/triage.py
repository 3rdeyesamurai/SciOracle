"""Default-deny context triage: strict command/data separation.

Tiers
  L3  Data      -- where every ingested byte lands. Never reaches a reasoning
                   context as free text; only structured, typed extractions
                   derived from it may travel upward.
  L2  Reviewed  -- passed forensics + injection scan; a neutralised, gated
                   summary exists. Usable as *quoted evidence*, still not
                   directive.
  L1  Command   -- reserved for operator-authored policy. Ingested evidence is
                   never promoted here; the tier exists so the boundary is
                   explicit and auditable rather than implied.

Promotion is a decision, is logged, and can be refused. Nothing is promoted by
default -- absence of a verdict is treated as a denial.
"""
import re
from dataclasses import dataclass, field

NEUTRALISERS = [
    (re.compile(r"<\s*/?\s*(system|assistant|tool_result|function_results|instructions?)\s*>", re.I), "[markup-neutralised]"),
    (re.compile(r"[​‌‍⁠﻿]"), ""),
]


@dataclass
class TriageDecision:
    tier: str
    promoted: bool
    quarantined: bool
    reasons: list[str] = field(default_factory=list)
    gated_summary: str = ""
    evidence_quotes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "tier": self.tier,
            "promoted": self.promoted,
            "quarantined": self.quarantined,
            "reasons": self.reasons,
            "gated_summary": self.gated_summary,
            "evidence_quotes": self.evidence_quotes,
        }


def neutralise(text: str) -> str:
    """Strip framing markup and invisible channels from quoted evidence.

    This is defence in depth, not the primary control: the primary control is
    that L3 text is never concatenated into a directive position at all.
    """
    out = text
    for pattern, replacement in NEUTRALISERS:
        out = pattern.sub(replacement, out)
    return out


def _summarise(text: str, max_sentences: int = 6) -> str:
    """Deterministic extractive gate.

    The summariser is intentionally non-generative: it selects sentences by
    evidentiary keyword density. A generative summariser reading hostile text is
    itself an injection surface, which is precisely what this tier exists to stop.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.;])\s+", neutralise(text)) if 40 <= len(s.strip()) <= 400]
    if not sentences:
        return neutralise(text)[:600]
    keywords = ("shall", "agree", "party", "parties", "obligation", "termination", "governing law",
                "liability", "warrant", "represent", "notice", "arbitration", "claim", "patent",
                "confidential", "indemnif", "effective date", "jurisdiction")
    scored = [(sum(k in s.lower() for k in keywords), -i, s) for i, s in enumerate(sentences)]
    scored.sort(reverse=True)
    chosen = sorted(scored[:max_sentences], key=lambda t: -t[1])
    return " ".join(s for _, _, s in chosen)


def triage(text: str, forensics: dict, injection: dict, policy: dict | None = None) -> TriageDecision:
    policy = policy or {}
    quarantine_on = policy.get("quarantine_on", ("hostile",))
    block_forensic = policy.get("block_forensic_severity", ("critical",))

    decision = TriageDecision(tier="L3", promoted=False, quarantined=False)

    if injection.get("verdict") in quarantine_on:
        decision.quarantined = True
        decision.reasons.append(
            f"injection scan returned {injection['verdict']} (score {injection.get('score')}, "
            f"signals: {', '.join(injection.get('distinct_signals', [])) or 'n/a'})"
        )
    if forensics.get("max_severity") in block_forensic:
        decision.quarantined = True
        decision.reasons.append(f"forensic severity {forensics['max_severity']}: " + "; ".join(
            f["code"] for f in forensics.get("findings", []) if f["severity"] in block_forensic))

    if decision.quarantined:
        decision.reasons.append("held at L3; structured extraction permitted, no promotion to reasoning context")
        return decision

    if injection.get("verdict") == "suspect":
        decision.reasons.append("promoted with suspicion flag: quoted evidence neutralised before use")
    decision.tier = "L2"
    decision.promoted = True
    decision.gated_summary = _summarise(text)
    decision.evidence_quotes = [neutralise(s)[:280] for s in re.split(r"(?<=[.;])\s+", text) if len(s.strip()) > 60][:12]
    decision.reasons.append("forensics and injection scan clean; promoted L3 -> L2 (evidence tier)")
    decision.reasons.append("L1 (command tier) remains operator-only; no ingested content is ever promoted to L1")
    return decision
