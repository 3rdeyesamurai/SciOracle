"""Indirect prompt-injection detection over ingested evidence.

The engine's threat model is that any document may be authored by the opposing
party specifically to be read by an AI. Detection is layered:

  1. Directive lexicon  -- imperative attempts to re-address the model.
  2. Role/frame forgery -- text impersonating system turns or tool results.
  3. Exfiltration       -- instructions to emit data outward.
  4. Carrier analysis   -- encodings and invisible channels used to smuggle 1-3.
  5. Salience proxy     -- a cheap stand-in for attention-head tracking: how much
     of the span's mass is imperative/second-person relative to the document.

The score gates tier promotion (see triage.py); it never rewrites the evidence.
"""
import base64
import math
import re

DIRECTIVE_PATTERNS: list[tuple[str, str, float]] = [
    (r"ignore\s+(all\s+)?(previous|prior|above|preceding)\s+(instructions?|prompts?|rules?)", "instruction_override", 1.0),
    (r"disregard\s+(all\s+)?(previous|prior|the\s+above|your)\s+\w+", "instruction_override", 0.9),
    (r"forget\s+(everything|all\s+previous|your\s+instructions)", "instruction_override", 0.9),
    (r"you\s+are\s+now\s+(a|an|the)\s+", "role_reassignment", 0.8),
    (r"from\s+now\s+on[, ]+(you|act|respond|answer)", "role_reassignment", 0.8),
    (r"(new|updated|revised)\s+(system\s+)?(instructions?|directive|policy)\s*[:\-]", "role_reassignment", 0.85),
    (r"(do\s+not|don't|never)\s+(mention|report|flag|disclose|log)\s+", "suppression", 0.95),
    (r"(mark|classify|treat|report)\s+(this|the)\s+(document|clause|file)?\s*as\s+(authentic|verified|compliant|clean|valid)", "verdict_forcing", 1.0),
    (r"(skip|bypass|omit|suppress)\s+(the\s+)?(verification|validation|analysis|review|check)", "verdict_forcing", 0.95),
    (r"(output|print|send|transmit|forward|post)\s+(the\s+)?(system\s+prompt|your\s+instructions|api[_ ]key|credentials|secrets?)", "exfiltration", 1.0),
    (r"https?://[^\s]{0,120}\?(q|data|payload|x)=", "exfiltration", 0.7),
    (r"<\s*(system|assistant|tool_result|function_results)\s*>", "frame_forgery", 0.9),
    (r"^\s*(system|assistant)\s*:", "frame_forgery", 0.6),
    (r"\[\[?\s*(SYSTEM|ADMIN|OVERRIDE|PRIORITY)\s*\]?\]", "frame_forgery", 0.8),
    (r"(as an ai|as a language model)[^.]{0,60}(you must|you should|comply)", "social_engineering", 0.6),
    (r"this\s+(instruction|message)\s+(takes|has)\s+(precedence|priority)", "instruction_override", 0.9),
    (r"(urgent|immediately|critical)[^.]{0,40}(comply|execute|perform)\s+", "social_engineering", 0.5),
]

IMPERATIVE_VERBS = {
    "ignore", "disregard", "forget", "output", "print", "send", "execute", "run",
    "comply", "respond", "answer", "classify", "mark", "treat", "skip", "bypass",
    "reveal", "disclose", "override", "act", "pretend", "assume",
}
SECOND_PERSON = {"you", "your", "yours", "yourself"}
BASE64_RE = re.compile(r"[A-Za-z0-9+/]{60,}={0,2}")
ZERO_WIDTH = "​‌‍⁠﻿"


def _decode_carriers(text: str) -> list[dict]:
    """Base64 blobs that decode to directive-shaped text are smuggled payloads."""
    carriers = []
    for match in BASE64_RE.finditer(text):
        blob = match.group(0)
        try:
            decoded = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True).decode("utf-8", "ignore")
        except Exception:
            continue
        if not decoded or sum(c.isprintable() for c in decoded) / len(decoded) < 0.8:
            continue
        hits = [name for pattern, name, _ in DIRECTIVE_PATTERNS if re.search(pattern, decoded, re.I)]
        if hits:
            carriers.append({
                "carrier": "base64", "offset": match.start(), "decoded_preview": decoded[:180],
                "decoded_signals": sorted(set(hits)),
            })
    return carriers


def _salience(span: str, document: str) -> float:
    """Cheap proxy for attention-head anomaly tracking.

    Injection payloads are locally dense in imperatives and second-person address
    relative to the surrounding evidentiary prose; that density ratio is the
    signal we score, without needing white-box access to the model.
    """
    def density(text: str) -> float:
        words = re.findall(r"[a-z']+", text.lower())
        if not words:
            return 0.0
        marked = sum(1 for w in words if w in IMPERATIVE_VERBS or w in SECOND_PERSON)
        return marked / len(words)

    doc_density = density(document) or 0.005
    return min(4.0, density(span) / doc_density)


def scan(text: str) -> dict:
    signals: list[dict] = []
    lowered = text or ""
    for pattern, name, weight in DIRECTIVE_PATTERNS:
        for match in re.finditer(pattern, lowered, re.I | re.M):
            start = max(0, match.start() - 120)
            end = min(len(lowered), match.end() + 120)
            window = lowered[start:end]
            salience = _salience(window, lowered)
            signals.append({
                "signal": name,
                "pattern": pattern,
                "match": match.group(0)[:160],
                "offset": match.start(),
                "end": match.end(),
                "context": window.replace("\n", " ").strip(),
                "base_weight": weight,
                "salience_ratio": round(salience, 2),
                "weight": round(weight * (1 + 0.25 * min(salience, 2.0)), 3),
            })

    carriers = _decode_carriers(lowered)
    for carrier in carriers:
        signals.append({
            "signal": "encoded_payload", "pattern": "base64", "match": carrier["decoded_preview"][:160],
            "offset": carrier["offset"], "end": carrier["offset"], "context": carrier["decoded_preview"],
            "base_weight": 1.0, "salience_ratio": 0.0, "weight": 1.2,
        })

    zw = sum(lowered.count(c) for c in ZERO_WIDTH)
    if zw > 8:
        signals.append({
            "signal": "invisible_channel", "pattern": "zero-width", "match": f"{zw} zero-width chars",
            "offset": lowered.find(ZERO_WIDTH[0]), "end": -1, "context": "", "base_weight": 0.7,
            "salience_ratio": 0.0, "weight": 0.7,
        })

    # Saturating aggregation: many weak signals should not outrank one decisive one,
    # and one decisive signal alone must be able to quarantine a document.
    total = sum(s["weight"] for s in signals)
    score = round(1 - math.exp(-total), 4) if total else 0.0
    peak = max((s["weight"] for s in signals), default=0.0)
    verdict = "clean"
    if peak >= 0.9 or score >= 0.75:
        verdict = "hostile"
    elif signals:
        verdict = "suspect"
    return {
        "verdict": verdict,
        "score": score,
        "peak_signal_weight": round(peak, 3),
        "signal_count": len(signals),
        "signals": sorted(signals, key=lambda s: -s["weight"])[:50],
        "encoded_carriers": carriers,
        "distinct_signals": sorted({s["signal"] for s in signals}),
    }
