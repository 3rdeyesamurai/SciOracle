"""System 1: fast, heuristic semantic extraction.

Deliberately non-generative. Every fact this layer produces carries the exact
character span it came from, because System 2 is only permitted to reason about
facts it can trace back to bytes in the L3 store.

If an LLM provider is configured the pipeline may enrich these extractions, but
the deterministic layer is always the floor: an unavailable model degrades
recall, never provenance.
"""
import re
from dataclasses import asdict, dataclass, field
from datetime import date

MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}
MONTH_RE = "|".join(MONTHS)

CLAUSE_HEAD_RE = re.compile(
    r"^\s*(?:(?:Article|Section|Clause|ARTICLE|SECTION|CLAUSE)\s+)?(\d+(?:\.\d+)*)\.?\s+(.{0,120}?)(?:\n|$)", re.M)

DATE_PATTERNS = [
    re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_RE})\s+(\d{{4}})\b", re.I),
    re.compile(rf"\b({MONTH_RE})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I),
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"),
]

DEFINED_TERM_PATTERNS = [
    re.compile(r"[\"“]([A-Z][A-Za-z0-9 \-]{2,60})[\"”]\s+(?:means|shall mean|refers to)\s+([^.;]{5,400})", re.I),
    re.compile(r"\b([A-Z][A-Za-z0-9\-]{2,40}(?:\s+[A-Z][A-Za-z0-9\-]{2,40}){0,3})\s+(?:means|shall mean)\s+([^.;]{5,400})"),
    re.compile(r"\(the\s+[\"“]([A-Z][A-Za-z0-9 \-]{2,60})[\"”]\)"),
]

OBLIGATION_RE = re.compile(
    r"\b(?P<subject>(?:The\s+)?[A-Z][A-Za-z ]{2,40}?|Each\s+Party|Either\s+Party|Neither\s+Party)\s+"
    r"(?P<modal>shall not|shall|must not|must|may not|may|will not|will|is required to|is entitled to|agrees to|undertakes to)\s+"
    r"(?P<predicate>[^.;]{5,240})", re.I)

MONEY_RE = re.compile(r"(?P<cur>USD|EUR|GBP|CHF|JPY|CNY|\$|€|£)\s?(?P<amt>[\d][\d,\.]{0,18})\s?(?P<scale>million|billion|thousand|m\b|bn\b)?", re.I)
DURATION_RE = re.compile(r"\b(?P<n>\d{1,4}|thirty|sixty|ninety|twelve|twenty-four|six|three|one|two|five|ten)\s*[\(\)]*\s*(?P<unit>day|days|business days|month|months|year|years|calendar days)\b", re.I)
PERCENT_RE = re.compile(r"\b(\d{1,3}(?:\.\d{1,3})?)\s?(?:%|per cent|percent)\b", re.I)

WORD_NUM = {"one": 1, "two": 2, "three": 3, "five": 5, "six": 6, "ten": 10, "twelve": 12,
            "twenty-four": 24, "thirty": 30, "sixty": 60, "ninety": 90}
SCALE = {"million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "thousand": 1e3}

GOVERNING_LAW_RE = re.compile(
    r"governed\s+by(?:\s+and\s+construed\s+in\s+accordance\s+with)?\s+the\s+laws?\s+of\s+"
    r"(?:the\s+)?(?P<jur>[A-Z][A-Za-z ,\.]{2,60}?)(?=[,\.;\n]|\s+and\b|\s+without\b)", re.I)
SEAT_RE = re.compile(r"(?:seat|place)\s+of\s+arbitration\s+(?:shall\s+be|is)\s+(?P<seat>[A-Z][A-Za-z ,\.]{2,50}?)(?=[,\.;\n]|\s+(?:under|pursuant|in\s+accordance|administered|and)\b)", re.I)
RULES_RE = re.compile(r"\b(UNCITRAL|ICC|LCIA|SIAC|HKIAC|ICSID|SCC|AAA|ICDR)\b(?:\s+(?:Arbitration\s+)?Rules)?")
NOTICE_RE = re.compile(r"(?:(?:written\s+)?notice\s+of\s+(?:not\s+less\s+than|at\s+least)?\s*|upon\s+)(?P<n>\d{1,4}|thirty|sixty|ninety)\s*[\(\)]*\s*(?P<unit>days?|months?)", re.I)
CAP_RE = re.compile(r"(?:aggregate\s+)?liability\s+(?:of\s+\w+\s+)?(?:shall\s+not\s+exceed|is\s+capped\s+at|limited\s+to)\s+(?P<val>[^.;]{3,80})", re.I)
PARTY_RE = re.compile(
    r"(?:between|among)\s+(?P<a>[A-Z][A-Za-z0-9&\.,\- ]{2,80}?)\s+(?:\(.{0,60}?\)\s*)?and\s+(?P<b>[A-Z][A-Za-z0-9&\.,\- ]{2,80}?)(?=[,\.;\n]|\s+\(|\s+(?:on|dated|with|having|as\s+of)\b)", re.S)


@dataclass
class Span:
    doc_id: str
    start: int
    end: int
    text: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Fact:
    kind: str
    key: str
    value: str
    normalized: str | float | None
    span: Span
    clause: str | None = None
    confidence: float = 0.8

    def as_dict(self) -> dict:
        out = asdict(self)
        out["span"] = self.span.as_dict()
        return out


@dataclass
class Extraction:
    doc_id: str
    clauses: list[dict] = field(default_factory=list)
    parties: list[dict] = field(default_factory=list)
    defined_terms: list[dict] = field(default_factory=list)
    obligations: list[dict] = field(default_factory=list)
    dates: list[dict] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _norm_number(token: str) -> float | None:
    token = token.strip().lower().replace(",", "")
    if token in WORD_NUM:
        return float(WORD_NUM[token])
    try:
        return float(token)
    except ValueError:
        return None


def segment_clauses(text: str) -> list[dict]:
    """Split on numbered headings; fall back to paragraph blocks."""
    heads = list(CLAUSE_HEAD_RE.finditer(text))
    clauses = []
    if heads:
        for i, m in enumerate(heads):
            start = m.start()
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            body = text[start:end].strip()
            if body:
                clauses.append({"number": m.group(1), "heading": m.group(2).strip(), "start": start,
                                "end": end, "text": body})
    if not clauses:
        offset = 0
        for block in re.split(r"\n\s*\n", text):
            stripped = block.strip()
            if stripped:
                start = text.find(stripped, offset)
                clauses.append({"number": None, "heading": stripped[:60], "start": start,
                                "end": start + len(stripped), "text": stripped})
                offset = start + len(stripped)
    return clauses


def clause_at(clauses: list[dict], offset: int) -> str | None:
    for clause in clauses:
        if clause["start"] <= offset < clause["end"]:
            return clause["number"] or clause["heading"][:40]
    return None


def _iso_date(match: re.Match, pattern_index: int) -> str | None:
    try:
        if pattern_index == 0:
            d, mo, y = int(match.group(1)), MONTHS[match.group(2).lower()], int(match.group(3))
        elif pattern_index == 1:
            mo, d, y = MONTHS[match.group(1).lower()], int(match.group(2)), int(match.group(3))
        elif pattern_index == 2:
            y, mo, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
        else:
            d, mo, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return date(y, mo, d).isoformat()
    except (ValueError, KeyError):
        return None


def extract(doc_id: str, text: str) -> Extraction:
    clauses = segment_clauses(text)
    ex = Extraction(doc_id=doc_id, clauses=[{k: c[k] for k in ("number", "heading", "start", "end")} for c in clauses])

    def span(m: re.Match, group: int | str = 0) -> Span:
        return Span(doc_id=doc_id, start=m.start(group), end=m.end(group), text=m.group(group)[:400])

    def add_fact(kind: str, key: str, value: str, normalized, m: re.Match, group=0, confidence: float = 0.8):
        s = span(m, group)
        ex.facts.append(Fact(kind, key, value.strip(), normalized, s, clause_at(clauses, s.start), confidence).as_dict())

    for m in PARTY_RE.finditer(text[:6000]):
        for label in ("a", "b"):
            name = re.sub(r"\s+", " ", m.group(label)).strip(" ,.")
            if 2 < len(name) < 90:
                ex.parties.append({"name": name, "span": span(m, label).as_dict()})

    for pattern in DEFINED_TERM_PATTERNS:
        for m in pattern.finditer(text):
            term = m.group(1).strip()
            definition = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else ""
            ex.defined_terms.append({"term": term, "definition": definition, "span": span(m).as_dict(),
                                     "clause": clause_at(clauses, m.start())})

    for i, pattern in enumerate(DATE_PATTERNS):
        for m in pattern.finditer(text):
            iso = _iso_date(m, i)
            if iso:
                context = text[max(0, m.start() - 90):m.end() + 60].replace("\n", " ")
                ex.dates.append({"iso": iso, "raw": m.group(0), "context": context.strip(),
                                 "span": span(m).as_dict(), "clause": clause_at(clauses, m.start())})

    for m in OBLIGATION_RE.finditer(text):
        modal = m.group("modal").lower()
        ex.obligations.append({
            "subject": re.sub(r"\s+", " ", m.group("subject")).strip(),
            "modal": modal,
            "polarity": "negative" if "not" in modal else "positive",
            "deontic": "prohibition" if "not" in modal else ("permission" if modal.startswith("may") else "obligation"),
            "predicate": re.sub(r"\s+", " ", m.group("predicate")).strip(),
            "span": span(m).as_dict(),
            "clause": clause_at(clauses, m.start()),
        })

    for m in GOVERNING_LAW_RE.finditer(text):
        jur = re.sub(r"\s+", " ", m.group("jur")).strip(" ,.")
        add_fact("attribute", "governing_law", jur, jur.lower(), m, confidence=0.9)
    for m in SEAT_RE.finditer(text):
        seat = re.sub(r"\s+", " ", m.group("seat")).strip(" ,.")
        add_fact("attribute", "arbitration_seat", seat, seat.lower(), m, confidence=0.9)
    for m in RULES_RE.finditer(text):
        add_fact("attribute", "arbitration_rules", m.group(1), m.group(1).upper(), m, confidence=0.85)
    for m in NOTICE_RE.finditer(text):
        n = _norm_number(m.group("n"))
        if n is None:
            continue
        days = n * (30 if m.group("unit").lower().startswith("month") else 1)
        add_fact("attribute", "notice_period_days", m.group(0), days, m, confidence=0.8)
    for m in CAP_RE.finditer(text):
        money = MONEY_RE.search(m.group("val"))
        normalized = None
        if money:
            amount = _norm_number(money.group("amt"))
            if amount is not None:
                normalized = amount * SCALE.get((money.group("scale") or "").lower(), 1)
        add_fact("attribute", "liability_cap", m.group("val").strip(), normalized, m, confidence=0.75)
    for m in MONEY_RE.finditer(text):
        amount = _norm_number(m.group("amt"))
        if amount is None:
            continue
        add_fact("quantity", "money", m.group(0), amount * SCALE.get((m.group("scale") or "").lower(), 1), m, confidence=0.7)
    for m in PERCENT_RE.finditer(text):
        add_fact("quantity", "percentage", m.group(0), float(m.group(1)), m, confidence=0.7)
    for m in DURATION_RE.finditer(text):
        n = _norm_number(m.group("n"))
        if n is None:
            continue
        unit = m.group("unit").lower()
        mult = 365 if unit.startswith("year") else (30 if unit.startswith("month") else 1)
        add_fact("quantity", "duration_days", m.group(0), n * mult, m, confidence=0.65)

    return ex


def chronology(extractions: list[Extraction], doc_titles: dict[str, str] | None = None) -> list[dict]:
    """Merge every dated assertion into one source-linked timeline."""
    doc_titles = doc_titles or {}
    events = []
    for ex in extractions:
        for d in ex.dates:
            events.append({
                "date": d["iso"],
                "document_id": ex.doc_id,
                "document": doc_titles.get(ex.doc_id, ex.doc_id),
                "clause": d.get("clause"),
                "assertion": d["context"],
                "raw": d["raw"],
                "span": d["span"],
            })
    return sorted(events, key=lambda e: (e["date"], e["document"]))
