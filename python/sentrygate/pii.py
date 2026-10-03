"""Deterministic, stdlib-only PII detection and reversible masking.

No ML model, no external dependencies: just `re` plus a couple of small
validators (Luhn for cards). Each recognizer turns raw text into a list of
`PIISpan`s; `detect` merges them into a non-overlapping, sorted list; `mask`
swaps spans for typed placeholders and hands back a reversible mapping that
`unmask` applies to put the originals back.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PIISpan:
    start: int
    end: int
    type: str
    text: str


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _luhn_ok(digits: str) -> bool:
    """Luhn checksum: True when the digit string is a valid card number."""
    nums = [int(c) for c in digits if c.isdigit()]
    if len(nums) < 12:
        return False
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# US numbers (optional +1, separators) and simple international (+NN ...).
_PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]?\d{3}[\s.\-]?\d{4}\b"
)

# Candidate card: 13-19 digits in groups separated by spaces/dashes. Luhn-gated.
_CARD_RE = re.compile(
    r"(?<![\w.])(?:\d[ \-]?){13,19}(?![\w.])"
)

_SSN_RE = re.compile(
    r"(?<![\w\-])(?!000|666|9\d\d)\d{3}-\d{2}-\d{4}(?![\w\-])"
    r"|(?<![\w\-])\d{9}(?![\w\-])"
)

_IPV4_RE = re.compile(
    r"(?<![\w.])"
    r"(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(?![\w.])"
)

# IPv6: enough to catch full and compressed forms without matching prose.
_IPV6_RE = re.compile(
    r"(?<![\w:])"
    r"(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}"
    r"|(?:[A-Fa-f0-9]{1,4}:){1,7}:"
    r"|(?:[A-Fa-f0-9]{1,4}:){1,6}:[A-Fa-f0-9]{1,4}"
    r"|(?:[A-Fa-f0-9]{1,4}:){1,5}(?::[A-Fa-f0-9]{1,4}){1,2}"
    r"|(?:[A-Fa-f0-9]{1,4}:){1,4}(?::[A-Fa-f0-9]{1,4}){1,3}"
    r"|(?:[A-Fa-f0-9]{1,4}:){1,3}(?::[A-Fa-f0-9]{1,4}){1,4}"
    r"|(?:[A-Fa-f0-9]{1,4}:){1,2}(?::[A-Fa-f0-9]{1,4}){1,5}"
    r"|[A-Fa-f0-9]{1,4}:(?::[A-Fa-f0-9]{1,4}){1,6}"
    r"|:(?:(?::[A-Fa-f0-9]{1,4}){1,7}|:)"
)

# IBAN: 2 letters, 2 check digits, then 11-30 alphanumerics.
_IBAN_RE = re.compile(
    r"\b[A-Z]{2}\d{2}[A-Za-z0-9]{11,30}\b"
)

# Known API-key prefixes, plus a context-guarded generic long token.
_API_KEY_PREFIXED_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9]{16,}"
    r"|AKIA[A-Z0-9]{16}"
    r"|ghp_[A-Za-z0-9]{36,}"
    r"|xoxb-[A-Za-z0-9\-]{10,})\b"
)
# Generic: only when it looks assigned to a key-like name (api_key=..., token: ...).
_API_KEY_GENERIC_RE = re.compile(
    r"(?i)(?:api[_\- ]?key|secret|token|access[_\- ]?key|auth)"
    r"\s*[:=]\s*[\"']?([A-Za-z0-9_\-]{32,})[\"']?"
)


# Recognizers in priority order (earlier = higher priority on ties).
# Each returns a list of PIISpan for the given text.

def _spans_email(text):
    return [PIISpan(m.start(), m.end(), "EMAIL", m.group())
            for m in _EMAIL_RE.finditer(text)]


def _spans_api_key(text):
    out = []
    for m in _API_KEY_PREFIXED_RE.finditer(text):
        out.append(PIISpan(m.start(), m.end(), "API_KEY", m.group()))
    for m in _API_KEY_GENERIC_RE.finditer(text):
        out.append(PIISpan(m.start(1), m.end(1), "API_KEY", m.group(1)))
    return out


def _spans_credit_card(text):
    out = []
    for m in _CARD_RE.finditer(text):
        digits = re.sub(r"[ \-]", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            out.append(PIISpan(m.start(), m.end(), "CREDIT_CARD", m.group()))
    return out


def _spans_iban(text):
    return [PIISpan(m.start(), m.end(), "IBAN", m.group())
            for m in _IBAN_RE.finditer(text)]


def _spans_ssn(text):
    return [PIISpan(m.start(), m.end(), "SSN", m.group())
            for m in _SSN_RE.finditer(text)]


def _spans_phone(text):
    return [PIISpan(m.start(), m.end(), "PHONE", m.group())
            for m in _PHONE_RE.finditer(text)]


def _spans_ipv4(text):
    return [PIISpan(m.start(), m.end(), "IPV4", m.group())
            for m in _IPV4_RE.finditer(text)]


def _spans_ipv6(text):
    out = []
    for m in _IPV6_RE.finditer(text):
        g = m.group()
        if ":" in g and len(g) >= 3:  # avoid matching a lone "::" artifacts aside
            out.append(PIISpan(m.start(), m.end(), "IPV6", g))
    return out


# Priority order: more specific / higher-value first.
_RECOGNIZERS = [
    _spans_email,
    _spans_api_key,
    _spans_iban,
    _spans_credit_card,
    _spans_ssn,
    _spans_ipv4,
    _spans_ipv6,
    _spans_phone,
]
_PRIORITY = {
    "EMAIL": 0, "API_KEY": 1, "IBAN": 2, "CREDIT_CARD": 3,
    "SSN": 4, "IPV4": 5, "IPV6": 6, "PHONE": 7,
}


def detect(text: str) -> list[PIISpan]:
    """Run every recognizer, resolve overlaps, return sorted non-overlapping spans.

    On overlap the longer span wins; on a length tie the higher-priority
    (earlier-registered) type wins.
    """
    spans: list[PIISpan] = []
    for rec in _RECOGNIZERS:
        spans.extend(rec(text))

    def better(a: PIISpan, b: PIISpan) -> PIISpan:
        la, lb = a.end - a.start, b.end - b.start
        if la != lb:
            return a if la > lb else b
        pa = _PRIORITY.get(a.type, 99)
        pb = _PRIORITY.get(b.type, 99)
        if pa != pb:
            return a if pa < pb else b
        return a if a.start <= b.start else b

    # Greedily keep the best span, dropping anything that overlaps it.
    spans.sort(key=lambda s: (s.start, -(s.end - s.start), _PRIORITY.get(s.type, 99)))
    kept: list[PIISpan] = []
    for s in spans:
        conflict = None
        for i, k in enumerate(kept):
            if s.start < k.end and k.start < s.end:  # overlap
                conflict = i
                break
        if conflict is None:
            kept.append(s)
        else:
            win = better(kept[conflict], s)
            if win is s:
                kept[conflict] = s
    kept.sort(key=lambda s: s.start)
    return kept


@dataclass
class MaskResult:
    masked_text: str
    mapping: dict = field(default_factory=dict)   # placeholder -> original value
    spans: list = field(default_factory=list)     # list[PIISpan]


def mask(text: str, types: set | None = None) -> MaskResult:
    """Replace detected PII spans with typed placeholders `<TYPE_n>`.

    The same original value always maps to the same placeholder (dedupe by
    value). `mapping` maps placeholder -> original. `types` optionally restricts
    which entity types are masked.
    """
    spans = detect(text)
    if types is not None:
        spans = [s for s in spans if s.type in types]

    value_to_ph: dict = {}        # (type, value) -> placeholder
    mapping: dict = {}            # placeholder -> value
    counters: dict = {}          # type -> next n

    # Assign placeholders in first-appearance order (spans already sorted).
    for s in spans:
        key = (s.type, s.text)
        if key not in value_to_ph:
            n = counters.get(s.type, 0) + 1
            counters[s.type] = n
            ph = f"<{s.type}_{n}>"
            value_to_ph[key] = ph
            mapping[ph] = s.text

    # Rebuild the text, replacing spans back-to-front is unnecessary since we
    # walk left to right tracking the cursor.
    out = []
    cursor = 0
    for s in spans:
        out.append(text[cursor:s.start])
        out.append(value_to_ph[(s.type, s.text)])
        cursor = s.end
    out.append(text[cursor:])

    return MaskResult(masked_text="".join(out), mapping=mapping, spans=spans)


def unmask(text: str, mapping: dict) -> str:
    """Reverse `mask`: replace each placeholder with its original value.

    Longest placeholder first, so `<EMAIL_1>` is never clipped by `<EMAIL_12>`.
    """
    for ph in sorted(mapping, key=len, reverse=True):
        text = text.replace(ph, mapping[ph])
    return text
