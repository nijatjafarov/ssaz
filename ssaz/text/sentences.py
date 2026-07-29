"""
Rule-based Azerbaijani sentence splitter.

Standard splitters trained on English text break on Azerbaijani
abbreviations, legal clause numbering, and agglutinative suffixes attached
to numerals. This splitter is deliberately conservative:
it never splits inside a known abbreviation, after a clause number, or
after a single-letter initial, and it is used by the recursive chunker only
as one separator level: the structure-aware chunker does not depend on
sentence detection at all.
"""

from __future__ import annotations

import re
from typing import List

# Common Azerbaijani abbreviations that end with a period but do not
# terminate a sentence.
_ABBREVIATIONS = {
    "məs",      # məsələn (for example)
    "b",        # bax (see)
    "s",        # səhifə (page)
    "səh",      # səhifə
    "c",        # cild (volume)
    "e.ə",      # eramızdan əvvəl (BC)
    "h",        # hicri
    "m",        # miladi
    "t",        # tarix
    "prof",     # professor
    "akad",     # akademik
    "dos",      # dosent
    "dr",       # doktor
    "müəl",     # müəllim
    "qəs",      # qəsəbə
    "r",        # rayon
    "ş",        # şəhər
    "küç",      # küçə
    "pr",       # prospekt
    "man",      # manat
    "qəp",      # qəpik
    "mln",      # milyon
    "mlrd",     # milyard
    "min",
    "və s",     # və sair (etc.)
    "və i.a",   # və ilaxır
}

"""
Sentence-terminal punctuation followed by whitespace and an uppercase
Azerbaijani letter, a digit, or an opening quote/dash starting the next
sentence
"""

_BOUNDARY_RE = re.compile(
    r"(?<=[.!?…])[)\"']*\s+(?=[A-ZÇƏĞİÖŞÜXQ0-9\"'(\-])"
)

_UPPER = "A-ZÇƏĞİÖŞÜXQ"


def _is_abbreviation(prefix: str) -> bool:
    """True if ``prefix`` (text before a candidate boundary) ends in a
    non-terminal abbreviation or an initial like ``H.`` in ``H. Əliyev``."""
    prefix = prefix.rstrip()
    if not prefix.endswith("."):
        return False
    stem = prefix[:-1]
    # Single uppercase initial: "M." / "Ə."
    m = re.search(r"(?:^|\s)([%s])$" % _UPPER, stem)
    if m:
        return True
    # Known abbreviation (case-insensitive on the last token)
    m = re.search(r"(?:^|\s)([\wçəğıöşü.]+)$", stem, re.IGNORECASE)
    if m:
        token = m.group(1).replace("İ", "i").replace("I", "ı").lower()
        if token in _ABBREVIATIONS:
            return True
    # Clause / list numbering: "3." or "2.1." — not a sentence end
    if re.search(r"(?:^|\s)\d+(?:\.\d+)*$", stem):
        return True
    return False


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences using Azerbaijani-aware rules."""
    if not text or not text.strip():
        return []
    sentences: List[str] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        boundary = match.start() + len(match.group(0).rstrip()) or match.start()
        candidate = text[start:match.end()].rstrip()
        prefix = text[start:match.start() + 1]
        if _is_abbreviation(prefix):
            continue
        if candidate:
            sentences.append(text[start:match.end()].strip())
            start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences
