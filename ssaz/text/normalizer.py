"""
Text normalization with Azerbaijani-specific casing and character repairs.

Azerbaijani Latin script has four case pairs that
mishandles or that appear mis-encoded in web corpora, and 
interpretation of Cyrillic in legacy text
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Legacy characters observed in scraped Azerbaijani corpora,
# mapped to the standard Azerbaijani Latin alphabet.
_CHAR_REPAIRS = {
    "Ǝ": "Ə",
    "ǝ": "ə",
    "Ә": "Ə",
    "ә": "ə",
    "‘": "'",
    "’": "'",
    "ʼ": "'",
    "´": "'",
    "`": "'",
    "“": '"',
    "”": '"',
    "«": '"',
    "»": '"',
    "–": "-",
    "—": "-",
    " ": " ",
}

_REPAIR_TABLE = str.maketrans(_CHAR_REPAIRS)

_WS_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def azerbaijani_lower(text: str) -> str:
    """Lowercase ``text`` using Azerbaijani casing rules"""
    return text.replace("İ", "i").replace("I", "ı").lower()


def azerbaijani_upper(text: str) -> str:
    """Uppercase ``text`` using Azerbaijani casing rules"""
    return text.replace("i", "İ").replace("ı", "I").upper()


@dataclass
class AzNormalizer:
    """Configurable normalizer applied before chunking and indexing.

    Args:
        repair_characters: Fix legacy schwa/apostrophe/quote encodings.
        collapse_whitespace: Collapse runs of spaces/tabs and 3+ newlines.
        lowercase: Apply Azerbaijani-aware lowercasing. Off by default for
            indexing, since the embedding models are case-aware.
        strip_controls: Remove non-printable control characters.
        transliterate_cyrillic: Convert Azerbaijani Cyrillic
            alphabet to the modern Latin alphabet, e.g.
            ``Азәрбајҹан -> Azərbaycan``. Latin text passes through
            unchanged, so mixed-script corpora are safe.
        unicode_form: Unicode normalization form applied first.
        extra_steps: Custom ``str -> str`` callables run at the end of
            the pipeline, in order, the extension point for corpus-
            specific cleanup:

                norm = AzNormalizer(extra_steps=[my_ocr_fixer])
    """

    repair_characters: bool = True
    collapse_whitespace: bool = True
    lowercase: bool = False
    strip_controls: bool = True
    transliterate_cyrillic: bool = True
    unicode_form: str = "NFC"
    extra_steps: tuple = ()

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize(self.unicode_form, text)
        if self.strip_controls:
            text = _CONTROL_RE.sub(" ", text)
        if self.transliterate_cyrillic:
            # Imported lazily to avoid a circular import at module load
            from ssaz.text.translit import cyrillic_to_latin
            text = cyrillic_to_latin(text)
        if self.repair_characters:
            text = text.translate(_REPAIR_TABLE)
        if self.collapse_whitespace:
            text = _WS_RE.sub(" ", text)
            text = _MULTI_NL_RE.sub("\n\n", text)
            text = "\n".join(line.strip() for line in text.split("\n"))
            text = text.strip()
        if self.lowercase:
            text = azerbaijani_lower(text)
        for step in self.extra_steps:
            text = step(text)
        return text

    def __call__(self, text: str) -> str:
        return self.normalize(text)
