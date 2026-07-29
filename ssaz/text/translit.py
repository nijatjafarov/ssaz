"""
Azerbaijani Cyrillic to Latin transliteration.
For example ``Азәрбајҹан -> Azərbaycan``, ``Гарабағ -> Qarabağ``.
"""

from __future__ import annotations

import re

from ssaz.text.normalizer import azerbaijani_upper

_CYR_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "q", "ғ": "ğ", "д": "d",
    "е": "e", "ә": "ə", "ж": "j", "з": "z", "и": "i", "ы": "ı",
    "ј": "y", "к": "k", "ҝ": "g", "л": "l", "м": "m", "н": "n",
    "о": "o", "ө": "ö", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ү": "ü", "ф": "f", "х": "x", "һ": "h", "ч": "ç",
    "ҹ": "c", "ш": "ş",
    # Russian-era letters kept in loanwords and pre-1958 orthography.
    "й": "y", "э": "e", "ю": "yu", "я": "ya", "ё": "yo",
    "ц": "ts", "щ": "şş", "ъ": "", "ь": "",
}

_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")


def contains_cyrillic(text: str) -> bool:
    """True if ``text`` contains any Cyrillic-block character"""
    return bool(_CYRILLIC_RE.search(text))


def cyrillic_to_latin(text: str) -> str:
    """Transliterate Azerbaijani Cyrillic to the modern Latin alphabet"""
    result = []
    for i, char in enumerate(text):
        lower = char.lower()
        latin = _CYR_TO_LAT.get(lower)
        if latin is None:
            result.append(char)
        elif char == lower:
            result.append(latin)
        elif len(latin) <= 1:
            result.append(azerbaijani_upper(latin))
        else:
            following = text[i + 1] if i + 1 < len(text) else ""
            if following and following.isupper():
                result.append(azerbaijani_upper(latin))
            else:
                result.append(azerbaijani_upper(latin[0]) + latin[1:])
    return "".join(result)
