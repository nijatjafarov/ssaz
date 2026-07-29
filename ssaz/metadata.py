"""Metadata enrichment, a required part of the SAZ indexing step"""

from __future__ import annotations

import re
from typing import List, Optional

from ssaz.chunking.base import Chunk
from ssaz.documents import Document

_MONTHS = {
    "yanvar": 1, "fevral": 2, "mart": 3, "aprel": 4, "may": 5, "iyun": 6,
    "iyul": 7, "avqust": 8, "sentyabr": 9, "oktyabr": 10, "noyabr": 11,
    "dekabr": 12,
}

_DATE_TEXT_RE = re.compile(
    r"\b(\d{1,2})\s+(yanvar|fevral|mart|aprel|may|iyun|iyul|avqust|"
    r"sentyabr|oktyabr|noyabr|dekabr)\s*(\d{4})?",
    re.IGNORECASE,
)
_DATE_NUM_RE = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b")
_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def extract_date(text: str, default_year: Optional[int] = None) -> Optional[str]:
    """Extract the first date mention and return it"""
    m = _DATE_ISO_RE.search(text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = _DATE_NUM_RE.search(text)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
    m = _DATE_TEXT_RE.search(text)
    if m:
        day = int(m.group(1))
        month = _MONTHS[m.group(2).replace("İ", "i").replace("I", "ı").lower()]
        year = int(m.group(3)) if m.group(3) else default_year
        if year and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
    return None


class MetadataEnricher:
    """Attaches domain, section heading, document date, and position to
    each chunk, and builds the text variant used for embedding.

    Args:
        embed_metadata_header: If True, ``embedding_text`` for a
            chunk is ``"[domain | heading | date]\\n" + text``. The stored
            chunk text is never modified.
        default_year: Year assumed for news datelines that omit the year.
    """

    def __init__(self, embed_metadata_header: bool = True,
                 default_year: Optional[int] = None) -> None:
        self.embed_metadata_header = embed_metadata_header
        self.default_year = default_year

    def enrich(self, document: Document, chunks: List[Chunk]) -> List[Chunk]:
        doc_date = document.date or extract_date(document.text[:500],
                                                 self.default_year)
        total = len(chunks)
        for position, chunk in enumerate(chunks):
            meta = chunk.metadata
            meta.setdefault("domain", document.domain)
            meta["doc_id"] = document.doc_id
            meta["position"] = position
            meta["n_chunks"] = total
            if document.title:
                meta.setdefault("title", document.title)
            if doc_date:
                meta.setdefault("date", doc_date)
            for key, value in document.metadata.items():
                meta.setdefault(key, value)
            meta["embedding_text"] = self._embedding_text(chunk)
        return chunks

    def _embedding_text(self, chunk: Chunk) -> str:
        if not self.embed_metadata_header:
            return chunk.text
        parts = [chunk.metadata.get("domain")]
        if chunk.metadata.get("title"):
            parts.append(chunk.metadata["title"])
        if chunk.metadata.get("section_heading"):
            parts.append(chunk.metadata["section_heading"])
        if chunk.metadata.get("date"):
            parts.append(chunk.metadata["date"])
        header = " | ".join(str(p) for p in parts if p)
        return f"[{header}]\n{chunk.text}" if header else chunk.text
