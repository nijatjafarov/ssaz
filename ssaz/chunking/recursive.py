"""Character-level recursive fallback chunker.

Splits on paragraph, line, Azerbaijani sentence boundary, word, 
character until every piece fits the size budget, 
then greedily merges adjacent pieces up to ``chunk_size`` with
``chunk_overlap`` characters of overlap. The character-level last resort
guarantees progress on problematic input (no mid-morpheme split is ever
*forced* except when a single unbroken token exceeds the budget).
"""

from __future__ import annotations

from typing import List

from ssaz.chunking.base import BaseChunker, Chunk
from ssaz.documents import Document
from ssaz.text.sentences import split_sentences


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # splitting

    def _split(self, text: str, level: int = 0) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        if level == 0:
            parts = [p for p in text.split("\n\n") if p.strip()]
        elif level == 1:
            parts = [p for p in text.split("\n") if p.strip()]
        elif level == 2:
            parts = split_sentences(text)
        elif level == 3:
            parts = text.split(" ")
        else:
            # Character-level last resort.
            return [text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size)]
        if len(parts) <= 1:
            return self._split(text, level + 1)
        result: List[str] = []
        for part in parts:
            if len(part) > self.chunk_size:
                result.extend(self._split(part, level + 1))
            elif part.strip():
                result.append(part)
        return result

    def _merge(self, pieces: List[str]) -> List[str]:
        merged: List[str] = []
        current = ""
        for piece in pieces:
            candidate = (current + " " + piece).strip() if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                    overlap = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                    # Start the overlap at a word boundary when possible
                    space = overlap.find(" ")
                    if 0 <= space < len(overlap) - 1:
                        overlap = overlap[space + 1:]
                    current = (overlap + " " + piece).strip()
                    if len(current) > self.chunk_size:
                        current = piece
                else:
                    current = piece
        if current:
            merged.append(current)
        return merged

    def split_text(self, text: str) -> List[str]:
        """Split raw text into chunk-sized strings."""
        return self._merge(self._split(text))

    # BaseChunker

    def chunk(self, document: Document) -> List[Chunk]:
        chunks: List[Chunk] = []
        cursor = 0
        for i, piece in enumerate(self.split_text(document.text)):
            # Offset recovery, attempt to make overlap regions exact
            # Offsets ambiguous, so we search from just before the cursor
            search_from = max(0, cursor - self.chunk_overlap - 8)
            start = document.text.find(piece[:64], search_from)
            if start < 0:
                start = cursor
            end = start + len(piece)
            cursor = end
            chunks.append(Chunk(
                chunk_id=f"{document.doc_id}::{i:04d}",
                doc_id=document.doc_id,
                text=piece,
                metadata={"domain": document.domain},
                start=start,
                end=end,
            ))
        return chunks
