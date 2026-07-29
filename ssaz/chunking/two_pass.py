"""The SAZ default chunker: structural pass + recursive fallback.

Sections found by the structural pass that fit the size budget become
chunks directly, carrying their heading and marker metadata. Oversized or
unmarked sections are handed to the recursive character-level fallback.
The section's structural metadata is inherited by every fallback chunk so
retrieval filters and metadata enrichment keep working.
"""

from __future__ import annotations

from typing import List, Optional

from ssaz.chunking.base import BaseChunker, Chunk
from ssaz.chunking.recursive import RecursiveChunker
from ssaz.chunking.structure import MarkerRule, StructuralChunker
from ssaz.documents import Document


class TwoPassChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        min_section_size: int = 40,
        domain: Optional[str] = None,
        structural_rules: Optional[List[MarkerRule]] = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.min_section_size = min_section_size
        self.structural = StructuralChunker(domain=domain,
                                            extra_rules=structural_rules)
        self.recursive = RecursiveChunker(chunk_size=chunk_size,
                                          chunk_overlap=chunk_overlap)

    def chunk(self, document: Document) -> List[Chunk]:
        sections = self.structural.split(document)
        chunks: List[Chunk] = []
        index = 0

        def emit(text: str, start: int, end: int, section) -> None:
            nonlocal index
            metadata = {
                "domain": document.domain,
                "chunking": "structural" if section.marker_type else "recursive",
            }
            if section.heading:
                metadata["section_heading"] = section.heading
            if section.marker_type:
                metadata["marker_type"] = section.marker_type
            metadata.update(section.metadata)
            chunks.append(Chunk(
                chunk_id=f"{document.doc_id}::{index:04d}",
                doc_id=document.doc_id,
                text=text,
                metadata=metadata,
                start=start,
                end=end,
            ))
            index += 1

        pending = None  # small section buffered for merging with the next
        for section in sections:
            if pending is not None:
                # Merge undersized section into the following one
                section.text = pending.text + "\n" + section.text
                section.start = pending.start
                if pending.heading and not section.heading:
                    section.heading = pending.heading
                    section.marker_type = pending.marker_type
                    section.metadata = {**pending.metadata, **section.metadata}
                pending = None
            if len(section.text) < self.min_section_size:
                pending = section
                continue
            if len(section.text) <= self.chunk_size:
                emit(section.text, section.start, section.end, section)
            else:
                offset = section.start
                for piece in self.recursive.split_text(section.text):
                    start = section.text.find(piece[:64])
                    start = section.start + (start if start >= 0 else 0)
                    emit(piece, start, start + len(piece), section)
        if pending is not None and pending.text.strip():
            emit(pending.text, pending.start, pending.end, pending)
        return chunks
