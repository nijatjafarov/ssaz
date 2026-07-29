"""Chunk container and chunker interface"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ssaz.documents import Document


@dataclass
class Chunk:
    """A retrievable unit produced by the chunking stage.

    Attributes:
        chunk_id: ``"<doc_id>::<index>"`` — stable across re-runs on the
            same document and chunker configuration.
        doc_id: Identifier of the source document.
        text: Chunk text as indexed.
        metadata: Enriched metadata (domain, section heading, date, ...).
        start: Character offset of the chunk in the source document.
        end: End character offset (exclusive).
    """

    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start: int = 0
    end: int = 0

    def __len__(self) -> int:
        return len(self.text)


class BaseChunker(ABC):
    """Interface implemented by all chunkers"""

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document into an ordered list of chunks"""

    def chunk_many(self, documents: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk(document))
        return chunks
