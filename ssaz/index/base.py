"""
Adapter interface for vector store backends
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ssaz.chunking.base import Chunk


@dataclass
class IndexHit:
    """A single nearest-neighbour result from a vector index"""

    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorIndex(ABC):
    @abstractmethod
    def add(self, chunks: List[Chunk], vectors: List[Sequence[float]]) -> None:
        """Insert chunks with their (L2-normalized) vectors"""

    @abstractmethod
    def query(self, vector: Sequence[float], k: int = 5,
              where: Optional[Dict[str, Any]] = None) -> List[IndexHit]:
        """Return top-``k`` hits by cosine similarity, optionally filtered
        by exact-match metadata constraints in ``where``"""

    @abstractmethod
    def count(self) -> int:
        """Number of indexed chunks"""

    @abstractmethod
    def all_chunks(self) -> List[Chunk]:
        """Return every stored chunk"""

    def existing_ids(self, ids: Sequence[str]) -> set:
        """Return the subset of ids already stored in the index"""
        return set()

    def clear(self) -> None:
        raise NotImplementedError
