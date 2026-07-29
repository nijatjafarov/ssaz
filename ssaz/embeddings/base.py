"""Embedder interface. All vectors are L2-normalized lists of floats so
cosine similarity reduces to a dot product in every index backend"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Sequence

Vector = List[float]


def l2_normalize(vector: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return list(vector)
    return [x / norm for x in vector]


class BaseEmbedder(ABC):
    """Common interface for all embedding backends.

    Implementations must return L2-normalized vectors. ``embed_query`` is
    separate from ``embed_documents`` because several models (e.g. Arctic
    Embed) require an instruction prefix on queries but not on documents.
    """

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[Vector]:
        """Embed document and chunk texts"""

    def embed_query(self, text: str) -> Vector:
        """Embed a search query. Defaults to document embedding"""
        return self.embed_documents([text])[0]
