"""
Dense retriever: embeds the query and ranks chunks by cosine
similarity in the vector index. Returns ``List[RetrievedChunk]`` sorted
best-first, the single result contract consumed by the engine, the
evaluation harness, and the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ssaz.embeddings.base import BaseEmbedder
from ssaz.index.base import VectorIndex


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class DenseRetriever:
    def __init__(self, embedder: BaseEmbedder, index: VectorIndex) -> None:
        self.embedder = embedder
        self.index = index

    def retrieve(self, query: str, k: int = 5,
                 where: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        vector = self.embedder.embed_query(query)
        hits = self.index.query(vector, k=k, where=where)
        return [RetrievedChunk(chunk_id=h.chunk_id, text=h.text,
                               score=h.score, metadata=h.metadata)
                for h in hits]
