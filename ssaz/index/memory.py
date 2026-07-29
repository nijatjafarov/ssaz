"""
Dependency-free in-memory vector index with JSON persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from ssaz.chunking.base import Chunk
from ssaz.index.base import IndexHit, VectorIndex


class InMemoryIndex(VectorIndex):
    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._vectors: List[List[float]] = []
        self._ids: Dict[str, int] = {}

    def add(self, chunks: List[Chunk], vectors: List[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        for chunk, vector in zip(chunks, vectors):
            if chunk.chunk_id in self._ids:
                # Upsert: replace the existing record
                i = self._ids[chunk.chunk_id]
                self._chunks[i] = chunk
                self._vectors[i] = list(vector)
            else:
                self._ids[chunk.chunk_id] = len(self._chunks)
                self._chunks.append(chunk)
                self._vectors.append(list(vector))

    def query(self, vector: Sequence[float], k: int = 5,
              where: Optional[Dict[str, Any]] = None) -> List[IndexHit]:
        query = list(vector)
        scored = []
        for chunk, stored in zip(self._chunks, self._vectors):
            if where and any(chunk.metadata.get(key) != value
                             for key, value in where.items()):
                continue
            score = sum(a * b for a, b in zip(query, stored))
            scored.append((score, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [IndexHit(chunk_id=c.chunk_id, score=s, text=c.text,
                         metadata=c.metadata)
                for s, c in scored[:k]]

    def count(self) -> int:
        return len(self._chunks)

    def existing_ids(self, ids) -> set:
        return {chunk_id for chunk_id in ids if chunk_id in self._ids}

    def all_chunks(self) -> List[Chunk]:
        return list(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()
        self._ids.clear()

    # persistence
    def save(self, path: Union[str, Path]) -> None:
        payload = {
            "chunks": [
                {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text,
                 "metadata": c.metadata, "start": c.start, "end": c.end}
                for c in self._chunks
            ],
            "vectors": self._vectors,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "InMemoryIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls()
        chunks = [Chunk(**record) for record in payload["chunks"]]
        index.add(chunks, payload["vectors"])
        return index
