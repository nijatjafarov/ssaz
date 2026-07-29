"""ChromaDB backend (optional extra ``chroma``)"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ssaz.chunking.base import Chunk
from ssaz.index.base import IndexHit, VectorIndex

# Chroma metadata values must be scalar, everything else is stringified
_SCALARS = (str, int, float, bool)


def _sanitize(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {key: (value if isinstance(value, _SCALARS) else str(value))
            for key, value in metadata.items() if value is not None}


class ChromaIndex(VectorIndex):
    def __init__(self, collection_name: str = "ssaz",
                 persist_directory: Optional[str] = None) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "chromadb is required for the chroma backend. "
                "Install with:  pip install ssaz[chroma]"
            ) from exc
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: List[Chunk], vectors: List[Sequence[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[list(v) for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[_sanitize({**c.metadata, "doc_id": c.doc_id,
                                  "start": c.start, "end": c.end})
                       for c in chunks],
        )

    def query(self, vector: Sequence[float], k: int = 5,
              where: Optional[Dict[str, Any]] = None) -> List[IndexHit]:
        result = self._collection.query(
            query_embeddings=[list(vector)],
            n_results=min(k, max(self.count(), 1)),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        hits: List[IndexHit] = []
        for chunk_id, document, metadata, distance in zip(
            result["ids"][0], result["documents"][0],
            result["metadatas"][0], result["distances"][0],
        ):
            hits.append(IndexHit(
                chunk_id=chunk_id,
                score=1.0 - distance,
                text=document,
                metadata=metadata or {},
            ))
        return hits

    def count(self) -> int:
        return self._collection.count()

    def existing_ids(self, ids) -> set:
        found = self._collection.get(ids=list(ids), include=[])
        return set(found["ids"])

    def all_chunks(self) -> List[Chunk]:
        result = self._collection.get(include=["documents", "metadatas"])
        chunks: List[Chunk] = []
        for chunk_id, document, metadata in zip(
            result["ids"], result["documents"], result["metadatas"],
        ):
            metadata = dict(metadata or {})
            doc_id = str(metadata.pop("doc_id", chunk_id.split("::")[0]))
            start = int(metadata.pop("start", 0))
            end = int(metadata.pop("end", 0))
            chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id,
                                text=document, metadata=metadata,
                                start=start, end=end))
        return chunks

    def clear(self) -> None:
        existing = self._collection.get()["ids"]
        if existing:
            self._collection.delete(ids=existing)
