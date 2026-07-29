"""
Pinecone backend: the SSAZ default vector store
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from ssaz.chunking.base import Chunk
from ssaz.index.base import IndexHit, VectorIndex

# Pinecone metadata values must be str, number, bool, or list of str
_SCALARS = (str, int, float, bool)

_UPSERT_BATCH = 100
_FETCH_BATCH = 100

_TRANSIENT_MARKERS = (
    "connection", "timeout", "timed out", "getaddrinfo", "temporarily",
    "unavailable", "reset by peer", "broken pipe", "too many requests",
    "502", "503", "504", "eof occurred",
)


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    haystack = f"{type(exc).__name__} {exc}".lower()
    return any(marker in haystack for marker in _TRANSIENT_MARKERS)


def _sanitize(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {key: (value if isinstance(value, _SCALARS) else str(value))
            for key, value in metadata.items() if value is not None}


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a Pinecone response that may be a plain dict
    or an SDK model object"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    value = getattr(obj, name, default)
    return default if value is None else value


class PineconeIndex(VectorIndex):
    def __init__(
        self,
        index_name: str = "ssaz",
        api_key: Optional[str] = None,
        namespace: str = "",
        dimension: Optional[int] = None,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        max_retries: int = 5,
        retry_backoff: float = 2.0,
    ) -> None:
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise ImportError(
                "The 'pinecone' package (a core ssaz dependency) is "
                "missing. Reinstall with:  pip install ssaz"
            ) from exc
        api_key = api_key or os.environ.get("PINECONE_API_KEY", "")
        if not api_key:
            raise ValueError("Set PINECONE_API_KEY or pass api_key=")
        self.namespace = namespace
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._pc = Pinecone(api_key=api_key)
        existing = {index.name: index for index in self._pc.list_indexes()}
        if index_name not in existing:
            if dimension is None:
                raise ValueError(
                    f"Pinecone index {index_name!r} does not exist; pass "
                    "dimension= (the embedder's output size) to create it")
            self._pc.create_index(
                name=index_name, dimension=dimension, metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region))
        elif dimension is not None:
            actual = _field(existing[index_name], "dimension")
            if actual is not None and int(actual) != int(dimension):
                raise ValueError(
                    f"Pinecone index {index_name!r} stores "
                    f"{int(actual)}-dimensional vectors, but this embedder "
                    f"produces {int(dimension)}. An index holds one model's "
                    f"vectors only — they are not comparable across models. "
                    f"Use a separate index for this model, e.g. "
                    f"index_name='{index_name}-{int(dimension)}', or keep "
                    f"this index and match its size with "
                    f"dim={int(actual)} on the embedder.")
        self._index = self._pc.Index(index_name)

    def _retry(self, operation: Callable, description: str):
        """Run ``operation``, retrying transient network failures
        with exponential backoff"""
        delay = self.retry_backoff
        for attempt in range(1, self.max_retries + 1):
            try:
                return operation()
            except Exception as exc:
                if attempt == self.max_retries or not _is_transient(exc):
                    if _is_transient(exc):
                        raise ConnectionError(
                            f"Pinecone {description} failed after "
                            f"{self.max_retries} attempts: {exc}. Already "
                            "stored vectors are kept — re-running resumes "
                            "from where it stopped."
                        ) from exc
                    raise
                time.sleep(delay)
                delay *= 2

    def add(self, chunks: List[Chunk], vectors: List[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        records = [
            {
                "id": chunk.chunk_id,
                "values": list(vector),
                "metadata": _sanitize({
                    **chunk.metadata,
                    "text": chunk.text,
                    "doc_id": chunk.doc_id,
                    "start": chunk.start,
                    "end": chunk.end,
                }),
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        for i in range(0, len(records), _UPSERT_BATCH):
            batch = records[i:i + _UPSERT_BATCH]
            self._retry(
                lambda batch=batch: self._index.upsert(
                    vectors=batch, namespace=self.namespace),
                "upsert")

    def query(self, vector: Sequence[float], k: int = 5,
              where: Optional[Dict[str, Any]] = None) -> List[IndexHit]:
        pinecone_filter = ({key: {"$eq": value}
                            for key, value in where.items()}
                           if where else None)
        response = self._retry(
            lambda: self._index.query(
                vector=list(vector), top_k=k, namespace=self.namespace,
                filter=pinecone_filter, include_metadata=True),
            "query")
        hits: List[IndexHit] = []
        for match in _field(response, "matches", []):
            metadata = dict(_field(match, "metadata", {}) or {})
            text = str(metadata.pop("text", ""))
            hits.append(IndexHit(chunk_id=_field(match, "id"),
                                 score=float(_field(match, "score", 0.0)),
                                 text=text, metadata=metadata))
        return hits

    def count(self) -> int:
        stats = self._retry(self._index.describe_index_stats, "stats")
        namespaces = _field(stats, "namespaces", {}) or {}
        if self.namespace or namespaces:
            entry = (namespaces.get(self.namespace or "")
                     if isinstance(namespaces, dict)
                     else _field(namespaces, self.namespace or ""))
            return int(_field(entry, "vector_count", 0)) if entry else 0
        return int(_field(stats, "total_vector_count", 0))

    def _list_ids(self):
        """Yield pages of plain id strings from ``Index.list``"""
        pages = self._retry(
            lambda: list(self._index.list(namespace=self.namespace)), "list")
        for page in pages:
            yield [item if isinstance(item, str)
                   else (item["id"] if isinstance(item, dict)
                         else getattr(item, "id", str(item)))
                   for item in page]

    def existing_ids(self, ids: Sequence[str]) -> set:
        """Intersect ``ids`` with everything stored in the namespace:
        one paginated id listing, far cheaper than re-embedding"""
        wanted = set(ids)
        found: set = set()
        for id_page in self._list_ids():
            found.update(set(id_page) & wanted)
        return found

    def all_chunks(self) -> List[Chunk]:
        """Full scan via paginated id listing + fetch. Intended for
        inspection and small research corpora, not production traffic"""
        chunks: List[Chunk] = []
        for ids in self._list_ids():
            for i in range(0, len(ids), _FETCH_BATCH):
                page = ids[i:i + _FETCH_BATCH]
                fetched = self._retry(
                    lambda page=page: self._index.fetch(
                        ids=page, namespace=self.namespace),
                    "fetch")
                vectors = _field(fetched, "vectors", {}) or {}
                for chunk_id, record in vectors.items():
                    metadata = dict(_field(record, "metadata", {}) or {})
                    text = str(metadata.pop("text", ""))
                    doc_id = str(metadata.pop("doc_id",
                                              chunk_id.split("::")[0]))
                    start = int(metadata.pop("start", 0))
                    end = int(metadata.pop("end", 0))
                    chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id,
                                        text=text, metadata=metadata,
                                        start=start, end=end))
        return chunks

    def clear(self) -> None:
        self._index.delete(delete_all=True, namespace=self.namespace)
