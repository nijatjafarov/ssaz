"""
AzSearchEngine: the unified public API of the SSAZ library.

normalize -> chunk -> enrich -> embed -> index -> retrieve
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ssaz.chunking import BaseChunker, Chunk, TwoPassChunker, get_chunker
from ssaz.documents import Document
from ssaz.embeddings import BaseEmbedder, get_embedder
from ssaz.index import InMemoryIndex, VectorIndex, get_index
from ssaz.metadata import MetadataEnricher
from ssaz.retrieval.retriever import DenseRetriever, RetrievedChunk
from ssaz.text import AzNormalizer


@dataclass
class SearchResult:
    """A single search hit returned to the user"""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class AzSearchEngine:
    """End-to-end Azerbaijani semantic search engine.

    Args:
        embedder: Embedder instance or registry name (``openai``,
            ``bge-m3``, ``arctic``, ``hf-api``, ``qwen3``, ``gemini``).
            Defaults to ``openai`` (text-embedding-3-small, works with
            the base install, needs ``OPENAI_API_KEY``). Use ``bge-m3``
            for fully-local inference (``pip install ssaz[dense]``).
        backend: Vector index backend name (``pinecone`` (default, works
            with the base install, needs ``PINECONE_API_KEY``),
            ``memory``, ``chroma``, or any name added via
            :func:`ssaz.register_index_backend`)
        chunker: Chunker instance, registered chunker name (``two-pass``,
            ``recursive``, or any name added via
            :func:`ssaz.register_chunker`); defaults to the two-pass
            structure-aware chunker.
        normalizer: Any ``str -> str`` callable applied to documents and
            queries; defaults to :class:`AzNormalizer`.
        enricher: Metadata enricher (required stage). Any object with an
            ``enrich(document, chunks) -> chunks`` method works.
        backend_options: Extra kwargs for the index backend factory
            (e.g. ``collection_name``, ``persist_directory`` for chroma;
            ``index_name``, ``namespace``, ``dimension`` for pinecone).
        embedder_options: Extra kwargs for the embedder factory.
    """

    def __init__(
        self,
        embedder: Union[str, BaseEmbedder] = "openai",
        backend: Union[str, VectorIndex] = "pinecone",
        chunker: Union[str, BaseChunker, None] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        enricher: Optional[MetadataEnricher] = None,
        backend_options: Optional[Dict[str, Any]] = None,
        embedder_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.embedder = (embedder if isinstance(embedder, BaseEmbedder)
                         else get_embedder(embedder, **(embedder_options or {})))
        if isinstance(backend, VectorIndex):
            self.index = backend
        else:
            backend_options = dict(backend_options or {})
            if backend == "pinecone" and self.embedder.dim:
                # A new Pinecone index needs the vector dimension; take
                # it from the embedder unless the caller overrides it.
                backend_options.setdefault("dimension", self.embedder.dim)
            self.index = get_index(backend, **backend_options)
        if isinstance(chunker, str):
            chunker = get_chunker(chunker)
        self.chunker = chunker or TwoPassChunker()
        if not callable(getattr(self.chunker, "chunk", None)):
            raise TypeError(
                f"chunker must implement BaseChunker.chunk(document) -> "
                f"List[Chunk]; got {type(self.chunker).__name__}. All "
                f"built-in strategies (two-pass, recursive, structural) "
                f"share this interface.")
        self.normalizer = normalizer or AzNormalizer()
        self.enricher = enricher or MetadataEnricher()
        self._dense = DenseRetriever(self.embedder, self.index)

    # indexing

    def add_documents(self, documents: List[Document],
                      batch_size: int = 64, progress: bool = False,
                      skip_existing: bool = True) -> int:
        """Chunk, enrich, embed, and index ``documents``.

        Args:
            documents: Documents to index.
            batch_size: Chunks per embedding request batch.
            progress: Show built-in progress bars for the
                chunking and embedding phases.
            skip_existing: Skip chunks whose ids are already indexed.
                Pass ``False`` to force re-embedding.

        Returns the number of chunks newly embedded and indexed.
        """
        from ssaz.progress import ProgressBar

        chunk_bar = (ProgressBar("Chunking", len(documents))
                     if progress else None)
        all_chunks: List[Chunk] = []
        for document in documents:
            document.text = self.normalizer(document.text)
            chunks = self.chunker.chunk(document)
            chunks = self.enricher.enrich(document, chunks)
            all_chunks.extend(chunks)
            if chunk_bar:
                chunk_bar.update(extra=f"{len(all_chunks)} chunks")
        if chunk_bar:
            chunk_bar.finish()

        skipped = 0
        if skip_existing and all_chunks:
            existing = self.index.existing_ids(
                [chunk.chunk_id for chunk in all_chunks])
            if existing:
                all_chunks = [chunk for chunk in all_chunks
                              if chunk.chunk_id not in existing]
                skipped = len(existing)

        embed_bar = (ProgressBar("Embedding", len(all_chunks))
                     if progress and all_chunks else None)
        suffix = f"({skipped} already indexed)" if skipped else ""
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [c.metadata.get("embedding_text", c.text) for c in batch]
            vectors = self.embedder.embed_documents(texts)
            self.index.add(batch, vectors)
            if embed_bar:
                embed_bar.update(min(i + batch_size, len(all_chunks)),
                                 extra=suffix)
        if embed_bar:
            embed_bar.finish()
        return len(all_chunks)

    def add_texts(self, texts: List[str], domain: str = "general",
                  ids: Optional[List[str]] = None) -> int:
        """Convenience wrapper: index raw strings as documents"""
        documents = [
            Document(doc_id=(ids[i] if ids else f"doc{i:05d}"),
                     text=text, domain=domain)
            for i, text in enumerate(texts)
        ]
        return self.add_documents(documents)

    # search

    def search(self, query: str, k: int = 5,
               where: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Dense semantic search over the index.

        Args:
            query: Natural-language Azerbaijani query.
            k: Number of results.
            where: Exact-match metadata filter.
        """
        query = self.normalizer(query)
        hits = self._dense.retrieve(query, k=k, where=where)
        return [self._to_result(hit, rank) for rank, hit in enumerate(hits, 1)]

    @staticmethod
    def _to_result(hit: RetrievedChunk, rank: int) -> SearchResult:
        metadata = {key: value for key, value in hit.metadata.items()
                    if key != "embedding_text"}
        doc_id = str(metadata.get("doc_id") or hit.chunk_id.split("::")[0])
        return SearchResult(chunk_id=hit.chunk_id, doc_id=doc_id,
                            text=hit.text, score=hit.score, rank=rank,
                            metadata=metadata)

    # Persistence
    def save(self, path: Union[str, Path]) -> None:
        if not isinstance(self.index, InMemoryIndex):
            raise TypeError("save() is only supported for the memory backend; "
                            "chroma and pinecone persist server-side")
        self.index.save(path)

    def load(self, path: Union[str, Path]) -> None:
        if not isinstance(self.index, InMemoryIndex):
            raise TypeError("load() is only supported for the memory backend")
        self.index = InMemoryIndex.load(path)
        self._dense = DenseRetriever(self.embedder, self.index)

    def count(self) -> int:
        return self.index.count()

    def export_chunks(self, path: Union[str, Path],
                      format: Optional[str] = None) -> Path:
        """Export every indexed chunk"""
        from ssaz.export import export_data
        return export_data(self.index.all_chunks(), path, format=format)
