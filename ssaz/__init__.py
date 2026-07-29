"""SAZ — Semantic search infrastructure for the Azerbaijani language.

SAZ provides a configurable retrieval pipeline in which every stage —
chunker, normalizer, embedder, indexer, and retriever — exposes
language-aware defaults calibrated for Azerbaijani, unified behind the
:class:`~ssaz.engine.AzSearchEngine` class and a companion CLI.

Quickstart::

    from ssaz import AzSearchEngine, Document

    engine = AzSearchEngine()          # bge-m3 embedder + in-memory index
    engine.add_documents([
        Document(doc_id="c1", text="Maddə 1. Azərbaycan dövləti ...",
                 domain="legal"),
    ])
    results = engine.search("dövlətin əsası nədir?", k=5)

Every component family is extensible through a registry — see
``register_embedder``, ``register_index_backend``, ``register_chunker``,
and ``register_corpus_format``, plus ``field_map`` on the data loaders,
``structural_rules`` on the chunker, and ``extra_steps`` on the
normalizer.
"""

from ssaz.chunking import (
    Chunk,
    RecursiveChunker,
    StructuralChunker,
    TwoPassChunker,
    get_chunker,
    register_chunker,
)
from ssaz.data import (
    load_documents,
    load_queries,
    register_corpus_format,
)
from ssaz.display import ResultPresenter, show_results
from ssaz.documents import Document
from ssaz.embeddings import BaseEmbedder, get_embedder, register_embedder
from ssaz.engine import AzSearchEngine, SearchResult
from ssaz.export import export_data, register_export_format
from ssaz.evaluation import EvalQuery, EvaluationHarness
from ssaz.index import VectorIndex, get_index, register_index_backend
from ssaz.metadata import MetadataEnricher
from ssaz.progress import ProgressBar, track
from ssaz.text import AzNormalizer, cyrillic_to_latin, split_sentences

__version__ = "0.1.0"

__all__ = [
    "AzSearchEngine",
    "AzNormalizer",
    "BaseEmbedder",
    "Chunk",
    "Document",
    "EvalQuery",
    "EvaluationHarness",
    "MetadataEnricher",
    "ProgressBar",
    "RecursiveChunker",
    "ResultPresenter",
    "SearchResult",
    "StructuralChunker",
    "TwoPassChunker",
    "VectorIndex",
    "cyrillic_to_latin",
    "export_data",
    "get_chunker",
    "get_embedder",
    "get_index",
    "load_documents",
    "load_queries",
    "register_chunker",
    "register_corpus_format",
    "register_embedder",
    "register_export_format",
    "register_index_backend",
    "show_results",
    "split_sentences",
    "track",
    "__version__",
]
