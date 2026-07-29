"""
Vector index backends behind a thin adapter interface.

Extension point: implement :class:`~ssaz.index.base.VectorIndex` and
register a factory to plug in any vector store:

    from ssaz import register_index_backend

    @register_index_backend("qdrant")
    def build_qdrant(**kwargs):
        return MyQdrantIndex(**kwargs)   # implements VectorIndex

    engine = AzSearchEngine(backend="qdrant",
                            backend_options={"collection": "az-docs"})
"""

from __future__ import annotations

from typing import Callable, Optional

from ssaz.index.base import IndexHit, VectorIndex
from ssaz.index.memory import InMemoryIndex
from ssaz.registry import Registry

__all__ = [
    "IndexHit",
    "VectorIndex",
    "InMemoryIndex",
    "get_index",
    "register_index_backend",
    "available_backends",
]

_BACKENDS = Registry("index backend")


def register_index_backend(name: str,
                           factory: Optional[Callable[..., VectorIndex]] = None):
    """Register an index backend factory (usable as a decorator)."""
    return _BACKENDS.register(name, factory)


def get_index(backend: str, **kwargs) -> VectorIndex:
    """Build an index backend by name.

    Built-in names: ``memory``, ``chroma``, ``pinecone`` — plus anything
    added via :func:`register_index_backend`.
    """
    return _BACKENDS.create(backend, **kwargs)


def available_backends() -> list:
    return _BACKENDS.names()


@register_index_backend("memory")
def _memory_factory(**kwargs) -> VectorIndex:
    return InMemoryIndex(**kwargs)


@register_index_backend("chroma")
def _chroma_factory(**kwargs) -> VectorIndex:
    from ssaz.index.chroma import ChromaIndex
    return ChromaIndex(**kwargs)


@register_index_backend("pinecone")
def _pinecone_factory(**kwargs) -> VectorIndex:
    from ssaz.index.pinecone import PineconeIndex
    return PineconeIndex(**kwargs)
