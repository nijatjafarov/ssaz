"""Two-pass structure-aware chunking for Azerbaijani documents.

Extension points:

* subclass :class:`BaseChunker` and register it to select it by name::

    from ssaz import register_chunker

    @register_chunker("sliding-window")
    def build(**kwargs):
        return SlidingWindowChunker(**kwargs)   # implements BaseChunker

    engine = AzSearchEngine(chunker="sliding-window")
"""

from __future__ import annotations

from typing import Callable, Optional

from ssaz.chunking.base import Chunk, BaseChunker
from ssaz.chunking.recursive import RecursiveChunker
from ssaz.chunking.structure import Section, StructuralChunker
from ssaz.chunking.two_pass import TwoPassChunker
from ssaz.registry import Registry

__all__ = [
    "BaseChunker",
    "Chunk",
    "RecursiveChunker",
    "Section",
    "StructuralChunker",
    "TwoPassChunker",
    "get_chunker",
    "register_chunker",
    "available_chunkers",
]

_CHUNKERS = Registry("chunker")


def register_chunker(name: str,
                     factory: Optional[Callable[..., BaseChunker]] = None):
    """Register a chunker factory (usable as a decorator)"""
    return _CHUNKERS.register(name, factory)


def get_chunker(name: str, **kwargs) -> BaseChunker:
    """Build a chunker by name: ``two-pass`` (default), ``recursive``,
    ``structural``: plus anything added via :func:`register_chunker`.

    All strategies implement the same :class:`BaseChunker` interface
    (``chunk(document) -> List[Chunk]``), so they are interchangeable
    everywhere a chunker is accepted."""
    return _CHUNKERS.create(name, **kwargs)


def available_chunkers() -> list:
    return _CHUNKERS.names()


register_chunker("two-pass", TwoPassChunker)
register_chunker("recursive", RecursiveChunker)
register_chunker("structural", StructuralChunker)
