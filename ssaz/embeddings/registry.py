"""Name-based embedder factory so engine/CLI users select models by string.

Extension point: register your own embedder:

    from ssaz import register_embedder

    @register_embedder("labse")
    def build_labse(**kwargs):
        from ssaz.embeddings.sentence_transformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder("sentence-transformers/LaBSE", **kwargs)

    engine = AzSearchEngine(embedder="labse")
"""

from __future__ import annotations

from typing import Callable, Optional

from ssaz.embeddings.base import BaseEmbedder
from ssaz.registry import Registry

_EMBEDDERS = Registry("embedder")


def register_embedder(name: str,
                      factory: Optional[Callable[..., BaseEmbedder]] = None):
    """Register an embedder factory (usable as a decorator)."""
    return _EMBEDDERS.register(name, factory)


def get_embedder(name: str, **kwargs) -> BaseEmbedder:
    """Build an embedder by registry name.

    Built-in names: ``openai`` (default), ``bge-m3``, ``arctic``,
    ``hf-api``, ``qwen3``, ``gemini``, plus anything added via
    :func:`register_embedder`.
    """
    return _EMBEDDERS.create(name, **kwargs)


def available_embedders() -> list:
    return _EMBEDDERS.names()


def _st_factory(preset: str) -> Callable[..., BaseEmbedder]:
    def factory(**kwargs) -> BaseEmbedder:
        from ssaz.embeddings.sentence_transformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder.from_preset(preset, **kwargs)
    return factory


def _hf_factory(**kwargs) -> BaseEmbedder:
    from ssaz.embeddings.api import HFInferenceEmbedder
    return HFInferenceEmbedder(**kwargs)


def _qwen_factory(**kwargs) -> BaseEmbedder:
    from ssaz.embeddings.api import Qwen3Embedder
    return Qwen3Embedder(**kwargs)


def _gemini_factory(**kwargs) -> BaseEmbedder:
    from ssaz.embeddings.api import GeminiEmbedder
    return GeminiEmbedder(**kwargs)


def _openai_factory(**kwargs) -> BaseEmbedder:
    from ssaz.embeddings.api import OpenAIEmbedder
    return OpenAIEmbedder(**kwargs)


register_embedder("openai", _openai_factory)
register_embedder("bge-m3", _st_factory("bge-m3"))
register_embedder("arctic", _st_factory("arctic"))
register_embedder("hf-api", _hf_factory)
register_embedder("qwen3", _qwen_factory)
register_embedder("gemini", _gemini_factory)
