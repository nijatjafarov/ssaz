"""Shared test fixtures.

FakeEmbedder is a deterministic hashed character-n-gram embedder used
only in tests: it needs no API key, no network, and no model download,
yet exercises the exact indexing and retrieval code paths the real
embedders use, and its scores are stable across runs.

The autouse fixture below registers it under the built-in embedder
names, so tests that ask for ``embedder="openai"`` stay offline, free and
deterministic. Registry entries are restored after each test.
"""

from __future__ import annotations

import hashlib
import math
from typing import List

import pytest

from ssaz.embeddings.base import BaseEmbedder, Vector, l2_normalize
from ssaz.embeddings.registry import _EMBEDDERS, register_embedder
from ssaz.text.normalizer import azerbaijani_lower

STUBBED_EMBEDDERS = ("openai", "gemini", "qwen3", "hf-api")


class FakeEmbedder(BaseEmbedder):
    name = "fake"

    def __init__(self, dim: int = 256, **kwargs) -> None:
        self.dim = dim

    def _embed(self, text: str) -> Vector:
        vector = [0.0] * self.dim
        text = " " + azerbaijani_lower(text) + " "
        for n in (3, 4, 5):
            for i in range(max(0, len(text) - n + 1)):
                gram = text[i:i + n]
                digest = hashlib.blake2b(gram.encode("utf-8"),
                                         digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[bucket] += sign
        vector = [math.copysign(math.log1p(abs(x)), x) for x in vector]
        return l2_normalize(vector)

    def embed_documents(self, texts: List[str]) -> List[Vector]:
        return [self._embed(t) for t in texts]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture(autouse=True)
def stub_api_embedders():
    """Keep the suite offline: API embedder names build FakeEmbedder."""
    saved = dict(_EMBEDDERS._factories)
    for name in STUBBED_EMBEDDERS:
        register_embedder(name, lambda **kwargs: FakeEmbedder(**kwargs))
    yield
    _EMBEDDERS._factories.clear()
    _EMBEDDERS._factories.update(saved)
