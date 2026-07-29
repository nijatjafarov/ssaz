"""Local neural embedders via sentence-transformers (optional extra ``dense``).

Presets for the two models:

* ``bge-m3``: BAAI/bge-m3 (Chen et al., 2024)
* ``arctic``: Snowflake/snowflake-arctic-embed-l-v2.0
  requires the ``query:`` instruction prefix on queries
"""

from __future__ import annotations

from typing import List, Optional

from ssaz.embeddings.base import BaseEmbedder, Vector

PRESETS = {
    "bge-m3": {
        "model_name": "BAAI/bge-m3",
        "query_prefix": "",
    },
    "arctic": {
        "model_name": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "query_prefix": "query: ",
    },
}


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Local neural embedders (bge-m3, arctic) need the heavy "
                "'dense' extra (sentence-transformers + torch, ~2 GB). "
                "Install with:  pip install ssaz[dense]\n"
                "If you'd rather avoid large local models, use an API "
                "embedder (already included in the base install): pass "
                "embedder='openai', 'gemini', 'qwen3', or 'hf-api'."
            ) from exc
        self.model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()

    @classmethod
    def from_preset(cls, preset: str, **kwargs) -> "SentenceTransformerEmbedder":
        if preset not in PRESETS:
            raise KeyError(f"Unknown preset {preset!r}; options: {sorted(PRESETS)}")
        return cls(**{**PRESETS[preset], **kwargs})

    def embed_documents(self, texts: List[str]) -> List[Vector]:
        vectors = self.model.encode(
            texts, batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> Vector:
        vector = self.model.encode(
            [self.query_prefix + text],
            normalize_embeddings=True, show_progress_bar=False,
        )[0]
        return vector.tolist()
