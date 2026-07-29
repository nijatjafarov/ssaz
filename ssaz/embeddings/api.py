"""API-hosted embedders (included in the base install): OpenAI (the SSAZ
default), Hugging Face Inference, Qwen3-Embedding via DashScope, and
Gemini Embedding via Google AI.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

from ssaz.embeddings.base import BaseEmbedder, Vector, l2_normalize


def estimate_tokens(texts: List[str]) -> int:
    """Crude token estimate (chars/4) for pilot cost estimation"""
    return sum(len(t) for t in texts) // 4


def _require_requests():
    try:
        import requests
        return requests
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'requests' package (a core ssaz dependency) is missing. "
            "Reinstall with:  pip install ssaz"
        ) from exc


class EmbeddingAPIError(RuntimeError):
    """An embedding API rejected a request; carries the HTTP status and
    the provider's error body so failures are diagnosable"""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class _RestEmbedder(BaseEmbedder):
    """Shared robustness logic for REST embedding APIs"""

    batch_size = 16
    max_retries = 4
    max_input_chars = 16000

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        requests = _require_requests()
        delay = 2.0
        for attempt in range(self.max_retries):
            last = attempt == self.max_retries - 1
            try:
                response = requests.post(url, headers=headers, json=payload,
                                         timeout=120)
            except requests.exceptions.RequestException as exc:
                if last:
                    raise ConnectionError(
                        f"Embedding request to {url} failed after "
                        f"{self.max_retries} attempts: {exc}") from exc
                time.sleep(delay)
                delay *= 2
                continue
            if (response.status_code == 429
                    or 500 <= response.status_code < 600) and not last:
                time.sleep(delay)
                delay *= 2
                continue
            if not response.ok:
                raise EmbeddingAPIError(
                    response.status_code,
                    f"Embedding API returned {response.status_code} for "
                    f"{url}: {response.text[:500]}")
            return response.json()
        raise RuntimeError("unreachable")

    def _embed_batch(self, texts: List[str]) -> List[Vector]:  # pragma: no cover
        raise NotImplementedError

    def _prepare(self, texts: List[str]) -> List[str]:
        prepared = []
        for text in texts:
            text = text.strip() or " "
            if self.max_input_chars and len(text) > self.max_input_chars:
                text = text[:self.max_input_chars]
            prepared.append(text)
        return prepared

    def _embed_bisect(self, texts: List[str]) -> List[Vector]:
        try:
            return self._embed_batch(texts)
        except EmbeddingAPIError as exc:
            if len(texts) > 1 and exc.status_code in (400, 413):
                middle = len(texts) // 2
                return (self._embed_bisect(texts[:middle])
                        + self._embed_bisect(texts[middle:]))
            raise

    def embed_documents(self, texts: List[str]) -> List[Vector]:
        texts = self._prepare(texts)
        vectors: List[Vector] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_bisect(texts[i:i + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]


class HFInferenceEmbedder(_RestEmbedder):
    """Hugging Face Inference API, the zero-cost fallback from the proposal"""

    def __init__(self, model_name: str = "BAAI/bge-m3",
                 api_key: Optional[str] = None, dim: int = 1024) -> None:
        self.name = f"hf:{model_name}"
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("HF_TOKEN", "")
        self.dim = dim
        if not self.api_key:
            raise ValueError("Set HF_TOKEN or pass api_key=")

    def _embed_batch(self, texts: List[str]) -> List[Vector]:
        data = self._post(
            f"https://router.huggingface.co/hf-inference/models/"
            f"{self.model_name}/pipeline/feature-extraction",
            {"Authorization": f"Bearer {self.api_key}"},
            {"inputs": texts},
        )
        return [l2_normalize(v) for v in data]


class OpenAIEmbedder(_RestEmbedder):
    """OpenAI embeddings API, the SSAZ default embedder"""

    batch_size = 64

    def __init__(self, model_name: str = "text-embedding-3-small",
                 api_key: Optional[str] = None, dim: int = 1536) -> None:
        self.name = f"openai:{model_name}"
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.dim = dim
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY or pass api_key=")

    def _embed_batch(self, texts: List[str]) -> List[Vector]:
        data = self._post(
            "https://api.openai.com/v1/embeddings",
            {"Authorization": f"Bearer {self.api_key}"},
            {"model": self.model_name, "input": texts,
             "dimensions": self.dim, "encoding_format": "float"},
        )
        items = sorted(data["data"], key=lambda d: d["index"])
        return [l2_normalize(item["embedding"]) for item in items]


class Qwen3Embedder(_RestEmbedder):
    """Qwen3-Embedding via the DashScope OpenAI-compatible endpoint"""

    def __init__(self, model_name: str = "text-embedding-v4",
                 api_key: Optional[str] = None, dim: int = 1024) -> None:
        self.name = f"qwen3:{model_name}"
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.dim = dim
        if not self.api_key:
            raise ValueError("Set DASHSCOPE_API_KEY or pass api_key=")

    def _embed_batch(self, texts: List[str]) -> List[Vector]:
        data = self._post(
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/embeddings",
            {"Authorization": f"Bearer {self.api_key}"},
            {"model": self.model_name, "input": texts,
             "dimensions": self.dim, "encoding_format": "float"},
        )
        items = sorted(data["data"], key=lambda d: d["index"])
        return [l2_normalize(item["embedding"]) for item in items]


class GeminiEmbedder(_RestEmbedder):
    """Gemini Embedding via the Google AI REST API"""

    batch_size = 32

    def __init__(self, model_name: str = "gemini-embedding-2",
                 api_key: Optional[str] = None, dim: int = 768) -> None:
        self.name = f"gemini:{model_name}"
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.dim = dim
        if not self.api_key:
            raise ValueError("Set GEMINI_API_KEY or pass api_key=")

    def _embed(self, texts: List[str], task_type: str) -> List[Vector]:
        data = self._post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:batchEmbedContents?key={self.api_key}",
            {"Content-Type": "application/json"},
            {"requests": [
                {"model": f"models/{self.model_name}",
                 "content": {"parts": [{"text": t}]},
                 "taskType": task_type,
                 "outputDimensionality": self.dim}
                for t in texts
            ]},
        )
        return [l2_normalize(e["values"]) for e in data["embeddings"]]

    def _embed_batch(self, texts: List[str]) -> List[Vector]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> Vector:
        return self._embed([text], "RETRIEVAL_QUERY")[0]