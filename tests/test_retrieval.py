"""Tests for the ranking metrics and the dense retriever."""

import math

from ssaz.chunking.base import Chunk
from ssaz.evaluation.metrics import mrr, ndcg_at_k, recall_at_k, reciprocal_rank
from ssaz.index.memory import InMemoryIndex
from ssaz.retrieval import DenseRetriever

from conftest import FakeEmbedder


class TestDenseRetriever:
    def _build(self):
        embedder = FakeEmbedder()
        index = InMemoryIndex()
        chunks = [
            Chunk(chunk_id="c0", doc_id="d0",
                  text="Azərbaycan tarixi qədim dövrlərə gedib çıxır.",
                  metadata={"domain": "encyclopedic"}),
            Chunk(chunk_id="c1", doc_id="d1",
                  text="Mülkiyyət hüququ konstitusiya ilə qorunur.",
                  metadata={"domain": "legal"}),
            Chunk(chunk_id="c2", doc_id="d2",
                  text="Bakı Xəzər dənizinin sahilində yerləşir.",
                  metadata={"domain": "encyclopedic"}),
        ]
        vectors = embedder.embed_documents([c.text for c in chunks])
        index.add(chunks, vectors)
        return DenseRetriever(embedder, index)

    def test_relevant_chunk_ranks_first(self):
        retriever = self._build()
        results = retriever.retrieve("mülkiyyət hüququ", k=3)
        assert results[0].chunk_id == "c1"

    def test_scores_descending(self):
        retriever = self._build()
        results = retriever.retrieve("Xəzər dənizi", k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_where_filter(self):
        retriever = self._build()
        results = retriever.retrieve("Azərbaycan", k=3,
                                     where={"domain": "legal"})
        assert results
        assert all(r.metadata["domain"] == "legal" for r in results)


class TestMetrics:
    def test_reciprocal_rank(self):
        assert reciprocal_rank(["x", "gold", "y"], {"gold"}) == 0.5
        assert reciprocal_rank(["x", "y"], {"gold"}) == 0.0

    def test_mrr(self):
        value = mrr([["gold"], ["x", "gold"]], [{"gold"}, {"gold"}])
        assert math.isclose(value, (1.0 + 0.5) / 2)

    def test_recall_at_k(self):
        assert recall_at_k(["a", "b", "c"], {"a", "z"}, k=3) == 0.5
        assert recall_at_k(["a", "z"], {"a", "z"}, k=2) == 1.0

    def test_recall_duplicates_not_double_counted(self):
        assert recall_at_k(["a", "a", "a"], {"a"}, k=3) == 1.0

    def test_ndcg_perfect(self):
        assert math.isclose(ndcg_at_k(["g1", "g2"], {"g1", "g2"}, k=2), 1.0)

    def test_ndcg_worse_when_lower(self):
        high = ndcg_at_k(["g", "x"], {"g"}, k=2)
        low = ndcg_at_k(["x", "g"], {"g"}, k=2)
        assert high > low
