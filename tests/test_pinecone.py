"""Tests for the Pinecone backend driven by the Gemini embedder"""

import sys
import types

import pytest

from ssaz.chunking.base import Chunk
from ssaz.embeddings.api import GeminiEmbedder

from conftest import FakeEmbedder

GEMINI_MODEL = "gemini-embedding-002"
DIM = 256


# -- pinecone SDK stub ------------------------------------------------------

class _Model:
    """Stand-in for the SDK's response models: attribute access only,
    no dict interface — the shape that broke a naive adapter."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class _StubIndexClient:
    """In-memory imitation of pinecone.Index that mirrors the real
    SDK's *object* responses (ListItem, QueryResponse, FetchResponse)
    rather than plain dicts."""

    def __init__(self):
        self._records = {}  # namespace -> {id: (values, metadata)}

    def _ns(self, namespace):
        return self._records.setdefault(namespace or "", {})

    def upsert(self, vectors, namespace=""):
        for record in vectors:
            self._ns(namespace)[record["id"]] = (record["values"],
                                                 record["metadata"])

    def query(self, vector, top_k, namespace="", filter=None,
              include_metadata=True):
        matches = []
        for record_id, (values, metadata) in self._ns(namespace).items():
            if filter and any(metadata.get(key) != spec["$eq"]
                              for key, spec in filter.items()):
                continue
            score = sum(a * b for a, b in zip(vector, values))
            matches.append(_Model(id=record_id, score=score,
                                  metadata=dict(metadata)))
        matches.sort(key=lambda m: m.score, reverse=True)
        return _Model(matches=matches[:top_k])

    def describe_index_stats(self):
        return _Model(
            namespaces={ns: _Model(vector_count=len(records))
                        for ns, records in self._records.items()},
            total_vector_count=sum(len(r) for r in self._records.values()),
        )

    def list(self, namespace=""):
        # The real SDK yields pages of ListItem objects, not strings.
        ids = list(self._ns(namespace))
        if ids:
            yield [_Model(id=i) for i in ids]

    def fetch(self, ids, namespace=""):
        records = self._ns(namespace)
        return _Model(vectors={i: _Model(metadata=dict(records[i][1]))
                               for i in ids if i in records})

    def delete(self, delete_all=False, namespace=""):
        if delete_all:
            self._ns(namespace).clear()


class _StubPinecone:
    shared_indexes = {}
    created = []  # (name, dimension) pairs, for assertions

    def __init__(self, api_key):
        self.api_key = api_key

    def list_indexes(self):
        return [types.SimpleNamespace(name=name)
                for name in self.shared_indexes]

    def create_index(self, name, dimension, metric, spec):
        type(self).created.append((name, dimension))
        self.shared_indexes[name] = _StubIndexClient()

    def Index(self, name):
        return self.shared_indexes[name]


@pytest.fixture
def stub_pinecone(monkeypatch):
    _StubPinecone.shared_indexes = {}
    _StubPinecone.created = []
    monkeypatch.setitem(sys.modules, "pinecone", types.SimpleNamespace(
        Pinecone=_StubPinecone,
        ServerlessSpec=lambda cloud, region: (cloud, region)))
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    return _StubPinecone


# -- gemini embedder with mocked REST layer ---------------------------------

@pytest.fixture
def gemini_embedder(monkeypatch):
    """Real GeminiEmbedder ('Gemini Embedding 2') with only the HTTP call
    replaced; captured requests are exposed as ``embedder.calls``."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    embedder = GeminiEmbedder(model_name=GEMINI_MODEL, dim=DIM)
    hasher = FakeEmbedder(dim=DIM)
    calls = []

    def fake_post(url, headers, payload):
        calls.append({"url": url, "payload": payload})
        vectors = [hasher._embed(req["content"]["parts"][0]["text"])
                   for req in payload["requests"]]
        return {"embeddings": [{"values": v} for v in vectors]}

    monkeypatch.setattr(embedder, "_post", fake_post)
    embedder.calls = calls
    return embedder


class TestGeminiEmbedder:
    def test_documents_use_retrieval_document_task(self, gemini_embedder):
        vectors = gemini_embedder.embed_documents(["Sənəd mətni."])
        assert len(vectors) == 1 and len(vectors[0]) == DIM
        request = gemini_embedder.calls[-1]["payload"]["requests"][0]
        assert request["taskType"] == "RETRIEVAL_DOCUMENT"
        assert request["model"] == f"models/{GEMINI_MODEL}"
        assert request["outputDimensionality"] == DIM

    def test_query_uses_retrieval_query_task(self, gemini_embedder):
        gemini_embedder.embed_query("sorğu mətni")
        request = gemini_embedder.calls[-1]["payload"]["requests"][0]
        assert request["taskType"] == "RETRIEVAL_QUERY"

    def test_model_name_in_endpoint_url(self, gemini_embedder):
        gemini_embedder.embed_query("sorğu")
        assert f"{GEMINI_MODEL}:batchEmbedContents" in \
            gemini_embedder.calls[-1]["url"]

    def test_batching(self, gemini_embedder):
        texts = [f"Mətn {i}" for i in range(70)]
        vectors = gemini_embedder.embed_documents(texts)
        assert len(vectors) == 70
        # batch_size=32 -> 32 + 32 + 6
        sizes = [len(c["payload"]["requests"]) for c in gemini_embedder.calls]
        assert sizes == [32, 32, 6]


# Pinecone adapter driven by gemini vectors

def _build_index(stub_pinecone):
    from ssaz.index.pinecone import PineconeIndex
    return PineconeIndex(index_name="ssaz-test", dimension=DIM)


def _indexed_chunks(index, embedder):
    chunks = [
        Chunk(chunk_id="d0::0000", doc_id="d0",
              text="Mülkiyyət hüququ konstitusiya ilə qorunur.",
              metadata={"domain": "legal", "position": 0}),
        Chunk(chunk_id="d1::0000", doc_id="d1",
              text="Bakı Xəzər dənizinin sahilində yerləşir.",
              metadata={"domain": "encyclopedic", "position": 0}),
    ]
    index.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    return chunks


class TestPineconeIndex:
    def test_missing_index_requires_dimension(self, stub_pinecone):
        from ssaz.index.pinecone import PineconeIndex
        with pytest.raises(ValueError, match="dimension"):
            PineconeIndex(index_name="absent")

    def test_add_and_count(self, stub_pinecone, gemini_embedder):
        index = _build_index(stub_pinecone)
        _indexed_chunks(index, gemini_embedder)
        assert index.count() == 2

    def test_query_returns_text_and_metadata(self, stub_pinecone,
                                             gemini_embedder):
        index = _build_index(stub_pinecone)
        _indexed_chunks(index, gemini_embedder)
        hits = index.query(gemini_embedder.embed_query("mülkiyyət hüququ"),
                           k=2)
        assert hits[0].chunk_id == "d0::0000"
        assert "Mülkiyyət" in hits[0].text
        assert hits[0].metadata["domain"] == "legal"
        assert "text" not in hits[0].metadata

    def test_where_filter_translated_to_eq(self, stub_pinecone,
                                           gemini_embedder):
        index = _build_index(stub_pinecone)
        _indexed_chunks(index, gemini_embedder)
        hits = index.query(gemini_embedder.embed_query("Azərbaycan"),
                           k=5, where={"domain": "encyclopedic"})
        assert hits
        assert all(h.metadata["domain"] == "encyclopedic" for h in hits)

    def test_all_chunks_roundtrip(self, stub_pinecone, gemini_embedder):
        index = _build_index(stub_pinecone)
        _indexed_chunks(index, gemini_embedder)
        chunks = index.all_chunks()
        assert {c.chunk_id for c in chunks} == {"d0::0000", "d1::0000"}
        by_id = {c.chunk_id: c for c in chunks}
        assert by_id["d0::0000"].doc_id == "d0"
        assert "Mülkiyyət" in by_id["d0::0000"].text

    def test_existing_ids(self, stub_pinecone, gemini_embedder):
        index = _build_index(stub_pinecone)
        _indexed_chunks(index, gemini_embedder)
        found = index.existing_ids(["d0::0000", "d1::0000", "absent::0000"])
        assert found == {"d0::0000", "d1::0000"}

    @pytest.mark.parametrize("page", [
        ["a", "b"],
        [_Model(id="a"), _Model(id="b")],
        [{"id": "a"}, {"id": "b"}],
    ])
    def test_list_id_shapes_normalized(self, stub_pinecone, page,
                                       monkeypatch):
        index = _build_index(stub_pinecone)
        monkeypatch.setattr(index._index, "list",
                            lambda namespace="": iter([page]))
        assert index.existing_ids(["a", "z"]) == {"a"}

    def test_transient_upsert_error_retried(self, stub_pinecone,
                                            gemini_embedder, monkeypatch):
        index = _build_index(stub_pinecone)
        index.retry_backoff = 0.0
        real_upsert = index._index.upsert
        attempts = {"n": 0}

        def flaky_upsert(**kwargs):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("[Errno 11001] getaddrinfo failed")
            return real_upsert(**kwargs)

        monkeypatch.setattr(index._index, "upsert", flaky_upsert)
        _indexed_chunks(index, gemini_embedder)
        assert attempts["n"] == 3
        assert index.count() == 2

    def test_permanent_error_not_retried(self, stub_pinecone,
                                         gemini_embedder, monkeypatch):
        index = _build_index(stub_pinecone)
        index.retry_backoff = 0.0
        attempts = {"n": 0}

        def bad_key(**kwargs):
            attempts["n"] += 1
            raise RuntimeError("401 Unauthorized: invalid API key")

        monkeypatch.setattr(index._index, "upsert", bad_key)
        with pytest.raises(RuntimeError, match="Unauthorized"):
            _indexed_chunks(index, gemini_embedder)
        assert attempts["n"] == 1

    def test_exhausted_retries_mention_resume(self, stub_pinecone,
                                              gemini_embedder, monkeypatch):
        index = _build_index(stub_pinecone)
        index.retry_backoff = 0.0
        index.max_retries = 2

        def always_down(**kwargs):
            raise RuntimeError("Connection refused")

        monkeypatch.setattr(index._index, "upsert", always_down)
        with pytest.raises(ConnectionError, match="re-running resumes"):
            _indexed_chunks(index, gemini_embedder)

    def test_clear(self, stub_pinecone, gemini_embedder):
        index = _build_index(stub_pinecone)
        _indexed_chunks(index, gemini_embedder)
        index.clear()
        assert index.count() == 0


# end-to-end: engine + gemini + pinecone

class TestEngineGeminiPinecone:
    def test_full_pipeline(self, stub_pinecone, gemini_embedder):
        from ssaz import AzSearchEngine, Document

        engine = AzSearchEngine(
            embedder=gemini_embedder,
            backend="pinecone",
            backend_options={"index_name": "ssaz-e2e"},
        )
        assert ("ssaz-e2e", DIM) in stub_pinecone.created

        engine.add_documents([
            Document(doc_id="law-mulkiyyat", domain="legal",
                     text=("Maddə 13. Mülkiyyət\n"
                           "Mülkiyyət toxunulmazdır və dövlət tərəfindən "
                           "müdafiə olunur.")),
            Document(doc_id="news-metro", domain="news",
                     text=("Bakı, 15 iyun (AZƏRTAC)\n"
                           "Paytaxtda yeni metro stansiyası istifadəyə "
                           "verilib.")),
        ])
        assert engine.count() >= 2

        results = engine.search("mülkiyyət toxunulmazdır", k=2)
        assert results[0].doc_id == "law-mulkiyyat"
        last = gemini_embedder.calls[-1]["payload"]["requests"][0]
        assert last["taskType"] == "RETRIEVAL_QUERY"
