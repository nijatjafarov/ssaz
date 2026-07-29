"""End-to-end tests for AzSearchEngine using the deterministic
FakeEmbedder from conftest and the in-memory index, the exact code
paths the neural embedders and server backends share."""

import json

from ssaz import AzSearchEngine, Document
from ssaz.evaluation import EvalQuery, EvaluationHarness
from ssaz.metadata import MetadataEnricher


CORPUS = [
    Document(
        doc_id="law-mulkiyyat",
        text=("Maddə 13. Mülkiyyət\n"
              "Azərbaycan Respublikasında mülkiyyət toxunulmazdır və dövlət "
              "tərəfindən müdafiə olunur. Mülkiyyət dövlət mülkiyyəti, xüsusi "
              "mülkiyyət və bələdiyyə mülkiyyəti növündə ola bilər."),
        domain="legal",
    ),
    Document(
        doc_id="news-metro",
        text=("Bakı, 15 iyun (AZƏRTAC)\n"
              "Paytaxtda yeni metro stansiyası istifadəyə verilib. Stansiya "
              "gündəlik on minlərlə sərnişinə xidmət göstərəcək."),
        domain="news",
    ),
    Document(
        doc_id="wiki-xezer",
        text=("Xəzər dənizi\n\n== Coğrafiya ==\n"
              "Xəzər dənizi dünyanın ən böyük qapalı su hövzəsidir və "
              "Azərbaycanın şərq sahillərini yuyur."),
        domain="encyclopedic",
    ),
]


def build_engine():
    engine = AzSearchEngine(embedder="openai", backend="memory")
    engine.add_documents([Document(doc_id=d.doc_id, text=d.text,
                                   domain=d.domain) for d in CORPUS])
    return engine


class TestIndexing:
    def test_chunks_indexed(self):
        engine = build_engine()
        assert engine.count() >= 3

    def test_metadata_enrichment_applied(self):
        engine = build_engine()
        chunks = engine.index.all_chunks()
        legal = [c for c in chunks if c.doc_id == "law-mulkiyyat"]
        assert legal
        assert legal[0].metadata["domain"] == "legal"
        assert "position" in legal[0].metadata
        heading = " ".join(c.metadata.get("section_heading", "")
                           for c in legal)
        assert "Maddə 13" in heading

    def test_news_date_extracted(self):
        engine = AzSearchEngine(embedder="openai", backend="memory",
                                enricher=MetadataEnricher(default_year=2026))
        engine.add_documents([Document(doc_id="n", text=CORPUS[1].text,
                                       domain="news")])
        chunk = engine.index.all_chunks()[0]
        assert chunk.metadata.get("date") == "2026-06-15"

    def test_cyrillic_document_normalized_before_indexing(self):
        engine = AzSearchEngine(embedder="openai", backend="memory")
        engine.add_documents([Document(doc_id="cyr",
                                       text="Азәрбајҹан дили һаггында мәтн.",
                                       domain="general")])
        chunk = engine.index.all_chunks()[0]
        assert "Azərbaycan dili" in chunk.text


class TestSearch:
    def test_search_returns_ranked_results(self):
        engine = build_engine()
        results = engine.search("mülkiyyət hüququ", k=3)
        assert results
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_relevant_doc_found(self):
        engine = build_engine()
        results = engine.search("mülkiyyət toxunulmazdır", k=3)
        assert results[0].doc_id == "law-mulkiyyat"

    def test_domain_filter(self):
        engine = build_engine()
        results = engine.search("Azərbaycan", k=5, where={"domain": "news"})
        assert results
        assert all(r.metadata["domain"] == "news" for r in results)

    def test_embedding_text_not_leaked(self):
        engine = build_engine()
        for result in engine.search("mülkiyyət", k=3):
            assert "embedding_text" not in result.metadata


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        engine = build_engine()
        expected = [r.chunk_id for r in engine.search("mülkiyyət", k=3)]
        path = tmp_path / "index.json"
        engine.save(path)
        fresh = AzSearchEngine(embedder="openai", backend="memory")
        fresh.load(path)
        assert fresh.count() == engine.count()
        assert [r.chunk_id for r in fresh.search("mülkiyyət", k=3)] == expected


class TestEvaluationHarness:
    def test_report_shape(self):
        engine = build_engine()
        harness = EvaluationHarness(engine)
        queries = [
            EvalQuery(query="mülkiyyət toxunulmazdır",
                      relevant_ids={"law-mulkiyyat"}, domain="legal"),
            EvalQuery(query="yeni metro stansiyası",
                      relevant_ids={"news-metro"}, domain="news"),
            EvalQuery(query="Xəzər dənizi hövzəsi",
                      relevant_ids={"wiki-xezer"}, domain="encyclopedic"),
        ]
        report = harness.evaluate(queries, k=5)
        assert report.n_queries == 3
        assert set(report.per_domain) == {"legal", "news", "encyclopedic"}
        assert 0.0 <= report.aggregate["mrr"] <= 1.0
        assert report.aggregate["recall"] == 1.0
        assert "| all |" in report.to_markdown()

    def test_multi_chunk_doc_metrics_bounded(self):
        engine = AzSearchEngine(embedder="openai", backend="memory")
        engine.add_documents([Document(
            doc_id="law-multi",
            domain="legal",
            text=("Maddə 1. Mülkiyyət hüququ\nMülkiyyət toxunulmazdır.\n\n"
                  "Maddə 2. Mülkiyyət növləri\nMülkiyyət üç növdə olur: "
                  "dövlət, xüsusi və bələdiyyə mülkiyyəti."),
        )])
        harness = EvaluationHarness(engine)
        report = harness.evaluate(
            [EvalQuery(query="mülkiyyət", relevant_ids={"law-multi"},
                       domain="legal")],
            k=5)
        assert report.aggregate["recall"] <= 1.0
        assert report.aggregate["ndcg"] <= 1.0

    def test_load_queries(self, tmp_path):
        path = tmp_path / "queries.json"
        path.write_text(json.dumps([
            {"query": "test", "relevant_ids": ["d1"], "domain": "legal"},
        ]), encoding="utf-8")
        queries = EvaluationHarness.load_queries(path)
        assert queries[0].relevant_ids == {"d1"}
