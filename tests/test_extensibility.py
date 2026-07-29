"""Tests for every SSAZ extension point: registries (embedder, index
backend, chunker, corpus format), loader field maps, custom structural
marker rules, and custom normalizer steps."""

import json
import re

import pytest

from ssaz import (
    AzSearchEngine,
    AzNormalizer,
    Document,
    TwoPassChunker,
    get_chunker,
    get_embedder,
    get_index,
    load_documents,
    load_queries,
    register_chunker,
    register_corpus_format,
    register_index_backend,
)
from ssaz.chunking import BaseChunker, Chunk
from ssaz.index.memory import InMemoryIndex


class TestIndexBackendRegistry:
    def test_register_custom_backend(self):
        @register_index_backend("test-memory")
        def build(**kwargs):
            return InMemoryIndex(**kwargs)

        index = get_index("test-memory")
        assert isinstance(index, InMemoryIndex)

    def test_engine_uses_custom_backend(self):
        register_index_backend("test-memory2", lambda **kw: InMemoryIndex())
        engine = AzSearchEngine(embedder="openai",
                                backend="test-memory2")
        engine.add_texts(["Sınaq mətni."], domain="news")
        assert engine.count() == 1


class TestChunkerRegistry:
    def test_builtin_names(self):
        assert isinstance(get_chunker("two-pass"), TwoPassChunker)

    def test_register_custom_chunker(self):
        class WholeDocChunker(BaseChunker):
            def chunk(self, document):
                return [Chunk(chunk_id=f"{document.doc_id}::0000",
                              doc_id=document.doc_id,
                              text=document.text,
                              metadata={"domain": document.domain})]

        register_chunker("whole-doc", WholeDocChunker)
        engine = AzSearchEngine(embedder="openai", backend="memory",
                                chunker="whole-doc")
        engine.add_documents([Document(
            doc_id="d1", domain="legal",
            text="Maddə 1. Birinci.\n\nMaddə 2. İkinci.")])
        assert engine.count() == 1


class TestCorpusFormatRegistry:
    def test_register_csv_reader(self, tmp_path):
        @register_corpus_format(".csv")
        def read_csv(path):
            import csv
            with open(path, encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f))

        path = tmp_path / "corpus.csv"
        path.write_text("doc_id,text,domain\n"
                        "d1,Birinci mətn.,news\n"
                        "d2,İkinci mətn.,legal\n", encoding="utf-8")
        docs = load_documents(path)
        assert [d.doc_id for d in docs] == ["d1", "d2"]
        assert docs[1].domain == "legal"

    def test_unregistered_suffix_raises(self, tmp_path):
        path = tmp_path / "corpus.xml"
        path.write_text("<docs/>", encoding="utf-8")
        with pytest.raises(ValueError, match="register_corpus_format"):
            load_documents(path)


class TestLoaderFieldMaps:
    def test_document_field_map(self, tmp_path):
        path = tmp_path / "corpus.json"
        path.write_text(json.dumps([
            {"article_no": "A-17", "body": "Qərar mətni.",
             "issued_on": "2026-01-05", "court": "Ali Məhkəmə"},
        ]), encoding="utf-8")
        docs = load_documents(path, default_domain="legal",
                              field_map={"doc_id": "article_no",
                                         "text": "body",
                                         "date": "issued_on"})
        assert docs[0].doc_id == "A-17"
        assert docs[0].text == "Qərar mətni."
        assert docs[0].date == "2026-01-05"
        assert docs[0].metadata["court"] == "Ali Məhkəmə"

    def test_query_field_map(self, tmp_path):
        path = tmp_path / "golden.json"
        path.write_text(json.dumps([
            {"sual": "Sual mətni?", "gold_docs": ["A-17"]},
        ]), encoding="utf-8")
        queries = load_queries(path, field_map={"query": "sual",
                                                "relevant_ids": "gold_docs"})
        assert queries[0].query == "Sual mətni?"
        assert queries[0].relevant_ids == {"A-17"}

    def test_unknown_canonical_field_raises(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(KeyError, match="Options"):
            load_documents(path, field_map={"body_text": "x"})


class TestStructuralRules:
    def test_custom_marker_rule(self):
        def qerar_rule(line, domain):
            m = re.match(r"^\s*QƏRAR\s+№\s*(\S+)", line)
            if m:
                return "decision", line.strip(), {"decision_no": m.group(1)}
            return None

        chunker = TwoPassChunker(structural_rules=[qerar_rule],
                                 min_section_size=10)
        doc = Document(
            doc_id="court1", domain="legal",
            text=("QƏRAR № 2026-14\n"
                  "Məhkəmə qərara aldı ki, iddia təmin edilsin.\n\n"
                  "QƏRAR № 2026-15\n"
                  "İkinci qərarın mətni burada davam edir."))
        chunks = chunker.chunk(doc)
        decision_nos = [c.metadata.get("decision_no") for c in chunks
                        if c.metadata.get("marker_type") == "decision"]
        assert decision_nos == ["2026-14", "2026-15"]

    def test_custom_rule_runs_before_builtins(self):
        def article_override(line, domain):
            if line.startswith("Maddə"):
                return "custom", line.strip(), {"overridden": True}
            return None

        chunker = TwoPassChunker(structural_rules=[article_override],
                                 min_section_size=10)
        doc = Document(doc_id="d", domain="legal",
                       text="Maddə 1. Mətn\nBu maddənin məzmunu buradadır.")
        chunks = chunker.chunk(doc)
        assert chunks[0].metadata["marker_type"] == "custom"
        assert chunks[0].metadata["overridden"] is True


class TestNormalizerExtraSteps:
    def test_extra_steps_applied_in_order(self):
        norm = AzNormalizer(extra_steps=(
            lambda t: t.replace("[SƏHV]", ""),
            lambda t: t.strip(),
        ))
        assert norm("[SƏHV] Təmiz mətn") == "Təmiz mətn"

    def test_engine_accepts_plain_callable_normalizer(self):
        engine = AzSearchEngine(embedder="openai", backend="memory",
                                normalizer=lambda t: t.upper())
        engine.add_texts(["salam dünya"], domain="news")
        chunk = engine.index.all_chunks()[0]
        assert chunk.text == "SALAM DÜNYA"
