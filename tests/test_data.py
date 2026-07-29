"""Tests for the default data loaders in ssaz.data"""

import json

import pytest

from ssaz.data import load_documents, load_queries
from ssaz.documents import Document


class TestLoadDocuments:
    def test_json_array_with_aliases(self, tmp_path):
        path = tmp_path / "corpus.json"
        path.write_text(json.dumps([
            {"doc_id": "d1", "domain": "legal", "title": "Qanun",
             "published_at": "2026-04-03", "source": "eqanun",
             "url": "https://e-qanun.az/x", "text": "Maddə 1. Mətn."},
            {"id": "d2", "content": "Sadə mətn."},
        ]), encoding="utf-8")
        docs = load_documents(path)
        assert [d.doc_id for d in docs] == ["d1", "d2"]
        assert docs[0].date == "2026-04-03"
        assert docs[0].metadata["source"] == "eqanun"
        assert docs[0].metadata["url"].startswith("https://")
        assert docs[1].text == "Sadə mətn."
        assert docs[1].domain == "general"

    def test_jsonl(self, tmp_path):
        path = tmp_path / "corpus.jsonl"
        path.write_text('{"doc_id": "a", "text": "Birinci."}\n'
                        '{"doc_id": "b", "text": "İkinci."}\n',
                        encoding="utf-8")
        docs = load_documents(path, default_domain="news")
        assert len(docs) == 2
        assert all(d.domain == "news" for d in docs)

    def test_txt_directory(self, tmp_path):
        (tmp_path / "one.txt").write_text("Mətn bir.", encoding="utf-8")
        (tmp_path / "two.txt").write_text("Mətn iki.", encoding="utf-8")
        docs = load_documents(tmp_path, default_domain="encyclopedic")
        assert {d.doc_id for d in docs} == {"one", "two"}

    def test_missing_id_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([{"text": "no id"}]), encoding="utf-8")
        with pytest.raises(ValueError, match="doc_id"):
            load_documents(path)


class TestLoadQueries:
    def _golden(self, tmp_path):
        path = tmp_path / "golden.json"
        path.write_text(json.dumps([
            {"id": "q1", "question": "Sual bir?", "answerable": True,
             "source_doc_ids": ["d1"]},
            {"id": "q2", "question": "Sual iki?", "answerable": False,
             "source_doc_ids": ["d1"]},
            {"id": "q3", "question": "Sual üç?", "answerable": True,
             "source_doc_ids": ["missing-doc"]},
        ]), encoding="utf-8")
        return path

    def test_golden_format(self, tmp_path):
        queries = load_queries(self._golden(tmp_path))
        assert [q.query_id for q in queries] == ["q1", "q3"]
        assert queries[0].relevant_ids == {"d1"}

    def test_golden_filtered_and_domain_derived(self, tmp_path):
        docs = [Document(doc_id="d1", text="x", domain="legal")]
        queries = load_queries(self._golden(tmp_path), documents=docs)
        assert [q.query_id for q in queries] == ["q1"]
        assert queries[0].domain == "legal"

    def test_simple_format(self, tmp_path):
        path = tmp_path / "queries.json"
        path.write_text(json.dumps([
            {"query": "test", "relevant_ids": ["d1"], "domain": "news"},
        ]), encoding="utf-8")
        queries = load_queries(path)
        assert queries[0].query == "test"
        assert queries[0].domain == "news"
