"""Tests for the export mechanism (ssaz.export)."""

import csv
import json

import pytest

from ssaz import AzSearchEngine, Document, export_data, register_export_format
from ssaz.chunking.base import Chunk
from ssaz.engine import SearchResult
from ssaz.evaluation import EvalQuery, EvaluationHarness


def build_engine():
    engine = AzSearchEngine(embedder="openai", backend="memory")
    engine.add_documents([
        Document(doc_id="law-13", domain="legal",
                 text=("Maddə 13. Mülkiyyət\n"
                       "Mülkiyyət toxunulmazdır və dövlət tərəfindən "
                       "müdafiə olunur.")),
        Document(doc_id="news-metro", domain="news",
                 text=("Bakı, 15 iyun (AZƏRTAC)\n"
                       "Paytaxtda yeni metro stansiyası açılıb.")),
    ])
    return engine


class TestExportData:
    def test_search_results_to_json(self, tmp_path):
        engine = build_engine()
        results = engine.search("mülkiyyət", k=3)
        path = export_data(results, tmp_path / "results.json")
        records = json.loads(path.read_text(encoding="utf-8"))
        assert len(records) == len(results)
        first = records[0]
        assert first["doc_id"] and first["chunk_id"] and first["text"]
        assert first["rank"] == 1
        assert "domain" in first["metadata"]

    def test_chunks_to_jsonl(self, tmp_path):
        engine = build_engine()
        path = engine.export_chunks(tmp_path / "chunks.jsonl")
        lines = [json.loads(line) for line in
                 path.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == engine.count()
        assert all("chunk_id" in r and "text" in r for r in lines)

    def test_csv_with_azerbaijani_text(self, tmp_path):
        engine = build_engine()
        results = engine.search("mülkiyyət", k=2)
        path = export_data(results, tmp_path / "results.csv")
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(results)
        assert "Mülkiyyət" in rows[0]["text"]
        assert json.loads(rows[0]["metadata"])["domain"] == "legal"

    def test_report_to_json_and_md(self, tmp_path):
        engine = build_engine()
        report = EvaluationHarness(engine).evaluate(
            [EvalQuery(query="mülkiyyət toxunulmazdır",
                       relevant_ids={"law-13"}, domain="legal")], k=3)
        json_path = export_data(report, tmp_path / "report.json")
        record = json.loads(json_path.read_text(encoding="utf-8"))[0]
        assert "aggregate" in record and "per_domain" in record
        md_path = export_data(report, tmp_path / "report.md")
        assert "aggregate" in md_path.read_text(encoding="utf-8")

    def test_eval_queries_sets_serialized(self, tmp_path):
        queries = [EvalQuery(query="sual", relevant_ids={"b", "a"},
                             domain="news")]
        path = export_data(queries, tmp_path / "queries.json")
        record = json.loads(path.read_text(encoding="utf-8"))[0]
        assert record["relevant_ids"] == ["a", "b"]

    def test_explicit_format_overrides_suffix(self, tmp_path):
        path = export_data([{"a": 1}], tmp_path / "out.dat", format="json")
        assert json.loads(path.read_text(encoding="utf-8")) == [{"a": 1}]

    def test_unknown_suffix_raises_with_guidance(self, tmp_path):
        with pytest.raises(ValueError, match="register_export_format"):
            export_data([{"a": 1}], tmp_path / "out.xyz")

    def test_custom_format_registration(self, tmp_path):
        @register_export_format(".tsv")
        def write_tsv(records, path):
            fields = list(records[0])
            lines = ["\t".join(fields)]
            lines += ["\t".join(str(r[f]) for f in fields) for r in records]
            path.write_text("\n".join(lines), encoding="utf-8")

        path = export_data([{"x": 1, "y": "mətn"}], tmp_path / "out.tsv")
        assert path.read_text(encoding="utf-8") == "x\ty\n1\tmətn"

    def test_unsupported_object_raises(self, tmp_path):
        with pytest.raises(TypeError, match="Cannot export"):
            export_data([object()], tmp_path / "out.json")
