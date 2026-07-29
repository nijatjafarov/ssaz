"""Tests for the customizable result presenter (ssaz.display)"""

import pytest

from ssaz import ResultPresenter, show_results
from ssaz.engine import SearchResult


def make_results():
    return [
        SearchResult(chunk_id="d1::0000", doc_id="d1", rank=1, score=0.91,
                     text="Mülkiyyət toxunulmazdır və dövlət tərəfindən "
                          "müdafiə olunur." * 10,
                     metadata={"domain": "legal",
                               "section_heading": "Maddə 13. Mülkiyyət",
                               "title": "Konstitusiya"}),
        SearchResult(chunk_id="d2::0001", doc_id="d2", rank=2, score=0.55,
                     text="Paytaxtda yeni metro stansiyası açılıb.",
                     metadata={"domain": "news"}),
    ]


class TestResultPresenter:
    def test_detailed_marks_answer_and_hits(self):
        output = ResultPresenter().render("sual?", make_results(),
                                          relevant_ids={"d1", "d2"})
        assert output.startswith("Q: sual?")
        assert "CAVAB" in output
        assert "HIT" in output
        assert "doc=d1" in output and "chunk=d1::0000" in output
        assert "[Maddə 13. Mülkiyyət]" in output
        assert "Mülkiyyət toxunulmazdır" in output

    def test_text_truncated_to_budget(self):
        output = ResultPresenter(max_text_chars=50).render(
            "q", make_results())
        text_lines = [l for l in output.splitlines()
                      if l.startswith("        ")]
        assert all(len(l.strip()) <= 50 for l in text_lines)
        assert "…" in output

    def test_text_hidden_when_zero(self):
        output = ResultPresenter(max_text_chars=0).render("q", make_results())
        assert "Mülkiyyət toxunulmazdır" not in output

    def test_compact_one_line_per_hit(self):
        output = ResultPresenter(style="compact", max_text_chars=40).render(
            "q", make_results())
        assert len(output.splitlines()) == 1 + len(make_results())

    def test_markdown_table(self):
        output = ResultPresenter(style="markdown").render(
            "q", make_results(), relevant_ids={"d2"})
        lines = output.splitlines()
        assert lines[1].startswith("| ") and "rank" in lines[1]
        assert lines[2].startswith("|---")
        assert "| CAVAB | 1 |" in output

    def test_custom_labels_and_fields(self):
        presenter = ResultPresenter(answer_label=">>>", hit_label="*",
                                    show_chunk_id=False,
                                    metadata_fields=("title",))
        output = presenter.render("q", make_results(), relevant_ids={"d2"})
        assert ">>>" in output and "* " in output
        assert "chunk=" not in output
        assert "title=Konstitusiya" in output

    def test_mark_answer_disabled(self):
        output = ResultPresenter(mark_answer=False).render(
            "q", make_results(), relevant_ids={"d1"})
        assert "CAVAB" not in output
        assert "HIT" in output

    def test_invalid_style_rejected(self):
        with pytest.raises(ValueError, match="style"):
            ResultPresenter(style="fancy")

    def test_show_results_convenience(self, capsys):
        show_results("sual?", make_results(), style="compact",
                     answer_label="ANSWER")
        captured = capsys.readouterr().out
        assert "Q: sual?" in captured
        assert "ANSWER" in captured

    def test_format_result_override(self):
        class OneLiner(ResultPresenter):
            def format_result(self, result, relevant_ids):
                return [f"{result.rank}) {result.doc_id}"]

        output = OneLiner().render("q", make_results())
        assert "1) d1" in output and "2) d2" in output