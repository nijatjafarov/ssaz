"""
Customizable demonstration of search results.

Configured use:

    from ssaz import ResultPresenter

    presenter = ResultPresenter(style="compact", max_text_chars=0,
                                answer_label=">>>", hit_label="*")
    print(presenter.render(question, results, relevant_ids={"eqa_000014"}))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set

from ssaz.engine import SearchResult

_STYLES = ("detailed", "compact", "markdown")


@dataclass
class ResultPresenter:
    """Configurable renderer for ranked search results.

    Args:
        style: ``detailed`` | ``compact`` | ``markdown``.
        max_text_chars: Chunk text budget per hit; ``0`` hides the text,
            ``None`` shows it in full.
        show_score / show_doc_id / show_chunk_id / show_heading: Toggle
            the corresponding field.
        metadata_fields: Extra chunk-metadata keys to display (e.g.
            ``("title", "date")``).
        answer_label: Marker for the rank-1 hit — the answer chunk.
        hit_label: Marker for other hits whose document is in
            ``relevant_ids``.
        miss_label: Marker for the remaining hits.
        mark_answer: If False, rank 1 is marked like any other hit.
    """

    style: str = "detailed"
    max_text_chars: Optional[int] = 220
    show_score: bool = True
    show_doc_id: bool = True
    show_chunk_id: bool = True
    show_heading: bool = True
    metadata_fields: Sequence[str] = ()
    answer_label: str = "CAVAB"
    hit_label: str = "HIT"
    miss_label: str = ""
    mark_answer: bool = True

    def __post_init__(self) -> None:
        if self.style not in _STYLES:
            raise ValueError(f"Unknown style {self.style!r}; "
                             f"options: {_STYLES}")

    # Building blocks

    def marker(self, result: SearchResult, relevant_ids: Set[str]) -> str:
        if self.mark_answer and result.rank == 1:
            return self.answer_label
        if result.doc_id in relevant_ids:
            return self.hit_label
        return self.miss_label

    def text_of(self, result: SearchResult) -> str:
        text = result.text.replace("\n", " ").strip()
        if self.max_text_chars == 0:
            return ""
        if self.max_text_chars and len(text) > self.max_text_chars:
            return text[:self.max_text_chars - 1] + "…"
        return text

    def fields_of(self, result: SearchResult) -> List[str]:
        parts: List[str] = []
        if self.show_score:
            parts.append(f"score={result.score:.4f}")
        if self.show_doc_id:
            parts.append(f"doc={result.doc_id}")
        if self.show_chunk_id:
            parts.append(f"chunk={result.chunk_id}")
        if self.show_heading and result.metadata.get("section_heading"):
            parts.append(f"[{result.metadata['section_heading']}]")
        for key in self.metadata_fields:
            if result.metadata.get(key) is not None:
                parts.append(f"{key}={result.metadata[key]}")
        return parts

    def format_result(self, result: SearchResult,
                      relevant_ids: Set[str]) -> List[str]:
        """Lines for one hit; override for a fully custom layout"""
        width = max(len(self.answer_label), len(self.hit_label),
                    len(self.miss_label))
        marker = self.marker(result, relevant_ids).ljust(width)
        head = f"  {marker} #{result.rank}  " + "  ".join(
            self.fields_of(result))
        if self.style == "compact":
            text = self.text_of(result)
            return [head + (f"  {text}" if text else "")]
        lines = [head]
        text = self.text_of(result)
        if text:
            lines.append(f"        {text}")
        return lines

    # Rendering

    def _render_markdown(self, results: Iterable[SearchResult],
                         relevant_ids: Set[str]) -> List[str]:
        header = ["", "rank"]
        header += (["score"] if self.show_score else [])
        header += (["doc"] if self.show_doc_id else [])
        header += (["chunk"] if self.show_chunk_id else [])
        header += list(self.metadata_fields)
        header += (["text"] if self.max_text_chars != 0 else [])
        lines = ["| " + " | ".join(header) + " |",
                 "|" + "---|" * len(header)]
        for result in results:
            row = [self.marker(result, relevant_ids), str(result.rank)]
            if self.show_score:
                row.append(f"{result.score:.4f}")
            if self.show_doc_id:
                row.append(result.doc_id)
            if self.show_chunk_id:
                row.append(result.chunk_id)
            for key in self.metadata_fields:
                row.append(str(result.metadata.get(key, "")))
            if self.max_text_chars != 0:
                row.append(self.text_of(result).replace("|", "\\|"))
            lines.append("| " + " | ".join(row) + " |")
        return lines

    def render(self, question: str, results: Iterable[SearchResult],
               relevant_ids: Optional[Set[str]] = None) -> str:
        """Render one question's ranked results as a string."""
        relevant_ids = relevant_ids or set()
        lines = [f"Q: {question}"] if question else []
        if self.style == "markdown":
            lines += self._render_markdown(results, relevant_ids)
        else:
            for result in results:
                lines += self.format_result(result, relevant_ids)
        return "\n".join(lines)

    def show(self, question: str, results: Iterable[SearchResult],
             relevant_ids: Optional[Set[str]] = None) -> None:
        """Print :meth:`render` output (with a leading blank line)"""
        print()
        print(self.render(question, results, relevant_ids))


def show_results(question: str, results: Iterable[SearchResult],
                 relevant_ids: Optional[Set[str]] = None,
                 **options) -> None:
    ResultPresenter(**options).show(question, results, relevant_ids)
