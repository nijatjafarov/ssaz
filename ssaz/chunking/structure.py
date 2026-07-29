"""Structural segmentation on Azerbaijani document markers.

Detection rules cover the three domains:

* **legal** — article headers (``Maddə 12.``), chapter/section headers
  (``Fəsil III``, ``Bölmə 2``), and hierarchical clause numbers (``12.3.``);
* **news** — datelines (``Bakı, 15 iyun (AZƏRTAC)``, ``Bakı. 21 mart.
  REPORT.AZ:``) and all-caps agency headers;
* **encyclopedic** — wiki-style ``== Heading ==`` markers, Markdown
  headings, and short standalone heading lines.

Rules operate on whole lines, never on sentence boundaries, which makes
the pass robust to agglutinative sentence-splitting errors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ssaz.chunking.base import BaseChunker, Chunk
from ssaz.documents import Document

_MONTHS = (
    "yanvar|fevral|mart|aprel|may|iyun|iyul|avqust|"
    "sentyabr|oktyabr|noyabr|dekabr"
)
_UPPER = "A-ZÇƏĞİÖŞÜ"
_LOWER = "a-zçəğıöşü"

# legal
LEGAL_ARTICLE_RE = re.compile(r"^\s*Maddə\s+(\d+)[.\-]?\s*(.*)$")
LEGAL_CHAPTER_RE = re.compile(
    r"^\s*(Fəsil|Bölmə|Hissə)\s+([IVXLCDM]+|\d+)[.\-]?\s*(.*)$", re.IGNORECASE
)
LEGAL_CLAUSE_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\.\s+")

# news
NEWS_DATELINE_RE = re.compile(
    r"^\s*[%s][%s]+\s*[.,]\s*\d{1,2}\s+(?:%s)\b" % (_UPPER, _LOWER, _MONTHS)
)
NEWS_AGENCY_RE = re.compile(
    r"^\s*[%s][%s]+\s*[.,][^\n]{0,60}?"
    r"(?:\(([A-ZƏÇĞİÖŞÜ][A-ZƏÇĞİÖŞÜ.\-]+)\)|(?:REPORT|APA|AZƏRTAC|Trend)[.\w]*:)"
    % (_UPPER, _LOWER)
)

# encyclopedic
WIKI_HEADING_RE = re.compile(r"^\s*(={2,6})\s*(.+?)\s*={2,6}\s*$")
MD_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")


def _is_bare_heading(line: str) -> bool:
    """Short standalone line with no terminal punctuation, likely a heading"""
    stripped = line.strip()
    if not stripped or len(stripped) > 70:
        return False
    if stripped[-1] in ".!?…:;,":
        return False
    if not re.match(r"^[%s\d\"']" % _UPPER, stripped):
        return False
    # A heading has few words and no verb-final sentence shape we can cheaply
    # detect; the word-count cap keeps false positives rare.
    return len(stripped.split()) <= 8


@dataclass
class Section:
    """A structurally delimited region of a document"""

    text: str
    start: int
    end: int
    heading: Optional[str] = None
    marker_type: Optional[str] = None  # article | chapter | clause | dateline | heading | None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _classify_line(line: str, domain: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Return ``(marker_type, heading_text, extra_metadata)`` if ``line``
    opens a new section, else ``None``"""
    if domain in ("legal", "general"):
        m = LEGAL_ARTICLE_RE.match(line)
        if m:
            title = m.group(2).strip()
            heading = f"Maddə {m.group(1)}" + (f". {title}" if title else "")
            return "article", heading, {"article_number": int(m.group(1))}
        m = LEGAL_CHAPTER_RE.match(line)
        if m:
            heading = line.strip()
            return "chapter", heading, {}
        m = LEGAL_CLAUSE_RE.match(line)
        if m:
            return "clause", m.group(1), {"clause_number": m.group(1)}
    if domain in ("news", "general"):
        if NEWS_DATELINE_RE.match(line) or NEWS_AGENCY_RE.match(line):
            return "dateline", line.strip(), {}
    if domain in ("encyclopedic", "general"):
        m = WIKI_HEADING_RE.match(line)
        if m:
            return "heading", m.group(2).strip(), {"heading_level": len(m.group(1))}
        m = MD_HEADING_RE.match(line)
        if m:
            return "heading", m.group(2).strip(), {"heading_level": len(m.group(1))}
    if domain == "encyclopedic" and _is_bare_heading(line):
        return "heading", line.strip(), {}
    return None


# Custom rules run BEFORE the built-in ones, so they can
# also override built-in behavior for a line.
MarkerRule = Callable[[str, str], Optional[Tuple[str, str, Dict[str, Any]]]]


class StructuralChunker(BaseChunker):
    """Segments a document at structural marker lines.

    Args:
        chunk_size: Size budget in characters; sections beyond it are
            windowed. ``None`` disables the budget (sections verbatim).
        chunk_overlap: Overlap between consecutive windows of an
            oversized section.
        domain: Overrides the per-document domain for rule selection.
        extra_rules: Custom :data:`MarkerRule` callables for markers the
            built-ins don't know (court decisions, medical protocols,
            religious texts, ...), e.g.::

                def qerar_rule(line, domain):
                    m = re.match(r"^\\s*QƏRAR\\s+№\\s*(\\S+)", line)
                    if m:
                        return "decision", line.strip(), {"decision_no": m.group(1)}
                    return None

                chunker = TwoPassChunker(structural_rules=[qerar_rule])
    """

    def __init__(self, domain: Optional[str] = None,
                 extra_rules: Optional[List[MarkerRule]] = None,
                 chunk_size: Optional[int] = 800,
                 chunk_overlap: int = 120) -> None:
        if chunk_size is not None and chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.domain = domain
        self.extra_rules: List[MarkerRule] = list(extra_rules or [])
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _classify(self, line: str, domain: str):
        for rule in self.extra_rules:
            hit = rule(line, domain)
            if hit:
                return hit
        return _classify_line(line, domain)

    def _windows(self, text: str) -> List[Tuple[int, str]]:
        """Split ``text`` into (offset, window) pairs of at most
        ``chunk_size`` characters with ``chunk_overlap`` overlap, window
        starts are snapped forward to a word boundary when possible"""
        if self.chunk_size is None or len(text) <= self.chunk_size:
            return [(0, text)]
        windows: List[Tuple[int, str]] = []
        start = 0
        while True:
            end = min(start + self.chunk_size, len(text))
            windows.append((start, text[start:end]))
            if end >= len(text):
                return windows
            start = end - self.chunk_overlap
            # Avoid starting an overlap mid-word (and mid-morpheme).
            space = text.find(" ", start, end)
            if 0 <= space < end - 1:
                start = space + 1

    def chunk(self, document: Document) -> List[Chunk]:
        """BaseChunker interface: one chunk per structural section,
        oversized sections windowed under the size budget"""
        chunks: List[Chunk] = []
        index = 0
        for section in self.split(document):
            metadata: Dict[str, Any] = {"domain": document.domain,
                                        "chunking": "structural"}
            if section.heading:
                metadata["section_heading"] = section.heading
            if section.marker_type:
                metadata["marker_type"] = section.marker_type
            metadata.update(section.metadata)
            for offset, window in self._windows(section.text):
                chunks.append(Chunk(
                    chunk_id=f"{document.doc_id}::{index:04d}",
                    doc_id=document.doc_id,
                    text=window,
                    metadata=dict(metadata),
                    start=section.start + offset,
                    end=section.start + offset + len(window),
                ))
                index += 1
        return chunks

    def split(self, document: Document) -> List[Section]:
        domain = self.domain or document.domain or "general"
        text = document.text
        lines = text.split("\n")

        # Compute line start offsets for character-accurate section spans
        offsets: List[int] = []
        pos = 0
        for line in lines:
            offsets.append(pos)
            pos += len(line) + 1

        boundaries: List[Tuple[int, str, str, Dict[str, Any]]] = []
        for i, line in enumerate(lines):
            hit = self._classify(line, domain)
            if hit:
                marker_type, heading, extra = hit
                boundaries.append((i, marker_type, heading, extra))

        sections: List[Section] = []

        def add_section(start_line: int, end_line: int,
                        heading: Optional[str], marker_type: Optional[str],
                        extra: Dict[str, Any]) -> None:
            start = offsets[start_line]
            end = (offsets[end_line - 1] + len(lines[end_line - 1])
                   if end_line > start_line else start)
            segment = text[start:end].strip()
            if segment:
                sections.append(Section(
                    text=segment, start=start, end=end,
                    heading=heading, marker_type=marker_type,
                    metadata=dict(extra),
                ))

        if not boundaries:
            add_section(0, len(lines), None, None, {})
            return sections

        first_line = boundaries[0][0]
        if first_line > 0:
            add_section(0, first_line, None, None, {})

        for idx, (line_no, marker_type, heading, extra) in enumerate(boundaries):
            next_line = (boundaries[idx + 1][0]
                         if idx + 1 < len(boundaries) else len(lines))
            add_section(line_no, next_line, heading, marker_type, extra)

        return sections
