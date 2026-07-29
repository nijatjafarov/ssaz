"""Tests for the structural, recursive, and two-pass chunkers"""

from ssaz.chunking import RecursiveChunker, StructuralChunker, TwoPassChunker
from ssaz.documents import Document

LEGAL_TEXT = """Azərbaycan Respublikasının Qanunu

Fəsil I. Ümumi müddəalar

Maddə 1. Əsas anlayışlar
Bu qanunda istifadə olunan anlayışlar aşağıdakı mənaları daşıyır.
Burada əlavə izahatlar verilir.

Maddə 2. Qanunun tətbiq dairəsi
Bu qanun bütün hüquqi və fiziki şəxslərə şamil edilir.
"""

NEWS_TEXT = """Bakı, 15 iyun (AZƏRTAC)
Paytaxtda yeni metro stansiyası istifadəyə verilib. Tədbirdə rəsmi şəxslər iştirak edib.
"""

WIKI_TEXT = """Azərbaycan dili

== Tarixi ==
Azərbaycan dili türk dilləri ailəsinə daxildir və zəngin tarixə malikdir.

== Qrammatika ==
Dil aqqlütinativ quruluşa malikdir və şəkilçilər sözün sonuna əlavə olunur.
"""


class TestStructuralChunker:
    def test_legal_articles_detected(self):
        doc = Document(doc_id="law1", text=LEGAL_TEXT, domain="legal")
        sections = StructuralChunker().split(doc)
        headings = [s.heading for s in sections if s.heading]
        assert any(h.startswith("Maddə 1") for h in headings)
        assert any(h.startswith("Maddə 2") for h in headings)

    def test_article_number_metadata(self):
        doc = Document(doc_id="law1", text=LEGAL_TEXT, domain="legal")
        sections = StructuralChunker().split(doc)
        numbers = [s.metadata.get("article_number") for s in sections
                   if s.marker_type == "article"]
        assert numbers == [1, 2]

    def test_news_dateline_detected(self):
        doc = Document(doc_id="news1", text=NEWS_TEXT, domain="news")
        sections = StructuralChunker().split(doc)
        assert any(s.marker_type == "dateline" for s in sections)

    def test_wiki_headings_detected(self):
        doc = Document(doc_id="wiki1", text=WIKI_TEXT, domain="encyclopedic")
        sections = StructuralChunker().split(doc)
        headings = [s.heading for s in sections if s.marker_type == "heading"]
        assert "Tarixi" in headings
        assert "Qrammatika" in headings

    def test_unmarked_document_single_section(self):
        doc = Document(doc_id="plain", text="Sadə mətn. Heç bir marker yoxdur.",
                       domain="legal")
        sections = StructuralChunker().split(doc)
        assert len(sections) == 1
        assert sections[0].marker_type is None

    def test_implements_common_chunk_interface(self):
        # Every strategy exposes BaseChunker.chunk(document) -> List[Chunk].
        doc = Document(doc_id="law1", text=LEGAL_TEXT, domain="legal")
        chunks = StructuralChunker().chunk(doc)
        assert chunks
        assert all(c.chunk_id.startswith("law1::") for c in chunks)
        headings = [c.metadata.get("section_heading") for c in chunks]
        assert any(h and h.startswith("Maddə 1") for h in headings)
        assert all(c.metadata["chunking"] == "structural" for c in chunks)

    def test_oversized_section_windowed_with_overlap(self):
        body = "Bu cümlə təkrar olunur və mətni uzadır. " * 40  # ~1600 chars
        doc = Document(doc_id="law", domain="legal",
                       text="Maddə 1. Uzun maddə\n" + body)
        chunker = StructuralChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        # Budget respected, no truncation: full section text is covered.
        assert all(len(c.text) <= 500 for c in chunks)
        assert chunks[-1].end >= len(doc.text.strip()) - 1
        # Consecutive windows overlap.
        assert chunks[1].start < chunks[0].end
        # Every window inherits the section metadata.
        assert all(c.metadata.get("section_heading", "").startswith("Maddə 1")
                   for c in chunks)
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_no_budget_keeps_sections_verbatim(self):
        body = "Uzun mətn davam edir. " * 100
        doc = Document(doc_id="law", domain="legal",
                       text="Maddə 1. Başlıq\n" + body)
        chunks = StructuralChunker(chunk_size=None).chunk(doc)
        assert len(chunks) == 1
        assert len(chunks[0].text) > 2000

    def test_overlap_validation(self):
        try:
            StructuralChunker(chunk_size=100, chunk_overlap=100)
            assert False, "expected ValueError"
        except ValueError:
            pass

class TestRecursiveChunker:
    def test_short_text_single_chunk(self):
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        doc = Document(doc_id="d", text="Qısa mətn.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1

    def test_respects_size_budget(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        text = " ".join(f"söz{i}" for i in range(200))
        for piece in chunker.split_text(text):
            assert len(piece) <= 100

    def test_pathological_unbroken_text(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        pieces = chunker.split_text("a" * 500)
        assert pieces
        assert all(len(p) <= 50 for p in pieces)

    def test_overlap_validation(self):
        try:
            RecursiveChunker(chunk_size=100, chunk_overlap=100)
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestTwoPassChunker:
    def test_structural_metadata_on_chunks(self):
        chunker = TwoPassChunker(chunk_size=800, min_section_size=10)
        doc = Document(doc_id="law1", text=LEGAL_TEXT, domain="legal")
        chunks = chunker.chunk(doc)
        headings = {c.metadata.get("section_heading") for c in chunks}
        assert any(h and h.startswith("Maddə 1") for h in headings)

    def test_oversized_section_falls_back(self):
        big_article = ("Maddə 1. Uzun maddə\n" +
                       "Bu cümlə təkrarlanır. " * 100)
        chunker = TwoPassChunker(chunk_size=300, chunk_overlap=50)
        doc = Document(doc_id="law2", text=big_article, domain="legal")
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        # All fallback chunks inherit the article heading.
        assert all(c.metadata.get("section_heading", "").startswith("Maddə 1")
                   for c in chunks)

    def test_chunk_ids_stable_and_unique(self):
        chunker = TwoPassChunker()
        doc = Document(doc_id="law1", text=LEGAL_TEXT, domain="legal")
        ids1 = [c.chunk_id for c in chunker.chunk(doc)]
        ids2 = [c.chunk_id for c in chunker.chunk(doc)]
        assert ids1 == ids2
        assert len(set(ids1)) == len(ids1)

    def test_domain_carried(self):
        chunker = TwoPassChunker()
        doc = Document(doc_id="n1", text=NEWS_TEXT, domain="news")
        for chunk in chunker.chunk(doc):
            assert chunk.metadata["domain"] == "news"
