"""Tests for normalization, sentence splitting, and stemming"""

from ssaz.text import (
    AzNormalizer,
    azerbaijani_lower,
    contains_cyrillic,
    cyrillic_to_latin,
    split_sentences,
)


class TestAzerbaijaniLower:
    def test_dotted_i(self):
        assert azerbaijani_lower("İstanbul") == "istanbul"

    def test_dotless_i(self):
        assert azerbaijani_lower("QIZIL") == "qızıl"

    def test_mixed(self):
        assert azerbaijani_lower("AZƏRBAYCAN DİLİ") == "azərbaycan dili"

    def test_no_combining_dot(self):
        assert "̇" not in azerbaijani_lower("İİİ")


class TestNormalizer:
    def test_schwa_repair(self):
        norm = AzNormalizer()
        assert norm("Ǝlifba vǝ ǝdǝbiyyat") == "Əlifba və ədəbiyyat"

    def test_cyrillic_schwa_repair(self):
        norm = AzNormalizer()
        assert norm("mәdәniyyәt") == "mədəniyyət"

    def test_whitespace_collapse(self):
        norm = AzNormalizer()
        assert norm("a  b\t c") == "a b c"

    def test_paragraph_breaks_preserved(self):
        norm = AzNormalizer()
        assert norm("a\n\n\n\nb") == "a\n\nb"

    def test_lowercase_option(self):
        norm = AzNormalizer(lowercase=True)
        assert norm("Bakı Şəhəri") == "bakı şəhəri"

    def test_empty(self):
        assert AzNormalizer()("") == ""


class TestCyrillicTransliteration:
    def test_country_name(self):
        assert cyrillic_to_latin("Азәрбајҹан") == "Azərbaycan"

    def test_g_letters_disambiguated(self):
        assert cyrillic_to_latin("Гарабағ") == "Qarabağ"
        assert cyrillic_to_latin("ҝөзәл") == "gözəl"

    def test_dotted_and_dotless_i(self):
        assert cyrillic_to_latin("гыз") == "qız"
        assert cyrillic_to_latin("иш") == "iş"
        assert cyrillic_to_latin("Ил") == "İl"
        assert cyrillic_to_latin("ЫЛДЫЗ") == "ILDIZ"

    def test_j_is_y(self):
        assert cyrillic_to_latin("Јени јол") == "Yeni yol"

    def test_h_and_x(self):
        assert cyrillic_to_latin("һәр хәбәр") == "hər xəbər"

    def test_multichar_casing(self):
        assert cyrillic_to_latin("июн") == "iyun"
        assert cyrillic_to_latin("Юбилей") == "Yubiley"
        assert cyrillic_to_latin("ЮНЕСКО") == "YUNESKO"
        
    def test_latin_passthrough(self):
        text = "Azərbaycan Respublikası, 2026-cı il"
        assert cyrillic_to_latin(text) == text

    def test_mixed_script(self):
        assert cyrillic_to_latin("Bakı шәһәри") == "Bakı şəhəri"

    def test_contains_cyrillic(self):
        assert contains_cyrillic("мәтн")
        assert not contains_cyrillic("mətn")

    def test_normalizer_integration(self):
        norm = AzNormalizer()
        assert norm("Азәрбајҹан дили") == "Azərbaycan dili"

    def test_normalizer_opt_out(self):
        norm = AzNormalizer(transliterate_cyrillic=False,
                            repair_characters=False)
        assert norm("мәтн") == "мәтн"


class TestSentenceSplitter:
    def test_basic_split(self):
        sentences = split_sentences("Bu birinci cümlədir. Bu ikinci cümlədir.")
        assert len(sentences) == 2

    def test_abbreviation_not_split(self):
        text = "Məs. bu bir nümunədir və davam edir."
        assert len(split_sentences(text)) == 1

    def test_initial_not_split(self):
        text = "H. Əliyev adına mərkəz açıldı. Sonra tədbir keçirildi."
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert sentences[0].startswith("H. Əliyev")

    def test_clause_number_not_split(self):
        text = "Qanunun 3. maddəsində qeyd olunur ki, hüquqlar qorunur."
        assert len(split_sentences(text)) == 1

    def test_question_and_exclamation(self):
        sentences = split_sentences("Bu nədir? Bu kitabdır! Oxu.")
        assert len(sentences) == 3

    def test_empty(self):
        assert split_sentences("") == []
