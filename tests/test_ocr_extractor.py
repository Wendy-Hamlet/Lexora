"""OCR extractor — offline core (no engine/model needed).

Pins the parts the verbatim contract depends on: reading-order assembly, page
confidence, and — most importantly — that filling an image-only page slot with
OCR text recomputes the GLOBAL char offsets so every page still slices its own
text out of the assembled document (the invariant CanonicalSpan relies on).
A fake engine / monkeypatched extract_ocr keeps this independent of rapidocr.
"""
from __future__ import annotations

from lexora.extract import ocr_extractor as oe
from lexora.extract.pdf_text_extractor import PAGE_SEPARATOR, PdfPage, assemble_global_text


def test_reading_order_sorts_top_to_bottom_then_left_to_right():
    def box(x, y):
        return [[x, y], [x + 10, y], [x + 10, y + 8], [x, y + 8]]

    # Out of order: bottom-left, top-right, top-left.
    items = [
        [box(5, 100), "third", 0.9],
        [box(200, 10), "second", 0.8],
        [box(5, 10), "first", 0.95],
    ]
    assert [t for t, _ in oe._reading_order(items)] == ["first", "second", "third"]


def test_page_text_and_conf_means_line_scores():
    text, conf = oe._page_text_and_conf([("a", 0.8), ("b", 1.0)])
    assert text == "a\nb"
    assert abs(conf - 0.9) < 1e-9
    assert oe._page_text_and_conf([]) == ("", 0.0)


def test_ocr_fill_pages_noop_when_all_pages_have_text_layer():
    pages = [
        PdfPage(1, "alpha", 0, 5, True),
        PdfPage(2, "bravo", 7, 12, True),
    ]
    out, conf = oe.ocr_fill_pages(pages, b"%PDF-")
    assert out is pages and conf == {}


def test_ocr_fill_pages_reflows_offsets_and_reports_confidence(monkeypatch):
    # Page 2 is image-only (blank text layer). OCR fills it; the global offsets of
    # page 2 (and any later page) must be recomputed so each page still slices its
    # own text out of the assembled document.
    pages = [
        PdfPage(1, "alpha", 0, 5, True),
        PdfPage(2, "", 7, 7, False),       # blank slot
        PdfPage(3, "charlie", 9, 16, True),
    ]

    def fake_extract_ocr(source, languages=None, *, dpi, engine, page_numbers):
        assert page_numbers == {2}  # only the blank page is OCR'd
        return [oe.OcrPage(2, "bravo scan", 0, 10, 0.72, "fake:1")]

    monkeypatch.setattr(oe, "extract_ocr", fake_extract_ocr)
    out, conf = oe.ocr_fill_pages(pages, b"%PDF-")

    assert [p.text for p in out] == ["alpha", "bravo scan", "charlie"]
    assert conf == {2: 0.72}
    # The verbatim invariant: every page's [char_start:char_end] slices its own
    # text out of the reassembled global document.
    global_text = assemble_global_text(out)
    for p in out:
        assert global_text[p.char_start:p.char_end] == p.text
    # Offsets are contiguous with the page separator between them.
    assert out[1].char_start == out[0].char_end + len(PAGE_SEPARATOR)
    assert out[2].char_start == out[1].char_end + len(PAGE_SEPARATOR)
    # The filled page is now treated as carrying text.
    assert out[1].has_text_layer is True


def test_ocr_fill_pages_marks_empty_ocr_result_as_no_text_layer(monkeypatch):
    pages = [PdfPage(1, "", 0, 0, False)]

    def fake_extract_ocr(source, languages=None, *, dpi, engine, page_numbers):
        return [oe.OcrPage(1, "", 0, 0, 0.0, "fake:1")]  # OCR found nothing

    monkeypatch.setattr(oe, "extract_ocr", fake_extract_ocr)
    out, conf = oe.ocr_fill_pages(pages, b"%PDF-")
    assert out[0].has_text_layer is False  # still blank -> can't back a citation
    assert conf == {1: 0.0}


def test_make_engine_defaults_to_rapidocr_name(monkeypatch):
    # Dispatch only: select paddleocr via env without constructing it (would import
    # paddle). We assert the branch is reached by stubbing the engine classes.
    built = {}

    class Stub:
        def __init__(self, *a, **k):
            built["cls"] = type(self).__name__

    monkeypatch.setattr(oe, "_RapidEngine", Stub)
    monkeypatch.setattr(oe, "_PaddleEngine", Stub)
    monkeypatch.delenv("LEXORA_OCR_ENGINE", raising=False)
    oe.make_engine()
    assert built["cls"] == "Stub"
    oe.make_engine("paddleocr")
    assert built["cls"] == "Stub"


def test_scanned_pdf_running_head_is_stripped_after_ocr_fills_the_pages(monkeypatch):
    """An image-only PDF is BLANK when the running-head cleaner first runs, so on a
    scan it saw nothing and every page header survived into the text — and then into
    the verbatim quotes ("Act 762 Access to place or premises 81. (1) ..."). Once OCR
    has filled the pages the repetition is visible, so it must be cleaned here."""
    pages = [PdfPage(i, "", 0, 0, False) for i in range(1, 7)]

    # a realistic page: a header band, then a body of several lines
    bodies = [
        ["1. This Act may be cited as the Goods and Services Tax Act.",
         "(a) it comes into operation on a date appointed by the Minister;",
         "(b) different dates may be appointed for different provisions.",
         "and the Minister shall publish the notice in the Gazette.",
         "This section binds the Government."],
        ["2. In this Act, unless the context otherwise requires -",
         '(a) "taxable person" means a person registered under this Act;',
         '(b) "supply" has the meaning given by section 4.',
         "and any reference to a supply is a reference to a taxable supply.",
         "The Minister may by order amend this section."],
        ["81. (1) Any senior officer of goods and services tax shall have access",
         "to any place or premises where a taxable person carries on business.",
         "(2) The officer may inspect and take copies of any record.",
         "(3) The occupier shall provide all reasonable facilities.",
         "(4) This section applies despite any other written law."],
        ["82. The Director General may require any person to furnish information",
         "within such time as may be specified in the notice.",
         "(2) A person who fails to comply commits an offence.",
         "(3) The notice may be served by registered post.",
         "(4) Service is deemed effected on the third day."],
        ["83. No person shall obstruct any officer in the exercise of his powers",
         "under this Act or its subsidiary legislation.",
         "(2) Obstruction includes refusing access to any premises.",
         "(3) An officer shall on demand produce his authority card.",
         "(4) Failure to produce it does not invalidate the exercise."],
        ["84. Any person who contravenes this section commits an offence.",
         "and shall on conviction be liable to a fine not exceeding thirty",
         "thousand ringgit or to imprisonment for a term not exceeding two years.",
         "(2) The court may order forfeiture of any goods seized.",
         "(3) This section does not limit any other remedy."],
    ]

    def fake_extract_ocr(source, languages=None, *, dpi, engine, page_numbers):
        # every page carries the same running head + its own page number
        return [
            oe.OcrPage(i, "Act 762\n{}\n{}".format(i + 10, "\n".join(b)), 0, 0, 0.9, "fake:1")
            for i, b in enumerate(bodies, 1)
        ]

    monkeypatch.setattr(oe, "extract_ocr", fake_extract_ocr)
    out, _ = oe.ocr_fill_pages(pages, b"%PDF-")

    text = assemble_global_text(out)
    assert "Act 762" not in text          # the running head
    assert "\n11\n" not in text           # its page-number companion
    for page in bodies:                   # every operative line survives
        for line in page:
            assert line in text
    # The verbatim invariant still holds against the CLEANED text.
    for p in out:
        assert text[p.char_start:p.char_end] == p.text
