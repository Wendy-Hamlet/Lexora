"""Text-layer PDF extractor (PyMuPDF).

Returns one PdfPage per page with the *global* character offsets needed to
build CanonicalSpans later. The global text of the document is the
concatenation of `page.text` with a single "\n\n" separator between pages —
the same convention used by `structure.legal_parser`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

PAGE_SEPARATOR = "\n\n"


@dataclass
class PdfPage:
    page_number: int  # 1-indexed
    text: str
    char_start: int
    char_end: int
    has_text_layer: bool


def extract_pdf_text(pdf_path: Path) -> list[PdfPage]:
    """Extract text-layer PDF pages with running global char offsets.

    A page whose text layer is empty/whitespace is returned with
    `has_text_layer=False` and empty text; the OCR extractor is expected to
    fill it in later (Slice 0 leaves these blank and downstream stages skip
    them — but the page slot is preserved so later OCR fills don't shift
    char offsets of subsequent pages).
    """
    pdf_path = Path(pdf_path)
    pages: list[PdfPage] = []
    cursor = 0
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            text = text.rstrip("\f")
            has_text = bool(text.strip())
            page_text = text if has_text else ""
            char_start = cursor
            char_end = cursor + len(page_text)
            pages.append(
                PdfPage(
                    page_number=idx,
                    text=page_text,
                    char_start=char_start,
                    char_end=char_end,
                    has_text_layer=has_text,
                )
            )
            cursor = char_end + len(PAGE_SEPARATOR)
    return pages


def assemble_global_text(pages: list[PdfPage]) -> str:
    """Reconstruct the global document text whose offsets match each page's
    `char_start`/`char_end`."""
    return PAGE_SEPARATOR.join(p.text for p in pages)


__all__ = ["PdfPage", "PAGE_SEPARATOR", "extract_pdf_text", "assemble_global_text"]
