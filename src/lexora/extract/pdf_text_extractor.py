"""Text-layer PDF extractor (PyMuPDF)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PdfPage:
    page_number: int
    text: str
    char_start: int
    char_end: int
    has_text_layer: bool


def extract_pdf_text(pdf_path: Path) -> list[PdfPage]:  # pragma: no cover
    """Extract text-layer PDF pages with running global offsets.

    TODO: use pymupdf (fitz). If `page.get_text("text")` returns empty for any
    page, mark `has_text_layer=False` and route those pages to the OCR
    extractor.
    """
    raise NotImplementedError("Implement PDF text extractor.")
