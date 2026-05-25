"""OCR pipeline with page-level confidence triage.

Confidence triage (page-level, not token-level):
    >= CITABLE_THRESHOLD    → page is citable
    <  CITABLE_THRESHOLD    → page is marked UNVERIFIED_SCAN; clauses on it
                              cannot back a citation until manually corrected
    Medium-confidence pages may be re-OCRed once with an alternate engine /
    higher DPI; persistent low confidence → human review.

VLM fallback for medium-confidence pages is optional and pluggable; it is
NOT required for the core demo.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class OcrPage:
    page_number: int
    text: str
    char_start: int
    char_end: int
    confidence: float       # 0..1, page-level mean
    engine: str             # e.g. "tesseract:5.3", "paddleocr:2.7"


def extract_ocr(pdf_path: Path, languages: list[str]) -> list[OcrPage]:  # pragma: no cover
    """OCR every page of a scanned PDF. Return per-page text + confidence.

    TODO: shell out to OCRmyPDF (recommended) or use pytesseract directly.
    Compute page-level mean confidence from per-word confidences.
    """
    raise NotImplementedError("Implement OCR extractor.")
