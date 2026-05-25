"""Stage 2 — extraction.

Turns RawDocument bytes into canonical per-page text plus character offsets,
bounding boxes when available, and page-level OCR confidence.

Dispatch by content_type:
    text/html               → html_extractor
    application/pdf (text)  → pdf_text_extractor
    application/pdf (scan)  → ocr_extractor

Pages below LEXORA_OCR_CITABLE_THRESHOLD are marked non-citable.
"""
