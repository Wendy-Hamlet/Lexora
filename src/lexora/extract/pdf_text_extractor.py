"""Text-layer PDF extractor (PyMuPDF).

Returns one PdfPage per page with the *global* character offsets needed to
build CanonicalSpans later. The global text of the document is the
concatenation of `page.text` with a single "\n\n" separator between pages —
the same convention used by `structure.legal_parser`.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

PAGE_SEPARATOR = "\n\n"

_DIGITS = re.compile(r"\d+")
_BARE_NUM = re.compile(r"\s*\d{1,4}\s*")  # a standalone page-number line


@dataclass
class PdfPage:
    page_number: int  # 1-indexed
    text: str
    char_start: int
    char_end: int
    has_text_layer: bool


def _strip_running_lines(page_texts: list[str]) -> list[str]:
    """Remove running page headers/footers that repeat across pages.

    A consolidated PDF stamps the same running head/foot on most pages — e.g. the
    Act title, "2020 Ed.", "Informal Consolidation – version in force from …" and
    a page number. When pages are concatenated these land *inside* a provision that
    spans a page break, polluting the verbatim quote and lowering similarity to the
    official text. They are detected by cross-page repetition (digit-stripped, so a
    line that differs only by page number still matches) and removed BEFORE char
    offsets are assigned, so the canonical text and every span stay consistent.

    Conservative: needs ≥4 pages; only a distinctive line (≥10 non-space chars after
    digit-stripping) repeating on ≥40% of pages is treated as a running line, plus a
    short page-number / "Ed." companion line directly adjacent to one. Operative
    clause text does not repeat verbatim on 40% of pages, so it is never removed."""
    n = len(page_texts)
    if n < 4:
        return page_texts

    def norm(line: str) -> str:
        return _DIGITS.sub("", line).strip().lower()

    page_counts: Counter[str] = Counter()
    for t in page_texts:
        seen = {norm(ln) for ln in t.split("\n") if len(norm(ln)) >= 10}
        page_counts.update(seen)
    threshold = max(3, int(0.4 * n))
    running = {key for key, c in page_counts.items() if c >= threshold}
    if not running:
        return page_texts

    cleaned: list[str] = []
    for t in page_texts:
        lines = t.split("\n")
        drop = [len(norm(ln)) >= 10 and norm(ln) in running for ln in lines]
        # Also drop a short page-number / "Ed." line touching a dropped running line
        # (the footer block is a number + "Ed." + the long phrase, in any order).
        for i, ln in enumerate(lines):
            if drop[i]:
                continue
            stripped = ln.strip()
            short = len(stripped) <= 12 and (
                _BARE_NUM.fullmatch(ln) or norm(ln) in running
                or re.fullmatch(r"\d{0,4}\s*ed\.?", stripped, re.I)
            )
            if short and ((i > 0 and drop[i - 1]) or (i + 1 < len(lines) and drop[i + 1])):
                drop[i] = True
        cleaned.append("\n".join(ln for ln, d in zip(lines, drop, strict=True) if not d))
    return cleaned


def _pages_from_doc(doc: fitz.Document) -> list[PdfPage]:
    raw = [(page.get_text("text") or "").rstrip("\f") for page in doc]
    raw = _strip_running_lines(raw)
    pages: list[PdfPage] = []
    cursor = 0
    for idx, text in enumerate(raw, start=1):
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


def extract_pdf_text(pdf_path: Path) -> list[PdfPage]:
    """Extract text-layer PDF pages with running global char offsets.

    A page whose text layer is empty/whitespace is returned with
    `has_text_layer=False` and empty text; the OCR extractor is expected to
    fill it in later (Slice 0 leaves these blank and downstream stages skip
    them — but the page slot is preserved so later OCR fills don't shift
    char offsets of subsequent pages).
    """
    with fitz.open(Path(pdf_path)) as doc:
        return _pages_from_doc(doc)


def extract_pdf_bytes(data: bytes) -> list[PdfPage]:
    """Same as `extract_pdf_text` but from in-memory bytes (e.g. a live fetch)."""
    with fitz.open(stream=data, filetype="pdf") as doc:
        return _pages_from_doc(doc)


def assemble_global_text(pages: list[PdfPage]) -> str:
    """Reconstruct the global document text whose offsets match each page's
    `char_start`/`char_end`."""
    return PAGE_SEPARATOR.join(p.text for p in pages)


__all__ = [
    "PdfPage",
    "PAGE_SEPARATOR",
    "extract_pdf_text",
    "extract_pdf_bytes",
    "assemble_global_text",
]
