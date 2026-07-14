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
_BAND = 2  # lines at the top / bottom of a page that a running head can occupy


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

    Needs ≥4 pages. Furniture is convicted on two independent signatures, because
    neither alone covers the real documents:

    1. DISTINCTIVE + REPEATED — a long line (≥10 non-space chars after digit-
       stripping) on ≥40% of pages, anywhere on the page. Operative clause text does
       not repeat verbatim on 40% of pages, so prose is never caught here.

    2. POSITIONAL + REPEATED — a line in the page's top or bottom band on ≥40% of
       pages, AT ANY LENGTH. Length alone cannot convict a short line (body text is
       full of short lines), but *position* can: a header sits at the top of every
       page; "(a)" does not. This is the only rule that reaches a header made solely
       of short lines — Malaysia's statutes head each page with a bare "Act 762",
       which rule 1 normalises to "act" (3 chars) and lets through.

    Then the REST OF THE BLOCK: a short line adjacent to one already dropped, which
    also repeats across pages. Headers wrap — Singapore's SSO stamps "Personal Data
    Protection" / "Act 2012" as two lines, and dropping only the long half left
    "Act 2012" behind, a fragment that reads exactly like a law number and which the
    metadata extractor duly copied ("Act 2012" for "Act 26 of 2012"). Adjacency is
    what keeps this safe: a short line only qualifies if it touches a line already
    proven to be furniture. It subsumes the page-number and "2020 Ed." companions.

    All of it runs BEFORE char offsets are assigned, so the canonical text and every
    span stay consistent. On a scanned PDF the pages are blank here and nothing can
    be detected; `ocr_extractor.ocr_fill_pages` re-runs this once OCR has filled them.
    """
    n = len(page_texts)
    if n < 4:
        return page_texts

    def norm(line: str) -> str:
        return _DIGITS.sub("", line).strip().lower()

    threshold = max(3, int(0.4 * n))

    def band_indices(lines: list[str]) -> set[int]:
        """Indices of the top/bottom band. Empty when the page is too short for the
        band to mean anything — on a 4-line page it would span the whole page, and
        position would convict body text."""
        filled = [i for i, ln in enumerate(lines) if norm(ln)]
        if len(filled) <= 2 * _BAND:
            return set()
        return set(filled[:_BAND]) | set(filled[-_BAND:])

    anywhere: Counter[str] = Counter()
    in_band: Counter[str] = Counter()
    for t in page_texts:
        lines = t.split("\n")
        anywhere.update({norm(ln) for ln in lines if norm(ln)})
        in_band.update({norm(lines[i]) for i in band_indices(lines)})

    running = {k for k, c in anywhere.items() if c >= threshold and len(k) >= 10}
    positional = {k for k, c in in_band.items() if c >= threshold}
    repeating = {k for k, c in anywhere.items() if c >= threshold}
    if not running and not positional:
        return page_texts

    cleaned: list[str] = []
    for t in page_texts:
        lines = t.split("\n")
        band = band_indices(lines)
        drop = [
            (len(norm(ln)) >= 10 and norm(ln) in running)
            or (i in band and norm(ln) in positional)
            for i, ln in enumerate(lines)
        ]
        for i, ln in enumerate(lines):
            if drop[i]:
                continue
            stripped = ln.strip()
            companion = len(stripped) <= 12 and (
                _BARE_NUM.fullmatch(ln) or norm(ln) in repeating
                or re.fullmatch(r"\d{0,4}\s*ed\.?", stripped, re.I)
            )
            if companion and (
                (i > 0 and drop[i - 1]) or (i + 1 < len(lines) and drop[i + 1])
            ):
                drop[i] = True
        cleaned.append("\n".join(ln for ln, d in zip(lines, drop, strict=True) if not d))
    return cleaned


# Document-final, non-operative trailing matter that the last section would
# otherwise absorb: a government-printer imprint/colophon (e.g. Malaysia's
# "DICETAK OLEH PERCETAKAN NASIONAL MALAYSIA BERHAD … BAGI PIHAK DAN DENGAN
# PERINTAH KERAJAAN MALAYSIA") and the consolidated-reprint amendment tables
# ("LIST OF AMENDMENTS" / "LIST OF SECTIONS AMENDED" / "TABLE OF AMENDMENTS").
# These markers head end-matter only; none opens an operative provision.
_DOC_TAIL = re.compile(
    r"(?im)^[ \t]*(?:dicetak\s+oleh|printed\s+by|printed\s+for|by\s+authority"
    r"|percetakan\s+nasional|bagi\s+pihak\s+dan\s+dengan\s+perintah"
    r"|government\s+printer|list\s+of\s+(?:sections\s+)?amendments?"
    r"|list\s+of\s+sections\s+amended|table\s+of\s+amendments)\b")


def _strip_trailing_matter(page_texts: list[str]) -> list[str]:
    """Remove document-final non-operative trailing matter (printer imprint /
    colophon, consolidated-reprint amendment tables) so the final section does not
    absorb it. Walks back over the final contiguous run of pages that each carry an
    end-matter marker; truncates the first such page at its marker (keeping any
    operative prose before it) and blanks the rest. A page whose pre-marker text
    has no sentence terminator is pure end-matter (running title + table heading)
    and is blanked entirely. Operative text — which always ends in a sentence — is
    never removed."""
    n = len(page_texts)
    cut: int | None = None
    for idx in range(n - 1, -1, -1):
        if not page_texts[idx].strip():
            continue  # skip trailing blank pages
        if _DOC_TAIL.search(page_texts[idx]):
            cut = idx
            continue
        break  # first substantive (operative) page from the end ends the run
    if cut is None:
        return page_texts
    m = _DOC_TAIL.search(page_texts[cut])
    assert m is not None
    before = page_texts[cut][: m.start()]
    page_texts[cut] = before.rstrip() if re.search(r"[.;]", before) else ""
    for idx in range(cut + 1, n):
        page_texts[idx] = ""
    return page_texts


# legislation.gov.au stamps a per-page navigational header that names the current
# Section/Clause and the enclosing Part/Division/Schedule, e.g. "Section 26WR",
# "Notification of eligible data breaches  Part IIIC", "Schedule 1  Australian
# Privacy Principles". Because each line names a different unit it appears on only
# a handful of pages, so the cross-page-repetition cleaner (≥40% of pages) leaves
# it, and it then bleeds into provisions that span a page break. It is recognised
# structurally instead: a lone "Section/Clause N" line, or a TWO-SPACE-celled line
# one of whose cells is a bare "Part/Division/Schedule N" token. The two-space
# layout and Title-case distinguish it from a real divisional heading, which the
# parser needs: SG's is ALL-CAPS ("PART 4") and AU's carries an em dash
# ("Part 3—Dealing with personal information"); both are preserved by the dash
# guard and the case-sensitive token.
_NAV_SECTION = re.compile(r"(?:Section|Clause)\s+\d+[A-Z]*\Z")
_NAV_STRUCT = re.compile(r"(?:Part|Division|Schedule)\s+[0-9IVXLC]+[A-Z]?\Z")


def _is_nav_header(line: str) -> bool:
    s = line.strip()
    if not s or "—" in s or "–" in s:  # a dash marks a real divisional heading
        return False
    if _NAV_SECTION.fullmatch(s):
        return True
    if "  " in s:
        cells = [c for c in re.split(r"\s{2,}", s) if c.strip()]
        if len(cells) == 2 and any(_NAV_STRUCT.fullmatch(c.strip()) for c in cells):
            return True
    return False


def _strip_nav_headers(page_texts: list[str]) -> list[str]:
    """Drop legislation.gov.au navigational running-header lines (see `_is_nav_header`)
    that the repetition cleaner misses because each names a different unit."""
    return [
        "\n".join(ln for ln in t.split("\n") if not _is_nav_header(ln))
        for t in page_texts
    ]


def _pages_from_doc(doc: fitz.Document) -> list[PdfPage]:
    raw = [(page.get_text("text") or "").rstrip("\f") for page in doc]
    raw = _strip_running_lines(raw)
    raw = _strip_nav_headers(raw)
    raw = _strip_trailing_matter(raw)
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
