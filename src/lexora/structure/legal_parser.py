"""Reconstruct legal document structure into Clause objects.

Slice 0/1 implements the Singapore Statutes Online style: numbered sections of
the form ``26.—(1)`` or ``26.``, with optional subsections ``(2)``, ``(3)`` on
continuation lines. Each clause is a section-or-subsection unit; its
CanonicalSpan points back into the global document text by char offset, and
carries either a page number (PDF) or a DOM anchor (HTML) for the citation's
location reference.

The parser is source-agnostic: `parse_structure` works on PDF pages and
`parse_structure_html` on HTML blocks; both delegate to the same core, since
the PDF page separator and the HTML block separator are identical ("\\n\\n").

Civil-law packs (Article N. / 第N条) and ambiguity flagging are reserved for
later slices.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from lexora.extract.html_extractor import BLOCK_SEPARATOR, HtmlBlock
from lexora.extract.pdf_text_extractor import PAGE_SEPARATOR, PdfPage
from lexora.models.clause import CanonicalSpan, Clause

SECTION_OPENER = re.compile(
    r"""
    (?P<sec>\d+[A-Z]?)        # section number, e.g. 26 or 26A
    \.                        # literal dot
    (?:                       # optional subsection in the same line
        (?:—|--|-)?\s*   # em dash / double dash / hyphen / nothing
        \((?P<sub>\d+[A-Z]?)\)
    )?
    \s+                       # at least one whitespace before the body
    """,
    re.VERBOSE,
)

# locate(offset) -> (page_number | None, dom_anchor | None)
Locator = Callable[[int], "tuple[int | None, str | None]"]


@dataclass
class _Segment:
    char_start: int
    char_end: int
    page: int | None
    anchor: str | None


def _make_locator(segments: Sequence[_Segment]) -> Locator:
    def locate(offset: int) -> tuple[int | None, str | None]:
        for seg in segments:
            if seg.char_start <= offset < seg.char_end or (
                offset == seg.char_end and seg is segments[-1]
            ):
                return seg.page, seg.anchor
        return None, None

    return locate


def parse_structure(document_id: str, pages: Iterable[PdfPage]) -> list[Clause]:
    """Parse PDF pages into a flat list of Clause objects."""
    pages = list(pages)
    if not pages:
        return []
    global_text = PAGE_SEPARATOR.join(p.text for p in pages)
    segments = [_Segment(p.char_start, p.char_end, p.page_number, None) for p in pages]
    return _parse(document_id, global_text, _make_locator(segments))


def parse_structure_html(document_id: str, blocks: Iterable[HtmlBlock]) -> list[Clause]:
    """Parse HTML blocks into a flat list of Clause objects (DOM-anchored)."""
    blocks = list(blocks)
    if not blocks:
        return []
    global_text = BLOCK_SEPARATOR.join(b.text for b in blocks)
    segments = [_Segment(b.char_start, b.char_end, None, b.dom_anchor) for b in blocks]
    return _parse(document_id, global_text, _make_locator(segments))


def _parse(document_id: str, global_text: str, locate: Locator) -> list[Clause]:
    """Core: detect section/subsection boundaries in the global text and emit
    Clauses whose spans reference that same global text by char offset."""
    boundaries: list[tuple[int, str, str | None]] = []  # (offset, section, subsection)
    for match in SECTION_OPENER.finditer(global_text):
        sec = match.group("sec")
        sub = match.group("sub")
        if match.start() > 0 and global_text[match.start() - 1] not in {"\n", "\f"}:
            continue
        boundaries.append((match.start(), sec, sub))

    if not boundaries:
        return []

    clauses: list[Clause] = []
    seq = 0
    for i, (start, sec, sub) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(global_text)
        body = global_text[start:end].rstrip()
        true_end = start + len(body)

        if sub is None:
            sub_chunks = _split_subsections(body, start, sec)
            if sub_chunks:
                for chunk_sec, chunk_sub, chunk_start, chunk_end in sub_chunks:
                    seq += 1
                    page, anchor = locate(chunk_start)
                    clauses.append(
                        _make_clause(
                            document_id=document_id,
                            section=chunk_sec,
                            subsection=chunk_sub,
                            char_start=chunk_start,
                            char_end=chunk_end,
                            text=global_text[chunk_start:chunk_end],
                            page=page,
                            dom_anchor=anchor,
                            seq=seq,
                        )
                    )
                continue

        seq += 1
        page, anchor = locate(start)
        clauses.append(
            _make_clause(
                document_id=document_id,
                section=sec,
                subsection=sub,
                char_start=start,
                char_end=true_end,
                text=global_text[start:true_end],
                page=page,
                dom_anchor=anchor,
                seq=seq,
            )
        )

    return clauses


def _split_subsections(
    body: str,
    body_start: int,
    section: str,
) -> list[tuple[str, str | None, int, int]]:
    """If `body` contains line-start `(N) ...` subsection markers, split into
    one chunk per subsection. Otherwise return []."""
    chunks: list[tuple[str, str | None, int, int]] = []
    positions: list[tuple[int, str | None]] = []
    for line_match in re.finditer(r"\n[ \t]*\((?P<sub>\d+[A-Z]?)\)\s+", body):
        positions.append((line_match.start() + 1, line_match.group("sub")))
    if not positions:
        return []
    first_pos = positions[0][0]
    if first_pos > 0:
        head_text = body[:first_pos].rstrip()
        if head_text.strip():
            chunks.append((section, None, body_start, body_start + len(head_text)))
    for i, (pos, sub) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(body)
        slice_text = body[pos:end].rstrip()
        chunks.append((section, sub, body_start + pos, body_start + pos + len(slice_text)))
    return chunks


def _make_clause(
    *,
    document_id: str,
    section: str,
    subsection: str | None,
    char_start: int,
    char_end: int,
    text: str,
    page: int | None,
    dom_anchor: str | None,
    seq: int,
) -> Clause:
    if subsection is not None:
        structural_path = f"Section {section}({subsection})"
        clause_id = f"{document_id}::s{section}-{subsection}"
        span_id = f"{document_id}::span::s{section}-{subsection}"
        paragraph_number = subsection
    else:
        structural_path = f"Section {section}"
        clause_id = f"{document_id}::s{section}"
        span_id = f"{document_id}::span::s{section}"
        paragraph_number = None

    span = CanonicalSpan(
        span_id=span_id,
        document_id=document_id,
        page_number=page,
        dom_anchor=dom_anchor,
        char_start=char_start,
        char_end=char_end,
        text=text,
    )
    return Clause(
        clause_id=clause_id,
        document_id=document_id,
        structural_path=structural_path,
        section_number=section,
        paragraph_number=paragraph_number,
        span=span,
        is_citable=True,
    )


__all__ = ["parse_structure", "parse_structure_html"]
