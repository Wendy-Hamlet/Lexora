"""Reconstruct legal document structure into Clause objects.

Slice 0 implements the Singapore Statutes Online style: numbered sections of
the form ``26.—(1)`` or ``26.``, with optional subsections ``(2)``, ``(3)``
on continuation lines. Each clause is a section-or-subsection unit; its
CanonicalSpan points back into the global document text by char offset.

Civil-law packs (Article N. / 第N条) and ambiguity flagging are reserved for
later slices; this Slice 0 parser is good enough to drive a working end-to-end
demo on the SG PDPA. Anything that does not match a section opener is folded
into the running clause body, so prelude/recital text is preserved without
losing alignment with downstream offsets.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

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

SUBSECTION_OPENER = re.compile(r"^\s*\((?P<sub>\d+[A-Z]?)\)\s+")


def _page_for_offset(pages: Sequence[PdfPage], offset: int) -> int | None:
    """Return the 1-indexed page number that contains the given global offset."""
    for page in pages:
        if page.char_start <= offset < page.char_end or (
            page.char_start <= offset and offset == page.char_end and page is pages[-1]
        ):
            return page.page_number
    return None


def parse_structure(
    document_id: str,
    pages: Iterable[PdfPage],
) -> list[Clause]:
    """Parse PDF pages into a flat list of Clause objects.

    Each clause's span text and char offsets reference the *global* document
    text (pages joined by `PAGE_SEPARATOR`), so cite-stage validation can
    confirm the quote byte-for-byte (after NFKC + whitespace normalization)
    against the same global string.
    """
    pages = list(pages)
    if not pages:
        return []
    global_text = PAGE_SEPARATOR.join(p.text for p in pages)

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
        page = _page_for_offset(pages, start)

        if sub is None:
            sub_chunks = _split_subsections(body, start, sec)
            if sub_chunks:
                for chunk_sec, chunk_sub, chunk_start, chunk_end in sub_chunks:
                    seq += 1
                    clauses.append(
                        _make_clause(
                            document_id=document_id,
                            section=chunk_sec,
                            subsection=chunk_sub,
                            char_start=chunk_start,
                            char_end=chunk_end,
                            text=global_text[chunk_start:chunk_end],
                            page=_page_for_offset(pages, chunk_start),
                            seq=seq,
                        )
                    )
                continue

        seq += 1
        clauses.append(
            _make_clause(
                document_id=document_id,
                section=sec,
                subsection=sub,
                char_start=start,
                char_end=true_end,
                text=global_text[start:true_end],
                page=page,
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
        chunks.append((section, None, body_start, body_start + first_pos - 0))
        # Trim trailing whitespace from the head chunk if it would be empty after strip.
        head_text = body[:first_pos].rstrip()
        if not head_text.strip():
            chunks.pop()
        else:
            chunks[-1] = (section, None, body_start, body_start + len(head_text))
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


__all__ = ["parse_structure"]
