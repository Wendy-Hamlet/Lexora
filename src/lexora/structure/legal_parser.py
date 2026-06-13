"""Reconstruct legal document structure into Clause objects.

Common-law statutes number their provisions in two dominant styles:

* **dotted** — Singapore SSO / Malaysia AGC: ``26.``, ``26.—(1)``, ``129. (1)``
  (section number, a dot, optional inline subsection).
* **spaced** — Australia's Federal Register: ``13  Short title``, ``2A  Objects``
  (section number, two-plus spaces, a heading; no dot).

The parser auto-detects which style dominates a document and rejects year-like
false positives (e.g. ``2010.`` / ``1975.`` in citations or a table of contents,
which previously mis-parsed AU Acts). Each clause is a section-or-subsection
unit; its CanonicalSpan points back into the global document text by char offset
and carries a page number (PDF) or DOM anchor (HTML).

**Schedule-aware.** A consolidated Act re-starts numbering inside each Schedule,
so a flat namespace makes Schedule 1 ``s1`` collide with the main body's ``s1``
(same ``clause_id`` -> one silently clobbers the other). The parser therefore
splits the text into the main body plus one part per ``Schedule N`` header and
namespaces every Schedule clause (``::sch1-s1``, "Schedule 1 > Section 1"). The
main body keeps its original un-prefixed ids, so existing citations are stable.

Australia's Privacy Principles live *inside* Schedule 1 of the Privacy Act 1988
and are numbered ``Australian Privacy Principle 8`` with ``8.1``/``8.2`` items —
not as sections — so they were previously unaddressable. An APP-bearing schedule
is parsed in its own style: each principle becomes a citable clause
(``::sch1-app8``, "Schedule 1 > Australian Privacy Principle 8") and its items
become sub-clauses, making indicators that target a specific APP (e.g. AU 6.4 =
APP 8 cross-border disclosure) verbatim-citable.

The parser is source-agnostic: `parse_structure` works on PDF pages and
`parse_structure_html` on HTML blocks; both delegate to the same core, since the
PDF page separator and the HTML block separator are identical ("\\n\\n").

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

# Dotted style (SG/MY): "26.", "26.—(1)", "129. (1)".
# The section number takes a multi-letter suffix (AU inserts amending sections as
# 6A, 6AA, 6AB, …); allowing only one letter folded 6AA's body into section 6.
_DOTTED = re.compile(
    r"""
    (?P<sec>\d+[A-Z]{0,3})    # section number, e.g. 26, 26A, 6AA
    \.                        # literal dot
    (?:                       # optional subsection in the same line
        (?:—|--|-)?\s*   # em dash / double dash / hyphen / nothing
        \((?P<sub>\d+[A-Z]?)\)
    )?
    \s+                       # at least one whitespace before the body
    """,
    re.VERBOSE,
)

# Spaced style (AU): "13  Short title", "2A  Objects", "6AA  ..." (no dot).
_SPACED = re.compile(r"(?P<sec>\d+[A-Z]{0,3})\s{2,}(?=\S)")

# Section numbers that are really 4-digit years are almost always false positives
# (a year in a citation, a TOC dotted-leader line, a commencement date).
_YEARISH = re.compile(r"(?:19|20)\d{2}")

# Schedule heading: line-start "Schedule 1—Title". The dash (em/en/hyphen) after
# the number is the load-bearing signal: a consolidated Act repeats a *running*
# page header ("Schedule 1  Australian Privacy Principles", two spaces, no dash)
# on every page and lists schedules in a dotted TOC ("Schedule 2.......347") — the
# dash matches only the genuine divisional heading.
_SCHEDULE = re.compile(r"Schedule\s+(?P<num>\d+[A-Z]?)\s*[—–-]", re.I)

# Ordinal-word schedule heading (SG/MY consolidated Acts): "FIRST SCHEDULE",
# "SECOND SCHEDULE — Additional bases …". Mapped to a digit so it namespaces like
# the numeric form (Schedule 1). Running page headers ("FIRST SCHEDULE —
# continued") repeat on every page and must be rejected (see `_split_parts`).
_ORD_SCHEDULE = re.compile(
    r"(?P<ord>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)"
    r"\s+schedule\b",
    re.I,
)
_ORDINALS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
             "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10"}
_SCHEDULE_CONTINUED = re.compile(r"[—–-]\s*continued\b", re.I)

# Australian Privacy Principle heading inside a schedule:
# "Australian Privacy Principle 8—cross-border disclosure of personal information".
# The dash likewise separates the real heading from prose cross-references
# ("Australian Privacy Principle 8 sets out ...") and running page headers.
_APP = re.compile(r"Australian\s+Privacy\s+Principle\s+(?P<num>\d+)\s*[—–-]", re.I)

# An APP item line: "8.1 ...", "8.2  ..." (the dotted N.M numbering APPs use).
_APP_ITEM = re.compile(r"\n[ \t]*(?P<app>\d+)\.(?P<item>\d+)\s+(?=\S)")

# A table-of-contents entry: a title trailed by a dotted leader and/or page no.,
# e.g. "Schedule 1—Australian Privacy Principles .......... 55". Such a line is a
# pointer, not the real heading, so it must not open a Schedule part.
_TOC_TAIL = re.compile(r"(?:\.{2,}\s*\d*|\s{2,}\d+)\s*$")

# --- Civil-law article openers (G-1c, multi-script) --------------------------
# Civil-law codes number provisions as "articles", not common-law "sections", in
# the document's own script. Each pattern captures the number in its native form;
# `_normalize_numeral` folds it to ASCII digits for a stable clause_id (::aN). The
# winner-takes-all in `_detect_boundaries` only adopts these when they yield more
# boundaries than the dotted/spaced section styles, so an English Act that merely
# mentions "Article" is unaffected.
_ARTICLE_ZH = re.compile(r"第\s*(?P<num>[0-9０-９〇零一二三四五六七八九十百千两]+)\s*[条條]")
_ARTICLE_TH = re.compile(r"มาตรา\s*(?P<num>[0-9๐-๙]+)")
_ARTICLE_LATIN = re.compile(r"Art(?:icle|ículo|icolo|igo|\.)?\s+(?P<num>\d+)", re.I)

# All section/article openers, each tagged with the structural kind it yields.
_SECTION_OPENERS = ((_DOTTED, "section"), (_SPACED, "section"))
_ARTICLE_OPENERS = (
    (_ARTICLE_ZH, "article"), (_ARTICLE_TH, "article"), (_ARTICLE_LATIN, "article"),
)
_ALL_OPENERS = _SECTION_OPENERS + _ARTICLE_OPENERS

_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９",
                                  "0123456789")
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙",
                             "0123456789")
_CN_DIGIT = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}


def _cn_to_int(s: str) -> int:
    """Parse a Chinese numeral (十三 -> 13, 二十四 -> 24, 一百零五 -> 105). Article
    numbers stay well under 1000, so multi-section composition is enough."""
    total = current = 0
    for ch in s:
        if ch in _CN_DIGIT:
            current = _CN_DIGIT[ch]
        elif ch in _CN_UNIT:
            total += (current or 1) * _CN_UNIT[ch]
            current = 0
    return total + current


def _normalize_numeral(raw: str) -> str:
    """Fold a native-script article number to ASCII digits, or '' if unparseable."""
    s = raw.strip().translate(_FULLWIDTH_DIGITS).translate(_THAI_DIGITS)
    if s.isdigit():
        return str(int(s))
    value = _cn_to_int(s)
    return str(value) if value else ""

# Back-compat alias (the dotted opener was the original public name).
SECTION_OPENER = _DOTTED

# locate(offset) -> (page_number | None, dom_anchor | None)
Locator = Callable[[int], "tuple[int | None, str | None]"]


@dataclass
class _Segment:
    char_start: int
    char_end: int
    page: int | None
    anchor: str | None


@dataclass
class _Part:
    """A top-level division of the document: the main body, or one Schedule."""

    schedule: str | None  # None == main body
    start: int            # global offset where this part begins
    end: int


@dataclass
class _Spec:
    """A resolved clause location (global offsets), pre-id-assignment."""

    section: str | None
    subsection: str | None
    app: str | None
    item: str | None
    char_start: int
    char_end: int
    article: str | None = None  # civil-law article number (G-1c)


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


def _line_start(text: str, pos: int) -> bool:
    return pos == 0 or text[pos - 1] in {"\n", "\f"}


def _line_at(text: str, pos: int) -> str:
    end = text.find("\n", pos)
    return text[pos:] if end == -1 else text[pos:end]


def _dedupe_keep_last(headers: Iterable[tuple[int, str]]) -> list[tuple[int, str]]:
    """Collapse repeated headings for the same number to a single occurrence — the
    last one. A consolidated Act lists each Schedule/Principle once in a front
    contents block and again as the real heading further down, so the last
    occurrence is the body; this drops the contents/overview duplicates without
    modelling the TOC. Result is sorted by position."""
    last: dict[str, int] = {}
    for pos, num in headers:
        last[num] = pos
    return sorted((pos, num) for num, pos in last.items())


def _split_parts(global_text: str) -> list[_Part]:
    """Divide the document into the main body and one part per Schedule.

    Only line-start ``Schedule N`` headers count, and table-of-contents pointers
    (a header line trailed by a dotted leader / page number) are rejected. When a
    schedule number appears more than once (a TOC entry *and* the real heading),
    the last occurrence wins — the body always follows its TOC line — which drops
    TOC entries without needing to model the TOC explicitly.
    """
    candidates = [
        (m.start(), m.group("num"))
        for m in _SCHEDULE.finditer(global_text)
        if _line_start(global_text, m.start())
        and not _TOC_TAIL.search(_line_at(global_text, m.start()))
    ]
    # Ordinal-word schedules (FIRST/SECOND/…), rejecting "— continued" running page
    # headers and dotted-leader TOC pointers.
    for m in _ORD_SCHEDULE.finditer(global_text):
        line = _line_at(global_text, m.start())
        # Require the heading to be UPPERCASE ("FIRST SCHEDULE"): that is the real
        # divisional heading's style, and it excludes the mixed-case contents-list
        # entries ("First Schedule") and prose cross-references ("First Schedule or
        # Part 2 of the Second Schedule") that otherwise drag the split to the wrong
        # offset (and drop text). "— continued" running page headers are rejected too.
        if (m.group(0).isupper()
                and _line_start(global_text, m.start())
                and not _TOC_TAIL.search(line)
                and not _SCHEDULE_CONTINUED.search(line)):
            candidates.append((m.start(), _ORDINALS[m.group("ord").lower()]))
    headers = _dedupe_keep_last(candidates)
    if not headers:
        return [_Part(None, 0, len(global_text))]

    parts: list[_Part] = []
    if headers[0][0] > 0:
        parts.append(_Part(None, 0, headers[0][0]))
    for i, (start, num) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(global_text)
        parts.append(_Part(num, start, end))
    return parts


def _detect_boundaries(global_text: str) -> list[tuple[int, str, str | None, str]]:
    """Find section/article boundaries, auto-selecting the numbering style.

    Each candidate must start a line and (for sections) not be a 4-digit year.
    The dotted (SG/MY) and spaced (AU) section styles plus the civil-law article
    styles (第N条 / มาตรา N / Article N) are all scanned; the style yielding the
    most valid boundaries wins, so a document is parsed in its own style — common
    law as sections, civil law as articles — rather than a hard-coded one. Returns
    ``(pos, number, subsection_or_None, kind)`` where kind is "section"|"article".
    """
    best: list[tuple[int, str, str | None, str]] = []
    for opener, kind in _ALL_OPENERS:
        has_sub = "sub" in opener.groupindex
        numgroup = "sec" if "sec" in opener.groupindex else "num"
        found: list[tuple[int, str, str | None, str]] = []
        for match in opener.finditer(global_text):
            if not _line_start(global_text, match.start()):
                continue
            raw = match.group(numgroup)
            if kind == "section":
                if _YEARISH.fullmatch(raw):
                    continue
                num: str | None = raw
            else:
                num = _normalize_numeral(raw) or None
            if not num:
                continue
            found.append((match.start(), num, match.group("sub") if has_sub else None, kind))
        if len(found) > len(best):
            best = found
    return best


def _dedupe_boundaries(
    boundaries: list[tuple[int, str, str | None, str]],
) -> list[tuple[int, str, str | None, str]]:
    """Collapse a provision that appears twice — once in the front contents list
    and again as the real text — to its last (body) occurrence. Keyed by
    (kind, number, inline-subsection) so distinct inline subsections like
    ``26.—(1)`` and ``26.—(2)`` are preserved while a duplicated plain ``26``
    heading is not."""
    last: dict[tuple[str, str, str | None], tuple[int, str, str | None, str]] = {}
    for pos, num, sub, kind in boundaries:
        last[(kind, num, sub)] = (pos, num, sub, kind)
    # An "Arrangement of Sections" contents list (no dotted-leader/page-number, so
    # _TOC_TAIL misses it) lists each provision as a bare "N" with no operative
    # text; the body then re-introduces it as "N.—(1)". Keying by sub keeps both,
    # leaving the contents entry as a stray heading-only clause. Drop a bare
    # (kind, num, None) when the body also yields a subsectioned (kind, num, sub)
    # occurrence — the bare one is the contents-list entry. (A genuinely
    # subsection-less section, e.g. SG PDPA s.13/s.24, has no such sibling and is
    # kept; its TOC duplicate is already collapsed by the keep-last above.)
    subsectioned = {(k, n) for (k, n, s) in last if s is not None}
    pruned = [
        val for key, val in last.items()
        if not (key[2] is None and (key[0], key[1]) in subsectioned)
    ]
    return sorted(pruned)


# Lines that are NOT a section's marginal heading: page footers / running headers.
_HEADING_NOISE = re.compile(
    r"informal consolidation|version in force|\bed\.|^\d+$", re.I)


def _pull_heading_start(text: str, pos: int) -> int:
    """If a short marginal heading sits on the line directly above the section
    boundary at ``pos``, return that line's start so the heading attaches to THIS
    section instead of bleeding onto the previous clause's tail (SG/MY put the
    heading on its own line above the number: "Advisory committees\\n6.—(1) …").
    Conservative — one line back, only for a heading-like line: short, no terminal
    sentence/subsection punctuation, not itself a numbered or structural line, not
    a page footer. Otherwise return ``pos`` unchanged."""
    if pos <= 0:
        return pos
    line_end = pos - 1  # pos is a line start; pos-1 is the preceding newline
    line_start = text.rfind("\n", 0, line_end) + 1
    line = text[line_start:line_end].strip()
    if (not line or len(line) > 80
            or line[-1] in ".;:)’”\"'"
            or line[0].isdigit() or line[0] in "(["
            or re.match(r"(?i)(part|division|chapter|schedule)\b", line)
            or _HEADING_NOISE.search(line)):
        return pos
    return line_start


def _section_specs(part_text: str, offset: int) -> list[_Spec]:
    """Resolve section/article clauses within a part (offsets globalised)."""
    boundaries = _dedupe_boundaries(_detect_boundaries(part_text))
    # Re-attach each section's marginal heading (the line above its number) to that
    # section, so a clause ends at its own last sentence rather than absorbing the
    # next section's heading. Guarded so a pull never reorders/overlaps boundaries.
    pulled: list[tuple[int, str, str | None, str]] = []
    prev = -1
    for start, num, sub, kind in boundaries:
        s = _pull_heading_start(part_text, start) if kind == "section" else start
        if s <= prev:
            s = start
        pulled.append((s, num, sub, kind))
        prev = s
    boundaries = pulled
    specs: list[_Spec] = []
    for i, (start, num, sub, kind) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(part_text)
        body = part_text[start:end].rstrip()
        true_end = start + len(body)
        if kind == "article":
            # Civil-law articles are kept atomic (no common-law (N) sub-split):
            # their paragraphs (款) are usually unmarked, so the article is the
            # citable unit. char-span correctness is preserved either way.
            specs.append(_Spec(None, None, None, None, offset + start, offset + true_end, num))
            continue
        if sub is None:
            chunks = _split_markers(body, start, num, _SUBSECTION)
            if chunks:
                for csec, csub, cstart, cend in chunks:
                    specs.append(_Spec(csec, csub, None, None, offset + cstart, offset + cend))
                continue
        specs.append(_Spec(num, sub, None, None, offset + start, offset + true_end))
    return specs


def _app_specs(part_text: str, offset: int) -> list[_Spec]:
    """Resolve Australian Privacy Principle clauses within a schedule part."""
    headers = _dedupe_keep_last(
        (m.start(), m.group("num"))
        for m in _APP.finditer(part_text)
        if _line_start(part_text, m.start())
    )
    if not headers:
        return _section_specs(part_text, offset)
    specs: list[_Spec] = []
    for i, (start, app) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(part_text)
        body = part_text[start:end].rstrip()
        true_end = start + len(body)
        items = _split_app_items(body, start, app)
        if items:
            for capp, citem, cstart, cend in items:
                specs.append(_Spec(None, None, capp, citem, offset + cstart, offset + cend))
        else:
            specs.append(_Spec(None, None, app, None, offset + start, offset + true_end))
    return specs


def _parse(document_id: str, global_text: str, locate: Locator) -> list[Clause]:
    """Core: split into parts, detect boundaries per part, and emit Clauses whose
    spans reference the global text by char offset (ids namespaced by Schedule)."""
    clauses: list[Clause] = []
    for part in _split_parts(global_text):
        part_text = global_text[part.start:part.end]
        if part.schedule is not None and _APP.search(part_text):
            specs = _app_specs(part_text, part.start)
        else:
            specs = _section_specs(part_text, part.start)
        for spec in specs:
            page, anchor = locate(spec.char_start)
            clauses.append(
                _make_clause(
                    document_id=document_id,
                    schedule=part.schedule,
                    spec=spec,
                    text=global_text[spec.char_start:spec.char_end],
                    page=page,
                    dom_anchor=anchor,
                )
            )
    return _ensure_unique_ids(clauses)


def _ensure_unique_ids(clauses: list[Clause]) -> list[Clause]:
    """Guarantee clause_id is unique within a document. Dedup-by-last removes the
    bulk TOC/contents duplicates, but a single section can still carry the same
    marker twice (e.g. a definitions section that restarts a nested ``(1)`` list).
    Both chunks are real text, so rather than drop one we disambiguate the later
    id with a ``~N`` suffix — the pipeline keys clauses by id, so a collision
    would otherwise silently clobber a citable span."""
    seen: set[str] = set()
    for c in clauses:
        if c.clause_id not in seen:
            seen.add(c.clause_id)
            continue
        n = 2
        while f"{c.clause_id}~{n}" in seen:
            n += 1
        c.clause_id = f"{c.clause_id}~{n}"
        c.span.span_id = f"{c.span.span_id}~{n}"
        seen.add(c.clause_id)
    return clauses


# Line-start "(N) ..." subsection markers (SG/MY/AU bodies).
_SUBSECTION = re.compile(r"\n[ \t]*\((?P<key>\d+[A-Z]?)\)\s+")


def _split_markers(
    body: str,
    body_start: int,
    section: str,
    pattern: re.Pattern[str],
) -> list[tuple[str, str | None, int, int]]:
    """If `body` carries line-start marker lines (e.g. ``(2)`` subsections), split
    into one chunk per marker, with a leading head chunk for the pre-marker text.
    Otherwise return [].  Offsets are relative to `body_start`."""
    positions = [(m.start() + 1, m.group("key")) for m in pattern.finditer(body)]
    if not positions:
        return []
    chunks: list[tuple[str, str | None, int, int]] = []
    first_pos = positions[0][0]
    head_text = body[:first_pos].rstrip()
    if head_text.strip():
        chunks.append((section, None, body_start, body_start + len(head_text)))
    for i, (pos, key) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(body)
        slice_text = body[pos:end].rstrip()
        chunks.append((section, key, body_start + pos, body_start + pos + len(slice_text)))
    return chunks


def _split_app_items(
    body: str,
    body_start: int,
    app: str,
) -> list[tuple[str, str | None, int, int]]:
    """Split an Australian Privacy Principle body into ``N.M`` item sub-clauses
    (head chunk for the principle's preamble). Offsets relative to `body_start`."""
    positions = [
        (m.start() + 1, m.group("app"), m.group("item")) for m in _APP_ITEM.finditer(body)
    ]
    if not positions:
        return []
    chunks: list[tuple[str, str | None, int, int]] = []
    first_pos = positions[0][0]
    head_text = body[:first_pos].rstrip()
    if head_text.strip():
        chunks.append((app, None, body_start, body_start + len(head_text)))
    for i, (pos, app_no, item) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(body)
        slice_text = body[pos:end].rstrip()
        chunks.append((app_no, item, body_start + pos, body_start + pos + len(slice_text)))
    return chunks


def _ids_and_path(
    document_id: str, schedule: str | None, spec: _Spec
) -> tuple[str, str, str, str | None, str, str | None]:
    """Build (structural_path, clause_id, span_id, section_number, paragraph_number,
    article_number) for a spec, namespaced by Schedule when present."""
    sch_path = f"Schedule {schedule} > " if schedule is not None else ""
    sch_id = f"sch{schedule}-" if schedule is not None else ""

    if spec.article is not None:
        path = f"{sch_path}Article {spec.article}"
        suffix = f"a{spec.article}"
        return (path, f"{document_id}::{sch_id}{suffix}",
                f"{document_id}::span::{sch_id}{suffix}", None, None, spec.article)

    if spec.app is not None:
        if spec.item is not None:
            path = f"{sch_path}Australian Privacy Principle {spec.app}.{spec.item}"
            suffix = f"app{spec.app}-{spec.item}"
            return (path, f"{document_id}::{sch_id}{suffix}",
                    f"{document_id}::span::{sch_id}{suffix}", spec.app, spec.item, None)
        path = f"{sch_path}Australian Privacy Principle {spec.app}"
        suffix = f"app{spec.app}"
        return (path, f"{document_id}::{sch_id}{suffix}",
                f"{document_id}::span::{sch_id}{suffix}", spec.app, None, None)

    sec = spec.section
    if spec.subsection is not None:
        path = f"{sch_path}Section {sec}({spec.subsection})"
        suffix = f"s{sec}-{spec.subsection}"
        return (path, f"{document_id}::{sch_id}{suffix}",
                f"{document_id}::span::{sch_id}{suffix}", sec, spec.subsection, None)
    path = f"{sch_path}Section {sec}"
    suffix = f"s{sec}"
    return (path, f"{document_id}::{sch_id}{suffix}",
            f"{document_id}::span::{sch_id}{suffix}", sec, None, None)


def _make_clause(
    *,
    document_id: str,
    schedule: str | None,
    spec: _Spec,
    text: str,
    page: int | None,
    dom_anchor: str | None,
) -> Clause:
    path, clause_id, span_id, section_number, paragraph_number, article_number = _ids_and_path(
        document_id, schedule, spec
    )
    span = CanonicalSpan(
        span_id=span_id,
        document_id=document_id,
        page_number=page,
        dom_anchor=dom_anchor,
        char_start=spec.char_start,
        char_end=spec.char_end,
        text=text,
    )
    return Clause(
        clause_id=clause_id,
        document_id=document_id,
        structural_path=path,
        article_number=article_number,
        section_number=section_number,
        paragraph_number=paragraph_number,
        span=span,
        is_citable=True,
    )


__all__ = ["parse_structure", "parse_structure_html"]
