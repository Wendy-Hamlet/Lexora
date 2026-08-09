"""CSV exporter — emits the official OUTPUT_TEMPLATE submission schema.

Column names and order MUST match OUTPUT_TEMPLATE_31MAY.xlsx exactly ("Do not
rename columns"). One row per Citation. A separate `to_audit_csv` keeps the
richer provenance columns for internal review.
"""
from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from pathlib import Path

from lexora.export.law_name import normalize_law_name
from lexora.models.citation import Citation

logger = logging.getLogger(__name__)

# A spreadsheet cell holds 32,767 characters; Excel truncates a longer one on paste,
# without saying so. The official template is an xlsx, so a provision longer than this
# reaches a reviewer silently cut off — and the Verbatim Snippet is the column they check
# by hand. Measured on the round-1 submission: 2 of 669 rows exceed it (48,078 and 42,165
# characters, both OAIC guidance chapters that parsed into one enormous "section").
#
# Deliberately a WARNING, not a truncation. The snippet is the clause's exact span and is
# published beside its char offsets; cutting it here would break the correspondence that
# makes it checkable. Better that the run says so than that a spreadsheet decides quietly.
_SPREADSHEET_CELL_LIMIT = 32_767

# (header label, Citation attribute) — order is the submission order.
SUBMISSION_COLUMNS: list[tuple[str, str]] = [
    ("Economy", "economy"),
    ("Law Name", "title"),
    ("Law Number / Ref", "law_number"),
    ("Last Amended", "last_amended"),
    ("Indicator ID", "indicator_id"),
    ("Article / Section", "article_path"),
    ("Discovery Tag", "discovery_tag"),
    ("Location Reference", "page_or_dom_anchor"),
    ("Verbatim Snippet", "quote"),
    ("Mapping Rationale", "mapping_rationale"),
    ("Source URL", "source_url"),
    ("Confidence", "confidence"),
    ("Notes", "notes"),
]


def _value(citation: Citation, attr: str) -> str:
    v = getattr(citation, attr)
    if v is None:
        return ""
    if attr == "title":
        return normalize_law_name(v)
    # enums -> their value; everything else -> str
    return getattr(v, "value", str(v))


def to_csv(citations: Iterable[Citation], out_path: Path) -> int:
    """Write the submission CSV (exact 13-column template). Returns rows written."""
    headers = [label for label, _ in SUBMISSION_COLUMNS]
    n = 0
    oversized: list[tuple[str, str, int]] = []
    # Column health, tracked while writing. O(1) per column: the first value seen, whether
    # anything ever differed from it, and how many cells were blank.
    seen: dict[str, tuple[str, bool, int]] = {}
    with Path(out_path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for c in citations:
            row = [_value(c, attr) for _, attr in SUBMISSION_COLUMNS]
            writer.writerow(row)
            n += 1
            for (label, _), value in zip(SUBMISSION_COLUMNS, row, strict=True):
                if len(value) > _SPREADSHEET_CELL_LIMIT:
                    oversized.append((label, c.indicator_id, len(value)))
                first, varied, blanks = seen.get(label, (value, False, 0))
                seen[label] = (first, varied or value != first,
                               blanks + (not value.strip()))
    if oversized:
        worst = max(oversized, key=lambda t: t[2])
        logger.warning(
            "%d cell(s) exceed the %d-character spreadsheet limit and will be silently "
            "truncated if this CSV is opened in Excel (largest: %s on %s, %d chars). The "
            "data is intact in the CSV and the JSON sidecar.",
            len(oversized), _SPREADSHEET_CELL_LIMIT, worst[0], worst[1], worst[2],
        )
    _report_dead_columns(seen, n)
    return n


# Columns that are legitimately the same on every row of a single-economy run.
_EXPECTED_CONSTANT = {"Economy"}
# `Notes` carries per-row caveats. Having none is the correct, healthy state, not a dead
# column, so an empty Notes says nothing is wrong.
_MAY_BE_BLANK = {"Notes"}
# Below this, "every row is the same" is not evidence of anything -- a 2-row export is
# constant in almost every column by construction, and firing there would train a reader to
# skip the warning on the 700-row run where it means something. A warning people learn to
# ignore protects nothing, which is the same reason `Economy` is excluded above.
_MIN_ROWS_FOR_COLUMN_HEALTH = 20


def _report_dead_columns(seen: dict[str, tuple[str, bool, int]], rows: int) -> None:
    """Say when a column carries no information.

    A column that is blank on every row does not read as "we did not look" -- it reads as
    "there is nothing there", which for `Law Number / Ref` means a reader concludes the Act
    has no number. Measured 2026-08-09: without `--metadata-llm`, `Law Number / Ref` and
    `Last Amended` are empty on all 181 rows of a Singapore run, because the portal
    publishes no structured metadata and nothing else fills them.

    This is deliberately at export time rather than in a test. The lesson it encodes is the
    one that cost us the 2026-08-03 pitch: every column of a deliverable has to be looked at
    before it is handed over, and "someone remembers to look" is not a mechanism. A unit
    test cannot see the file a real run just produced.
    """
    if rows < _MIN_ROWS_FOR_COLUMN_HEALTH:
        return
    for label, (_first, varied, blanks) in seen.items():
        if blanks == rows and label not in _MAY_BE_BLANK:
            logger.warning(
                "column %r is EMPTY on all %d row(s) -- a reader cannot tell that from "
                "'this instrument has no such value'. Check the layer that fills it.",
                label, rows,
            )
        elif not varied and label not in _EXPECTED_CONSTANT:
            logger.warning(
                "column %r is the same value on all %d row(s); a constant column carries "
                "no information and may mean the gate or scale behind it is inert.",
                label, rows,
            )


AUDIT_FIELDS = [
    "economy", "title", "law_number", "last_amended", "indicator_id",
    "article_path", "discovery_tag", "page_or_dom_anchor", "quote",
    "mapping_rationale", "source_url", "confidence", "notes",
    "clause_id", "jurisdiction", "legal_form", "coverage",
    "char_start", "char_end", "document_hash", "retrieval_timestamp",
    "review_status",
    "currency_status", "amended_by", "amendments_incorporated_to", "amendment_text",
    "source_version",
]


def to_audit_csv(citations: Iterable[Citation], out_path: Path) -> int:
    """Write the richer internal audit CSV (all provenance columns)."""
    n = 0
    with Path(out_path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for c in citations:
            row = c.model_dump(mode="json")
            writer.writerow({k: row.get(k, "") for k in AUDIT_FIELDS})
            n += 1
    return n
