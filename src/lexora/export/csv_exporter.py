"""CSV exporter — emits the official OUTPUT_TEMPLATE submission schema.

Column names and order MUST match OUTPUT_TEMPLATE_31MAY.xlsx exactly ("Do not
rename columns"). One row per Citation. A separate `to_audit_csv` keeps the
richer provenance columns for internal review.
"""
from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from lexora.export.law_name import normalize_law_name
from lexora.models.citation import Citation

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
    with Path(out_path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for c in citations:
            writer.writerow([_value(c, attr) for _, attr in SUBMISSION_COLUMNS])
            n += 1
    return n


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
