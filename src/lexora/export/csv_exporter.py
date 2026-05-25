"""CSV exporter — one row per Citation."""
from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from lexora.models.citation import Citation

CSV_FIELDS = [
    "indicator_id",
    "clause_id",
    "jurisdiction",
    "legal_form",
    "title",
    "article_path",
    "page_or_dom_anchor",
    "quote",
    "source_url",
    "document_hash",
    "retrieval_timestamp",
    "confidence",
    "review_status",
]


def to_csv(citations: Iterable[Citation], out_path: Path) -> int:
    n = 0
    with Path(out_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for c in citations:
            row = c.model_dump(mode="json")
            writer.writerow({k: row[k] for k in CSV_FIELDS})
            n += 1
    return n
