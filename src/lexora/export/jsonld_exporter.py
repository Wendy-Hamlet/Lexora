"""JSON-LD exporter — newline-delimited JSON-LD records."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from lexora.models.citation import Citation

JSONLD_CONTEXT = {
    "@vocab": "https://lexora.example/schema#",
    "indicator_id": "rdtii:indicator",
    "clause_id": "lex:clause",
    "source_url": "schema:url",
    "document_hash": "lex:sha256",
    "quote": "schema:text",
}


def to_jsonld(citations: Iterable[Citation], out_path: Path) -> int:
    """Write one JSON-LD object per line. Returns count written."""
    n = 0
    with Path(out_path).open("w", encoding="utf-8") as f:
        for c in citations:
            obj = {"@context": JSONLD_CONTEXT, **c.model_dump(mode="json")}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n
