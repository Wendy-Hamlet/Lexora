"""Database helpers.

MVP uses SQLite. Schema covers:
    documents       — RawDocument
    pages           — extracted pages with offsets + OCR confidence
    clauses         — structural units with canonical spans
    indicators      — RDTII catalog (loaded from yaml)
    claims          — EvidenceClaim per (clause, indicator)
    citations       — validated Citation
    audit_events    — every validation pass/fail for reproducibility
"""
from __future__ import annotations

from pathlib import Path


def init_db(db_path: Path) -> None:  # pragma: no cover
    """Create tables if missing.

    TODO: implement with SQLAlchemy or raw sqlite3. Schema lives in
    scripts/init_db.py.
    """
    raise NotImplementedError("Implement DB init.")
