"""Reconstruct legal document structure (title / chapter / article / section /
paragraph) into Clause objects."""
from __future__ import annotations

from collections.abc import Iterable

from lexora.models.clause import Clause


def parse_structure(document_id: str, raw_text: str) -> Iterable[Clause]:  # pragma: no cover
    """Walk raw text, detect structural headings, yield Clause objects.

    Heuristics differ by jurisdiction / language. Suggested approach:
      1. Try language-specific regex packs first (e.g. "Article \\d+", "第\\d+条").
      2. Fall back to indentation / numbering patterns.
      3. When ambiguous, emit a Clause flagged for manual review rather than
         silently merging into the parent.

    TODO: implement language-aware heading detection.
    """
    raise NotImplementedError("Implement legal structure parser.")
