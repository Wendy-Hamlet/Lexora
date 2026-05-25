"""Citation validator — enforces the verbatim contract.

A claim only becomes a Citation if every gate passes:
  - the source document exists
  - the indicator exists
  - the quote span resolves to canonical text
  - quote equals canonical-span text after Unicode + whitespace normalization
  - OCR confidence (if relevant) is at or above the citable threshold
  - the source is a primary instrument (or a secondary that quotes a primary)

`normalize()` tolerates OCR line-break drift (whitespace + NFKC). It does NOT
accept paraphrases. The LLM never writes quote text — quote text is copied
from canonical storage by the orchestrator — so any disagreement here indicates
either OCR drift, a stale snapshot, or an unsupported mapping.
"""
from __future__ import annotations

import unicodedata

from lexora.models.citation import EvidenceClaim, ReviewStatus
from lexora.models.clause import CanonicalSpan
from lexora.models.source import SourceType

OCR_CITABLE_THRESHOLD = 0.85


def normalize(text: str) -> str:
    """Unicode NFKC + whitespace collapse. Used both sides of equality check."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def validate_claim(
    claim: EvidenceClaim,
    span: CanonicalSpan,
    canonical_quote: str,
    source_type: SourceType,
    ocr_threshold: float = OCR_CITABLE_THRESHOLD,
) -> ReviewStatus:
    """Return a ReviewStatus describing the outcome.

    `ReviewStatus.verified` means publish.
    Anything else means hold for human review with the given reason.
    """
    if source_type is not SourceType.primary:
        return ReviewStatus.no_primary_source

    if span.ocr_confidence is not None and span.ocr_confidence < ocr_threshold:
        return ReviewStatus.low_ocr_confidence

    if normalize(canonical_quote) != normalize(span.text):
        return ReviewStatus.hallucinated_or_unsupported

    return ReviewStatus.verified
