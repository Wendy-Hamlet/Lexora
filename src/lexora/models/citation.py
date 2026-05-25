"""Citation + claim models — the on-the-wire contract.

`EvidenceClaim` is what the LLM verifier returns. Notice it has NO quote text —
quote text is copied from canonical storage by the orchestrator, never written
by the model.

`Citation` is what reaches the audit UI / JSON-LD export, after the validator
has confirmed all gates.

See docs/citation_schema.md and docs/anti_hallucination.md.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


class ClaimLabel(str, Enum):
    match = "match"
    no_match = "no_match"
    uncertain = "uncertain"


class ReviewStatus(str, Enum):
    verified = "VERIFIED"
    low_ocr_confidence = "LOW_OCR_CONFIDENCE"
    conflict_review = "CONFLICT_REVIEW"
    no_primary_source = "NO_PRIMARY_SOURCE_FOUND"
    hallucinated_or_unsupported = "HALLUCINATED_OR_UNSUPPORTED_MAPPING"


class EvidenceClaim(BaseModel):
    """The LLM verifier's output.

    The model selects IDs and a label; the orchestrator fills the quote from
    canonical storage. The model cannot produce free text by design.
    """

    indicator_id: str
    clause_id: str
    quote_span_id: str
    label: ClaimLabel
    confidence: float = Field(ge=0.0, le=1.0)


class Citation(BaseModel):
    """Final, validated citation released to the audit UI / exports."""

    indicator_id: str
    clause_id: str
    source_url: HttpUrl
    retrieval_timestamp: datetime
    document_hash: str  # sha256:<hex>
    title: str
    jurisdiction: str
    legal_form: str  # statute | regulation | gazette | treaty | guideline
    article_path: str
    page_or_dom_anchor: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    quote: str  # copied from canonical span by the orchestrator
    confidence: float = Field(ge=0.0, le=1.0)
    review_status: ReviewStatus = ReviewStatus.verified
