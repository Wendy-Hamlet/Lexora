"""Build a Citation from a validated EvidenceClaim + canonical span.

The orchestrator calls this function. The LLM verifier never does.
"""
from __future__ import annotations

from datetime import datetime

from lexora.models.citation import Citation, EvidenceClaim, ReviewStatus
from lexora.models.clause import CanonicalSpan
from lexora.models.source import RawDocument


def build_citation(
    claim: EvidenceClaim,
    span: CanonicalSpan,
    document: RawDocument,
    legal_form: str,
    article_path: str,
    review_status: ReviewStatus = ReviewStatus.verified,
) -> Citation:
    """Assemble the final Citation. Quote text is copied from the canonical span
    — NOT from the LLM."""
    return Citation(
        indicator_id=claim.indicator_id,
        clause_id=claim.clause_id,
        source_url=document.source_url,
        retrieval_timestamp=document.retrieval_timestamp,
        document_hash=document.sha256,
        title=document.title or "",
        jurisdiction=document.jurisdiction,
        legal_form=legal_form,
        article_path=article_path,
        page_or_dom_anchor=str(span.page_number) if span.page_number is not None else (span.dom_anchor or ""),
        char_start=span.char_start,
        char_end=span.char_end,
        quote=span.text,
        confidence=claim.confidence,
        review_status=review_status,
    )


def now_utc() -> datetime:
    return datetime.utcnow()
