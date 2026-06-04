"""Build a Citation from a validated EvidenceClaim + canonical span.

The orchestrator calls this function. The LLM verifier never does.
"""
from __future__ import annotations

from datetime import datetime

from lexora.models.citation import (
    Citation,
    Coverage,
    DiscoveryTag,
    EvidenceClaim,
    ReviewStatus,
)
from lexora.models.clause import CanonicalSpan
from lexora.models.source import RawDocument


def build_citation(
    claim: EvidenceClaim,
    span: CanonicalSpan,
    document: RawDocument,
    legal_form: str,
    article_path: str,
    *,
    economy: str = "",
    law_number: str = "",
    last_amended: str = "",
    discovery_tag: DiscoveryTag = DiscoveryTag.known,
    mapping_rationale: str = "",
    notes: str = "",
    coverage: Coverage | None = None,
    review_status: ReviewStatus = ReviewStatus.verified,
) -> Citation:
    """Assemble the final Citation. Quote text is copied from the canonical span
    — NOT from the LLM. ``claim.indicator_id`` is expected to be the submission
    code (e.g. "P6-I4")."""
    return Citation(
        economy=economy or document.jurisdiction,
        title=document.title or "",
        law_number=law_number,
        last_amended=last_amended,
        indicator_id=claim.indicator_id,
        article_path=article_path,
        discovery_tag=discovery_tag,
        page_or_dom_anchor=str(span.page_number) if span.page_number is not None else (span.dom_anchor or ""),
        quote=span.text,
        mapping_rationale=mapping_rationale,
        source_url=document.source_url,
        confidence=claim.confidence,
        notes=notes,
        clause_id=claim.clause_id,
        retrieval_timestamp=document.retrieval_timestamp,
        document_hash=document.sha256,
        jurisdiction=document.jurisdiction,
        legal_form=legal_form,
        coverage=coverage,
        char_start=span.char_start,
        char_end=span.char_end,
        review_status=review_status,
    )


def now_utc() -> datetime:
    return datetime.utcnow()
