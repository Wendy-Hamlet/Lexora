"""Tests for the verbatim contract — the core anti-hallucination boundary."""
from __future__ import annotations

from lexora.cite.validator import normalize, validate_claim
from lexora.models.citation import ReviewStatus
from lexora.models.source import SourceType


def test_normalize_collapses_whitespace_and_linebreaks():
    raw = "Cross-border\n  transfer of\t personal data."
    assert normalize(raw) == "Cross-border transfer of personal data."


def test_normalize_applies_nfkc():
    # full-width digits and parens should collapse to ascii under NFKC
    raw = "Article（１）"
    assert normalize(raw) == "Article(1)"


def test_validate_claim_passes_when_quote_matches(evidence_claim, canonical_span):
    status = validate_claim(
        claim=evidence_claim,
        span=canonical_span,
        canonical_quote=canonical_span.text,
        source_type=SourceType.primary,
    )
    assert status is ReviewStatus.verified


def test_validate_claim_tolerates_ocr_whitespace_drift(evidence_claim, canonical_span):
    drifted = canonical_span.text.replace("personal data", "personal\n data")
    status = validate_claim(
        claim=evidence_claim,
        span=canonical_span,
        canonical_quote=drifted,
        source_type=SourceType.primary,
    )
    assert status is ReviewStatus.verified


def test_validate_claim_rejects_paraphrase(evidence_claim, canonical_span):
    paraphrase = "An organisation may not export personal data outside Singapore."
    status = validate_claim(
        claim=evidence_claim,
        span=canonical_span,
        canonical_quote=paraphrase,
        source_type=SourceType.primary,
    )
    assert status is ReviewStatus.hallucinated_or_unsupported


def test_validate_claim_rejects_low_ocr_confidence(evidence_claim, canonical_span):
    low_conf_span = canonical_span.model_copy(update={"ocr_confidence": 0.5})
    status = validate_claim(
        claim=evidence_claim,
        span=low_conf_span,
        canonical_quote=low_conf_span.text,
        source_type=SourceType.primary,
    )
    assert status is ReviewStatus.low_ocr_confidence


def test_validate_claim_rejects_secondary_source(evidence_claim, canonical_span):
    status = validate_claim(
        claim=evidence_claim,
        span=canonical_span,
        canonical_quote=canonical_span.text,
        source_type=SourceType.secondary,
    )
    assert status is ReviewStatus.no_primary_source
