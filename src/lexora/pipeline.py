"""End-to-end Slice 0 pipeline orchestrator.

Drives the six stages without an LLM in the loop:

    ingest_local_file  →  extract_pdf_text  →  parse_structure
                       →  build_index + retrieve_candidates
                       →  validate_claim + build_citation
                       →  to_jsonld

The orchestrator owns the anti-hallucination invariant: the EvidenceClaim
carries only IDs (`clause_id`, `quote_span_id`), and quote text is copied
verbatim from the canonical CanonicalSpan via `build_citation`. Even though
Slice 0 has no LLM yet, every quote that reaches the export already passes
through the same validator the LLM verifier will use in Slice 1+.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lexora.cite.citation_builder import build_citation
from lexora.cite.validator import validate_claim
from lexora.classify.retrieval import BM25Index, build_index, retrieve_candidates
from lexora.collect.crawler import ingest_local_file
from lexora.extract.pdf_text_extractor import PdfPage, extract_pdf_text
from lexora.models.citation import Citation, ClaimLabel, EvidenceClaim, ReviewStatus
from lexora.models.clause import Clause
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import RawDocument, SourceProfile, SourceType
from lexora.structure.legal_parser import parse_structure


@dataclass
class DemoArtifacts:
    """Everything Slice 0 produces, kept together so callers (CLI, tests, UI)
    can inspect each stage."""

    document: RawDocument
    pages: list[PdfPage]
    clauses: list[Clause]
    citations: list[Citation]


def run_demo_pipeline(
    *,
    pdf_path: Path,
    profile: SourceProfile,
    indicators: list[RDTIIIndicator],
    source_url: str,
    portal_name: str,
    title: str | None = None,
    legal_form: str = "statute",
    dest_dir: Path | None = None,
    top_k: int = 1,
    min_score: float = 0.1,
) -> DemoArtifacts:
    """Run collect→extract→structure→retrieve→cite→(returned) for one PDF."""
    dest_dir = dest_dir or (Path("data") / "raw" / profile.iso_code.lower())

    document = ingest_local_file(
        pdf_path,
        jurisdiction=profile.iso_code,
        portal_name=portal_name,
        source_url=source_url,
        source_type=SourceType.primary,
        dest_dir=dest_dir,
        title=title,
    )

    pages = extract_pdf_text(pdf_path)
    clauses = parse_structure(document.document_id, pages)

    citations: list[Citation] = []
    if not clauses:
        return DemoArtifacts(document=document, pages=pages, clauses=clauses, citations=citations)

    index: BM25Index = build_index(clauses)
    clause_by_id = {c.clause_id: c for c in clauses}

    for indicator in indicators:
        hits = retrieve_candidates(indicator, profile, index, top_k=top_k)
        for hit in hits:
            if hit.score < min_score:
                continue
            clause = clause_by_id[hit.clause_id]
            citation = _materialize(
                indicator=indicator,
                clause=clause,
                document=document,
                legal_form=legal_form,
                economy=profile.jurisdiction,
                bm25_score=hit.score,
            )
            if citation is not None:
                citations.append(citation)

    return DemoArtifacts(document=document, pages=pages, clauses=clauses, citations=citations)


def _materialize(
    *,
    indicator: RDTIIIndicator,
    clause: Clause,
    document: RawDocument,
    legal_form: str,
    economy: str,
    bm25_score: float,
) -> Citation | None:
    """One-clause-one-indicator: synthesize a verifier-shaped claim and run it
    through the same validator the real LLM verifier will use. The claim carries
    the official submission code (e.g. "P6-I4") as its indicator_id."""
    confidence = _normalize_score(bm25_score)
    claim = EvidenceClaim(
        indicator_id=indicator.submission_id,
        clause_id=clause.clause_id,
        quote_span_id=clause.span.span_id,
        label=ClaimLabel.match,
        confidence=confidence,
    )
    status = validate_claim(
        claim=claim,
        span=clause.span,
        canonical_quote=clause.span.text,
        source_type=document.source_type,
    )
    if status is ReviewStatus.hallucinated_or_unsupported:
        return None
    return build_citation(
        claim=claim,
        span=clause.span,
        document=document,
        legal_form=legal_form,
        article_path=clause.structural_path,
        economy=economy,
        review_status=status,
    )


def _normalize_score(score: float) -> float:
    """Squash unbounded BM25 scores into [0, 1] for the Pydantic confidence
    field. A tanh-on-fifth maps a typical strong-match BM25 score (~5+) into
    the high-0.9s and never saturates to 1.0."""
    import math

    return float(max(0.0, min(1.0, math.tanh(score / 5.0))))


__all__ = ["DemoArtifacts", "run_demo_pipeline"]
