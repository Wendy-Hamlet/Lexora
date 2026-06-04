"""End-to-end pipeline orchestrator.

Drives the six stages without an LLM in the loop:

    fetch / ingest  →  extract (PDF text | HTML)  →  parse_structure
                    →  build_index + retrieve_candidates
                    →  validate_claim + build_citation
                    →  export

The orchestrator owns the anti-hallucination invariant: the EvidenceClaim
carries only IDs (`clause_id`, `quote_span_id`), and quote text is copied
verbatim from the canonical CanonicalSpan via `build_citation`. Even without an
LLM yet, every quote that reaches the export already passes through the same
validator the LLM verifier will use.

Two entry points:
  * `run_demo_pipeline`     — from a local PDF (manual-upload fallback).
  * `run_pipeline_from_url` — from a LIVE fetch (the path that earns crawl
                              points), routing PDF vs HTML by content type.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from lexora.cite.citation_builder import build_citation
from lexora.cite.validator import validate_claim
from lexora.classify.retrieval import BM25Index, build_index, retrieve_candidates
from lexora.collect.crawler import fetch, ingest_local_file
from lexora.extract.html_extractor import HtmlBlock, extract_html
from lexora.extract.pdf_text_extractor import PdfPage, extract_pdf_bytes, extract_pdf_text
from lexora.models.citation import Citation, ClaimLabel, EvidenceClaim, ReviewStatus
from lexora.models.clause import Clause
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import RawDocument, SourceProfile, SourceType
from lexora.structure.legal_parser import parse_structure, parse_structure_html


@dataclass
class DemoArtifacts:
    """Everything a run produces, kept together so callers (CLI, tests, UI) can
    inspect each stage. `pages` is set for PDF inputs, `blocks` for HTML."""

    document: RawDocument
    clauses: list[Clause]
    citations: list[Citation]
    pages: list[PdfPage] = field(default_factory=list)
    blocks: list[HtmlBlock] = field(default_factory=list)


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
    """Run extract→structure→retrieve→cite for one local PDF."""
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
    citations = _citations_from_clauses(
        clauses, document, profile, indicators, legal_form, top_k, min_score
    )
    return DemoArtifacts(document=document, clauses=clauses, citations=citations, pages=pages)


def run_pipeline_from_url(
    *,
    url: str,
    profile: SourceProfile,
    indicators: list[RDTIIIndicator],
    portal_name: str = "live-fetch",
    source_type: SourceType = SourceType.primary,
    title: str | None = None,
    legal_form: str = "statute",
    dest_dir: Path | None = None,
    top_k: int = 1,
    min_score: float = 0.1,
    **fetch_kwargs,
) -> DemoArtifacts:
    """Live fetch a URL and run the full pipeline, routing PDF vs HTML.

    A non-2xx response yields empty clauses (status captured on the document)
    instead of raising — the caller can inspect ``document.http_status`` and
    fall back to a browser engine in a later slice.
    """
    dest_dir = dest_dir or (Path("data") / "raw" / profile.iso_code.lower())
    result = fetch(
        url,
        jurisdiction=profile.iso_code,
        portal_name=portal_name,
        source_type=source_type,
        dest_dir=dest_dir,
        title=title,
        **fetch_kwargs,
    )
    document = result.document

    pages: list[PdfPage] = []
    blocks: list[HtmlBlock] = []
    clauses: list[Clause] = []

    if 200 <= document.http_status < 300:
        if result.is_pdf():
            pages = extract_pdf_bytes(result.body)
            clauses = parse_structure(document.document_id, pages)
        elif result.is_html():
            blocks = extract_html(result.body, url)
            clauses = parse_structure_html(document.document_id, blocks)

    citations = _citations_from_clauses(
        clauses, document, profile, indicators, legal_form, top_k, min_score
    )
    return DemoArtifacts(
        document=document, clauses=clauses, citations=citations, pages=pages, blocks=blocks
    )


def _citations_from_clauses(
    clauses: list[Clause],
    document: RawDocument,
    profile: SourceProfile,
    indicators: list[RDTIIIndicator],
    legal_form: str,
    top_k: int,
    min_score: float,
) -> list[Citation]:
    """Shared core: per indicator, retrieve top-k clauses and materialize the
    ones that pass the verbatim validator."""
    citations: list[Citation] = []
    if not clauses:
        return citations
    index: BM25Index = build_index(clauses)
    clause_by_id = {c.clause_id: c for c in clauses}
    for indicator in indicators:
        for hit in retrieve_candidates(indicator, profile, index, top_k=top_k):
            if hit.score < min_score:
                continue
            citation = _materialize(
                indicator=indicator,
                clause=clause_by_id[hit.clause_id],
                document=document,
                legal_form=legal_form,
                economy=profile.jurisdiction,
                bm25_score=hit.score,
            )
            if citation is not None:
                citations.append(citation)
    return citations


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
    claim = EvidenceClaim(
        indicator_id=indicator.submission_id,
        clause_id=clause.clause_id,
        quote_span_id=clause.span.span_id,
        label=ClaimLabel.match,
        confidence=_normalize_score(bm25_score),
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
    field. tanh(score/5) maps a strong-match BM25 score (~5+) into the high-0.9s
    and never saturates to 1.0."""
    return float(max(0.0, min(1.0, math.tanh(score / 5.0))))


__all__ = ["DemoArtifacts", "run_demo_pipeline", "run_pipeline_from_url"]
