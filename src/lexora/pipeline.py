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
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from lexora.cite.citation_builder import build_citation
from lexora.cite.metadata import MetadataExtractor
from lexora.cite.rationale import RationaleGenerator, template_rationale
from lexora.cite.validator import validate_claim
from lexora.classify.boundaries import admits_clause
from lexora.classify.retrieval import BM25Index, build_index, retrieve_candidates
from lexora.collect.crawler import fetch, ingest_local_file
from lexora.extract.html_extractor import HtmlBlock, extract_html
from lexora.extract.pdf_text_extractor import PdfPage, extract_pdf_bytes, extract_pdf_text
from lexora.models.citation import (
    Citation,
    ClaimLabel,
    DiscoveryTag,
    EvidenceClaim,
    ReviewStatus,
)
from lexora.models.clause import Clause
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import PortalSpec, RawDocument, SourceProfile, SourceType
from lexora.structure.legal_parser import parse_structure, parse_structure_html

if TYPE_CHECKING:
    from lexora.collect.discovery import DiscoveryResult


@dataclass
class DemoArtifacts:
    """Everything a run produces, kept together so callers (CLI, tests, UI) can
    inspect each stage. `pages` is set for PDF inputs, `blocks` for HTML."""

    document: RawDocument
    clauses: list[Clause]
    citations: list[Citation]
    pages: list[PdfPage] = field(default_factory=list)
    blocks: list[HtmlBlock] = field(default_factory=list)


@dataclass
class MapResult:
    """Output of a multi-instrument autonomous map: the working set of instruments
    discovered for the indicators, the per-document artifacts, and the aggregated,
    de-duplicated citations across all of them."""

    discovered: list[DiscoveryResult]
    documents: list[DemoArtifacts]
    citations: list[Citation]


def _map_use_dense() -> bool:
    """Whether mapping clause retrieval fuses the dense channel.

    Default OFF (BM25-only) — PROVISIONAL. The G-6.3 ablation favours BM25-only on
    AU (large margin) and MY, but the corrected SG PDPA run favours dense fusion
    (it recovers one tail gold), so the result is NOT unanimous and the gold is too
    small (~2-3 points/economy) to settle a global default — pending G-6.4 gold
    expansion. Dense fusion stays an explicit opt-in via ``LEXORA_MAP_DENSE=1``
    (e.g. for SG, or a cross-lingual doc once a multilingual embedder is active).
    Gates the MAPPING clause retrieval only; discovery's semantic re-rank is separate."""
    return os.environ.get("LEXORA_MAP_DENSE", "").lower() in ("1", "true", "yes", "on")


def _maybe_ocr_fill(pages: list[PdfPage], source: Path | bytes) -> list[PdfPage]:
    """Fill image-only page slots with OCR when ``LEXORA_OCR`` is set.

    Inert by default (like the LLM verifier): a scanned PDF still yields blank
    pages unless OCR is explicitly enabled, so offline tests and the default path
    never import the OCR backend. When any page lacks a text layer, OCR fills it
    and the global char offsets are recomputed so verbatim spans stay valid."""
    if not os.environ.get("LEXORA_OCR"):
        return pages
    if all(p.has_text_layer for p in pages):
        return pages
    from lexora.extract.ocr_extractor import ocr_fill_pages

    filled, _conf = ocr_fill_pages(pages, source)
    return filled


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
    verifier=None,
    rationale_gen: RationaleGenerator | None = None,
    meta_extractor: MetadataExtractor | None = None,
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
    pages = _maybe_ocr_fill(pages, pdf_path)
    clauses = parse_structure(document.document_id, pages)
    citations = _citations_from_clauses(
        clauses, document, profile, indicators, legal_form, top_k, min_score,
        verifier=verifier, rationale_gen=rationale_gen, meta_extractor=meta_extractor,
        document_text="\n\n".join(p.text for p in pages),
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
    discovery_tag: DiscoveryTag = DiscoveryTag.known,
    verifier=None,
    rationale_gen: RationaleGenerator | None = None,
    meta_extractor: MetadataExtractor | None = None,
    law_number: str = "",
    last_amended: str = "",
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
        law_number=law_number,
        last_amended=last_amended,
        **fetch_kwargs,
    )
    document = result.document

    pages: list[PdfPage] = []
    blocks: list[HtmlBlock] = []
    clauses: list[Clause] = []

    if 200 <= document.http_status < 300:
        if result.is_pdf():
            pages = extract_pdf_bytes(result.body)
            pages = _maybe_ocr_fill(pages, result.body)
            clauses = parse_structure(document.document_id, pages)
        elif result.is_html():
            blocks = extract_html(result.body, url)
            clauses = parse_structure_html(document.document_id, blocks)

    document_text = (
        "\n\n".join(p.text for p in pages) if pages
        else "\n".join(b.text for b in blocks)
    )
    citations = _citations_from_clauses(
        clauses, document, profile, indicators, legal_form, top_k, min_score, discovery_tag,
        verifier=verifier, rationale_gen=rationale_gen, meta_extractor=meta_extractor,
        document_text=document_text,
    )
    return DemoArtifacts(
        document=document, clauses=clauses, citations=citations, pages=pages, blocks=blocks
    )


def run_pipeline_autodiscover(
    *,
    portal: PortalSpec,
    profile: SourceProfile,
    indicators: list[RDTIIIndicator],
    query: str | None = None,
    dest_dir: Path | None = None,
    top_k: int = 1,
    min_score: float = 0.1,
    timeout: float = 60.0,
) -> tuple[DiscoveryResult | None, DemoArtifacts]:
    """Fully autonomous path: discover the top instrument on a portal, resolve its
    full text (PDF when available), then run the live pipeline on it.

    Returns ``(DiscoveryResult | None, DemoArtifacts)``. This is the end-to-end
    mandatory-crawl flow — no URL is handed in; Lexora searches, picks, fetches
    and maps on its own.
    """
    from lexora.collect.browser import DEFAULT_UA as BROWSER_UA
    from lexora.collect.discovery import discover, resolve_fulltext
    from lexora.models.source import FetchMethod

    force_browser = portal.fetch_method is FetchMethod.playwright
    dest_dir = dest_dir or (Path("data") / "raw" / profile.iso_code.lower())

    hits = discover(
        portal, query=query, known_instruments=profile.known_instruments,
        known_instrument_ids=profile.known_instrument_ids,
        force_browser=force_browser, timeout=timeout, limit=max(top_k, 3),
    )
    if not hits:
        empty = RawDocument(
            document_id=f"{profile.iso_code.lower()}:none",
            source_url=str(portal.url),
            retrieval_timestamp=datetime.now(timezone.utc),
            http_status=0, sha256="sha256:", content_type="", bytes_path="",
            portal_name=portal.name, jurisdiction=profile.iso_code,
            source_type=portal.source_type, title=None,
        )
        return None, DemoArtifacts(document=empty, clauses=[], citations=[])

    top = hits[0]
    fulltext = resolve_fulltext(top, query=query, force_browser=force_browser, timeout=timeout)
    target = fulltext or top.url
    # Fetch the full text with a browser UA so anti-bot portals (SG SSO) serve the
    # PDF; browser_fallback still covers the case where only HTML is reachable.
    artifacts = run_pipeline_from_url(
        url=target, profile=profile, indicators=indicators, portal_name=portal.name,
        source_type=portal.source_type, dest_dir=dest_dir, top_k=top_k,
        min_score=min_score, browser_fallback=force_browser, timeout=timeout,
        user_agent=BROWSER_UA,
    )
    return top, artifacts


def run_pipeline_map(
    *,
    portal: PortalSpec,
    profile: SourceProfile,
    indicators: list[RDTIIIndicator],
    query: str | None = None,
    per_indicator_limit: int = 8,
    max_queries_per_indicator: int = 3,
    budget: int = 20,
    dest_dir: Path | None = None,
    top_k: int = 1,
    min_score: float = 0.35,
    timeout: float = 60.0,
    verifier=None,
    rationale_gen: RationaleGenerator | None = None,
    meta_extractor: MetadataExtractor | None = None,
) -> MapResult:
    """Autonomous MULTI-instrument map (P0).

    Instead of mapping only the single top hit, discover a working set of
    instruments across all indicators' concept phrases (flagship law + sectoral
    statutes the indicator touches), then fetch, parse and map EACH one. Citations
    carry the per-instrument NEW/KNOWN tag and are de-duplicated by
    (indicator, clause) across documents; one clause may map to several indicators
    (non-mutually-exclusive), so we never dedup across indicators.

    ``min_score`` is the BM25 clause-relevance floor — higher than the single-doc
    default because budget×indicators candidate pairs would otherwise be noisy.
    The link-relevance floor inside discovery is separate (its own default).
    """
    from lexora.collect.browser import DEFAULT_UA as BROWSER_UA
    from lexora.collect.discovery import discover_for_indicators, resolve_fulltext
    from lexora.models.source import FetchMethod

    force_browser = portal.fetch_method is FetchMethod.playwright
    dest_dir = dest_dir or (Path("data") / "raw" / profile.iso_code.lower())

    hits = discover_for_indicators(
        portal, indicators, per_indicator_limit=per_indicator_limit,
        max_queries_per_indicator=max_queries_per_indicator, budget=budget,
        timeout=timeout, force_browser=force_browser,
        known_instruments=profile.known_instruments,
        known_instrument_ids=profile.known_instrument_ids,
    )

    documents: list[DemoArtifacts] = []
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        fulltext = resolve_fulltext(hit, force_browser=force_browser, timeout=timeout)
        target = fulltext or hit.url
        tag = DiscoveryTag.new if hit.discovery_tag == "NEW" else DiscoveryTag.known
        # Score the instrument only against the indicators whose query surfaced it
        # (a retention-query hit is a candidate for 7.3, not for all nine). For a
        # name-driven hit (AU OData has no full-text, so no surfacing indicator),
        # attribute via the profile's indicator->instrument-name hints; only fall
        # back to all indicators when nothing pins it down.
        wanted = set(hit.indicator_hits) or _attribute_by_name(hit, profile, indicators)
        ind_subset = [i for i in indicators if i.submission_id in wanted] or indicators
        artifacts = run_pipeline_from_url(
            url=target, profile=profile, indicators=ind_subset, portal_name=portal.name,
            source_type=portal.source_type, dest_dir=dest_dir, top_k=top_k,
            min_score=min_score, discovery_tag=tag, browser_fallback=force_browser,
            timeout=timeout, user_agent=BROWSER_UA, title=hit.title, verifier=verifier,
            rationale_gen=rationale_gen, meta_extractor=meta_extractor,
            law_number=hit.law_number, last_amended=hit.last_amended,
        )
        documents.append(artifacts)
        for c in artifacts.citations:
            key = (c.indicator_id, c.clause_id)
            if key in seen:
                continue
            seen.add(key)
            citations.append(c)

    return MapResult(discovered=hits, documents=documents, citations=citations)


def _attribute_by_name(hit, profile: SourceProfile, indicators: list[RDTIIIndicator]) -> set[str]:
    """Indicators a name-driven hit maps to, via the profile's per-indicator
    instrument-name hints (``keywords_by_indicator``). A keyword that strongly
    fuzzy-matches the instrument title is an instrument-name hint (e.g. AU
    "Telecommunications (Interception and Access) Act" hints 7.3 + 7.5), so an AU
    Act maps only to its relevant indicators instead of all nine. Returns an empty
    set when nothing matches (caller then falls back to all indicators)."""
    from rapidfuzz import fuzz

    title_l = hit.title.lower()
    wanted: set[str] = set()
    for ind in indicators:
        for kw in profile.keywords_by_indicator.get(ind.id, {}).get(profile.primary_language, []):
            if fuzz.token_set_ratio(kw.lower(), title_l) >= 80:
                wanted.add(ind.submission_id)
                break
    return wanted


def _resolve_instrument_meta(title: str | None, profile: SourceProfile) -> tuple[str, str]:
    """Return ``(last_amended, law_number)`` for the document's instrument, or
    ``("", "")`` when nothing matches (graceful — the columns stay blank).

    Backfills the two submission columns the fetched document does not carry, from
    the curated ``instrument_metadata`` (configs/jurisdictions/<iso>.yaml). Matches
    the document title fuzzily; for portals whose titles are filenames (MY Fess) it
    also resolves via a known Act number that appears in the title."""
    meta_map = profile.instrument_metadata
    if not title or not meta_map:
        return "", ""
    from rapidfuzz import fuzz

    title_l = title.lower()
    best_name, best_score = None, 0.0
    for name in meta_map:
        score = fuzz.token_set_ratio(name.lower(), title_l)
        if score > best_score:
            best_name, best_score = name, score
    if best_name is not None and best_score >= 85:
        meta = meta_map[best_name]
        return meta.last_amended, meta.law_number
    # Filename-title fallback: a known Act number embedded in the title.
    for number, name in profile.known_instrument_ids.items():
        if number and number in title and name in meta_map:
            meta = meta_map[name]
            return meta.last_amended, meta.law_number
    return "", ""


def _resolve_doc_metadata(
    document: RawDocument,
    document_text: str,
    profile: SourceProfile,
    meta_extractor: MetadataExtractor | None,
) -> tuple[str, str]:
    """Resolve ``(last_amended, law_number)`` for a document, once, by precedence:

    1. structured portal metadata captured at fetch time (most reliable; generalizes
       to NEW laws — e.g. the AU register's act number);
    2. the generic LLM extractor over the document text (source-verified) — the
       generalizer for portals without a structured connector;
    3. the curated, source-verified anchor for the known flagship instruments.

    Each tier only fills a field still missing, so the highest-confidence source wins
    and a citation always gets the best value available (or blank)."""
    last_amended, law_number = document.last_amended, document.law_number
    if not (last_amended and law_number) and meta_extractor is not None:
        ex_amended, ex_number = meta_extractor.extract(
            document_text, document.jurisdiction, document.title or ""
        )
        last_amended = last_amended or ex_amended
        law_number = law_number or ex_number
    if not (last_amended and law_number):
        anchor_amended, anchor_number = _resolve_instrument_meta(document.title, profile)
        last_amended = last_amended or anchor_amended
        law_number = law_number or anchor_number
    return last_amended, law_number


def _citations_from_clauses(
    clauses: list[Clause],
    document: RawDocument,
    profile: SourceProfile,
    indicators: list[RDTIIIndicator],
    legal_form: str,
    top_k: int,
    min_score: float,
    discovery_tag: DiscoveryTag = DiscoveryTag.known,
    verifier=None,
    rationale_gen: RationaleGenerator | None = None,
    meta_extractor: MetadataExtractor | None = None,
    document_text: str = "",
) -> list[Citation]:
    """Shared core: per indicator, retrieve top-k clauses and materialize the
    ones that pass the verbatim validator.

    When ``verifier`` is supplied (the optional LLM gate, P-3) it runs AFTER
    retrieval and verbatim as a tightening step: of the BM25-passing candidates
    it selects at most one clause that actually supports the indicator, or
    abstains (dropping the citation). It can only narrow the keyword result — it
    never adds a clause or relaxes a gate."""
    citations: list[Citation] = []
    if not clauses:
        return citations
    index: BM25Index = build_index(clauses)
    clause_by_id = {c.clause_id: c for c in clauses}
    # Document-level metadata (Law Number / Last Amended) resolved ONCE per document.
    doc_last_amended, doc_law_number = _resolve_doc_metadata(
        document, document_text, profile, meta_extractor
    )
    for indicator in indicators:
        # Gate on the NORMALIZED score so `min_score` is a portable [0, 1]
        # relevance floor (raw BM25 is unbounded and corpus-dependent — a
        # fixed raw cutoff prunes nothing on a big document).
        passing = [
            hit for hit in retrieve_candidates(
                indicator, profile, index, top_k=top_k, use_semantic=_map_use_dense()
            )
            if _normalize_score(hit.score) >= min_score
        ]
        # P6/P7 boundary discipline: drop clauses a scope rule excludes from this
        # indicator (e.g. a retention LIMITATION wrongly surfaced for 7.3). Tightening
        # only — it can never add a clause. See classify/boundaries.py.
        passing = [
            hit for hit in passing
            if admits_clause(indicator.rdtii_id, clause_by_id[hit.clause_id].span.text)
        ]
        if not passing:
            continue

        if verifier is not None:
            # One LLM judgement per indicator over its candidate clauses; it may
            # pick one (match/uncertain) or abstain. The score scale is unchanged
            # — confidence stays BM25-derived; the verdict only gates inclusion
            # and, for "uncertain", routes to human review.
            candidates = [clause_by_id[h.clause_id] for h in passing]
            claim = verifier.verify(indicator, candidates)
            if claim is None:
                continue
            hit = next(h for h in passing if h.clause_id == claim.clause_id)
            citation = _materialize(
                indicator=indicator, clause=clause_by_id[claim.clause_id],
                document=document, legal_form=legal_form, profile=profile,
                bm25_score=hit.score, discovery_tag=discovery_tag,
                review_label=claim.label, rationale_gen=rationale_gen,
                last_amended=doc_last_amended, law_number=doc_law_number,
            )
            if citation is not None:
                citations.append(citation)
            continue

        for hit in passing:
            citation = _materialize(
                indicator=indicator,
                clause=clause_by_id[hit.clause_id],
                document=document,
                legal_form=legal_form,
                profile=profile,
                bm25_score=hit.score,
                discovery_tag=discovery_tag,
                rationale_gen=rationale_gen,
                last_amended=doc_last_amended,
                law_number=doc_law_number,
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
    profile: SourceProfile,
    bm25_score: float,
    discovery_tag: DiscoveryTag = DiscoveryTag.known,
    review_label: ClaimLabel = ClaimLabel.match,
    rationale_gen: RationaleGenerator | None = None,
    last_amended: str = "",
    law_number: str = "",
) -> Citation | None:
    """One-clause-one-indicator: synthesize a verifier-shaped claim and run it
    through the same validator. The claim carries the official submission code
    (e.g. "P6-I4") as its indicator_id.

    ``review_label`` is the LLM verifier's verdict (default ``match`` keeps the
    pre-P-3 behaviour). An ``uncertain`` verdict still publishes the citation but
    routes it to human review (``CONFLICT_REVIEW``) rather than ``VERIFIED`` — the
    verbatim quote is sound, the *mapping* is what's in doubt."""
    claim = EvidenceClaim(
        indicator_id=indicator.submission_id,
        clause_id=clause.clause_id,
        quote_span_id=clause.span.span_id,
        label=review_label,
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
    notes = ""
    if status is ReviewStatus.verified and review_label is ClaimLabel.uncertain:
        status = ReviewStatus.conflict_review
        notes = "LLM verifier flagged the mapping as uncertain."
    rationale = (
        rationale_gen.generate(indicator, profile, clause, clause.structural_path)
        if rationale_gen is not None
        else template_rationale(indicator, profile, clause, clause.structural_path)
    )
    return build_citation(
        claim=claim,
        span=clause.span,
        document=document,
        legal_form=legal_form,
        article_path=clause.structural_path,
        economy=profile.jurisdiction,
        law_number=law_number,
        last_amended=last_amended,
        discovery_tag=discovery_tag,
        mapping_rationale=rationale,
        review_status=status,
        notes=notes,
    )


def _normalize_score(score: float) -> float:
    """Squash unbounded BM25 scores into [0, 1] for the Pydantic confidence
    field. tanh(score/5) maps a strong-match BM25 score (~5+) into the high-0.9s
    and never saturates to 1.0."""
    return float(max(0.0, min(1.0, math.tanh(score / 5.0))))


__all__ = [
    "DemoArtifacts",
    "MapResult",
    "run_demo_pipeline",
    "run_pipeline_from_url",
    "run_pipeline_autodiscover",
    "run_pipeline_map",
]
