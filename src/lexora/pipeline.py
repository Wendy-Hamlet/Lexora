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

import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from lexora.cite.citation_builder import build_citation
from lexora.cite.metadata import MetadataExtractor
from lexora.cite.rationale import RationaleGenerator, template_rationale
from lexora.cite.validator import validate_claim
from lexora.classify.boundaries import admits_clause
from lexora.classify.lifecycle import detect_status, is_enforced, merge_status
from lexora.classify.retrieval import (
    BM25Index,
    _maybe_reranker,
    build_index,
    retrieve_candidates,
)
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
from lexora.models.source import (
    InstrumentStatus,
    PortalSpec,
    RawDocument,
    SourceProfile,
    SourceType,
)
from lexora.structure.legal_parser import parse_structure, parse_structure_html

if TYPE_CHECKING:
    from lexora.collect.discovery import DiscoveryResult

logger = logging.getLogger(__name__)


@dataclass
class DemoArtifacts:
    """Everything a run produces, kept together so callers (CLI, tests, UI) can
    inspect each stage. `pages` is set for PDF inputs, `blocks` for HTML."""

    document: RawDocument
    clauses: list[Clause]
    citations: list[Citation]
    pages: list[PdfPage] = field(default_factory=list)
    blocks: list[HtmlBlock] = field(default_factory=list)
    # WS-2 per-document technical metadata for the JSON sidecar. `document_text` is
    # the canonical concatenated text (used to slice raw before/after context per
    # provision); the OCR fields come from `_maybe_ocr_fill`; `processing_time_seconds`
    # is set by the caller that times the per-document work (`run_pipeline_map`).
    document_text: str = ""
    pdf_is_scanned: bool = False
    ocr_quality_cer: float | None = None
    ocr_engine: str = ""
    processing_time_seconds: float | None = None


@dataclass
class MapResult:
    """Output of a multi-instrument autonomous map: the working set of instruments
    discovered for the indicators, the per-document artifacts, and the aggregated,
    de-duplicated citations across all of them."""

    discovered: list[DiscoveryResult]
    documents: list[DemoArtifacts]
    citations: list[Citation]
    # Secondary-source signals (WS-S) gathered for this economy, when --secondary
    # is on. Non-citable discovery aid — carried so the runner can print the
    # coverage cross-check; never exported as evidence.
    secondary_signals: list = field(default_factory=list)


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


def _map_use_rerank() -> bool:
    """Whether mapping clause retrieval applies the cross-encoder rerank stage.

    Default OFF. When ``LEXORA_MAP_RERANK=1`` the BM25 recall pool is reordered by a
    cross-encoder (RRF-blended with BM25 so a confident BM25 top-1 is not displaced)
    before truncation — the precision lane that lifts a buried on-point section
    (measured: AU APP8 rank 3 -> 1). Opt-in because it needs the fastembed
    cross-encoder backend + a downloaded model; falls back to BM25 if unavailable."""
    return os.environ.get("LEXORA_MAP_RERANK", "").lower() in ("1", "true", "yes", "on")


def _maybe_ocr_fill(
    pages: list[PdfPage], source: Path | bytes
) -> tuple[list[PdfPage], dict]:
    """Fill image-only page slots with OCR when ``LEXORA_OCR`` is set.

    Inert by default (like the LLM verifier): a scanned PDF still yields blank
    pages unless OCR is explicitly enabled, so offline tests and the default path
    never import the OCR backend. When any page lacks a text layer, OCR fills it
    and the global char offsets are recomputed so verbatim spans stay valid.

    Returns ``(pages, ocr_meta)``. ``ocr_meta`` records the document-level OCR audit
    trail for the WS-2 JSON sidecar — ``scanned`` (any image-only page was filled),
    ``ocr_quality_cer`` (mean page-level OCR confidence over the filled pages; see
    note below) and the engine name. ``scanned=False``/``ocr_quality_cer=None`` when
    OCR did not run, so a dropped scan is no longer invisible (the §3 audit gap).

    NOTE — ``ocr_quality_cer`` currently carries the mean OCR *confidence* (0..1,
    higher is better), NOT a reference-based Character Error Rate (we have no
    ground-truth transcript to score against). Kept under the official field name
    for schema compatibility; the JSON sidecar annotates this. This is a flagged
    point to revisit (see progress report §6)."""
    meta: dict = {"scanned": False, "ocr_quality_cer": None, "ocr_engine": ""}
    if not os.environ.get("LEXORA_OCR"):
        return pages, meta
    if all(p.has_text_layer for p in pages):
        return pages, meta
    from lexora.extract.ocr_extractor import ocr_fill_pages

    filled, page_conf = ocr_fill_pages(pages, source)
    if page_conf:
        meta["scanned"] = True
        meta["ocr_quality_cer"] = round(sum(page_conf.values()) / len(page_conf), 4)
        meta["ocr_engine"] = os.environ.get("LEXORA_OCR_ENGINE", "rapidocr").lower()
    return filled, meta


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
    enforced_only: bool = True,
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
    pages, ocr_meta = _maybe_ocr_fill(pages, pdf_path)
    clauses = parse_structure(document.document_id, pages)
    document_text = "\n\n".join(p.text for p in pages)
    citations = _citations_from_clauses(
        clauses, document, profile, indicators, legal_form, top_k, min_score,
        verifier=verifier, rationale_gen=rationale_gen, meta_extractor=meta_extractor,
        document_text=document_text, enforced_only=enforced_only,
    )
    return DemoArtifacts(
        document=document, clauses=clauses, citations=citations, pages=pages,
        document_text=document_text, pdf_is_scanned=ocr_meta["scanned"],
        ocr_quality_cer=ocr_meta["ocr_quality_cer"], ocr_engine=ocr_meta["ocr_engine"],
    )


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
    status: str = "UNKNOWN",
    enforced_only: bool = True,
    rel_floor: float = 0.0,
    llm_workers: int = 1,
    secondary_note_by_indicator: dict | None = None,
    brute_judge=None,
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
        instrument_status=status,
        **fetch_kwargs,
    )
    document = result.document

    pages: list[PdfPage] = []
    blocks: list[HtmlBlock] = []
    clauses: list[Clause] = []

    ocr_meta: dict = {"scanned": False, "ocr_quality_cer": None, "ocr_engine": ""}
    if 200 <= document.http_status < 300:
        if result.is_pdf():
            pages = extract_pdf_bytes(result.body)
            pages, ocr_meta = _maybe_ocr_fill(pages, result.body)
            clauses = parse_structure(document.document_id, pages)
        elif result.is_html():
            blocks = extract_html(result.body, url)
            clauses = parse_structure_html(document.document_id, blocks)

    document_text = (
        "\n\n".join(p.text for p in pages) if pages
        else "\n".join(b.text for b in blocks)
    )
    # Regime-2: a full-text 9-in-1 judge decides which of ALL indicators this law is
    # relevant to, replacing the discovery-attribution subset (which caps recall — a law
    # surfaced under one indicator is never tried for another). On total judge failure it
    # returns None and we keep the attribution subset we were called with.
    if brute_judge is not None and document_text:
        judged = brute_judge.subset(document_text, indicators)
        if judged:
            indicators = judged
    citations = _citations_from_clauses(
        clauses, document, profile, indicators, legal_form, top_k, min_score, discovery_tag,
        verifier=verifier, rationale_gen=rationale_gen, meta_extractor=meta_extractor,
        document_text=document_text, enforced_only=enforced_only, rel_floor=rel_floor,
        llm_workers=llm_workers, secondary_note_by_indicator=secondary_note_by_indicator,
    )
    return DemoArtifacts(
        document=document, clauses=clauses, citations=citations, pages=pages, blocks=blocks,
        document_text=document_text, pdf_is_scanned=ocr_meta["scanned"],
        ocr_quality_cer=ocr_meta["ocr_quality_cer"], ocr_engine=ocr_meta["ocr_engine"],
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
    top_k: int = 3,
    min_score: float = 0.35,
    rel_floor: float = 0.6,
    timeout: float = 60.0,
    verifier=None,
    rationale_gen: RationaleGenerator | None = None,
    meta_extractor: MetadataExtractor | None = None,
    enforced_only: bool = True,
    llm_workers: int = 1,
    doc_workers: int = 1,
    fetch_min_interval: float = 0.0,
    serial_fetch: bool = False,
    secondary_signals: list | None = None,
    discover_amendments: bool = True,
    amendment_extractor=None,
    brute_judge=None,
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

    # Regime-2 relevance judge (opt-in via LEXORA_BRUTE_JUDGE). Inert -> None, and the
    # mapper keeps its discovery-attribution subset. Reuses the production fetch/parse
    # (browser anti-bot, OCR, PDF) — only the relevance decision changes.
    if brute_judge is None:
        from lexora.classify.brute_judge import make_brute_judge

        brute_judge = make_brute_judge()

    # Secondary-source consumers (WS-S, S-2): seed discovery with the primary-law
    # names a tracker pointed to (USE 1), and prepare per-indicator provenance
    # notes to stamp on matching citations (USE 3). Signals are a discovery aid —
    # never evidence; the only conversions are the lossy seed/note helpers.
    from lexora.collect.secondary import (
        provenance_notes_by_indicator,
        to_discovery_seeds,
    )

    secondary_signals = secondary_signals or []
    seed_queries = to_discovery_seeds(secondary_signals)
    secondary_note_by_indicator = provenance_notes_by_indicator(secondary_signals)

    hits = discover_for_indicators(
        portal, indicators, per_indicator_limit=per_indicator_limit,
        max_queries_per_indicator=max_queries_per_indicator, budget=budget,
        timeout=timeout, force_browser=force_browser,
        known_instruments=profile.known_instruments,
        known_instrument_ids=profile.known_instrument_ids,
        extra_seed_queries=seed_queries,
    )
    logger.info("working set: %d instrument(s) to map", len(hits))

    import threading

    _progress = {"done": 0, "total": len(hits)}
    _progress_lock = threading.Lock()

    def _process(hit) -> DemoArtifacts:
        t0 = time.perf_counter()
        fulltext = resolve_fulltext(hit, force_browser=force_browser, timeout=timeout)
        target = fulltext or hit.url
        tag = DiscoveryTag.new if hit.discovery_tag == "NEW" else DiscoveryTag.known
        # Score the instrument only against the indicators whose query surfaced it
        # (a retention-query hit is a candidate for 7.3, not for all nine). For a
        # name-driven hit (AU OData has no full-text, so no surfacing indicator),
        # attribute via the profile's indicator->instrument-name hints; only fall
        # back to all indicators when nothing pins it down.
        # Per-clause 9-in-1 verifier decides relevance at the clause level against ALL
        # indicators, so pass the full set (it removes the attribution ceiling without a
        # noisy full-text skim). A full-text brute judge likewise wants all indicators.
        # Otherwise use the discovery-attribution subset (regime-1).
        per_clause = verifier is not None and getattr(verifier, "mode", "") == "per_clause"
        if brute_judge is not None or per_clause:
            ind_subset = indicators
        else:
            wanted = set(hit.indicator_hits) or _attribute_by_name(hit, profile, indicators)
            ind_subset = [i for i in indicators if i.submission_id in wanted] or indicators
        artifacts = run_pipeline_from_url(
            url=target, profile=profile, indicators=ind_subset, portal_name=portal.name,
            source_type=portal.source_type, dest_dir=dest_dir, top_k=top_k,
            min_score=min_score, discovery_tag=tag, browser_fallback=force_browser,
            timeout=timeout, user_agent=BROWSER_UA, title=hit.title, verifier=verifier,
            rationale_gen=rationale_gen, meta_extractor=meta_extractor,
            law_number=hit.law_number, last_amended=hit.last_amended,
            status=hit.status, enforced_only=enforced_only, rel_floor=rel_floor,
            llm_workers=llm_workers, min_interval=fetch_min_interval,
            serial_fetch=serial_fetch,
            secondary_note_by_indicator=secondary_note_by_indicator,
            brute_judge=brute_judge,
        )
        artifacts.processing_time_seconds = round(time.perf_counter() - t0, 3)
        with _progress_lock:
            _progress["done"] += 1
            n = _progress["done"]
        logger.info(
            "mapped %d/%d %-38s %d clause(s) -> %d citation(s) [%.1fs]",
            n, _progress["total"], (hit.title or hit.url)[:38],
            len(artifacts.clauses), len(artifacts.citations),
            artifacts.processing_time_seconds,
        )
        return artifacts

    # Document-level parallelism: process instruments concurrently. This is what
    # parallelizes the per-document metadata extraction (one LLM call each, the
    # serial floor of an LLM run) as well as fetch / OCR / rationale across docs.
    # SG full-text resolves by URL construction (no browser render in the loop), so
    # concurrent docs are safe. `map` preserves order; dedup runs sequentially after.
    if doc_workers > 1 and len(hits) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(doc_workers, len(hits))) as ex:
            documents = list(ex.map(_process, hits))
    else:
        documents = [_process(h) for h in hits]

    # Tag every fetched law original/amendment/consolidated and, for each ORIGINAL,
    # look for ITS amendments (queries derived from its own title — general, not a
    # fixed amendment list). Found amending Acts join the working set so the currency
    # pass below can adjudicate the principal's provisions against real instructions.
    if discover_amendments:
        documents = documents + _discover_amendments(
            documents, _process, portal, profile,
            force_browser=force_browser, timeout=timeout,
        )

    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for artifacts in documents:  # sequential dedup -> deterministic order
        for c in artifacts.citations:
            key = (c.indicator_id, c.clause_id)
            if key in seen:
                continue
            seen.add(key)
            citations.append(c)

    # Amendment-currency pass: flag citations whose source text may pre-date a later
    # amending instrument (needs the whole working set, so it runs once here). Tier-2
    # instruction extraction is LLM-first when an extractor is supplied (or
    # LEXORA_AMENDMENT_LLM is set), regex otherwise; the per-amending-Act LLM calls run
    # concurrently under the same `llm_workers` budget as the rest of the pipeline.
    if amendment_extractor is None:
        from lexora.cite.amendments_llm import llm_enabled, make_amendment_extractor

        amendment_extractor = make_amendment_extractor(use_llm=llm_enabled())
    _apply_currency_flags(
        documents, citations, profile,
        extractor=amendment_extractor, workers=llm_workers,
    )
    # Enforced-only also drops provisions a commenced amendment REPEALED (the legal
    # team's "deleted -> remove" case). Flagged-but-uncommenced repeals stay
    # (STALE_RISK), and the per-document artifacts retain the row for the audit trail.
    if enforced_only:
        citations = [c for c in citations if c.currency_status != "REPEALED"]

    return MapResult(
        discovered=hits, documents=documents, citations=citations,
        secondary_signals=list(secondary_signals),
    )


def _discover_amendments(
    documents: list[DemoArtifacts],
    process_fn,
    portal: PortalSpec,
    profile: SourceProfile,
    *,
    force_browser: bool,
    timeout: float,
    max_queries: int = 12,
    per_query: int = 4,
) -> list[DemoArtifacts]:
    """For every ORIGINAL or CONSOLIDATED law in the working set, look for ITS amendments.

    General by construction: amendment-search queries are derived from each law's own
    title (``amendment_search_queries``), not a hardcoded list of amending Acts — so a
    NEW law's amendments are pursued just like a known law's. Each candidate is
    fetched/parsed through the normal per-document path and kept only if it actually
    classifies as an amending Act (``AMENDMENT_DELTA``), so an unrelated law surfaced
    by the query is discarded. De-duplicated against what is already fetched.

    An ORIGINAL "as made" text incorporates no later amendments, so all of its
    amendments matter. A CONSOLIDATED text (e.g. an AU compilation) already folds in
    everything up to its compilation year, so only amendments newer than that year can
    render it stale — those are filtered by the candidate's title year before fetching,
    so a heavily-amended principal does not drag in dozens of already-incorporated Acts."""
    from lexora.cite.amendments import (
        VersionKind,
        amendment_search_queries,
        classify_version,
        detect_incorporated_to,
    )
    from lexora.collect.discovery import discover

    have_sha = {a.document.sha256 for a in documents}
    seen_url = {str(a.document.source_url) for a in documents}
    # query -> year FLOOR below which an amendment is uninteresting. An ORIGINAL
    # "as made" text folds in nothing, so it pursues ALL its amendments (floor None);
    # a CONSOLIDATED already incorporates everything up to its compilation year (AU
    # serves compiled text), so only amendments AFTER that year can make it stale.
    # When a query serves several principals, keep the loosest (lowest) floor.
    query_floor: dict[str, int | None] = {}
    for a in documents:
        kind = classify_version(a.document_text or "")
        if not a.document.title:
            continue
        if kind is VersionKind.original:
            floor: int | None = None
        elif kind is VersionKind.consolidated:
            floor = detect_incorporated_to(a.document_text or "")
        else:
            continue
        for q in amendment_search_queries(a.document.title):
            if q not in query_floor:
                query_floor[q] = floor
            elif floor is None or query_floor[q] is None:
                query_floor[q] = None
            else:
                query_floor[q] = min(query_floor[q], floor)
    queries = list(query_floor)[:max_queries]

    new_docs: list[DemoArtifacts] = []
    for q in queries:
        floor = query_floor[q]
        try:
            hits = discover(
                portal, query=q, limit=per_query, timeout=timeout,
                force_browser=force_browser, known_instruments=profile.known_instruments,
                known_instrument_ids=profile.known_instrument_ids,
            )
        except Exception:  # noqa: BLE001 — a failed amendment query never breaks a run
            continue
        for hit in hits:
            if hit.url in seen_url:
                continue
            seen_url.add(hit.url)
            # Skip an amendment a consolidated principal already incorporates, judged
            # by the title year BEFORE fetching (avoids fetching/parsing dead weight).
            hit_year = _year_of(hit.title)
            if floor is not None and hit_year is not None and hit_year <= floor:
                continue
            try:
                art = process_fn(hit)
            except Exception:  # noqa: BLE001
                continue
            if art.document.sha256 in have_sha:
                continue
            if classify_version(art.document_text or "") is VersionKind.amendment_delta:
                have_sha.add(art.document.sha256)
                new_docs.append(art)
    return new_docs


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


def _resolve_doc_metadata(
    document: RawDocument,
    document_text: str,
    profile: SourceProfile,
    meta_extractor: MetadataExtractor | None,
) -> tuple[str, str, str]:
    """Resolve ``(last_amended, law_number, review_note)`` for a document, once, by
    precedence:

    1. structured portal metadata captured at fetch time (most reliable; generalizes
       to NEW laws — e.g. the AU register's act number);
    2. the generic LLM extractor over the document text (source-verified, hybrid
       windowing — see :func:`lexora.cite.metadata._document_window`). A field the
       model cannot derive from the text comes back sentinelled (blank).

    Each tier only fills a field still missing, so the highest-confidence source wins
    and a citation always gets the best value available (or blank). ``review_note``
    is the extractor's own-knowledge channel (review-only, never an answer)."""
    last_amended, law_number = document.last_amended, document.law_number
    review_note = ""
    if not (last_amended and law_number) and meta_extractor is not None:
        ex_amended, ex_number, review_note = meta_extractor.extract(
            document_text, document.jurisdiction, document.title or ""
        )
        last_amended = last_amended or ex_amended
        law_number = law_number or ex_number
    return last_amended, law_number, review_note


def _year_of(value: str) -> int | None:
    """First 4-digit year in a string, or None."""
    import re

    m = re.search(r"\b(1[6-9]\d{2}|20\d{2})\b", value or "")
    return int(m.group(1)) if m else None


def _extract_amendment_instructions(text: str, extractor=None) -> list:
    """Instructions an amending Act performs — LLM-FIRST, regex fallback.

    The LLM path (:mod:`lexora.cite.amendments_llm`) is the more complete extractor on
    real drafting: on the Arbitration (Amendment) Act 2024 it recovered a section
    deletion and several per-subsection substitutions the regex parser collapses into a
    coarse ``amend`` or misses outright (LLM 24 vs regex 7 instructions), and it is
    source-verified so it cannot fabricate one. The deterministic
    :func:`lexora.cite.amendments.parse_amendment_instructions` backstops it whenever
    the LLM is unavailable, inert, or returns nothing (offline run, non-English lexicon
    the model declined, transient backend error) — so adjudication never loses the
    regex floor. ``extractor`` is ``None`` or the inert extractor when LLM is off."""
    from lexora.cite.amendments import parse_amendment_instructions

    if extractor is not None:
        instrs = extractor.extract(text)
        if instrs:
            return instrs
    return parse_amendment_instructions(text)


def _apply_currency_flags(
    documents: list[DemoArtifacts],
    citations: list[Citation],
    profile: SourceProfile,
    *,
    extractor=None,
    workers: int = 1,
) -> None:
    """Stamp each citation with an amendment-currency verdict, in place.

    Two layers. **Document-level** (Tier-1): the amendment chain for every law is
    built from the fetched corpus (Signal B), the portal channel (D) and the curated
    registry (C backstop); a citation whose source pre-dates a known amendment is
    flagged ``STALE_RISK``. **Provision-level** (Tier-2): when the amending Act's TEXT
    is in the corpus, its instructions are extracted (LLM-first via ``extractor``, regex
    fallback) and each citation's exact section is adjudicated — untouched sections are
    downgraded to ``CURRENT``, an amended section becomes ``AMENDED`` (carrying the
    amending Act's own verbatim), a deleted+commenced section becomes ``REPEALED``, and
    an act-wide term rename annotates any quote that uses the old term. Any non-CURRENT
    verdict raises a clean ``VERIFIED`` row to ``AMENDMENT_REVIEW``.

    ``extractor`` is the LLM amendment extractor (inert/``None`` -> regex only);
    ``workers`` > 1 runs the per-amending-Act LLM extraction concurrently — the LLM
    call is the wall-clock cost here, so this mirrors the document/rationale thread
    pools. See :mod:`lexora.cite.amendments`."""
    from lexora.cite.amendments import (
        AmendmentIndex,
        CurrencyStatus,
        adjudicate_provision,
        assess_currency,
        candidate_keys,
        classify_version,
        currency_note,
        detect_amends_target,
        detect_incorporated_to,
        is_commenced,
        parse_identity,
        section_of,
    )
    from lexora.cite.amendments import VersionKind as _VK

    index = AmendmentIndex()
    index.add_from_corpus(
        [(a.document.title or "", a.document_text) for a in documents if a.document_text]
    )
    index.add_registry(profile.amended_by)

    # Extract the instruction set of every amending Act in the corpus (Tier-2). The
    # extraction is LLM-first (regex fallback) and is the wall-clock cost of this pass,
    # so it runs CONCURRENTLY across amending Acts when an LLM extractor is in play —
    # mirroring the document/rationale thread pools (`map` preserves order). Results are
    # stored under ALL keys the principal may be cited by (Act-number AND title) so a
    # citation keyed by number matches an amendment that named the principal by title.
    amend_docs = [
        a for a in documents
        if classify_version(a.document_text or "") is _VK.amendment_delta
        and detect_amends_target(a.document_text or "") is not None
    ]
    texts = [a.document_text or "" for a in amend_docs]
    use_pool = (
        workers > 1 and len(amend_docs) > 1
        and extractor is not None and getattr(extractor, "_client", None) is not None
    )
    if use_pool:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(workers, len(amend_docs))) as ex:
            instr_lists = list(ex.map(
                lambda t: _extract_amendment_instructions(t, extractor), texts
            ))
    else:
        instr_lists = [_extract_amendment_instructions(t, extractor) for t in texts]

    instr_by_key: dict[str, list] = {}
    label_by_key: dict[str, list[str]] = {}
    commenced_by_key: dict[str, bool] = {}
    for a, instrs in zip(amend_docs, instr_lists, strict=True):
        text = a.document_text or ""
        target = detect_amends_target(text)  # not None by construction of amend_docs
        ident = parse_identity(text)
        label = ident.number or "amendment"
        if ident.year:
            label = f"{label} ({ident.year})"
        commenced = is_commenced(text)
        for key in candidate_keys(number=target[0], title=target[1]):
            instr_by_key.setdefault(key, []).extend(instrs)
            label_by_key.setdefault(key, []).append(label)
            commenced_by_key[key] = commenced_by_key.get(key, True) and commenced

    # Per-source-document candidate keys + incorporation cutoff + version tag. The
    # cutoff is how current the source text is: Signal A's in-doc consolidation point,
    # raised by any portal-reported amendment year (Signal D).
    by_hash: dict[str, tuple[list[str], int | None]] = {}
    ver_by_hash: dict[str, str] = {}
    for a in documents:
        doc = a.document
        ident = parse_identity(a.document_text or "")
        keys = candidate_keys(
            number=doc.law_number or ident.number, title=doc.title or ident.title
        )
        in_doc = detect_incorporated_to(a.document_text or "")
        portal_year = _year_of(doc.last_amended)
        cutoff = max([y for y in (in_doc, portal_year) if y is not None], default=None)
        by_hash[doc.sha256] = (keys, cutoff)
        ver_by_hash[doc.sha256] = classify_version(a.document_text or "").value

    def _flag(c: Citation, status: CurrencyStatus, amended_by: str, note: str,
              amendment_text: str = "") -> None:
        c.currency_status = status.value
        if amended_by:
            c.amended_by = amended_by
        if amendment_text:
            c.amendment_text = amendment_text
        if note:
            c.notes = f"{c.notes} | {note}" if c.notes else note
        if status in (CurrencyStatus.stale_risk, CurrencyStatus.amended,
                      CurrencyStatus.repealed) and c.review_status is ReviewStatus.verified:
            c.review_status = ReviewStatus.amendment_review

    for c in citations:
        c.source_version = ver_by_hash.get(c.document_hash, "")
        keys, cutoff = by_hash.get(c.document_hash, ([], None))
        if not keys:  # fall back to the citation's own resolved metadata
            keys = candidate_keys(number=c.law_number, title=c.title)
        if cutoff is None:
            cutoff = _year_of(c.last_amended)
        # A source that is itself a consolidation is CURRENT to its stated point when
        # no later amendment is known; an as-made original stays UNKNOWN (see
        # `assess_currency`). The version was classified per-document above.
        self_consolidated = ver_by_hash.get(c.document_hash) == _VK.consolidated.value
        assessment = assess_currency(
            keys=keys, incorporated_to=cutoff, index=index,
            self_consolidated=self_consolidated,
        )
        c.currency_status = assessment.status.value
        if assessment.incorporated_to is not None:
            c.amendments_incorporated_to = str(assessment.incorporated_to)

        # Gather the amending Act's instructions across every key the principal may be
        # referenced by (number + title).
        instrs: list = []
        labels: list[str] = []
        commenced = True
        for k in keys:
            if k in instr_by_key:
                instrs.extend(instr_by_key[k])
                labels.extend(label_by_key.get(k, []))
                commenced = commenced and commenced_by_key.get(k, True)
        if instrs:
            # Tier-2: we have the amending Act's text -> adjudicate this exact section.
            amend_label = "; ".join(dict.fromkeys(labels)) or assessment.amended_by_label()
            verdict = adjudicate_provision(
                section=section_of(c.article_path), quote=c.quote,
                instructions=instrs, amend_label=amend_label, commenced=commenced,
            )
            status = verdict.status if verdict.status is not None else assessment.status
            _flag(c, status, amend_label if status is not CurrencyStatus.current else "",
                  verdict.note, verdict.amendment_text)
        elif assessment.status is CurrencyStatus.stale_risk:
            # Tier-1 only: an amendment exists but its text is not in the corpus.
            _flag(c, CurrencyStatus.stale_risk, assessment.amended_by_label(),
                  currency_note(assessment))


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
    enforced_only: bool = True,
    rel_floor: float = 0.0,
    llm_workers: int = 1,
    secondary_note_by_indicator: dict | None = None,
) -> list[Citation]:
    """Shared core: per indicator, retrieve top-k clauses and materialize the
    ones that pass the verbatim validator.

    ``llm_workers`` > 1 generates the per-citation Mapping Rationale (the LLM call
    inside :func:`_materialize`) concurrently across this document's citations via
    a thread pool — the rationale request is I/O-bound, so this collapses the
    dominant wall-clock cost of an LLM run. Order is preserved; counts/tokens stay
    exact (the client locks its accounting). No effect without an LLM rationale
    generator (the template path is local and already fast).

    When ``verifier`` is supplied (the optional LLM gate, P-3) it runs AFTER
    retrieval and verbatim as a tightening step: of the BM25-passing candidates
    it selects at most one clause that actually supports the indicator, or
    abstains (dropping the citation). It can only narrow the keyword result — it
    never adds a clause or relaxes a gate.

    ``enforced_only`` (official scope, internal guide p.8) drops a whole
    instrument's citations when it is positively repealed/draft. The verdict
    merges the authoritative portal channel (``document.status``) with a
    conservative title/head-text signal; ``unknown`` is treated as enforced, so a
    law is only ever dropped on a clear retire/draft signal, never on doubt.

    ``rel_floor`` is the multi-section precision gate (WS-5). With ``top_k > 1`` an
    indicator can map to several sections of the same instrument (the official
    "one document, several relevant sections" case), but a secondary section is
    emitted only when its normalized relevance is at least ``rel_floor`` of the
    best surviving section's — so a broad indicator can't drag in weakly-related
    sections. ``rel_floor=0`` disables the gate (single-section behaviour)."""
    citations: list[Citation] = []
    if not clauses:
        return citations
    if enforced_only:
        portal_status = InstrumentStatus(document.status)
        text_status = detect_status(title=document.title or "", text=document_text)
        if not is_enforced(merge_status(portal_status, text_status)):
            return citations  # repealed/draft instrument — out of the inventory
    index: BM25Index = build_index(clauses)
    clause_by_id = {c.clause_id: c for c in clauses}
    # Optional cross-encoder rerank stage (LEXORA_MAP_RERANK), built ONCE per document
    # (the model is a cached singleton). None when disabled/unavailable -> BM25 path.
    reranker = _maybe_reranker(_map_use_rerank())
    # Document-level metadata (Law Number / Last Amended) resolved ONCE per document.
    doc_last_amended, doc_law_number, doc_meta_note = _resolve_doc_metadata(
        document, document_text, profile, meta_extractor
    )
    # Common kwargs for every _materialize call on this document.
    common = dict(
        document=document, legal_form=legal_form, profile=profile,
        discovery_tag=discovery_tag, rationale_gen=rationale_gen,
        last_amended=doc_last_amended, law_number=doc_law_number,
        meta_note=doc_meta_note,
    )
    # Collect materialization specs first (cheap, sequential — retrieval, boundary
    # filtering, the verifier verdict), then run the per-citation rationale calls
    # (the slow, I/O-bound part inside _materialize) — concurrently when asked.
    specs: list[dict] = []
    sec_notes = secondary_note_by_indicator or {}

    # Per-clause 9-in-1 relevance: pool candidate clauses across all indicators, then
    # judge each clause ONCE against all of them (the focused single-clause question a
    # full-text skim gets wrong). This IS the relevance decision, so it replaces the
    # per-indicator verifier loop below.
    if verifier is not None and getattr(verifier, "mode", "pick_one") == "per_clause":
        specs = _per_clause_specs(
            indicators, profile, index, clause_by_id, verifier,
            top_k=top_k, min_score=min_score, rel_floor=rel_floor,
            use_semantic=_map_use_dense(), reranker=reranker,
            sec_notes=sec_notes, common=common, llm_workers=llm_workers,
        )
        return _execute_specs(specs, llm_workers, rationale_gen)

    for indicator in indicators:
        secondary_note = sec_notes.get(indicator.submission_id, "")
        # Gate on the NORMALIZED score so `min_score` is a portable [0, 1]
        # relevance floor (raw BM25 is unbounded and corpus-dependent — a
        # fixed raw cutoff prunes nothing on a big document).
        passing = [
            hit for hit in retrieve_candidates(
                indicator, profile, index, top_k=top_k,
                use_semantic=_map_use_dense(), reranker=reranker,
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

        if verifier is not None and getattr(verifier, "mode", "pick_one") == "per_cell":
            # Universal precision lane: judge EVERY (clause × indicator) cell
            # keep/drop, then fall through to the normal multi-section emit path
            # with only the kept clauses. Kills the "broad statute scored against
            # all 9 indicators floods on shared vocabulary" failure mode (e.g. a
            # criminal-procedure clause wrongly surfaced for P6 localization). A
            # None verdict means the backend errored -> keep all (never worse than
            # the un-verified baseline); an empty keep-set is a real "drop all".
            candidates = [clause_by_id[h.clause_id] for h in passing]
            kept = verifier.judge_each(indicator, candidates)
            if kept is not None:
                passing = [h for h in passing if h.clause_id in kept]
            if not passing:
                continue
        elif verifier is not None:
            # One LLM judgement per indicator over its candidate clauses; it may
            # pick one (match/uncertain) or abstain. The score scale is unchanged
            # — confidence stays BM25-derived; the verdict only gates inclusion
            # and, for "uncertain", routes to human review.
            candidates = [clause_by_id[h.clause_id] for h in passing]
            claim = verifier.verify(indicator, candidates)
            if claim is None:
                continue
            hit = next(h for h in passing if h.clause_id == claim.clause_id)
            specs.append(dict(
                indicator=indicator, clause=clause_by_id[claim.clause_id],
                bm25_score=hit.score, review_label=claim.label,
                secondary_note=secondary_note, **common,
            ))
            continue

        # Multi-section precision gate (WS-5): keep the best section always, and a
        # secondary one only if it is genuinely competitive with it. `passing` is
        # rank-ordered (BM25/fused), so passing[0] is the strongest survivor.
        if rel_floor > 0.0 and len(passing) > 1:
            cutoff = rel_floor * _normalize_score(passing[0].score)
            passing = [passing[0]] + [
                h for h in passing[1:] if _normalize_score(h.score) >= cutoff
            ]

        for hit in passing:
            specs.append(dict(
                indicator=indicator, clause=clause_by_id[hit.clause_id],
                bm25_score=hit.score, secondary_note=secondary_note, **common,
            ))

    return _execute_specs(specs, llm_workers, rationale_gen)


def _execute_specs(
    specs: list[dict], llm_workers: int, rationale_gen: RationaleGenerator | None
) -> list[Citation]:
    """Materialize collected specs into citations. Rationale is the only
    network-bound step; parallelize it when ``llm_workers`` > 1 and an LLM
    generator is in play (the template path is local). ``map`` preserves order, so
    citation order is unchanged."""
    use_pool = (
        llm_workers > 1 and len(specs) > 1
        and rationale_gen is not None and rationale_gen._client is not None
    )
    if use_pool:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(llm_workers, len(specs))) as ex:
            results = list(ex.map(lambda s: _materialize(**s), specs))
    else:
        results = [_materialize(**s) for s in specs]
    return [c for c in results if c is not None]


def _per_clause_specs(
    indicators: list[RDTIIIndicator],
    profile: SourceProfile,
    index: BM25Index,
    clause_by_id: dict[str, Clause],
    verifier,
    *,
    top_k: int,
    min_score: float,
    rel_floor: float,
    use_semantic: bool,
    reranker,
    sec_notes: dict,
    common: dict,
    llm_workers: int,
) -> list[dict]:
    """Build materialization specs via the per-clause 9-in-1 judge.

    Pool the candidate clauses every indicator retrieves (union — so a clause
    surfaced under one indicator can still be judged for another, removing the
    attribution ceiling), judge each pooled clause ONCE against all indicators,
    then for each supported indicator keep the clause if it passes that indicator's
    boundary rule, capped to ``top_k`` per indicator by retrieval score and gated
    by ``rel_floor`` (the multi-section precision gate)."""
    ind_by_id = {i.submission_id: i for i in indicators}
    # Pool: clause_id -> best RAW retrieval score; plus the per-(indicator,clause)
    # raw score. Scores stay RAW (``_materialize`` normalizes); gate on normalized.
    pool_score: dict[str, float] = {}
    pair_score: dict[tuple[str, str], float] = {}
    for indicator in indicators:
        for hit in retrieve_candidates(
            indicator, profile, index, top_k=top_k,
            use_semantic=use_semantic, reranker=reranker,
        ):
            if _normalize_score(hit.score) < min_score:
                continue
            pool_score[hit.clause_id] = max(pool_score.get(hit.clause_id, 0.0), hit.score)
            pair_score[(indicator.submission_id, hit.clause_id)] = hit.score
    if not pool_score:
        return []

    pool_ids = list(pool_score)

    def _judge(cid: str) -> tuple[str, set[str] | None]:
        return cid, verifier.judge_clause(clause_by_id[cid], indicators)

    if llm_workers > 1 and len(pool_ids) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(llm_workers, len(pool_ids))) as ex:
            verdicts = dict(ex.map(_judge, pool_ids))
    else:
        verdicts = dict(_judge(cid) for cid in pool_ids)

    # Invert to indicator -> [(clause_id, score)], keeping only boundary-admitted
    # clauses; a judge-assigned indicator that never retrieved the clause uses the
    # clause's best pool score as its confidence proxy.
    by_indicator: dict[str, list[tuple[str, float]]] = {}
    for cid, sids in verdicts.items():
        if not sids:
            continue
        clause = clause_by_id[cid]
        for sid in sids:
            indicator = ind_by_id.get(sid)
            if indicator is None or not admits_clause(indicator.rdtii_id, clause.span.text):
                continue
            score = pair_score.get((sid, cid), pool_score[cid])
            by_indicator.setdefault(sid, []).append((cid, score))

    specs: list[dict] = []
    for sid, items in by_indicator.items():
        items.sort(key=lambda t: t[1], reverse=True)
        items = items[:top_k]
        if rel_floor > 0.0 and len(items) > 1:
            cutoff = rel_floor * _normalize_score(items[0][1])
            items = [items[0]] + [
                it for it in items[1:] if _normalize_score(it[1]) >= cutoff
            ]
        for cid, score in items:
            specs.append(dict(
                indicator=ind_by_id[sid], clause=clause_by_id[cid],
                bm25_score=score, secondary_note=sec_notes.get(sid, ""), **common,
            ))
    return specs


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
    meta_note: str = "",
    secondary_note: str = "",
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
    note_parts: list[str] = []
    if status is ReviewStatus.verified and review_label is ClaimLabel.uncertain:
        status = ReviewStatus.conflict_review
        note_parts.append("LLM verifier flagged the mapping as uncertain.")
    if rationale_gen is not None:
        rationale, rationale_note = rationale_gen.generate(
            indicator, profile, clause, clause.structural_path
        )
        if rationale_note:
            note_parts.append(rationale_note)
    else:
        rationale = template_rationale(indicator, profile, clause, clause.structural_path)
    # Document-level own-knowledge channel (same for every row of this document).
    if meta_note:
        note_parts.append(meta_note)
    # Secondary-source provenance (WS-S, USE 3): corroboration, never evidence.
    if secondary_note:
        note_parts.append(secondary_note)
    notes = " | ".join(note_parts)
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
