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
import threading
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
from lexora.console import console_safe
from lexora.export.law_name import resolve_law_name
from lexora.extract.html_extractor import HtmlBlock, extract_html
from lexora.extract.html_extractor import assemble_global_text as assemble_html_text
from lexora.extract.pdf_text_extractor import PdfPage, extract_pdf_bytes, extract_pdf_text
from lexora.extract.pdf_text_extractor import assemble_global_text as assemble_pdf_text
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


def _degraded_min_score() -> float:
    """Relevance floor applied only after the judge has been declared dead. See the call
    site for the measurement; ``LEXORA_DEGRADED_MIN_SCORE=0`` restores the shared floor."""
    try:
        return float(os.environ.get("LEXORA_DEGRADED_MIN_SCORE", "0.7"))
    except ValueError:
        return 0.7


def _judge_breaker_at() -> int:
    """Consecutive judge failures after which we stop calling it for this document.

    Default 8: high enough that a handful of flaky replies on a healthy endpoint never
    trips it (any success resets the counter), low enough that a dead endpoint is detected
    in seconds rather than after every clause has exhausted its retry ladder.
    ``LEXORA_JUDGE_BREAKER_AT=0`` disables the breaker.
    """
    try:
        return max(0, int(os.environ.get("LEXORA_JUDGE_BREAKER_AT", "8")))
    except ValueError:
        return 8


def _judge_is_dead(failed: int, asked: int) -> bool:
    """True when this document's judge failures look SYSTEMATIC rather than incidental.

    The per-clause lane drops a clause whose judgement errored, which is the right call for
    one flaky reply -- an outage must not invent a mapping. But apply it to every clause and
    a dead endpoint produces an empty document silently: zero calls, zero cost, zero
    citations, no exception. That has happened once already (a 403 that printed itself as a
    successful $0.0000 run), and an invalid key still reproduces it exactly.

    Above the threshold the caller degrades the whole document to the key-free ranking lane
    instead. Default 0.5: a single failure on a two-clause document is not evidence of an
    outage, so require a majority. ``LEXORA_JUDGE_DEGRADE_AT=1.1`` disables the gate (no
    ratio can exceed 1), restoring the old drop-everything behaviour.
    """
    if asked <= 0:
        return False
    try:
        threshold = float(os.environ.get("LEXORA_JUDGE_DEGRADE_AT", "0.5"))
    except ValueError:
        threshold = 0.5
    return failed / asked > threshold


def _map_pool_k() -> int:
    """How many clauses per indicator enter the per-clause judge's candidate pool.

    This is a RECALL GATE, not an output cap: whatever the judge admits from the pool
    ships (see :func:`_per_clause_specs`). It exists only so a 1804-clause Act (AU
    Criminal Code) is not judged clause-by-clause — the flagships (~140 clauses) are
    effectively judged whole at the default.

    40 because that is where the measured retrieval ceiling saturates over the SG/MY
    flagship gold (pool 3 -> 38%, 10 -> 62%, 20 -> 76%, 40 -> 95%); a gold section can
    sit deep (MY 7.1 s.45 at BM25 rank 38) because gold is a relevance SET, not a
    ranking. Raising it costs one LLM call per extra clause and can only add recall."""
    try:
        return max(1, int(os.environ.get("LEXORA_MAP_POOL_K", "40")))
    except ValueError:
        return 40


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
    from lexora.extract.ocr_extractor import make_engine, ocr_fill_pages

    # Build the engine here rather than letting ocr_fill_pages do it lazily, so the audit
    # trail can record the engine that ACTUALLY ran -- version and execution provider
    # included ("rapidocr:1.2.3+cuda" vs "rapidocr:1.2.3"). The field used to be filled
    # from LEXORA_OCR_ENGINE, i.e. the *configured* name, so it read "rapidocr" whether
    # the pages went through the GPU or the CPU, and a silent CPU fallback was invisible.
    # Both guards above have already passed, so importing the backend now is not eager.
    engine = make_engine()
    filled, page_conf = ocr_fill_pages(pages, source, engine=engine)
    # Average over the pages OCR actually produced TEXT for. `page_conf` carries an entry
    # for every page OCR was RUN on, and a blank or image-only cover page yields no lines
    # and therefore a confidence of 0.0. Averaging those in reported a document as
    # "scanned, mean OCR confidence 0.000" when its citable text had come from the text
    # layer all along and OCR had merely swept a few empty pages: 19 of the 51 scanned rows
    # of the round-1 submission said exactly that, above quotes of clean legal prose.
    #
    # No usable page means nothing was filled, so the document is not a scan at all and the
    # field stays null rather than claiming a measurement that was never made.
    text_by_page = {p.page_number: p.text for p in filled}
    usable = {n: c for n, c in page_conf.items() if text_by_page.get(n, "").strip()}
    if usable:
        meta["scanned"] = True
        meta["ocr_quality_cer"] = round(sum(usable.values()) / len(usable), 4)
        meta["ocr_engine"] = engine.name
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
    llm_workers: int = 1,
) -> DemoArtifacts:
    """Run extract→structure→retrieve→cite for one local PDF.

    ``llm_workers`` matters here for the same reason it does in the multi-instrument
    path: under the per-clause 0/1 judge the LLM is called once per pooled clause, so a
    serial run of a large Act waits on hundreds of round trips. It changes wall-clock
    only -- token counts, and therefore cost, are identical at any concurrency."""
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
    document_text = assemble_pdf_text(pages)  # same helper the clause offsets index
    citations = _citations_from_clauses(
        clauses, document, profile, indicators, legal_form, top_k, min_score,
        verifier=verifier, rationale_gen=rationale_gen, meta_extractor=meta_extractor,
        document_text=document_text, enforced_only=enforced_only,
        llm_workers=llm_workers,
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

    # Assembled by the extractors' own helpers, which are the definition of the text the
    # clause offsets index. Hand-rolling the join here used a SINGLE newline between HTML
    # blocks where the parser (and `html_extractor.assemble_global_text`) use
    # BLOCK_SEPARATOR = "\n\n", so from the second block onward every published
    # char_start/char_end was off by one per boundary and did not index this string.
    #
    # Visible in the round-1 submission: `raw_context_before`/`after` in the JSON sidecar
    # are located by `document_text.find(quote)`, and a clause spanning a block boundary
    # carries the "\n\n" the parser saw, which is not present here. 167 of 228 Australian
    # rows (73%) shipped with no context at all, and every one of them came through this
    # branch -- the AU EPUB/XHTML path. No PDF row was affected.
    document_text = (
        assemble_pdf_text(pages) if pages else assemble_html_text(blocks)
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
    per_indicator_limit: int = 8,
    max_queries_per_indicator: int = 4,
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
    from lexora.collect.discovery import (
        discover_for_indicators,
        reset_acquisition_log,
        reset_page_memo,
        resolve_fulltext,
    )
    from lexora.models.source import FetchMethod

    # One economy run = one acquisition tally and one page memo. Both are scoped here
    # rather than to the sweep, because the follow-on discovery passes below issue the
    # majority of a run's queries and belong to the same accounting.
    reset_acquisition_log()
    reset_page_memo()

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
    _processed: dict[tuple, DemoArtifacts] = {}
    _processed_lock = threading.Lock()
    # One lock per cache key, so two workers that draw the SAME document queue instead of
    # both doing the work. Checking the memo and filling it under separate locks is a
    # check-then-act race: at doc_workers=8 both threads miss, both fetch, both OCR and
    # both pay for a full set of judge calls, and the loser's artifacts object is a
    # distinct object that the id()-based de-dup below cannot see -- so the document is
    # also counted twice in `fetched_ok` / `docs_with_clauses` and written twice into the
    # JSON sidecar. The memo existed precisely to stop that work happening twice.
    _key_locks: dict[tuple, threading.Lock] = {}

    def _process(hit) -> DemoArtifacts:
        t0 = time.perf_counter()
        fulltext = resolve_fulltext(hit, force_browser=force_browser, timeout=timeout)
        target = fulltext or hit.url
        tag = DiscoveryTag.new if hit.discovery_tag == "NEW" else DiscoveryTag.known
        # Which indicators is this instrument scored against?
        #
        # Regime-1 (no LLM judge): only the indicators whose query surfaced it (a
        # retention-query hit is a candidate for 7.3, not for all nine). For a
        # name-driven hit (AU OData has no full-text, so no surfacing indicator),
        # attribute via the profile's indicator->instrument-name hints; only fall back
        # to all indicators when nothing pins it down.
        #
        # This discovery attribution is a RECALL CEILING: a law reached under one
        # indicator is never tried for another, so a provision that serves two pillars
        # is only ever cited under one. Measured on the 2026-07-12 run: MY PDPA was
        # attributed to 7.1/7.2/7.5 only, so s.129 (gold for BOTH 7.1 and 6.4) never got
        # scored for 6.4 — even though it ranks #1 there when simply asked. Same for SG
        # PDPA s.26 (6.4, rank #1) and s.11 (7.4, rank #1).
        #
        # A verifier that judges every (clause x indicator) pair itself — per_cell and
        # per_clause alike — makes the ceiling redundant AND lossy: it is built to reject
        # the wrong-indicator matches the ceiling was guarding against, so it can be
        # handed all nine indicators and decide for itself. A full-text brute judge
        # likewise wants all indicators.
        judges_all = verifier is not None and getattr(verifier, "mode", "") in (
            "per_cell", "per_clause")
        if brute_judge is not None or judges_all:
            ind_subset = indicators
        else:
            wanted = set(hit.indicator_hits) or _attribute_by_name(hit, profile, indicators)
            ind_subset = [i for i in indicators if i.submission_id in wanted] or indicators

        # Ask the same question of the same document once. Two discovery hits can
        # resolve to one full text (a law surfaced by both its name and a concept
        # phrase), and the follow-on passes re-reach laws the main sweep already
        # mapped — measured 2026-07-27: 8 of 122 map events were exact repeats, and
        # the amendment pass fetched, OCR'd, parsed and mapped each duplicate in full
        # before `_consider` discarded it on its hash.
        #
        # The key carries the discovery tag and the indicator subset, not just the
        # URL: under regime-1 attribution the same text scored against a different
        # indicator set is a DIFFERENT question, and reusing there would silently
        # narrow the second hit's coverage to the first one's.
        cache_key = (target, tag.value, tuple(i.submission_id for i in ind_subset))
        with _processed_lock:
            done = _processed.get(cache_key)
            key_lock = _key_locks.setdefault(cache_key, threading.Lock())
        if done is not None:
            return done

        with key_lock:
            # Re-check inside the key lock: whoever held it may have just finished.
            with _processed_lock:
                done = _processed.get(cache_key)
            if done is not None:
                return done
            return _map_one(hit, target, tag, ind_subset, cache_key, t0)

    def _map_one(hit, target, tag, ind_subset, cache_key, t0) -> DemoArtifacts:
        """Fetch, parse and map one instrument. Called with this key's lock held."""
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
        with _processed_lock:
            _processed[cache_key] = artifacts
        with _progress_lock:
            _progress["done"] += 1
            n = _progress["done"]
            total = _progress["total"]
        # `_process` is also the worker for the three follow-on passes below (amendments,
        # child regulations, regulator soft law), which grow the working set as they go.
        # Their counts are not known in advance, so once the initial set is exhausted the
        # denominator is dropped rather than printed as a number the run has passed.
        where = f"{n}/{total}" if total is not None and n <= total else f"{n} (follow-on)"
        logger.info(
            "mapped %-12s %-38s %d clause(s) -> %d citation(s) [%.1fs]",
            where, console_safe((hit.title or hit.url)[:38]),
            len(artifacts.clauses), len(artifacts.citations),
            artifacts.processing_time_seconds,
        )
        return artifacts

    # Document-level parallelism: process instruments concurrently. This is what
    # parallelizes the per-document metadata extraction (one LLM call each, the
    # serial floor of an LLM run) as well as fetch / OCR / rationale across docs.
    # SG full-text resolves by URL construction (no browser render in the loop), so
    # concurrent docs are safe. `map` preserves order; dedup runs sequentially after.
    skipped: list[str] = []

    def _process_isolated(hit):
        """One document's failure must not take the jurisdiction with it.

        `_fetch_in_order` has given the FOLLOW-ON passes exactly this since they were
        written; the primary working-set pass -- the one that maps everything discovery
        found -- never had it, so a single unfetchable URL raised straight out of
        `ex.map` and ended the economy. On 2026-07-31 three page-relative Malaysian
        download links did that: Malaysia contributed 0 rows to a run that exited 0 and
        wrote a submission CSV of 561 rows from the other two economies.

        Unlike the follow-on version this one is loud. There the candidates are
        speculative, so a quiet None is right; here every hit is something discovery
        decided was worth mapping, and losing one silently is how a run comes back
        smaller with nothing to say why.
        """
        try:
            return _process(hit)
        except Exception as exc:  # noqa: BLE001 — one document, not the jurisdiction
            url = getattr(hit, "url", "") or "?"
            skipped.append(url)
            logger.error("SKIPPED %s -- %s: %s", url[:110], type(exc).__name__, exc)
            return None

    if doc_workers > 1 and len(hits) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(doc_workers, len(hits))) as ex:
            documents = list(ex.map(_process_isolated, hits))
    else:
        documents = [_process_isolated(h) for h in hits]
    documents = [d for d in documents if d is not None]
    if skipped:
        logger.error("%d of %d discovered document(s) skipped after an unrecoverable "
                     "error; the rest of this jurisdiction is unaffected",
                     len(skipped), len(hits))
    # Two hits that resolved to one document now share one artifacts object; carry it
    # once so the JSON sidecar and the currency pass each see the document once.
    seen_artifacts: set[int] = set()
    documents = [d for d in documents
                 if id(d) not in seen_artifacts and not seen_artifacts.add(id(d))]
    with _progress_lock:
        _progress["total"] = None  # the discovered set is done; what follows is follow-on

    # Give every document its real name BEFORE anything reads the title.
    #
    # `_resolve_doc_metadata` has always called `resolve_law_name`, but it runs per
    # CITATION, hundreds of lines below — after the amendment pass, which builds its
    # search queries FROM the title. So a Malaysian Act the portal serves as an upload
    # name was searched for as `"Akta91y2006bi (Amendment) Act"`, matched nothing, and
    # its currency stayed UNKNOWN. The fix existed and was tested; it simply ran too
    # late to reach the caller that needed it. Stale law is one of the three failure
    # modes the organisers named for AI on this task, so a whole class of documents
    # silently unable to report amendments is not a cosmetic defect.
    #
    # The later call stays: the follow-on passes below ADD documents after this point,
    # and re-resolving an already-resolved name is a no-op (a real law name is not
    # filename-shaped, so the resolver returns it untouched).
    for artifact in documents:
        if artifact.document_text:
            artifact.document.title = resolve_law_name(
                artifact.document.title, artifact.document_text)

    # Tag every fetched law original/amendment/consolidated and, for each ORIGINAL,
    # look for ITS amendments (queries derived from its own title — general, not a
    # fixed amendment list). Found amending Acts join the working set so the currency
    # pass below can adjudicate the principal's provisions against real instructions.
    if discover_amendments:
        documents = documents + _discover_amendments(
            documents, _process, portal, profile,
            force_browser=force_browser, timeout=timeout, workers=doc_workers,
        )
        # AU only: pull in the principal REGULATIONS made under each Act. The brute
        # Act-enumeration skips delegated legislation (a different FRL collection), so
        # a regulation that carries an indicator (e.g. Telecommunications Regulations
        # 2021 -> P7-I5) is otherwise unreachable. Runs after the amendment pass so it
        # sees (and de-dups against) the amendments already added.
        documents = documents + _discover_child_regulations(
            documents, _process, profile=profile, timeout=timeout, workers=doc_workers,
        )
        # Regulator soft-law (codes of practice / standards) the statute portal does
        # not index — gold instruments (MY: PDP Codes of Practice + Standard 2015)
        # otherwise unreachable by the map pipeline.
        documents = documents + _discover_regulator_instruments(
            profile, _process, documents, timeout=timeout, workers=doc_workers,
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


def _fetch_in_order(process_fn, hits: list, workers: int) -> list:
    """Fetch/parse/map a batch of discovery hits, results in the batch's own order.

    The follow-on passes each discover candidates one query at a time (browser-bound
    and serial, under the portal's rate limit) and then need every candidate fetched,
    OCR'd, parsed and mapped (network- and CPU-bound, and independent per candidate).
    Doing the second inline with the first makes the whole pass run at the sum of both:
    measured 2026-07-27, the follow-on passes were 72% of a Singapore run and each of
    their ~130 fetches waited on the previous one.

    Order is preserved and acceptance stays with the caller, so which of two
    byte-identical candidates is kept never depends on thread timing. A candidate that
    raises yields ``None`` rather than losing the batch.
    """
    def _safe(hit):
        try:
            return process_fn(hit)
        except Exception:  # noqa: BLE001 — one bad candidate, not the pass
            return None

    if workers > 1 and len(hits) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(workers, len(hits))) as ex:
            return list(ex.map(_safe, hits))
    return [_safe(h) for h in hits]


def _discover_amendments(
    documents: list[DemoArtifacts],
    process_fn,
    portal: PortalSpec,
    profile: SourceProfile,
    *,
    force_browser: bool,
    timeout: float,
    max_queries: int | None = None,
    per_query: int = 4,
    workers: int = 1,
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
    so a heavily-amended principal does not drag in dozens of already-incorporated Acts.

    ``max_queries=None`` (the default) covers every principal in the working set; pass
    an int only to throttle. The old fixed cap silently limited STALE_RISK detection to
    the first handful of principals on large (e.g. brute-enumerated) working sets."""
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
        for q in amendment_search_queries(
            a.document.title, word_tokenised=portal.word_tokenised_search
        ):
            if q not in query_floor:
                query_floor[q] = floor
            elif floor is None or query_floor[q] is None:
                query_floor[q] = None
            else:
                query_floor[q] = min(query_floor[q], floor)
    # By default cover EVERY principal's amendment queries (max_queries=None) so
    # STALE_RISK detection is not silently truncated to the first few principals.
    # The OData search per query is cheap; the costly fetch is bounded by per_query
    # and dedup, and for CONSOLIDATED principals further year-gated below — so the
    # honest default is "no cap". A caller may still pass an int to throttle.
    queries = list(query_floor)
    if max_queries is not None:
        queries = queries[:max_queries]

    new_docs: list[DemoArtifacts] = []
    # Candidates are gathered first and fetched afterwards, on a pool. Discovering them
    # is browser-bound and strictly serial (one session, spaced under the portal's rate
    # limit); fetching and parsing them is network- and CPU-bound and independent per
    # candidate. Interleaving the two, as this used to, made the whole pass run at the
    # speed of the slower one: measured 2026-07-27, the follow-on passes were 72% of a
    # Singapore run and every one of their ~130 fetches waited for the previous.
    candidates: list = []

    def _collect(hit, floor: int | None) -> None:
        """Queue one amendment candidate, applying the checks that need no fetch."""
        if hit.url in seen_url:
            return
        seen_url.add(hit.url)
        # Skip an amendment a consolidated principal already incorporates, judged by
        # the title year BEFORE fetching (avoids fetching/parsing dead weight).
        hit_year = _year_of(hit.title)
        if floor is not None and hit_year is not None and hit_year <= floor:
            return
        candidates.append(hit)

    def _fetch_candidates() -> None:
        """Fetch the queued candidates and keep the ones that really are amendments."""
        if not candidates:
            return
        fetched = _fetch_in_order(process_fn, list(candidates), workers)
        candidates.clear()
        for art in fetched:
            if art is None or art.document.sha256 in have_sha:
                continue
            if classify_version(art.document_text or "") is VersionKind.amendment_delta:
                have_sha.add(art.document.sha256)
                new_docs.append(art)

    # Path 1 — title-derived name search (portal-agnostic). Works for amendments named
    # "<principal core> Amendment Act"; misses theme-named omnibus Acts (see Path 2).
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
            _collect(hit, floor)
    _fetch_candidates()

    # Path 2 — AU FRL reverse-lookup (recovers the omnibus amendments Path 1 cannot).
    # Australia's omnibus amendment Acts ("Surveillance Legislation Amendment (Identify
    # and Disrupt) Act 2021") are named by policy theme, not by the principal they
    # amend, so a title-derived query never surfaces them. The Federal Register's own
    # versions/affects graph lists every amending Act by id; reverse-look up each AU
    # principal in the working set (keyed by its FRL title id, which we already hold in
    # the document URL). The per-principal year floor is the SAME staleness rule as
    # Path 1 (ORIGINAL pursues all; CONSOLIDATED only amendments after its point).
    from lexora.collect.strategies import au_amendment_acts, frl_id_from_url

    for a in documents:
        fid = frl_id_from_url(str(a.document.source_url))
        if not fid:
            continue
        kind = classify_version(a.document_text or "")
        if kind is VersionKind.original:
            floor = None
        elif kind is VersionKind.consolidated:
            floor = detect_incorporated_to(a.document_text or "")
        else:
            continue
        try:
            amd_hits = au_amendment_acts(
                fid, timeout=timeout,
                known_instruments=profile.known_instruments,
                known_instrument_ids=profile.known_instrument_ids,
            )
        except Exception:  # noqa: BLE001
            continue
        for hit in amd_hits:
            _collect(hit, floor)
    _fetch_candidates()

    # Path 3 — SG SSO inline-annotation reverse-lookup (the SG analogue of Path 2).
    # SSO has no FRL-style affects API, but a consolidated Act's text annotates each
    # amending Act inline as "Act N of YYYY"; build each one's Acts Supplement URL.
    # SG CONSOLIDATES amendments into the principal, so the staleness year-floor does
    # NOT apply — the goal is to RECALL the standalone amendment instrument (a separate
    # gold item, e.g. the PDPA (Amendment) Act 2020), so pass floor=None and let the
    # AMENDMENT_DELTA filter discard the principal's own enactment and cross-references.
    from lexora.collect.strategies import sg_amendment_acts

    for a in documents:
        if "sso.agc.gov.sg" not in str(a.document.source_url):
            continue
        for hit in sg_amendment_acts(a.document_text or ""):
            _collect(hit, None)
    _fetch_candidates()

    return new_docs


def _discover_regulator_instruments(
    profile: SourceProfile,
    process_fn,
    existing: list[DemoArtifacts],
    *,
    timeout: float,
    workers: int = 1,
) -> list[DemoArtifacts]:
    """Harvest soft-law (codes of practice, standards) from the regulator portals
    in the profile and add the ones that fetch to real text.

    A regulator (SG PDPC, MY PDP, AU OAIC) publishes codes/standards/guidance that
    the primary STATUTE portal does not index — they are gold instruments for
    several indicators (MY: the PDP Codes of Practice + Standard 2015) yet invisible
    to the statute search. ``connector_for`` harvests each such portal's corpus; the
    map pipeline (unlike the ``discover`` CLI) did not run it, so these never reached
    the working set. Fetch each via the normal per-document path (the PDF resolves
    through ``resolve_fulltext``'s page .pdf-harvest) and keep any that parsed."""
    from lexora.collect.strategies import connector_for

    have_sha = {a.document.sha256 for a in existing}
    seen_url = {str(a.document.source_url) for a in existing}
    new_docs: list[DemoArtifacts] = []
    for portal in profile.portals:
        connector = connector_for(portal)
        if connector is None:
            continue
        try:
            hits = connector(
                portal, [], limit=40, timeout=timeout,
                known_instruments=profile.known_instruments,
                known_instrument_ids=profile.known_instrument_ids,
            )
        except Exception:  # noqa: BLE001 — a failed connector never breaks the run
            continue
        fresh = [h for h in hits if h.url not in seen_url]
        seen_url.update(h.url for h in fresh)
        for art in _fetch_in_order(process_fn, fresh, workers):
            if art is None or art.document.sha256 in have_sha:
                continue
            if art.document_text:  # kept iff it fetched to real text
                have_sha.add(art.document.sha256)
                new_docs.append(art)
    return new_docs


def _discover_child_regulations(
    documents: list[DemoArtifacts],
    process_fn,
    *,
    profile: SourceProfile | None = None,
    timeout: float,
    workers: int = 1,
) -> list[DemoArtifacts]:
    """For every AU principal Act in the working set, discover the principal
    REGULATIONS made under it and add the ones that fetch successfully.

    The brute Act-enumeration only judges the ``Act`` collection, so delegated
    legislation (regulations / rules, a separate Federal Register collection) is
    invisible — yet a regulation can be the sole carrier of an indicator (e.g.
    Telecommunications Regulations 2021 for P7-I5 government access). AU exposes no
    Act -> instruments navigation, so :func:`au_child_regulations` finds them by the
    "<Act stem> Regulations" naming convention and confirms each via its FRL
    ``authorisedBy`` edge (made-under this Act). Keyed by the principal's FRL title id,
    which is already in its document URL; non-AU documents are skipped. A regulation
    is kept when it fetched to real text — relevance is the mapper's call."""
    from lexora.collect.strategies import au_child_regulations, frl_id_from_url

    have_sha = {a.document.sha256 for a in documents}
    seen_url = {str(a.document.source_url) for a in documents}
    new_docs: list[DemoArtifacts] = []
    for a in documents:
        fid = frl_id_from_url(str(a.document.source_url))
        if not fid or not a.document.title:
            continue
        try:
            hits = au_child_regulations(
                fid, a.document.title, timeout=timeout,
                known_instruments=profile.known_instruments if profile else None,
                known_instrument_ids=profile.known_instrument_ids if profile else None,
            )
        except Exception:  # noqa: BLE001 — a failed lookup never breaks the run
            continue
        fresh = [h for h in hits if h.url not in seen_url]
        seen_url.update(h.url for h in fresh)
        for art in _fetch_in_order(process_fn, fresh, workers):
            if art is None or art.document.sha256 in have_sha:
                continue
            if art.document_text:  # kept iff it fetched to real text
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
    and a citation always gets the best value available (or blank). ``review_note`` is
    non-empty only when the extractor's separate recall channel CONTRADICTS the
    document (opt-in, review-only, never an answer)."""
    # Law Name comes from the portal, and Malaysia's AGC portal serves each Act as a file
    # named by whoever uploaded it -- eight of the eleven Malaysian NEW rows of the round-1
    # submission carried one ("Act 706 ori.pdf", "DRAF KEDUA AKTA 701 (final)(KU) (1).pdf")
    # in the column that says which instrument the row is about. Resolved here because this
    # is the one place that already holds the document's own text, where the Act states its
    # real title. A portal title that already reads like a law name is untouched.
    document.title = resolve_law_name(document.title, document_text)
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
    # An amending Act that ALSO surfaces a citation of its own (a provision it enacts,
    # mapped directly to an indicator the consolidated principal missed) is exported
    # under the amendment's name + its internal Schedule locator — never linked to the
    # principal in the fixed submission columns. Record what each amending document
    # amends (read from its OWN masthead, not guessed) so the citation can carry that
    # link in the Notes column. ``detect_amends_target`` is non-None by amend_docs's filter.
    amends_target_by_hash: dict[str, tuple[str, str]] = {
        a.document.sha256: detect_amends_target(a.document_text or "")  # type: ignore[misc]
        for a in amend_docs
    }
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

    def _principal_of(target: tuple[str, str] | None) -> str:
        """Render an amends-target ``(number, title)`` as a human reference, e.g.
        "Privacy Act 1988 (Act C2004A03712)". Either field may be empty."""
        if not target:
            return ""
        number, title = target
        if title and number and number.lower() not in title.lower():
            return f"{title} ({number})"
        return title or number

    for c in citations:
        c.source_version = ver_by_hash.get(c.document_hash, "")
        # If the citation's OWN source is an amending Act, the fixed submission columns
        # name the amendment and locate the snippet inside the amendment's Schedule —
        # nothing links it to the principal it amends. Add that link in Notes (an
        # official column), read from the amendment's masthead, so a reviewer reading
        # only the submission CSV can trace the provision to the consolidated principal.
        if c.source_version == _VK.amendment_delta.value:
            principal = _principal_of(amends_target_by_hash.get(c.document_hash))
            amd_note = (
                "Source is an amending Act (delta); snippet is the provision as enacted "
                "by this amendment"
                + (f" to {principal}" if principal else "")
                + " — refer to the consolidated principal for the in-force text."
            )
            c.notes = f"{c.notes} | {amd_note}" if c.notes else amd_note
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
    # (indicator, clause) pairs the judge already ruled on. Stays empty unless a degraded
    # document falls through to the ranking lane below, which must not re-emit them.
    judged_pairs: set[tuple[str, str]] = set()

    # Per-clause 9-in-1 relevance: pool candidate clauses across all indicators, then
    # judge each clause ONCE against all of them (the focused single-clause question a
    # full-text skim gets wrong). This IS the relevance decision, so it replaces the
    # per-indicator verifier loop below.
    if verifier is not None and getattr(verifier, "mode", "pick_one") == "per_clause":
        specs, judge_failed, judge_asked = _per_clause_specs(
            indicators, profile, index, clause_by_id, verifier,
            top_k=top_k, min_score=min_score, rel_floor=rel_floor,
            use_semantic=_map_use_dense(), reranker=reranker,
            sec_notes=sec_notes, common=common, llm_workers=llm_workers,
        )
        if not _judge_is_dead(judge_failed, judge_asked):
            return _execute_specs(specs, llm_workers, rationale_gen)
        # Systematic judge failure: fall through to the key-free ranking lane below by
        # dropping the verifier. That lane is measurably worse (BM25 + boundary rules reach
        # 24-29% gold recall against the judge's 85%) but it is REAL, verbatim-grounded
        # output instead of nothing -- and it is measurably better than any small-model
        # fallback we tested, at zero latency and zero spend. Every row says so in Notes,
        # because a degraded row that looks identical to a judged one is the failure this
        # gate exists to prevent.
        logger.warning(
            "judge unavailable for %s: %d/%d clause judgements failed -> DEGRADED to the "
            "BM25 ranking lane for this document (lower recall; rows are marked)",
            document.title or document.source_url, judge_failed, judge_asked,
        )
        verifier = None
        # The two lanes want OPPOSITE things from `min_score`, so the degraded lane gets its
        # own floor. For the judge, the pool is a pure RECALL gate -- a clause it never sees
        # it can never admit -- so the floor stays low (raising it to 0.7 would cut 29% of
        # the judge calls but also make one reachable gold section unreachable). The ranking
        # lane has no way to say "this indicator has nothing here": it emits top_k for every
        # indicator, so a weak floor is exactly how a document ends up with citations on
        # indicators the legal group marked N/A. Measured on the MY PDPA, lifting the floor
        # to 0.7 here drops those provably-wrong rows from 16 to 10 and total citations from
        # 25 to 19 with gold recall UNCHANGED at 38% -- pure precision, no cost.
        min_score = max(min_score, _degraded_min_score())
        # The judge usually answers for SOME clauses before it dies (the breaker opens after
        # 8 consecutive failures, so everything judged up to that point is a real verdict).
        # Those specs are kept -- throwing away evidence the run already paid for would be
        # worse -- but the fall-through gave them neither of the two things they need.
        #
        # They must not be re-emitted by the ranking lane: the same (indicator, clause) then
        # ships twice, and because `run_pipeline_map` de-dups on first-seen, the row that
        # SURVIVES into the CSV is the judged one, which carries no degrade marker at all.
        # A document declared degraded was quietly publishing unmarked rows.
        #
        # And they must say that the DOCUMENT is incomplete even though the row itself was
        # judged, because nothing else distinguishes a fully-judged document from the
        # surviving third of a broken one.
        judged_pairs = {(s["indicator"].submission_id, s["clause"].clause_id) for s in specs}
        partial_note = (
            f"PARTIALLY DEGRADED: the LLM judge failed on {judge_failed}/{judge_asked} "
            "clauses of this document; THIS row was judged, but the document's coverage "
            "is incomplete"
        )
        for spec in specs:
            spec["meta_note"] = " | ".join(
                filter(None, [spec.get("meta_note", ""), partial_note])
            )
        common["meta_note"] = " | ".join(filter(None, [
            common.get("meta_note", ""),
            f"DEGRADED: LLM judge unavailable ({judge_failed}/{judge_asked} judgements "
            f"failed); this row comes from BM25 retrieval + boundary rules, not the judge",
        ]))

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
            if (indicator.submission_id, hit.clause_id) in judged_pairs:
                continue  # the judge already ruled on this pair; don't ship it twice
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

    This lane asks a MEMBERSHIP question — "is this clause relevant to this indicator,
    yes/no" — not a RANKING one. That is the same question the legal group answered when
    they annotated gold: they worked from a three-channel union pool plus a whole-Act
    section index (``scripts/eval_mapping.py``), so gold is a *set* of relevant sections
    per indicator, not a top-3 list. A ranking cutoff cannot approximate a set: it forces
    exactly ``top_k`` citations onto every indicator, padding the ones with no relevant
    provision (false positives on indicators the legal group marked N/A) while truncating
    the ones with many (MY 7.1 has 9 gold sections; only 3 could ever be emitted).

    So retrieval is demoted to a pure RECALL GATE — take a wide pool (``pool_k``, default
    40, tunable via ``LEXORA_MAP_POOL_K``) — and the judge's 0/1 verdict is the ONLY thing
    that decides what ships. No ``top_k`` truncation, no ``rel_floor``.

    Measured on the flagships, whole-act 0/1 beats top_k=3 on BOTH axes at once:
    MY gold recall 24-29% -> 53-59% with citations 80 -> 32; SG 67% with ZERO off-gold
    picks; and on both, every indicator the legal group marked N/A came back EMPTY —
    the padding those N/A false positives came from is gone.

    ``min_score`` still gates pool entry, and each indicator's boundary rule still applies
    (both tightening-only). ``top_k``/``rel_floor`` are accepted for signature
    compatibility with the ranking lane and deliberately unused."""
    ind_by_id = {i.submission_id: i for i in indicators}
    # Recall gate: a wide per-indicator pool, unioned. Wide because the judge — not the
    # rank — decides: a gold section sitting at BM25 rank 38 (MY 7.1 s.45) is unreachable
    # at top_k=3 no matter how good the judge is. Measured retrieval ceiling over both
    # flagships: pool 3 -> 38%, 20 -> 76%, 40 -> 95%.
    pool_k = _map_pool_k()
    # Pool: clause_id -> best RAW retrieval score; plus the per-(indicator,clause)
    # raw score. Scores stay RAW (``_materialize`` normalizes); gate on normalized.
    pool_score: dict[str, float] = {}
    pair_score: dict[tuple[str, str], float] = {}
    for indicator in indicators:
        for hit in retrieve_candidates(
            indicator, profile, index, top_k=pool_k, pool_k=max(pool_k, 20),
            use_semantic=use_semantic, reranker=reranker,
        ):
            if _normalize_score(hit.score) < min_score:
                continue
            pool_score[hit.clause_id] = max(pool_score.get(hit.clause_id, 0.0), hit.score)
            pair_score[(indicator.submission_id, hit.clause_id)] = hit.score
    if not pool_score:
        return [], 0, 0

    pool_ids = list(pool_score)

    # Circuit breaker. Without one the degrade gate below is useless in practice: each
    # judgement retries the endpoint up to six times before giving up, so a dead backend
    # makes the run take LONGER than a healthy one and the gate only fires after every
    # clause has been through the full retry ladder. Measured on the MY PDPA: a 90-clause
    # document did not finish in ten minutes against an invalid key. After
    # ``_JUDGE_BREAKER_AT`` consecutive failures we stop calling for the rest of this
    # document and report the remainder as failed, which is exactly what they would be.
    breaker = {"consecutive": 0, "open": False}
    breaker_lock = threading.Lock()
    breaker_at = _judge_breaker_at()

    def _judge(cid: str) -> tuple[str, set[str] | None]:
        if breaker["open"]:
            return cid, None
        verdict = verifier.judge_clause(clause_by_id[cid], indicators)
        with breaker_lock:
            if verdict is None:
                breaker["consecutive"] += 1
                if breaker_at and breaker["consecutive"] >= breaker_at:
                    if not breaker["open"]:
                        logger.warning(
                            "judge failed %d times in a row -> not calling it again for "
                            "this document", breaker["consecutive"],
                        )
                    breaker["open"] = True
            else:
                # A success means the endpoint is alive; flaky replies must not accumulate
                # across an otherwise healthy document into a false outage.
                breaker["consecutive"] = 0
        return cid, verdict

    if llm_workers > 1 and len(pool_ids) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(llm_workers, len(pool_ids))) as ex:
            verdicts = dict(ex.map(_judge, pool_ids))
    else:
        verdicts = dict(_judge(cid) for cid in pool_ids)

    # A None verdict is a backend failure, and this lane deliberately DROPS such a clause
    # rather than keeping it: an outage must never fabricate a mapping. Individually that is
    # right; in bulk it is how a dead endpoint turns into an empty submission that looks like
    # a cheap successful run (calls 0, cost 0.00, citations 0). Count them so the caller can
    # tell "this document had nothing" apart from "we never got an answer".
    failed = sum(1 for v in verdicts.values() if v is None)

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

    # Every clause the judge admitted ships. No top_k truncation (it would cap an
    # indicator whose gold is a 9-section set) and no rel_floor (it would drop a genuinely
    # relevant provision merely for scoring below the best one — retrieval rank is a recall
    # aid here, not evidence). The judge already said no to everything else; sorting is
    # cosmetic, so the strongest-retrieved section still reads first.
    specs: list[dict] = []
    for sid, items in by_indicator.items():
        items.sort(key=lambda t: t[1], reverse=True)
        for cid, score in items:
            specs.append(dict(
                indicator=ind_by_id[sid], clause=clause_by_id[cid],
                bm25_score=score, secondary_note=sec_notes.get(sid, ""), **common,
            ))
    return specs, failed, len(pool_ids)


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
    # Document-level recall-vs-document conflict (same for every row of this document).
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


# Scale of the tanh squash below. The original 5.0 was calibrated against an assumed
# "strong match ~5+", which real documents do not resemble: measured on the MY PDPA, the
# per-indicator top-3 BM25 scores run 20-82. tanh(20/5) is already 0.9993, so EVERY score
# normalised to 1.000 and three things silently stopped working -- `min_score` and
# `rel_floor` became knobs that cannot gate at any setting below 1.0 (which is why
# rel_floor=0.6 and rel_floor=0 produced byte-identical output), and the Confidence column
# saturated: all 669 rows of the Round 1 submission carry ~1.0.
# 40.0 puts the observed range across a usable spread (20 -> 0.46, 40 -> 0.76, 80 -> 0.96).
_SCORE_SCALE = 40.0


def _normalize_score(score: float) -> float:
    """Squash unbounded BM25 scores into [0, 1] for the Pydantic confidence field and for
    the ``min_score`` / ``rel_floor`` gates, which are documented as portable [0, 1]
    relevance floors. See ``_SCORE_SCALE`` for why the scale is what it is."""
    return float(max(0.0, min(1.0, math.tanh(score / _SCORE_SCALE))))


__all__ = [
    "DemoArtifacts",
    "MapResult",
    "run_demo_pipeline",
    "run_pipeline_from_url",
    "run_pipeline_autodiscover",
    "run_pipeline_map",
]
