"""JSON submission sidecar — the official "CSV + JSON" deliverable (Round 1).

The CSV (:mod:`lexora.export.csv_exporter`) carries the 13 human-facing columns;
this JSON carries the SAME provision rows plus the technical metadata a CSV cell
cannot hold, per the official ``README_template.md`` Output Format section:
``source_pdf_path``, ``pdf_is_scanned``, ``ocr_quality_cer``,
``processing_time_seconds``, ``model_version``, ``retrieval_method`` and the raw
before/after context for human-in-the-loop review.

Structure is FLAT — one object per provision, so JSON rows == CSV rows and a judge
can validate the two against each other. The document-level technical fields
(source PDF path, OCR audit, timing, model version) are repeated on each of that
law's provisions.

``ocr_quality_cer`` carries the mean OCR *confidence* (0..1, higher is better), NOT
a reference-based Character Error Rate — we have no ground-truth transcript to score
against. It is emitted under the official field name for schema compatibility and
annotated with ``ocr_quality_note``. Flagged to revisit (progress report §6).
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

SCHEMA_VERSION = "lexora-submission-json/v1"

_OCR_NOTE = (
    "ocr_quality_cer carries mean OCR line confidence (0..1, higher is better), "
    "not a reference-based CER; no ground-truth transcript was available."
)


def _context(text: str, quote: str, window: int) -> tuple[str, str]:
    """Raw before/after context around the verbatim quote, located by exact match in
    the document text (robust to offset-convention differences). Empty when the quote
    is not found verbatim (e.g. HTML reflow)."""
    if not text or not quote:
        return "", ""
    idx = text.find(quote)
    if idx < 0:
        return "", ""
    before = text[max(0, idx - window): idx]
    after = text[idx + len(quote): idx + len(quote) + window]
    return before, after


def _retrieval_method(use_dense: bool, review_status: str) -> str:
    base = "bm25+dense" if use_dense else "bm25"
    # `verified` is the DEFAULT validated status of every citation (not an LLM signal),
    # so it tells us nothing about the verifier. Only CONFLICT_REVIEW is unambiguously
    # produced by the LLM verifier — mark just that to avoid over-claiming.
    if review_status == "CONFLICT_REVIEW":
        return f"{base}+llm-flagged"
    return base


def to_submission_json(
    documents: Iterable,
    out_path: Path,
    *,
    model_version: str = "",
    use_dense: bool = False,
    context_window: int = 200,
) -> int:
    """Write the flat JSON sidecar. ``documents`` is an iterable of per-document
    artifacts (duck-typed: ``document``, ``citations``, ``document_text``,
    ``pdf_is_scanned``, ``ocr_quality_cer``, ``ocr_engine``,
    ``processing_time_seconds``). Provisions are de-duplicated by
    ``(indicator_id, clause_id)`` across documents to mirror the submission CSV.
    Returns the number of provision objects written."""
    provisions: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for art in documents:
        doc = art.document
        text = getattr(art, "document_text", "") or ""
        ocr_cer = getattr(art, "ocr_quality_cer", None)
        engine = getattr(art, "ocr_engine", "") or ""
        doc_model_version = "; ".join(
            p for p in (model_version, f"ocr:{engine}" if engine else "") if p
        )
        for c in art.citations:
            key = (c.indicator_id, c.clause_id)
            if key in seen:
                continue
            seen.add(key)
            before, after = _context(text, c.quote, context_window)
            provisions.append({
                # --- the 13 CSV-equivalent fields ---
                "economy": c.economy,
                "law_name": c.title or "",
                "law_number_ref": c.law_number or "",
                "last_amended": c.last_amended or "",
                "indicator_id": c.indicator_id,
                "article": c.article_path,
                "discovery_tag": getattr(c.discovery_tag, "value", str(c.discovery_tag)),
                "location_reference": c.page_or_dom_anchor or "",
                "verbatim_snippet": c.quote,
                "mapping_rationale": c.mapping_rationale or "",
                "source_url": str(c.source_url),
                "confidence": c.confidence,
                "notes": c.notes or "",
                # --- amendment-currency audit (nested so the top level stays close to
                # the official example; see lexora.cite.amendments) ---
                "amendment_currency": {
                    "review_status": getattr(c.review_status, "value", str(c.review_status)),
                    "status": getattr(c, "currency_status", "") or "",
                    "amended_by": getattr(c, "amended_by", "") or "",
                    "incorporated_to": getattr(c, "amendments_incorporated_to", "") or "",
                    "amendment_text": getattr(c, "amendment_text", "") or "",
                },
                # --- technical metadata the CSV cannot hold ---
                "source_pdf_path": getattr(doc, "bytes_path", "") or "",
                "pdf_is_scanned": bool(getattr(art, "pdf_is_scanned", False)),
                "ocr_quality_cer": ocr_cer,
                "ocr_quality_note": _OCR_NOTE if ocr_cer is not None else "",
                "processing_time_seconds": getattr(art, "processing_time_seconds", None),
                "model_version": doc_model_version,
                "retrieval_method": _retrieval_method(
                    use_dense, getattr(c.review_status, "value", str(c.review_status))
                ),
                "raw_context_before": before,
                "raw_context_after": after,
            })

    payload = {
        "schema": SCHEMA_VERSION,
        "count": len(provisions),
        "provisions": provisions,
    }
    Path(out_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(provisions)
