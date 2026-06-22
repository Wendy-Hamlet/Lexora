"""WS-2 JSON submission sidecar."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from lexora.export.json_exporter import to_submission_json
from lexora.models.citation import Citation, DiscoveryTag, ReviewStatus


def _citation(indicator: str, clause_id: str, quote: str, **kw) -> Citation:
    base = dict(
        economy="Malaysia",
        title="Personal Data Protection Act 2010",
        law_number="Act 709",
        last_amended="2024",
        indicator_id=indicator,
        article_path="s.129",
        discovery_tag=DiscoveryTag.new,
        page_or_dom_anchor="p.51",
        quote=quote,
        mapping_rationale="Transfer outside Malaysia conditional on safeguards.",
        source_url="https://example.gov.my/act709.pdf",
        confidence=0.81,
        notes="",
        clause_id=clause_id,
        retrieval_timestamp=datetime.now(timezone.utc),
        jurisdiction="MY",
        legal_form="statute",
        document_hash="sha256:deadbeef",
        char_start=0,
        char_end=len(quote),
        review_status=ReviewStatus.verified,
    )
    base.update(kw)
    return Citation(**base)


@dataclass
class _Doc:
    document_id: str = "my:abc"
    bytes_path: str = r"data/raw/my/abc.pdf"
    source_url: str = "https://example.gov.my/act709.pdf"


@dataclass
class _Artifacts:
    document: _Doc
    citations: list
    document_text: str = ""
    pdf_is_scanned: bool = False
    ocr_quality_cer: float | None = None
    ocr_engine: str = ""
    processing_time_seconds: float | None = None
    clauses: list = field(default_factory=list)


def test_flat_one_object_per_provision(tmp_path):
    quote = "A data user shall not transfer any personal data outside Malaysia"
    text = f"PART X\n{quote} unless the conditions are met. Further provisions follow."
    art = _Artifacts(
        document=_Doc(),
        citations=[_citation("P6-I4", "c1", quote)],
        document_text=text,
        pdf_is_scanned=True,
        ocr_quality_cer=0.92,
        ocr_engine="rapidocr",
        processing_time_seconds=4.2,
    )
    out = tmp_path / "sub.json"
    n = to_submission_json([art], out, model_version="llm:gpt-x", use_dense=False)
    assert n == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["count"] == 1
    p = payload["provisions"][0]
    # 13 CSV-equivalent fields present
    assert p["indicator_id"] == "P6-I4"
    assert p["law_name"] == "Personal Data Protection Act 2010"
    assert p["verbatim_snippet"] == quote
    # technical metadata
    assert p["source_pdf_path"].endswith("abc.pdf")
    assert p["pdf_is_scanned"] is True
    assert p["ocr_quality_cer"] == 0.92
    assert "confidence" in p["ocr_quality_note"].lower()  # annotated, not silent CER
    assert p["processing_time_seconds"] == 4.2
    assert p["model_version"] == "llm:gpt-x; ocr:rapidocr"
    assert p["retrieval_method"] == "bm25"  # verified is the default status, not an LLM signal
    # raw context located by exact match around the quote
    assert p["raw_context_before"].endswith("PART X\n")
    assert p["raw_context_after"].startswith(" unless")


def test_dedup_across_documents(tmp_path):
    q = "consent must be obtained"
    c = _citation("P7-I1", "c1", q)
    a1 = _Artifacts(document=_Doc(), citations=[c], document_text=q)
    a2 = _Artifacts(document=_Doc(), citations=[c], document_text=q)  # same (indicator, clause)
    out = tmp_path / "sub.json"
    n = to_submission_json([a1, a2], out)
    assert n == 1  # de-duplicated by (indicator_id, clause_id)


def test_no_ocr_leaves_cer_null_and_unannotated(tmp_path):
    q = "the controller shall designate an officer"
    art = _Artifacts(document=_Doc(), citations=[_citation("P7-I4", "c1", q)], document_text=q)
    out = tmp_path / "sub.json"
    to_submission_json([art], out)
    p = json.loads(out.read_text(encoding="utf-8"))["provisions"][0]
    assert p["ocr_quality_cer"] is None
    assert p["pdf_is_scanned"] is False
    assert p["ocr_quality_note"] == ""
    assert p["retrieval_method"] == "bm25"


def test_quote_not_found_yields_empty_context(tmp_path):
    art = _Artifacts(
        document=_Doc(),
        citations=[_citation("P6-I4", "c1", "verbatim text not in the body")],
        document_text="completely different document text",
    )
    out = tmp_path / "sub.json"
    to_submission_json([art], out)
    p = json.loads(out.read_text(encoding="utf-8"))["provisions"][0]
    assert p["raw_context_before"] == ""
    assert p["raw_context_after"] == ""
