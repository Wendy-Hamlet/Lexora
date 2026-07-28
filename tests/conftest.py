"""Shared pytest fixtures."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lexora.models.citation import ClaimLabel, EvidenceClaim
from lexora.models.clause import CanonicalSpan
from lexora.models.source import RawDocument, SourceType


@pytest.fixture
def canonical_span() -> CanonicalSpan:
    return CanonicalSpan(
        span_id="sg.pdpa.s26.span0",
        document_id="sg.pdpa",
        page_number=42,
        char_start=14820,
        char_end=14988,
        text="An organisation shall not transfer any personal data to a country or territory outside Singapore except in accordance with requirements prescribed under this Act.",
        ocr_confidence=0.97,
    )


@pytest.fixture
def evidence_claim() -> EvidenceClaim:
    return EvidenceClaim(
        indicator_id="6.1",
        clause_id="sg.pdpa.s26",
        quote_span_id="sg.pdpa.s26.span0",
        label=ClaimLabel.match,
        confidence=0.92,
    )


@pytest.fixture
def raw_document() -> RawDocument:
    return RawDocument(
        document_id="sg.pdpa",
        source_url="https://sso.agc.gov.sg/Act/PDPA2012",
        retrieval_timestamp=datetime(2026, 6, 12, 8, 14, 23, tzinfo=timezone.utc),
        http_status=200,
        sha256="sha256:7f3a000000000000000000000000000000000000000000000000000000000000",
        content_type="text/html",
        bytes_path="data/raw/sg/pdpa.html",
        portal_name="Singapore Statutes Online",
        jurisdiction="SG",
        source_type=SourceType.primary,
        title="Personal Data Protection Act 2012",
    )


@pytest.fixture(autouse=True)
def _isolated_discovery_state():
    """Discovery memoises answered pages and tallies portal outcomes per RUN.

    Both are module-level by design (the sweep and the three follow-on passes have to
    share them), so without this the state leaks between tests and a test that counts
    fetches silently measures the previous test's cache.
    """
    from lexora.collect.discovery import reset_acquisition_log, reset_page_memo

    reset_page_memo()
    reset_acquisition_log()
    yield
    reset_page_memo()
    reset_acquisition_log()
