"""Crawler + live-fetch pipeline tests.

The offline tests use httpx.MockTransport so they are deterministic and need no
network. A single `live`-marked test exercises a real government PDF and is
skipped unless LEXORA_LIVE=1.
"""
from __future__ import annotations

import os
from pathlib import Path

import fitz  # PyMuPDF
import httpx
import pytest

from lexora.collect.crawler import fetch
from lexora.collect.profile_loader import load_profile
from lexora.indicators import load_indicators
from lexora.models.source import SourceType
from lexora.pipeline import run_pipeline_from_url

REPO = Path(__file__).resolve().parent.parent

HTML_ACT = (
    "<html><head><title>PDPA</title><style>x{}</style></head><body>"
    "<h1 id='t'>Personal Data Protection Act 2012</h1>"
    "<p id='s13'>13. An organisation must not collect, use or disclose personal "
    "data about an individual unless the individual gives consent under this Act.</p>"
    "<p id='s26'>26. An organisation must not transfer any personal data to a "
    "country or territory outside Singapore except in accordance with "
    "requirements prescribed under this Act.</p>"
    "<script>track()</script></body></html>"
)


def _pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(72, 72, 540, 770)
    page.insert_textbox(
        rect,
        "Personal Data Protection Act 2012\n\n"
        "26. An organisation must not transfer any personal data to a country or "
        "territory outside Singapore except in accordance with requirements "
        "prescribed under this Act.\n",
        fontsize=11,
        fontname="helv",
    )
    data = doc.tobytes()
    doc.close()
    return data


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/pdf"):
        return httpx.Response(200, content=_pdf_bytes(), headers={"content-type": "application/pdf"})
    if path.endswith("/blocked"):
        return httpx.Response(403, content=b"<html>blocked</html>", headers={"content-type": "text/html"})
    return httpx.Response(200, content=HTML_ACT.encode(), headers={"content-type": "text/html; charset=utf-8"})


@pytest.fixture
def mock_client() -> httpx.Client:
    client = httpx.Client(transport=httpx.MockTransport(_handler))
    yield client
    client.close()


@pytest.fixture
def sg_profile():
    return load_profile(REPO / "configs" / "jurisdictions" / "sg.yaml")


@pytest.fixture
def indicators():
    return load_indicators(REPO / "configs" / "rdtii_indicators.yaml")


def _fetch(url, client, tmp_path, **kw):
    return fetch(
        url,
        jurisdiction="SG",
        portal_name="mock",
        source_type=SourceType.primary,
        dest_dir=tmp_path,
        client=client,
        **kw,
    )


def test_fetch_routes_html(mock_client, tmp_path):
    res = _fetch("https://portal.test/act", mock_client, tmp_path)
    assert res.document.http_status == 200
    assert res.is_html() and not res.is_pdf()


def test_fetch_routes_pdf(mock_client, tmp_path):
    res = _fetch("https://portal.test/pdf", mock_client, tmp_path)
    assert res.is_pdf()
    assert res.document.bytes_path.endswith(".pdf")
    assert res.document.sha256.startswith("sha256:")


def test_fetch_captures_non_2xx_without_raising(mock_client, tmp_path):
    res = _fetch("https://portal.test/blocked", mock_client, tmp_path)
    assert res.document.http_status == 403  # captured, not raised


def test_pipeline_from_url_html_verbatim(mock_client, sg_profile, indicators, tmp_path):
    artifacts = run_pipeline_from_url(
        url="https://portal.test/act",
        profile=sg_profile,
        indicators=indicators,
        dest_dir=tmp_path,
        client=mock_client,
        min_score=-10.0,
    )
    assert artifacts.blocks, "HTML path should populate blocks"
    assert not artifacts.pages
    assert artifacts.citations
    for c in artifacts.citations:
        clause = next(cl for cl in artifacts.clauses if cl.clause_id == c.clause_id)
        assert c.quote == clause.span.text  # verbatim
        # HTML location reference is a DOM anchor, not a page number
        assert c.page_or_dom_anchor and not c.page_or_dom_anchor.isdigit()
    assert any("transfer any personal data" in c.quote for c in artifacts.citations)


def test_pipeline_from_url_pdf(mock_client, sg_profile, indicators, tmp_path):
    artifacts = run_pipeline_from_url(
        url="https://portal.test/pdf",
        profile=sg_profile,
        indicators=indicators,
        dest_dir=tmp_path,
        client=mock_client,
        min_score=-10.0,
    )
    assert artifacts.pages, "PDF path should populate pages"
    assert artifacts.citations
    assert all(c.quote == next(cl.span.text for cl in artifacts.clauses if cl.clause_id == c.clause_id)
               for c in artifacts.citations)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("LEXORA_LIVE"), reason="set LEXORA_LIVE=1 for network tests")
def test_live_fetch_real_government_pdf(tmp_path):
    # NZ legislation serves static PDFs over plain HTTP (a good live smoke target).
    res = fetch(
        "https://www.legislation.govt.nz/act/public/2020/0031/latest/096be8ed81f1a3e3.pdf",
        jurisdiction="NZ",
        portal_name="NZ Legislation",
        source_type=SourceType.primary,
        dest_dir=tmp_path,
    )
    assert res.document.http_status == 200
    assert res.is_pdf()
    assert len(res.body) > 10_000


def test_replayed_fetch_keeps_the_recorded_retrieval_time():
    """The audit trail must date a document to when the portal served it, not to now.

    ``retrieval_timestamp`` is exported beside every citation. Replaying a recording
    re-reads bytes the portal handed over earlier, possibly days earlier for a demo, so
    stamping the run's own clock onto them would quietly overstate how fresh the source
    is -- the one field a reviewer would use to check exactly that.
    """
    from datetime import datetime, timezone

    recorded = datetime(2026, 7, 20, 9, 15, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"<html>Act</html>",
            headers={"content-type": "text/html",
                     "x-lexora-recorded-at": recorded.isoformat()},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch("https://sso.agc.gov.sg/Act/PDPA2012", jurisdiction="SG",
                       portal_name="SSO", source_type=SourceType.primary, client=client)

    assert result.document.retrieval_timestamp == recorded
