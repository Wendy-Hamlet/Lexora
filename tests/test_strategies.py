"""Per-portal strategy tests (offline via httpx.MockTransport)."""
from __future__ import annotations

import os

import httpx
import pytest

from lexora.collect.discovery import DiscoveryResult
from lexora.collect.strategies import (
    au_legislation_api,
    au_resolve_fulltext,
    my_legislation_api,
    resolver_for,
    sg_resolve_fulltext,
    strategy_for,
)
from lexora.models.source import FetchMethod, PortalSpec, SourceType

_LIVE = pytest.mark.skipif(
    not os.environ.get("LEXORA_LIVE"), reason="set LEXORA_LIVE=1 for network tests"
)

AU_PORTAL = PortalSpec(
    name="Federal Register of Legislation",
    url="https://www.legislation.gov.au/",
    source_type=SourceType.primary,
    fetch_method=FetchMethod.api,
    search_query="privacy act 1988",
)

# A trimmed OData response: the principal Privacy Act 1988 plus an amendment.
AU_JSON = {
    "@odata.count": 2,
    "value": [
        {"id": "C2004A03712", "name": "Privacy Act 1988", "collection": "Act",
         "isPrincipal": True, "isInForce": True, "status": "InForce"},
        {"id": "C2099A00001", "name": "Privacy Amendment Act 1990", "collection": "Act",
         "isPrincipal": False, "isInForce": False, "status": "NotInForce"},
    ],
}


def test_strategy_for_matches_au_host():
    assert strategy_for(AU_PORTAL) is au_legislation_api


def test_strategy_for_returns_none_for_unknown_host():
    p = PortalSpec(name="x", url="https://sso.agc.gov.sg/", source_type=SourceType.primary)
    assert strategy_for(p) is None


def test_au_api_ranks_principal_inforce_first():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "api.prod.legislation.gov.au" in str(request.url)
        assert "privacy%20act%201988" in str(request.url).lower()
        return httpx.Response(200, json=AU_JSON)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    results = au_legislation_api(
        AU_PORTAL, query="privacy act 1988", limit=5, client=client,
        known_instruments=["Privacy Act 1988"],
    )
    client.close()
    assert results
    top = results[0]
    assert "C2004A03712" in top.url        # the principal Privacy Act 1988
    assert top.url.endswith("/latest")     # in force -> /latest
    assert top.via == "api"
    assert top.discovery_tag == "KNOWN"
    assert top.matched_instrument == "Privacy Act 1988"
    # the not-in-force amendment routes to /asmade
    amend = next(r for r in results if "C2099A00001" in r.url)
    assert amend.url.endswith("/asmade")


def test_au_api_handles_non_200():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    results = au_legislation_api(AU_PORTAL, query="privacy act", client=client)
    client.close()
    assert results == []  # degrades to empty -> discover() falls back to generic


@pytest.mark.live
@_LIVE
def test_au_api_live_finds_privacy_act():
    results = au_legislation_api(
        AU_PORTAL, query="privacy act 1988", limit=5,
        known_instruments=["Privacy Act 1988"],
    )
    assert any("C2004A03712" in r.url for r in results)


MY_PORTAL = PortalSpec(
    name="Laws of Malaysia",
    url="https://lom.agc.gov.my/",
    source_type=SourceType.primary,
    fetch_method=FetchMethod.api,
    search_query="personal data protection",
)

# A trimmed Fess/Solr proxy response (real field names: titleBI, actNo,
# legislationStatus "IBU" = principal; DOC2DOWNLOADBI carries the PDF link).
MY_JSON = {
    "response": {
        "numFound": 2,
        "docs": [
            {"_os_url": "https://lom.agc.gov.my/act-detail.php?act=709&lang=BI",
             "titleBI": "PERSONAL DATA PROTECTION ACT 2010", "legislationStatus": "IBU",
             "actNo": "709",
             "DOC2DOWNLOADBI": '<a href="https://lom.agc.gov.my/ilims/upload/Act709.pdf">PDF</a>'},
            {"_os_url": "https://lom.agc.gov.my/act-detail.php?act=A1727&lang=BI",
             "titleBI": "PERSONAL DATA PROTECTION (AMENDMENT) ACT 2024",
             "legislationStatus": "PINDAAN", "actNo": "A1727"},
        ],
    }
}


def test_strategy_for_matches_my_host():
    assert strategy_for(MY_PORTAL) is my_legislation_api


def test_my_api_finds_act_709_first():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "fess-proxy.php" in str(request.url)
        assert "q=personal+data+protection" in str(request.url)
        return httpx.Response(200, json=MY_JSON)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    results = my_legislation_api(
        MY_PORTAL, query="personal data protection", limit=5, client=client,
        known_instruments=["Personal Data Protection Act 2010"],
    )
    client.close()
    assert results
    # principal Act 709 ranks first (Solr relevance + IBU + known-name match)
    assert "act=709" in results[0].url
    assert results[0].via == "api"
    assert results[0].discovery_tag == "KNOWN"
    assert results[0].fulltext_url == "https://lom.agc.gov.my/ilims/upload/Act709.pdf"
    # the amendment collapses to a separate instrument, not a 709 duplicate
    assert sum("act=709" in r.url for r in results) == 1


def test_my_api_tags_known_by_act_id_on_filename_titles():
    # Document records carry filename titles that defeat fuzzy name matching;
    # the Act-number identity map must still tag them KNOWN.
    doc_json = {"response": {"docs": [
        {"_os_url": "https://lom.agc.gov.my/ilims/.../Act 709 ori.pdf",
         "titleBI": "JW515839 Act 709.indd", "legislationStatus": "IBU"},
    ]}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=doc_json)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    results = my_legislation_api(
        MY_PORTAL, query="transfer of personal data out of malaysia", limit=5,
        client=client,
        known_instruments=["Personal Data Protection Act 2010"],
        known_instrument_ids={"709": "Personal Data Protection Act 2010"},
    )
    client.close()
    assert results
    top = results[0]
    assert top.discovery_tag == "KNOWN"  # tagged by Act-number identity, not title
    assert top.matched_instrument == "Personal Data Protection Act 2010"


def test_my_api_handles_non_200():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(502)))
    results = my_legislation_api(MY_PORTAL, query="x", client=client)
    client.close()
    assert results == []


def test_my_api_retries_on_dropped_tls(monkeypatch):
    # MY's Fess proxy intermittently drops the TLS connection; the first GET
    # raises a transport error and the retry succeeds (P-5 robustness).
    monkeypatch.setattr("time.sleep", lambda *_: None)  # no real backoff wait
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("SSL handshake dropped")
        return httpx.Response(200, json=MY_JSON)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    results = my_legislation_api(
        MY_PORTAL, query="personal data protection", limit=5, client=client,
        known_instruments=["Personal Data Protection Act 2010"],
    )
    client.close()
    assert calls["n"] == 2          # retried once after the dropped TLS
    assert any("act=709" in r.url for r in results)


def test_my_api_gives_up_gracefully_after_retries(monkeypatch):
    # If every attempt drops, _get_with_retry exhausts and my_legislation_api
    # returns [] (so discover() falls back) rather than crashing the run.
    monkeypatch.setattr("time.sleep", lambda *_: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("down")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    results = my_legislation_api(MY_PORTAL, query="x", client=client)
    client.close()
    assert calls["n"] == 3          # initial + 2 retries
    assert results == []


def _disc(url, **kw):
    return DiscoveryResult(url=url, title="x", source_type=SourceType.primary,
                           score=1.0, via="api", is_pdf_link=False, **kw)


def test_sg_resolver_builds_pdf_url():
    r = _disc("https://sso.agc.gov.sg/Act/PDPA2012?ViewType=Advance&Phrase=x")
    assert sg_resolve_fulltext(r) == "https://sso.agc.gov.sg/Act/PDPA2012?ViewType=Pdf"


def test_sg_resolver_skips_non_act_urls():
    assert sg_resolve_fulltext(_disc("https://sso.agc.gov.sg/Browse/Act/Current")) \
        == "https://sso.agc.gov.sg/Browse/Act/Current?ViewType=Pdf"  # has /Act/
    assert sg_resolve_fulltext(_disc("https://sso.agc.gov.sg/")) is None


def test_resolver_for_matches_hosts():
    assert resolver_for("https://sso.agc.gov.sg/Act/PDPA2012") is sg_resolve_fulltext
    assert resolver_for("https://www.legislation.gov.au/C2004A03712/latest") is au_resolve_fulltext
    assert resolver_for("https://lom.agc.gov.my/act-detail.php?act=709") is None


@pytest.mark.live
@_LIVE
def test_my_api_live_finds_pdpa():
    results = my_legislation_api(
        MY_PORTAL, query="personal data protection", limit=10,
        known_instruments=["Personal Data Protection Act 2010"],
    )
    assert any("act=709" in r.url for r in results)
