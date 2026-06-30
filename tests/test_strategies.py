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


# --- AU amendment reverse-lookup (versions/affects graph) -------------------
def test_frl_id_from_url():
    from lexora.collect.strategies import frl_id_from_url

    assert frl_id_from_url("https://www.legislation.gov.au/C2004A05145/latest/text") == "C2004A05145"
    assert frl_id_from_url("https://www.legislation.gov.au/F2009L02014/asmade") == "F2009L02014"
    assert frl_id_from_url("https://example.com/no-id") == ""


# A trimmed versions response: the affecting Act lives under `affectedByTitle`
# (NOT `amendedByTitle`, which is always null), keyed off `affect == "Amend"`.
AU_VERSIONS_JSON = {
    "id": "C2004A05145",
    "versions": [
        {"compilationNumber": "1", "reasons": [
            {"affect": "Amend", "amendedByTitle": None, "affectedByTitle": {
                "titleId": "C2023A00017",
                "name": "Telecommunications Legislation Amendment (Information "
                        "Disclosure, National Interest and Other Measures) Act 2023",
                "provisions": "sch 1 (items 12-14)", "year": 2023, "number": 17}},
        ]},
        {"compilationNumber": "2", "reasons": [
            # A Repeal reason must be ignored (not an amendment).
            {"affect": "Repeal", "affectedByTitle": {
                "titleId": "C2099A09999", "name": "Some Repealing Act 2099",
                "provisions": "sch 9", "year": 2099, "number": 1}},
            # An older amendment — comes out AFTER the 2023 one (newest-first sort).
            {"affect": "Amend", "affectedByTitle": {
                "titleId": "C2004A05315", "name": "Telecommunications Legislation "
                "Amendment Act 1997", "provisions": "sch 1", "year": 1997, "number": 99}},
        ]},
    ],
}


def test_au_amendment_acts_reverse_lookup():
    from lexora.collect.strategies import au_amendment_acts

    def handler(request: httpx.Request) -> httpx.Response:
        assert "expand=versions" in request.url.query.decode()
        assert "C2004A05145" in str(request.url)
        return httpx.Response(200, json=AU_VERSIONS_JSON)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    results = au_amendment_acts("C2004A05145", client=client)
    client.close()
    titles = [r.title for r in results]
    # Both amendments surfaced; the Repeal reason filtered out.
    assert any("Information Disclosure" in t for t in titles)
    assert any(t == "Telecommunications Legislation Amendment Act 1997" for t in titles)
    assert not any("Repealing" in t for t in titles)
    # Newest first, canonical /latest URL by id, register law number.
    assert results[0].url == "https://www.legislation.gov.au/C2023A00017/latest"
    assert results[0].law_number == "No. 17 of 2023"
    assert results[0].discovery_tag == "NEW"


def test_au_amendment_acts_empty_id_and_non_200():
    from lexora.collect.strategies import au_amendment_acts

    assert au_amendment_acts("") == []
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    assert au_amendment_acts("C2004A05145", client=client) == []
    client.close()


# --- SG SSO inline-annotation amendment reverse-lookup ---
def test_sg_amendment_acts_builds_acts_supp_urls_newest_first():
    from lexora.collect.strategies import sg_amendment_acts

    # A consolidated PDPA 2012 text with SSO's inline amendment annotations (and its
    # own original enactment "Act 26 of 2012", plus a duplicate).
    text = ("... Personal Data Protection Act 2012 (Act 26 of 2012) ... amended by "
            "Act 40 of 2019 wef 01/02/2020 ... Act 40 of 2020 wef 01/02/2021 ... "
            "Act 25 of 2021 ... Act 40 of 2020 again ...")
    out = sg_amendment_acts(text)
    urls = [r.url for r in out]
    # Newest first, de-duplicated, Acts-Supplement URL form.
    assert urls[0] == "https://sso.agc.gov.sg/Acts-Supp/25-2021/"
    assert "https://sso.agc.gov.sg/Acts-Supp/40-2020/" in urls  # the gold amendment
    assert urls.count("https://sso.agc.gov.sg/Acts-Supp/40-2020/") == 1  # de-duped
    assert [r.title for r in out][:2] == ["Act 25 of 2021", "Act 40 of 2020"]
    for r in out:
        assert r.via == "sso-history" and r.source_type is SourceType.primary


def test_sg_amendment_acts_limit_and_empty():
    from lexora.collect.strategies import sg_amendment_acts

    assert sg_amendment_acts("") == []
    assert sg_amendment_acts("no citations here") == []
    text = " ".join(f"Act {n} of 20{n:02d}" for n in range(1, 20))
    assert len(sg_amendment_acts(text, limit=4)) == 4


# --- AU child-regulations discovery (Path 3: name stem + authorisedBy guard) ---
def test_act_title_core():
    from lexora.collect.strategies import _act_title_core

    assert _act_title_core("Telecommunications Act 1997") == "Telecommunications"
    assert _act_title_core("Personal Data Protection Act 2010") == "Personal Data Protection"
    assert _act_title_core("Something With No Stem") == "Something With No Stem"


# Name search for "<stem> regulations" returns the principal regulation made under
# this Act AND a same-named one made under a DIFFERENT Act; the authorisedBy guard
# must keep only the former.
AU_REG_SEARCH_JSON = {
    "value": [
        {"id": "F2021L00289", "name": "Telecommunications Regulations 2021",
         "collection": "LegislativeInstrument", "subCollection": "Regulations",
         "isPrincipal": True, "isInForce": True, "number": None, "year": 2021},
        {"id": "FWRONGACT01", "name": "Telecommunications Regulations (Other) 2019",
         "collection": "LegislativeInstrument", "subCollection": "Regulations",
         "isPrincipal": True, "isInForce": True, "number": None, "year": 2019},
        # noise: a Determination (not Regulations) — filtered by subCollection.
        {"id": "F2010DET001", "name": "Telecommunications Regulations Determination",
         "collection": "LegislativeInstrument", "subCollection": "Determinations",
         "isPrincipal": True, "isInForce": True},
    ],
}


def _au_reg_handler(act_id: str = "C2004A05145"):
    def handler(request: httpx.Request) -> httpx.Response:
        q = request.url.query.decode()
        if "expand=authorisedBy" in q:
            rid = str(request.url).split("('")[1].split("')")[0]
            # only F2021L00289 is authorised by act_id; the other reg is under a different Act
            affecting = act_id if rid == "F2021L00289" else "CDIFFERENT99"
            return httpx.Response(200, json={"id": rid, "authorisedBy": [
                {"affectingTitleId": affecting, "affectedTitleId": rid}]})
        return httpx.Response(200, json=AU_REG_SEARCH_JSON)

    return handler


def test_au_child_regulations_name_and_authorised_by_guard():
    from lexora.collect.strategies import au_child_regulations

    client = httpx.Client(transport=httpx.MockTransport(_au_reg_handler()))
    results = au_child_regulations("C2004A05145", "Telecommunications Act 1997", client=client)
    client.close()
    # Exactly the one regulation actually made under this Act survives.
    assert [r.url for r in results] == ["https://www.legislation.gov.au/F2021L00289/latest"]
    assert results[0].title == "Telecommunications Regulations 2021"
    assert results[0].discovery_tag == "NEW"


def test_au_child_regulations_skips_short_core_and_non_200():
    from lexora.collect.strategies import au_child_regulations

    assert au_child_regulations("C1", "Ab Act 1900") == []  # core "Ab" < 3 chars
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    assert au_child_regulations("C2004A05145", "Telecommunications Act 1997", client=client) == []
    client.close()


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


def test_au_resolve_prefers_epub_html_no_browser(monkeypatch):
    """The AU resolver harvests the EPUB full-text URL from the cheap ``/text``
    shell with a plain GET, never touching the browser. The shell is a TOC whose
    entries deep-link into the EPUB; the resolver returns that static XHTML URL so
    the structure parser can extract verbatim clauses (the shell yields none)."""
    epub = ("https://www.legislation.gov.au/C2004A03712/2026-06-04/2026-06-04/"
            "text/original/epub/OEBPS/document_1/document_1.html")
    shell_html = (
        '<html><body><a href="' + epub + '#_Toc1">1 Short title</a>'
        '<a href="' + epub + '#_Toc2">6 Interpretation</a></body></html>'
    )
    requested: list[str] = []

    def fake_get(self, url, *a, **k):  # noqa: ANN001
        requested.append(url)
        return httpx.Response(200, text=shell_html)

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    # A browser call would raise (no display in CI) — proves we never reach it.
    import lexora.collect.browser as browser
    monkeypatch.setattr(browser, "render", lambda *a, **k: pytest.fail("browser used"))

    result = DiscoveryResult(
        url="https://www.legislation.gov.au/C2004A03712/latest",
        title="Privacy Act 1988", source_type=SourceType.primary, score=1.0,
        via="api", is_pdf_link=False,
    )
    assert au_resolve_fulltext(result) == epub
    assert requested == ["https://www.legislation.gov.au/C2004A03712/latest/text"]


def test_au_resolve_falls_back_to_pdf_when_no_epub(monkeypatch):
    """When the shell carries no EPUB link (e.g. an image-only as-made
    instrument), the resolver falls back to the legacy browser → dated-PDF path."""
    pdf = "/C2004A03712/2026-06-04/2026-06-04/text/original/pdf"
    monkeypatch.setattr(
        httpx.Client, "get",
        lambda self, url, *a, **k: httpx.Response(200, text="<html>no epub here</html>"),
    )
    import lexora.collect.browser as browser
    monkeypatch.setattr(
        browser, "render",
        lambda *a, **k: type("R", (), {"html": f'<a href="{pdf}">PDF</a>'})(),
    )
    result = DiscoveryResult(
        url="https://www.legislation.gov.au/C2004A03712/latest",
        title="Privacy Act 1988", source_type=SourceType.primary, score=1.0,
        via="api", is_pdf_link=False,
    )
    assert au_resolve_fulltext(result) == "https://www.legislation.gov.au" + pdf


@pytest.mark.live
@_LIVE
def test_my_api_live_finds_pdpa():
    results = my_legislation_api(
        MY_PORTAL, query="personal data protection", limit=10,
        known_instruments=["Personal Data Protection Act 2010"],
    )
    assert any("act=709" in r.url for r in results)
