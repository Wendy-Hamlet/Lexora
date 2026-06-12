"""Discovery-layer tests.

The harvester is a pure function and fully offline. `discover()` is exercised
with httpx.MockTransport. A browser-marked test renders a real anti-bot portal
and is skipped unless a Chromium build is available.
"""
from __future__ import annotations

import os

import httpx
import pytest

from lexora.collect.discovery import (
    DiscoveryResult,
    _fetch_page,
    discover,
    discover_for_indicators,
    harvest_candidates,
    resolve_fulltext,
)
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import FetchMethod, PortalSpec, SourceType

_LIVE = pytest.mark.skipif(
    not os.environ.get("LEXORA_LIVE"), reason="set LEXORA_LIVE=1 for network tests"
)

# A results page mixing site chrome, a relevant Act (page + PDF) and noise.
RESULTS_HTML = """
<html><body>
  <nav><a href="/login">Login</a> <a href="/contact">Contact us</a></nav>
  <a href="https://twitter.com/agc">Follow us</a>
  <ul>
    <li><a href="/acts/PDPA2012">Personal Data Protection Act 2012</a></li>
    <li><a href="/acts/PDPA2012/download.pdf">Download PDPA 2012 (PDF)</a></li>
    <li><a href="/news/2024/office-party">Annual office party photos</a></li>
    <li><a href="#top">Back to top</a></li>
  </ul>
</body></html>
"""


def test_harvest_ranks_relevant_above_noise():
    results = harvest_candidates(
        RESULTS_HTML,
        base_url="https://sso.example.gov/search",
        query="personal data protection act",
        source_type=SourceType.primary,
    )
    urls = [r.url for r in results]
    # The PDPA links rank above the office-party news item.
    assert any("PDPA2012" in u for u in urls)
    pdpa_idx = next(i for i, u in enumerate(urls) if "PDPA2012" in u)
    party_idx = next((i for i, u in enumerate(urls) if "office-party" in u), len(urls))
    assert pdpa_idx < party_idx


def test_harvest_drops_chrome_and_fragments():
    results = harvest_candidates(
        RESULTS_HTML,
        base_url="https://sso.example.gov/search",
        query="personal data",
        source_type=SourceType.primary,
    )
    urls = " ".join(r.url for r in results)
    assert "login" not in urls
    assert "contact" not in urls
    assert "twitter.com" not in urls
    assert "#top" not in urls


def test_harvest_pdf_bonus_and_absolute_urls():
    results = harvest_candidates(
        RESULTS_HTML,
        base_url="https://sso.example.gov/search",
        query="personal data protection act",
        source_type=SourceType.primary,
    )
    pdf = next(r for r in results if r.is_pdf_link)
    assert pdf.url.startswith("https://sso.example.gov/")  # urljoin made it absolute
    assert pdf.source_type is SourceType.primary


def test_harvest_dedupes_repeated_urls():
    html = (
        '<a href="/acts/X2020">X Act 2020</a>'
        '<a href="/acts/X2020">X Act 2020 (again)</a>'
    )
    results = harvest_candidates(
        html, base_url="https://e.gov/", query="x act", source_type=SourceType.primary
    )
    assert len([r for r in results if r.url.endswith("/acts/X2020")]) == 1


def test_discover_uses_search_template_and_ranks(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, content=RESULTS_HTML.encode(), headers={"content-type": "text/html"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    portal = PortalSpec(
        name="SSO",
        url="https://sso.example.gov/",
        source_type=SourceType.primary,
        fetch_method=FetchMethod.http,
        search_query="personal data protection act",
        search_url_template="https://sso.example.gov/search?q={query}",
    )
    results = discover(portal, client=client, limit=5)
    client.close()
    assert "q=personal%20data%20protection%20act" in captured["url"]  # template + urlencode
    assert results and results[0].via == "http"
    assert any("PDPA2012" in r.url for r in results)


def test_discover_escalates_to_browser_on_403(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, content=b"blocked", headers={"content-type": "text/html"})

    rendered_html = RESULTS_HTML

    class _Rendered:
        html = rendered_html
        status = 200
        final_url = "https://sso.example.gov/"
        content_type = "text/html"

    # Stub the browser so the test needs no Chromium.
    import lexora.collect.browser as browser_mod

    monkeypatch.setattr(browser_mod, "render", lambda *a, **k: _Rendered())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    portal = PortalSpec(
        name="SSO",
        url="https://sso.example.gov/",
        source_type=SourceType.primary,
        search_query="personal data protection act",
    )
    results = discover(portal, client=client, limit=5)
    client.close()
    assert results and all(r.via == "browser" for r in results)
    assert any("PDPA2012" in r.url for r in results)


class _CapturingSession:
    """A fake BrowserSession that records the render kwargs (no Chromium)."""

    def __init__(self):
        self.render_kwargs = None

    def render(self, url, **kwargs):
        self.render_kwargs = kwargs

        class _R:
            html = RESULTS_HTML
            status = 200
            final_url = url
            content_type = "text/html"

        return _R()


def test_fetch_page_forwards_render_wait_until():
    # G-5: SG SSO long-polls, so the render must avoid `networkidle`. _fetch_page
    # threads the chosen wait_until to the browser session.
    session = _CapturingSession()
    html, via = _fetch_page(
        "https://sso.agc.gov.sg/Search/Content?Phrase=x",
        force_browser=True, client=None, timeout=5.0, user_agent="t",
        browser_session=session, render_wait_until="load",
    )
    assert via == "browser"
    assert session.render_kwargs["wait_until"] == "load"


def test_fetch_page_render_wait_until_defaults_to_networkidle():
    session = _CapturingSession()
    _fetch_page(
        "https://example.gov/", force_browser=True, client=None, timeout=5.0,
        user_agent="t", browser_session=session,
    )
    assert session.render_kwargs["wait_until"] == "networkidle"


def test_dedupes_instrument_variants_to_one():
    # One Act surfaced via many provision deep-links (SG SSO pattern).
    html = (
        '<a href="/Act/PDPA2012?ViewType=Advance">Personal Data Protection Act 2012</a>'
        '<a href="/Act/PDPA2012?ProvIds=P12-">Personal Data Protection Act 2012</a>'
        '<a href="/Act/PDPA2012?ProvIds=pr5-">Personal Data Protection Act 2012</a>'
    )
    results = harvest_candidates(
        html, base_url="https://sso.gov/", query="personal data protection act",
        source_type=SourceType.primary,
    )
    pdpa = [r for r in results if "PDPA2012" in r.url]
    assert len(pdpa) == 1                     # collapsed to one instrument
    assert pdpa[0].n_variants == 3            # but remembers it matched 3 links
    assert "ProvIds" not in pdpa[0].url       # representative is the clean Act URL


def test_known_tag_via_fuzzy_match():
    html = '<a href="/Act/PDPA2012">Personal Data Protection Act 2012</a>'
    results = harvest_candidates(
        html, base_url="https://sso.gov/", query="data protection",
        source_type=SourceType.primary,
        known_instruments=["Personal Data Protection Act 2012", "Cybersecurity Act 2018"],
    )
    assert results[0].discovery_tag == "KNOWN"
    assert results[0].matched_instrument == "Personal Data Protection Act 2012"


def test_new_tag_for_unlisted_instrument():
    # An instrument-like, query-relevant link clearly distinct from the known
    # instrument (which is about cybersecurity, not data protection).
    html = '<a href="/Act/DPA2025">Data Protection Act 2025</a>'
    results = harvest_candidates(
        html, base_url="https://sso.gov/", query="data protection act",
        source_type=SourceType.primary,
        known_instruments=["Cybersecurity Act 2018"],
    )
    assert results[0].discovery_tag == "NEW"
    assert results[0].matched_instrument is None


def test_cross_reference_in_snippet_does_not_spoof_known():
    # SG SSO result cards append "<Act name> Current version as at <date>
    # <provision snippet>". When that snippet quotes ANOTHER statute (here the
    # PDPA), the body text must not make this Act read as KNOWN — identity is the
    # name slot only. Regression: token_set_ratio on the full snippet returned
    # 1.0 against the quoted name, flooding the budget with false-KNOWN hits.
    html = (
        '<li><a href="/Act/FSMA2022">Financial Services and Markets Act 2022 '
        "Current version as at 10 Jun 2026 28M Application of sections 21 and 22 "
        "of Personal Data Protection Act 2012</a></li>"
    )
    results = harvest_candidates(
        html, base_url="https://sso.agc.gov.sg/", query="protection of personal data",
        source_type=SourceType.primary,
        known_instruments=["Personal Data Protection Act 2012", "Cybersecurity Act 2018"],
    )
    top = results[0]
    assert "FSMA2022" in top.url
    assert top.discovery_tag != "KNOWN"  # it is NOT the PDPA, only cites it
    assert top.matched_instrument is None


def test_context_recovers_title_outside_anchor():
    # AU-style result card: title in a heading, anchor text is just "View".
    html = (
        '<div class="search-result">'
        '<h3>Privacy Act 1988</h3>'
        '<a href="/C2004A03712/latest">View</a>'
        '</div>'
    )
    results = harvest_candidates(
        html, base_url="https://legislation.gov.au/", query="privacy act 1988",
        source_type=SourceType.primary,
        known_instruments=["Privacy Act 1988"],
    )
    top = results[0]
    assert "C2004A03712" in top.url
    assert "Privacy Act 1988" in top.title  # recovered from the card heading
    assert top.discovery_tag == "KNOWN"


def test_scores_are_normalized():
    results = harvest_candidates(
        RESULTS_HTML, base_url="https://sso.example.gov/search",
        query="personal data protection act", source_type=SourceType.primary,
    )
    assert results
    assert all(0.0 <= r.score <= 1.0 for r in results)


def _result(url, *, fulltext=None, is_pdf=False):
    return DiscoveryResult(
        url=url, title="Act", source_type=SourceType.primary, score=1.0,
        via="api", is_pdf_link=is_pdf, fulltext_url=fulltext,
    )


def test_resolve_fulltext_prefers_explicit_url():
    r = _result("https://e.gov/act-detail.php?act=709",
                fulltext="https://e.gov/Act709.pdf")
    assert resolve_fulltext(r) == "https://e.gov/Act709.pdf"  # no fetch needed


def test_resolve_fulltext_passes_through_pdf_result():
    r = _result("https://e.gov/act.pdf", is_pdf=True)
    assert resolve_fulltext(r) == "https://e.gov/act.pdf"


def test_resolve_fulltext_finds_pdf_on_instrument_page():
    page = (
        '<html><body>'
        '<a href="/act/709/print.pdf">Download whole Act (PDF)</a>'
        '<a href="/act/709/share">Share</a>'
        '</body></html>'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=page.encode(), headers={"content-type": "text/html"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    r = _result("https://e.gov/act/709")
    out = resolve_fulltext(r, client=client, query="act")
    client.close()
    assert out == "https://e.gov/act/709/print.pdf"


def test_resolve_fulltext_returns_none_when_no_pdf():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html><body><p>no downloads</p></body></html>",
                              headers={"content-type": "text/html"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = resolve_fulltext(_result("https://e.gov/act/709"), client=client)
    client.close()
    assert out is None


# ---- per-indicator multi-instrument discovery (P0) ----

def _ind(rid: str, sid: str, phrases: list[str]) -> RDTIIIndicator:
    return RDTIIIndicator(
        rdtii_id=rid, submission_id=sid, pillar=int(float(rid)),
        name=f"indicator {rid}", description="desc", discovery_queries=phrases,
    )


def test_query_phrases_prefers_discovery_queries_then_keywords_fallback():
    a = _ind("6.1", "P6-I1", ["alpha", "beta", "gamma", "delta"])
    assert a.query_phrases(limit=2) == ["alpha", "beta"]
    # No curated phrases -> fall back to name + keywords.
    b = RDTIIIndicator(rdtii_id="7.3", submission_id="P7-I3", pillar=7,
                       name="Retention", description="d", keywords=["keep", "retain"])
    assert b.query_phrases() == ["Retention", "keep", "retain"]


_MULTI_ACT_HTML = """
<ul>
  <li><a href="/Act/AAA2010">Alpha Data Act 2010</a></li>
  <li><a href="/Act/BBB2018">Beta Privacy Act 2018</a></li>
  <li><a href="/Act/CCC2020">Gamma Retention Act 2020</a></li>
</ul>
"""


def test_discover_for_indicators_unions_dedups_and_budgets():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, content=_MULTI_ACT_HTML.encode(),
                              headers={"content-type": "text/html"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    portal = PortalSpec(
        name="SSO", url="https://sso.example.gov/", source_type=SourceType.primary,
        search_url_template="https://sso.example.gov/search?q={query}",
    )
    inds = [_ind("6.1", "P6-I1", ["alpha"]), _ind("7.3", "P7-I3", ["beta"])]
    # min_score 0.3 keeps only the query-matched act per phrase, so attribution
    # is unambiguous (alpha->Alpha Act, beta->Beta Act).
    results = discover_for_indicators(portal, inds, client=client, budget=2, min_score=0.3)
    client.close()
    assert calls["n"] == 2                       # one fetch per distinct phrase
    assert len(results) == 2                      # capped to budget
    assert results[0].score >= results[1].score   # sorted by score
    by_url = {r.url.rsplit("/", 1)[-1]: r for r in results}
    # each instrument is credited to the indicator whose phrase surfaced it
    assert by_url["AAA2010"].indicator_hits == ["P6-I1"]
    assert by_url["BBB2018"].indicator_hits == ["P7-I3"]


def test_discover_for_indicators_fetches_a_shared_phrase_once():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, content=_MULTI_ACT_HTML.encode(),
                              headers={"content-type": "text/html"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    portal = PortalSpec(
        name="SSO", url="https://sso.example.gov/", source_type=SourceType.primary,
        search_url_template="https://sso.example.gov/search?q={query}",
    )
    # Both indicators share the phrase "alpha" -> deduped to a single fetch.
    inds = [_ind("6.1", "P6-I1", ["alpha"]), _ind("6.2", "P6-I2", ["alpha"])]
    discover_for_indicators(portal, inds, client=client, budget=5)
    client.close()
    assert calls["n"] == 1


def test_discover_for_indicators_name_driven_on_non_full_text_portal():
    # A name-only portal (AU OData): concept phrases are ignored; discovery is
    # driven by the jurisdiction's known instrument NAMES instead.
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        queries.append(str(request.url))
        return httpx.Response(200, content=_MULTI_ACT_HTML.encode(),
                              headers={"content-type": "text/html"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    portal = PortalSpec(
        name="FRL", url="https://e.gov/", source_type=SourceType.primary,
        full_text=False, search_url_template="https://e.gov/search?q={query}",
    )
    inds = [_ind("6.1", "P6-I1", ["alpha concept phrase"])]
    results = discover_for_indicators(
        portal, inds, client=client, budget=5, min_score=0.3,
        known_instruments=["Alpha Data Act 2010"],
    )
    client.close()
    # queried by the known NAME, not the indicator's concept phrase
    assert any("Alpha" in q for q in queries)
    assert not any("concept" in q for q in queries)
    # name-driven hits carry no per-indicator attribution (mapped against all)
    assert results and all(r.indicator_hits == [] for r in results)


@pytest.mark.live
@_LIVE
def test_browser_renders_real_anti_bot_portal():
    from lexora.collect.browser import is_available, render

    if not is_available():
        pytest.skip("Chromium not installed (pip install playwright; playwright install chromium)")
    res = render("https://sso.agc.gov.sg/", timeout=45.0)
    assert res.status in (200, 0)
    assert "<" in res.html and len(res.html) > 1000  # got a real rendered DOM
