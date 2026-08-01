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


def test_a_year_scoped_act_number_is_not_collapsed_across_years():
    """Singapore restarts Act numbering every year, so "Act 19" is not an identity.

    `_identity_key` collapses an Act's many PDF variants onto one working-set slot, and
    `_merge_into` keeps ONE representative per key -- so a key shared by different Acts
    drops the others before they are ever fetched, parsed or mapped. Measured on the
    round-1 submission: "act:19" collapsed Act 19 of 2021, of 2023 AND of 2024, and
    "act:6" collapsed Act 6 of 2024 with Act 6 of 2025.
    """
    from lexora.collect.discovery import DiscoveryResult, _identity_key

    def key(url: str, title: str = "") -> str:
        return _identity_key(DiscoveryResult(
            url=url, title=title, source_type=SourceType.primary, score=1.0,
            via="x", is_pdf_link=False))

    sg = [key("https://sso.agc.gov.sg/Acts-Supp/19-2021/", "Act 19 of 2021"),
          key("https://sso.agc.gov.sg/Acts-Supp/19-2023/", "Act 19 of 2023"),
          key("https://sso.agc.gov.sg/Acts-Supp/19-2024/", "Act 19 of 2024")]
    assert len(set(sg)) == 3, sg

    # The mechanism this key exists for is untouched: one Malaysian Act surfacing as
    # several PDF variants still collapses onto a single slot.
    my = [key("https://lom.agc.gov.my/x/ACT%20709.pdf", "Act 709"),
          key("https://lom.agc.gov.my/x/Act 709 ori.pdf", "Act 709"),
          key("https://lom.agc.gov.my/x/ACT 709-REPRINT 2023.pdf", "Act 709")]
    assert len(set(my)) == 1, my

    # And a title year is still not an Act number.
    assert "act:2012" not in key("https://sso.agc.gov.sg/Act/PDPA2012",
                                 "Personal Data Protection Act 2012")


def test_identity_requires_the_years_to_agree_but_relevance_does_not():
    """``token_set_ratio`` scores the INTERSECTION of the token sets, so a longer title
    that CONTAINS a known one scores a perfect match. Australia's theme-named omnibus Acts
    are exactly that shape: "Telecommunications Legislation Amendment (Information
    Disclosure ...) Act 2023" matched "Telecommunications Legislation Amendment Act 1997"
    — 26 years apart — so a genuine discovery was written off against the gold inventory,
    on the metric worth 20 of the 40 accuracy points.

    The guard is for IDENTITY only. Applied to relevance it scored "Privacy Amendment Act
    1990" at zero for the query "privacy act 1988" and dropped it from the results
    entirely, which is a worse failure than the mis-tag it prevents.
    """
    from lexora.collect.discovery import _fuzzy_known

    known = ["Telecommunications Legislation Amendment Act 1997"]
    omnibus = ("Telecommunications Legislation Amendment (Information Disclosure, "
               "National Interest and Other Measures) Act 2023")
    assert _fuzzy_known(omnibus, known)[0] >= 0.80                    # relevance: a match
    assert _fuzzy_known(omnibus, known, year_strict=True)[0] == 0.0   # identity: not it

    # The instrument still matches itself, and a compilation still matches its principal
    # (set intersection, not equality).
    assert _fuzzy_known("Telecommunications Legislation Amendment Act 1997", known,
                        year_strict=True)[0] >= 0.80
    assert _fuzzy_known("Privacy Act 1988 (Compilation No. 89, 2022)", ["Privacy Act 1988"],
                        year_strict=True)[0] >= 0.80
    # A known name carrying no year constrains nothing.
    assert _fuzzy_known("Personal Data Protection Code of Practice 2017",
                        ["Personal Data Protection Code of Practice"],
                        year_strict=True)[0] >= 0.80


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


def test_sweep_spaces_queries_on_a_portal_that_asks_for_it(monkeypatch):
    # The spacing used to be gated on holding a browser session, which made it
    # unreachable for an API portal -- exactly the kind that answers a burst with a
    # refusal (Malaysia's Fess proxy: one spaced POST fine, every request of the
    # sweep after it a 500). Assert the sleeps happen with no session in sight.
    slept: list[float] = []
    monkeypatch.setattr("lexora.collect.discovery.time.sleep", slept.append)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_MULTI_ACT_HTML.encode(),
                              headers={"content-type": "text/html"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    portal = PortalSpec(
        name="Fess", url="https://api.example.gov/", source_type=SourceType.primary,
        fetch_method=FetchMethod.http, sweep_interval=20.0,
        search_url_template="https://api.example.gov/search?q={query}",
    )
    inds = [_ind("6.1", "P6-I1", ["alpha"]), _ind("7.3", "P7-I3", ["beta"])]
    discover_for_indicators(portal, inds, client=client, budget=5)
    client.close()
    # Spacing goes BETWEEN queries, so two phrases cost one wait, not two.
    assert slept == [20.0]


def test_sweep_does_not_space_a_portal_that_did_not_ask(monkeypatch):
    # The knob is opt-in: an unconfigured portal must sweep at full speed, or every
    # jurisdiction pays for one portal's limiter.
    slept: list[float] = []
    monkeypatch.setattr("lexora.collect.discovery.time.sleep", slept.append)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_MULTI_ACT_HTML.encode(),
                              headers={"content-type": "text/html"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    portal = PortalSpec(
        name="Plain", url="https://plain.example.gov/", source_type=SourceType.primary,
        search_url_template="https://plain.example.gov/search?q={query}",
    )
    inds = [_ind("6.1", "P6-I1", ["alpha"]), _ind("7.3", "P7-I3", ["beta"])]
    discover_for_indicators(portal, inds, client=client, budget=5)
    client.close()
    assert slept == []


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


def test_an_answered_page_is_asked_for_once_per_run():
    """The sweep and the three follow-on passes reach `discover` independently.

    They re-issue queries each other has already asked; on a browser portal that is a
    3-10 s render for a page still in memory. Only ANSWERED pages are memoised.
    """
    from lexora.collect.discovery import discover

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
    first = discover(portal, query="alpha", client=client, limit=5)
    second = discover(portal, query="alpha", client=client, limit=5)
    client.close()

    assert calls["n"] == 1
    # A memo hit must be the result a re-fetch would have produced.
    assert [r.url for r in first] == [r.url for r in second]


def test_a_refused_page_is_never_memoised():
    """A refusal has to stay re-askable -- that is what the late retry is for."""
    from lexora.collect.discovery import discover

    calls = {"n": 0}
    blocked = (b"<html><head><title>ERROR: The request could not be satisfied</title>"
               b"</head><body><h1>403 ERROR</h1>Request blocked.</body></html>")

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, content=blocked,
                              headers={"content-type": "text/html"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    portal = PortalSpec(
        name="SSO", url="https://sso.example.gov/", source_type=SourceType.primary,
        search_url_template="https://sso.example.gov/search?q={query}",
    )
    discover(portal, query="alpha", client=client, limit=5)
    discover(portal, query="alpha", client=client, limit=5)
    client.close()

    assert calls["n"] == 2


def test_the_memo_does_not_confuse_two_queries():
    from lexora.collect.discovery import discover

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=_MULTI_ACT_HTML.encode(),
                              headers={"content-type": "text/html"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    portal = PortalSpec(
        name="SSO", url="https://sso.example.gov/", source_type=SourceType.primary,
        search_url_template="https://sso.example.gov/search?q={query}",
    )
    discover(portal, query="alpha", client=client, limit=5)
    discover(portal, query="beta", client=client, limit=5)
    client.close()
    assert len(seen) == 2 and seen[0] != seen[1]


def test_the_sso_sweep_does_not_pace_itself_against_a_recording(monkeypatch, tmp_path):
    """The SG sweep spaces navigations 1.5s apart to stay under SSO's cumulative
    per-IP limit. Under replay every render comes off a SQLite file on this disk, so
    that limiter is not in the loop -- yet the spacing fired anyway: measured on the
    real recording, 87s of the ~90s the Singapore discovery leg took was these sleeps.
    """
    from lexora.collect import discovery, http_cache

    monkeypatch.setenv("LEXORA_HTTP_CACHE", "replay")
    monkeypatch.setenv("LEXORA_HTTP_CACHE_DIR", str(tmp_path / "http_cache"))
    monkeypatch.setattr(http_cache, "_STORE", None)
    slept: list[float] = []
    monkeypatch.setattr(discovery.time, "sleep", lambda s: slept.append(s))

    portal = PortalSpec(
        name="SSO", url="https://sso.agc.gov.sg/", source_type=SourceType.primary,
        fetch_method=FetchMethod.playwright,
        search_url_template="https://sso.agc.gov.sg/Search?q={query}",
    )
    inds = [_ind("6.1", "P6-I1", ["alpha"]), _ind("7.3", "P7-I3", ["beta"])]

    discover_for_indicators(portal, inds, budget=5)
    assert slept == []

    # Same sweep with the recording out of the picture: the spacing is still there,
    # so the guard removed the delay and not the rate-limit protection.
    monkeypatch.setattr(discovery.http_cache, "serving_from_recording", lambda: False)
    discover_for_indicators(portal, inds, budget=5)
    # One gap between the two phrases, plus one before each of the two refused
    # queries the post-sweep retry pass picks up -- both spacing sites, both silenced.
    assert slept == [1.5, 1.5, 1.5]


def test_an_unentered_browser_session_says_so(monkeypatch):
    """`render` turns every failure into an empty page, which is how a refusing portal
    also looks. A session that was never entered has no context to render through, so
    it would report "no results" for every query in the sweep."""
    from lexora.collect.browser import BrowserSession, SessionNotStarted

    monkeypatch.delenv("LEXORA_HTTP_CACHE", raising=False)
    with pytest.raises(SessionNotStarted):
        BrowserSession().render("https://sso.agc.gov.sg/")


def test_portal_furniture_is_not_an_instrument():
    """Malaysia's LOM portal put five navigation labels into a 13-slot working set.

    On 2026-08-01 the Fess search backend answered 500 with an empty body. Discovery
    fell back to scraping the portal's own links and returned "Ordinance", "Search",
    "Translated", "Top Hit (Weekly)" and "See All..." as instruments. They were fetched,
    parsed, counted, and they crowded the statutes out of the working set -- the
    Personal Data Protection Act 2010 never made it in.

    The match must be on the WHOLE anchor text. As a substring, "search" would discard
    Criminal Procedure Code results about search and seizure and "ordinance" would
    discard every real Ordinance. Checked against all 470 distinct law names in our
    outputs and gold inventories: zero would be filtered.
    """
    from lexora.collect.discovery import _is_nav_label

    for junk in ("Ordinance", "Search", "Translated", "Top Hit (Weekly)",
                 "See All...", "see all", "  ADVANCED SEARCH  ", "View All"):
        assert _is_nav_label(junk), f"portal furniture not caught: {junk!r}"

    for real in ("Criminal Procedure Code", "Emergency Ordinance 1969",
                 "Search and Seizure Act 1988", "Personal Data Protection Act 2010",
                 "Ordinance 12 of 1955", "Home Affairs Act 1999"):
        assert not _is_nav_label(real), f"real law discarded: {real!r}"
