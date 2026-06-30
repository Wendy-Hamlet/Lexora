"""Regulator-portal guidance connectors (P-1) — offline.

The PDPC hub is JS-rendered live; here a fake browser session feeds fixture HTML
so the harvest logic (detail-link selection, title cleaning, NEW/KNOWN tagging,
secondary source type) is pinned without a network or a real browser.
"""
from __future__ import annotations

from dataclasses import dataclass

from lexora.collect.strategies import (
    _OAIC_INCLUDE,
    _clean_guidance_title,
    _collect_guidance,
    _my_is_code,
    connector_for,
    imda_guidance,
    my_pdp_guidance,
    oaic_guidance,
    pdpc_guidance,
)
from lexora.models.source import (
    FetchMethod,
    LegalSystem,
    PortalSpec,
    SourceProfile,
    SourceType,
)

_HUB_HTML = """
<html><body>
  <nav><a href="/about">About</a><a href="/contact-us">Contact</a></nav>
  <a href="/organisations/regulations-decisions/regulatory-guidance">Regulatory Guidance</a>
  <ul>
    <li><a href="/organisations/regulations-decisions/regulatory-guidance/advisory-guidelines-on-the-pdpa-for-childrens-personal-data-in-the-digital-environment">
        Advisory Guidelines 28 Mar 2024 Advisory Guidelines on the PDPA for Children's Personal Data in the Digital Environment</a></li>
    <li><a href="/organisations/regulations-decisions/regulatory-guidance/advisory-guidelines-on-key-concepts">
        Advisory Guidelines on Key Concepts</a></li>
    <li><a href="/organisations/resources/guidance-by-topic/guide-to-data-protection-impact-assessments">
        Publications 14 Sep 2021 Guide to Data Protection Impact Assessments</a></li>
    <li><a href="https://facebook.com/pdpcsg">Facebook</a></li>
  </ul>
</body></html>
"""


@dataclass
class _Rendered:
    html: str


class _FakeSession:
    def render(self, url, *, timeout=None):
        return _Rendered(html=_HUB_HTML)


def _pdpc_portal() -> PortalSpec:
    return PortalSpec(
        name="PDPC", url="https://www.pdpc.gov.sg/", source_type=SourceType.secondary,
        fetch_method=FetchMethod.http,
    )


def test_clean_guidance_title_strips_category_and_date():
    assert _clean_guidance_title(
        "Advisory Guidelines 28 Mar 2024 Advisory Guidelines on the PDPA for Children's Personal Data"
    ) == "Advisory Guidelines on the PDPA for Children's Personal Data"
    assert _clean_guidance_title("Publications 14 Sep 2021 Guide to DPIA") == "Guide to DPIA"
    # No prefix -> unchanged (just whitespace-normalized).
    assert _clean_guidance_title("Advisory Guidelines on Key Concepts") == "Advisory Guidelines on Key Concepts"


def test_pdpc_guidance_harvests_detail_links_as_secondary_new():
    hits = pdpc_guidance(_pdpc_portal(), [], browser_session=_FakeSession(), known_instruments=[])
    titles = {h.title for h in hits}
    assert "Advisory Guidelines on the PDPA for Children's Personal Data in the Digital Environment" in titles
    assert "Guide to Data Protection Impact Assessments" in titles
    # nav + social + the hub self-link are excluded
    assert not any("Facebook" in t or "About" in t for t in titles)
    for h in hits:
        assert h.source_type is SourceType.secondary
        assert h.discovery_tag == "NEW"  # guidance isn't in the statute known-list


def test_pdpc_guidance_tags_known_when_title_matches_known_list():
    hits = pdpc_guidance(
        _pdpc_portal(), [], browser_session=_FakeSession(),
        known_instruments=["Advisory Guidelines on Key Concepts"],
    )
    known = [h for h in hits if h.title == "Advisory Guidelines on Key Concepts"]
    assert known and known[0].discovery_tag == "KNOWN"


def test_pdpc_guidance_always_includes_the_dpia_guide():
    # The DPIA guide is a PDF outside the crawled hubs, so the link harvest cannot
    # reach it; pdpc_guidance must add it explicitly even when the hub HTML lacks it.
    hits = pdpc_guidance(_pdpc_portal(), [], browser_session=_FakeSession(), known_instruments=[])
    # The explicitly-seeded entry is the PDF (the live hub does not link it); assert
    # that PDF lands, distinct from any HTML detail page a hub might happen to expose.
    dpia_pdf = [h for h in hits
                if "Data Protection Impact Assessments" in h.title and h.url.endswith(".pdf")]
    assert dpia_pdf and dpia_pdf[0].is_pdf_link
    assert dpia_pdf[0].source_type is SourceType.secondary


def _imda_portal() -> PortalSpec:
    return PortalSpec(
        name="IMDA", url="https://www.imda.gov.sg/", source_type=SourceType.secondary,
        fetch_method=FetchMethod.http,
    )


def test_imda_guidance_seeds_the_two_telecom_instruments():
    hits = imda_guidance(_imda_portal(), [], known_instruments=[])
    titles = {h.title for h in hits}
    assert any("IP Telephony" in t for t in titles)
    assert any("Facilities-Based Operations" in t for t in titles)
    for h in hits:
        assert h.source_type is SourceType.secondary
        assert h.is_pdf_link and h.url.endswith(".pdf")


def test_connector_for_matches_imda_host():
    assert connector_for(_imda_portal()) is imda_guidance


def test_connector_for_matches_pdpc_host_only():
    assert connector_for(_pdpc_portal()) is pdpc_guidance
    other = PortalSpec(name="SSO", url="https://sso.agc.gov.sg/", source_type=SourceType.primary)
    assert connector_for(other) is None


def test_connector_for_matches_my_and_au_hosts():
    my = PortalSpec(name="JPDP", url="https://www.pdp.gov.my/", source_type=SourceType.secondary)
    au = PortalSpec(name="OAIC", url="https://www.oaic.gov.au/", source_type=SourceType.secondary)
    assert connector_for(my) is my_pdp_guidance
    assert connector_for(au) is oaic_guidance


# --- MY PDP codes of practice ---

_MY_HTML = """
<html><body>
  <nav><a href="/ppdpv1/en/akta/">Akta</a><a href="/ppdpv1/en/akta/code-of-practice/">Code of Practice</a></nav>
  <ul>
    <li><a href="/ppdpv1/en/akta/personal-data-protection-code-of-practice-for-banking-sector-and-financial-institutions/">
        Personal Data Protection Code of Practice For Banking Sector And Financial Institutions</a></li>
    <li><a href="/ppdpv1/en/akta/personal-data-protection-code-of-practice-for-the-communications-sector/">
        Personal Data Protection Code of Practice For the Communications Sector</a></li>
    <li><a href="/ppdpv1/en/akta/akta-pdp-2010-my/">AKTA 709</a></li>
  </ul>
</body></html>
"""


def test_my_is_code_excludes_bare_hub():
    assert _my_is_code("https://www.pdp.gov.my/ppdpv1/en/akta/personal-data-protection-code-of-practice-for-banking-sector-and-financial-institutions/")
    assert not _my_is_code("https://www.pdp.gov.my/ppdpv1/en/akta/code-of-practice/")
    assert not _my_is_code("https://www.pdp.gov.my/ppdpv1/en/akta/akta-pdp-2010-my/")


def test_collect_guidance_harvests_my_codes_excludes_hub_and_act():
    hub = "https://www.pdp.gov.my/ppdpv1/en/akta/code-of-practice/"
    out = _collect_guidance(
        _MY_HTML, hub, include=_my_is_code, known=[], hubs=(hub,),
    )
    titles = {r.title for r in out.values()}
    assert any("Banking Sector" in t for t in titles)
    assert any("Communications Sector" in t for t in titles)
    assert all("AKTA 709" not in t for t in titles)  # the Act, not a code
    assert all(r.source_type is SourceType.secondary for r in out.values())


def test_my_pdp_guidance_always_includes_the_standard(monkeypatch):
    # The PDP Standard 2015 is not in the code-of-practice sidebar, so the link
    # harvest cannot reach it; my_pdp_guidance must add it explicitly. Stub the
    # network with a page that has NO code links to prove the Standard still lands.
    import httpx

    from lexora.collect import strategies as S

    def _fake_get(self, url, *a, **k):
        return httpx.Response(200, text="<html><body><p>no codes here</p></body></html>")

    monkeypatch.setattr(httpx.Client, "get", _fake_get)
    portal = PortalSpec(
        name="JPDP", url="https://www.pdp.gov.my/", source_type=SourceType.secondary,
        fetch_method=FetchMethod.http, search_query="code of practice",
    )
    results = S.my_pdp_guidance(portal, [], limit=40, timeout=5.0)
    titles = [r.title for r in results]
    assert "Personal Data Protection Standard 2015" in titles
    std = next(r for r in results if "Standard 2015" in r.title)
    assert std.source_type is SourceType.secondary
    assert std.url.endswith("/personal-data-protection-standard-2015/")


# --- AU OAIC guidance ---

_OAIC_HTML = """
<html><body>
  <a href="/privacy">Privacy</a>
  <a href="/freedom-of-information/foi-guidance">FOI guidance</a>
  <ul>
    <li><a href="https://www.oaic.gov.au/privacy/privacy-guidance-for-organisations-and-government-agencies/privacy-impact-assessments">
        Privacy impact assessments</a></li>
    <li><a href="https://www.oaic.gov.au/privacy/australian-privacy-principles/australian-privacy-principles-guidelines/chapter-8-app-8-cross-border-disclosure-of-personal-information">
        Chapter 8: APP 8 Cross-border disclosure of personal information</a></li>
    <li><a href="https://www.oaic.gov.au/privacy/privacy-guidance-for-organisations-and-government-agencies">Government agencies</a></li>
  </ul>
</body></html>
"""


def test_collect_guidance_harvests_oaic_pia_and_app_skips_nav():
    out = _collect_guidance(
        _OAIC_HTML, "https://www.oaic.gov.au/privacy/x",
        include=lambda u: bool(_OAIC_INCLUDE.search(u)), known=[],
    )
    titles = {r.title for r in out.values()}
    assert "Privacy impact assessments" in titles
    assert any("APP 8 Cross-border" in t for t in titles)
    # cross-section FOI link and the "Government agencies" nav label are excluded
    assert all("FOI" not in t and t.lower() != "government agencies" for t in titles)


def test_oaic_guidance_returns_empty_offline(monkeypatch):
    # No network in the offline suite: the connector must degrade to [].
    import lexora.collect.strategies as strat

    class _Boom:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, *a, **k): raise RuntimeError("no network")

    monkeypatch.setattr(strat.httpx, "Client", lambda *a, **k: _Boom())
    portal = PortalSpec(name="OAIC", url="https://www.oaic.gov.au/", source_type=SourceType.secondary)
    assert oaic_guidance(portal, []) == []


def test_discover_secondary_returns_empty_without_connector_portals(monkeypatch):
    from lexora.collect import discovery

    profile = SourceProfile(
        jurisdiction="Singapore", iso_code="SG", primary_language="en",
        legal_system=LegalSystem.common,
        portals=[PortalSpec(name="SSO", url="https://sso.agc.gov.sg/", source_type=SourceType.primary)],
    )
    assert discovery.discover_secondary(profile, []) == []
