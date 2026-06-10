"""Regulator-portal guidance connectors (P-1) — offline.

The PDPC hub is JS-rendered live; here a fake browser session feeds fixture HTML
so the harvest logic (detail-link selection, title cleaning, NEW/KNOWN tagging,
secondary source type) is pinned without a network or a real browser.
"""
from __future__ import annotations

from dataclasses import dataclass

from lexora.collect.strategies import (
    _clean_guidance_title,
    connector_for,
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


def test_connector_for_matches_pdpc_host_only():
    assert connector_for(_pdpc_portal()) is pdpc_guidance
    other = PortalSpec(name="SSO", url="https://sso.agc.gov.sg/", source_type=SourceType.primary)
    assert connector_for(other) is None


def test_discover_secondary_returns_empty_without_connector_portals(monkeypatch):
    from lexora.collect import discovery

    profile = SourceProfile(
        jurisdiction="Singapore", iso_code="SG", primary_language="en",
        legal_system=LegalSystem.common,
        portals=[PortalSpec(name="SSO", url="https://sso.agc.gov.sg/", source_type=SourceType.primary)],
    )
    assert discovery.discover_secondary(profile, []) == []
