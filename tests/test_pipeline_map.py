"""Multi-instrument autonomous map (P0) — orchestration logic.

`run_pipeline_map`'s collaborators (discovery, full-text resolution, the
per-document pipeline) are stubbed so this test pins the *aggregation* behaviour
with no network: it maps every discovered instrument, tags each citation NEW or
KNOWN from its instrument, de-duplicates by (indicator, clause) across documents,
and never collapses one clause that legitimately maps to several indicators.
"""
from __future__ import annotations

from datetime import datetime, timezone

from lexora.collect.discovery import DiscoveryResult
from lexora.models.citation import Citation, DiscoveryTag
from lexora.models.source import (
    FetchMethod,
    LegalSystem,
    PortalSpec,
    RawDocument,
    SourceProfile,
    SourceType,
)


def _hit(url: str, tag: str, score: float) -> DiscoveryResult:
    return DiscoveryResult(
        url=url, title=f"Instrument {url[-3:]}", source_type=SourceType.primary,
        score=score, via="http", is_pdf_link=False, discovery_tag=tag,
    )


def _doc() -> RawDocument:
    return RawDocument(
        document_id="doc", source_url="https://e.gov/x", http_status=200,
        retrieval_timestamp=datetime.now(timezone.utc), sha256="sha256:abc",
        content_type="application/pdf", bytes_path="/tmp/x", portal_name="P",
        jurisdiction="SG", source_type=SourceType.primary,
    )


def _cite(indicator_id: str, clause_id: str, tag: DiscoveryTag, title: str) -> Citation:
    return Citation(
        title=title, indicator_id=indicator_id, article_path="S. 1",
        discovery_tag=tag, page_or_dom_anchor="1", quote="verbatim text",
        source_url="https://e.gov/x", confidence=0.9, clause_id=clause_id,
        retrieval_timestamp=datetime.now(timezone.utc), document_hash="sha256:abc",
        jurisdiction="SG", legal_form="statute", char_start=0, char_end=4,
    )


def _profile() -> SourceProfile:
    return SourceProfile(
        jurisdiction="Singapore", iso_code="SG", primary_language="en",
        legal_system=LegalSystem.common,
        portals=[PortalSpec(name="SSO", url="https://e.gov/", source_type=SourceType.primary,
                            fetch_method=FetchMethod.http)],
    )


def test_run_pipeline_map_aggregates_dedups_and_propagates_tags(monkeypatch):
    import lexora.collect.discovery as disc
    import lexora.pipeline as pipe

    hits = [_hit("https://e.gov/Act/AAA", "KNOWN", 0.9),
            _hit("https://e.gov/Act/BBB", "NEW", 0.5)]
    monkeypatch.setattr(disc, "discover_for_indicators", lambda *a, **k: hits)
    monkeypatch.setattr(disc, "resolve_fulltext", lambda hit, **k: hit.url + ".pdf")

    def fake_from_url(*, url, discovery_tag, title, **k):
        if "AAA" in url:
            cites = [_cite("P6-I1", "doc::c1", discovery_tag, title)]
        else:  # BBB re-finds the SAME (indicator, clause) -> must dedup; adds a new one
            cites = [_cite("P6-I1", "doc::c1", discovery_tag, title),
                     _cite("P7-I3", "doc::c2", discovery_tag, title)]
        return pipe.DemoArtifacts(document=_doc(), clauses=[], citations=cites)

    monkeypatch.setattr(pipe, "run_pipeline_from_url", fake_from_url)

    profile = _profile()
    result = pipe.run_pipeline_map(
        portal=profile.portals[0], profile=profile, indicators=[],
    )

    assert len(result.discovered) == 2
    assert len(result.documents) == 2
    # (P6-I1, doc::c1) appears in both docs -> collapsed once; (P7-I3, doc::c2) kept.
    keys = sorted((c.indicator_id, c.clause_id) for c in result.citations)
    assert keys == [("P6-I1", "doc::c1"), ("P7-I3", "doc::c2")]
    # The surviving P6-I1 came from the first (KNOWN) instrument; P7-I3 from the NEW one.
    by_ind = {c.indicator_id: c.discovery_tag for c in result.citations}
    assert by_ind["P6-I1"] is DiscoveryTag.known
    assert by_ind["P7-I3"] is DiscoveryTag.new


def test_attribute_by_name_maps_to_relevant_indicators_only():
    from lexora.models.indicator import RDTIIIndicator
    from lexora.pipeline import _attribute_by_name

    inds = [
        RDTIIIndicator(rdtii_id="7.2", submission_id="P7-I2", pillar=7,
                       name="cyber", description="d"),
        RDTIIIndicator(rdtii_id="7.3", submission_id="P7-I3", pillar=7,
                       name="retention", description="d"),
    ]
    profile = SourceProfile(
        jurisdiction="Australia", iso_code="AU", primary_language="en",
        legal_system=LegalSystem.common,
        keywords_by_indicator={
            "7.2": {"en": ["Security of Critical Infrastructure Act"]},
            "7.3": {"en": ["Telecommunications (Interception and Access) Act"]},
        },
    )
    hit = _hit("https://www.legislation.gov.au/C2004A02124/latest", "KNOWN", 0.9)
    hit.title = "Telecommunications (Interception and Access) Act 1979"
    # attributed to 7.3 (its name hint) only — not 7.2
    assert _attribute_by_name(hit, profile, inds) == {"P7-I3"}


def test_run_pipeline_map_empty_when_no_instruments(monkeypatch):
    import lexora.collect.discovery as disc
    import lexora.pipeline as pipe

    monkeypatch.setattr(disc, "discover_for_indicators", lambda *a, **k: [])
    profile = _profile()
    result = pipe.run_pipeline_map(portal=profile.portals[0], profile=profile, indicators=[])
    assert result.discovered == [] and result.documents == [] and result.citations == []


def test_map_use_dense_defaults_off_and_honors_env(monkeypatch):
    # G-6.3: mapping clause retrieval is BM25-only by default; dense fusion is an
    # explicit opt-in via LEXORA_MAP_DENSE.
    import lexora.pipeline as pipe

    monkeypatch.delenv("LEXORA_MAP_DENSE", raising=False)
    assert pipe._map_use_dense() is False
    for val in ("1", "true", "yes", "on", "ON"):
        monkeypatch.setenv("LEXORA_MAP_DENSE", val)
        assert pipe._map_use_dense() is True
    for val in ("0", "false", "no", ""):
        monkeypatch.setenv("LEXORA_MAP_DENSE", val)
        assert pipe._map_use_dense() is False
