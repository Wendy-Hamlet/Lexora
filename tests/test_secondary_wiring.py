"""WS-S S-2 — the three consumers wired into the run.

USE 1 recall booster (extra_seed_queries -> discovery), USE 2 coverage cross-check
(indicator_gaps), USE 3 provenance (note stamped on a citation, never evidence).
All offline.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import lexora.collect.discovery as disc
from lexora.collect.secondary import (
    Presence,
    SecondarySignal,
    SecondarySourceSpec,
    gather_signals,
    indicator_gaps,
    provenance_notes_by_indicator,
    register_source,
)
from lexora.collect.secondary import base as sb
from lexora.indicators import load_indicators
from lexora.models.clause import CanonicalSpan, Clause
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import (
    LegalSystem,
    PortalSpec,
    RawDocument,
    SourceProfile,
    SourceType,
)
from lexora.pipeline import _citations_from_clauses

REPO = Path(__file__).resolve().parent.parent
INDS = load_indicators(REPO / "configs" / "rdtii_indicators.yaml")


import pytest


@pytest.fixture
def tmp_cleanup_keys():
    """Collect registry keys to remove after the test so fakes don't leak."""
    keys: list[str] = []
    yield keys
    for k in keys:
        sb.SECONDARY_SOURCES.pop(k, None)


def _sig(**kw) -> SecondarySignal:
    base = dict(economy="SG", indicator_id="P7-I1", source_name="UNCTAD X",
                presence=Presence.yes)
    base.update(kw)
    return SecondarySignal(**base)


# --- gather (entry point) ---------------------------------------------------

def test_gather_runs_applicable_adapters_and_filters(tmp_cleanup_keys):
    @register_source("fake_track")
    def _fake(economy, indicators):
        return [SecondarySignal(economy=economy, indicator_id="P7-I1",
                                source_name="FAKE", presence=Presence.yes)]
    tmp_cleanup_keys.append("fake_track")

    in_scope = SecondarySourceSpec(key="fake_track", name="FAKE", indicators=("P7-I1",))
    out_scope = SecondarySourceSpec(key="fake_track", name="FAKE", indicators=("P6-I5",))  # not in INDS

    sigs = gather_signals("SG", INDS, specs=[in_scope])
    assert [s.source_name for s in sigs] == ["FAKE"]
    # a source covering only out-of-scope indicators is skipped
    assert gather_signals("SG", INDS, specs=[out_scope]) == []


def test_gather_skips_unbuilt_and_failing_sources(tmp_cleanup_keys):
    @register_source("boom")
    def _boom(economy, indicators):
        raise RuntimeError("flaky tracker")
    tmp_cleanup_keys.append("boom")

    ok = SecondarySourceSpec(key="boom", name="B", indicators=("P7-I1",))
    no_adapter = SecondarySourceSpec(key="never_registered", name="N", indicators=("P7-I1",))
    # a raising adapter and an unbuilt source both degrade to nothing, no crash
    assert gather_signals("SG", INDS, specs=[ok, no_adapter]) == []


# --- USE 3 provenance -------------------------------------------------------

def test_provenance_notes_dedupe_per_source_and_skip_absent():
    sigs = [
        _sig(source_name="UNCTAD A"),
        _sig(source_name="UNCTAD A"),  # dup source -> one note
        _sig(source_name="UNCTAD B", presence=Presence.draft),  # draft counts
        _sig(source_name="UNCTAD C", presence=Presence.no),  # absent -> skipped
    ]
    notes = provenance_notes_by_indicator(sigs)
    assert "P7-I1" in notes
    assert notes["P7-I1"].count("corroborated by secondary source") == 2  # A + B
    assert "UNCTAD C" not in notes["P7-I1"]


# --- USE 2 coverage cross-check ---------------------------------------------

def test_indicator_gaps_flags_uncovered_yes():
    sigs = [
        _sig(indicator_id="P7-I1"),          # covered below -> no gap
        _sig(indicator_id="P7-I2"),          # uncovered -> gap
        _sig(indicator_id="P7-I3", presence=Presence.no),  # absent -> never a gap
    ]
    gaps = indicator_gaps(sigs, covered_indicator_ids={"P7-I1"})
    assert [g.indicator_id for g in gaps] == ["P7-I2"]


# --- USE 1 recall booster ---------------------------------------------------

def test_extra_seed_queries_reach_discovery(monkeypatch):
    seen = []

    def fake_discover(portal, query=None, **kw):
        seen.append(query)
        return []

    monkeypatch.setattr(disc, "discover", fake_discover)
    portal = PortalSpec(name="p", url="https://x.example/", source_type=SourceType.primary,
                        full_text=True)
    inds = [RDTIIIndicator(rdtii_id="7.1", submission_id="P7-I1", pillar=7, name="n",
                           description="d")]
    disc.discover_for_indicators(portal, inds, extra_seed_queries=["Ghost Privacy Act"],
                                 use_semantic=False)
    assert "Ghost Privacy Act" in seen


# --- USE 3 end-to-end on a real citation ------------------------------------

_TEXT = ("An organisation must protect personal data in its possession by making "
         "reasonable security arrangements to prevent unauthorised access.")


def _profile():
    return SourceProfile(
        jurisdiction="Singapore", iso_code="SG", primary_language="en",
        legal_system=LegalSystem.common,
        keywords_by_indicator={"7.1": {"en": [
            "protect personal data reasonable security arrangements unauthorised access"]}})


def _doc():
    return RawDocument(
        document_id="d", source_url="https://e.gov/x",
        retrieval_timestamp=datetime.now(timezone.utc), http_status=200, sha256="sha256:x",
        content_type="application/pdf", bytes_path="/tmp/x", portal_name="P",
        jurisdiction="SG", source_type=SourceType.primary, title="Test Act")


def _clauses():
    return [Clause(clause_id="S. 24", document_id="d", structural_path="S. 24",
                   span=CanonicalSpan(span_id="s", document_id="d", char_start=0,
                                      char_end=len(_TEXT), text=_TEXT))]


def _ind():
    return [RDTIIIndicator(rdtii_id="7.1", submission_id="P7-I1", pillar=7,
                           name="Comprehensive data protection framework",
                           description="protect personal data security arrangements")]


def test_provenance_note_lands_in_citation_notes_not_as_evidence():
    note = "corroborated by secondary source: UNCTAD Global Cyberlaw Tracker (http://u)"
    cites = _citations_from_clauses(
        _clauses(), _doc(), _profile(), _ind(), "statute", top_k=1, min_score=0.0,
        secondary_note_by_indicator={"P7-I1": note})
    assert len(cites) == 1
    c = cites[0]
    assert note in c.notes  # provenance is in Notes
    assert c.quote == _TEXT  # the EVIDENCE is still the verbatim clause, untouched
    assert "corroborated" not in c.quote  # never leaks into the quote/evidence


def test_no_secondary_note_means_unchanged_notes():
    cites = _citations_from_clauses(
        _clauses(), _doc(), _profile(), _ind(), "statute", top_k=1, min_score=0.0)
    assert "corroborated" not in (cites[0].notes or "")
