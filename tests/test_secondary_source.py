"""Secondary-source infrastructure (WS-S, S-0).

Covers the model, the config catalogue, the registry, the disk cache helper, and
— most importantly — the RED LINE: a SecondarySignal is a non-citable discovery
aid and there is no path from it to an EvidenceClaim / Citation.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from lexora.collect.secondary import (
    CoverageGap,
    Presence,
    SecondarySignal,
    adapter_for,
    applicable_sources,
    cached_json,
    coverage_gaps,
    load_secondary_sources,
    register_source,
    to_discovery_seeds,
    to_provenance_note,
)
from lexora.collect.secondary import base as sb
from lexora.indicators import load_indicators
from lexora.models.citation import Citation, EvidenceClaim

REPO = Path(__file__).resolve().parent.parent
INDICATORS = REPO / "configs" / "rdtii_indicators.yaml"


def _signal(**kw) -> SecondarySignal:
    base = dict(economy="MY", indicator_id="P7-I1", source_name="UNCTAD DPP")
    base.update(kw)
    return SecondarySignal(**base)


# --- red line ---------------------------------------------------------------

def test_secondary_signal_is_not_a_citation():
    """The type itself must not be (or look like) evidence."""
    assert not issubclass(SecondarySignal, Citation)
    assert not issubclass(SecondarySignal, EvidenceClaim)
    fields = set(SecondarySignal.model_fields)
    # none of the verbatim-evidence fields may exist on a secondary signal
    assert not (fields & {"quote", "verbatim_snippet", "quote_span_id", "clause_id"})


def test_signal_can_only_leave_as_plain_strings():
    """The sanctioned exits are LOSSY — a signal can only become str / list[str]
    / a coverage row, never a Citation. This is the structural red-line guard."""
    s = _signal(presence=Presence.yes, primary_law_name="Personal Data Protection Act 2010")
    seeds = to_discovery_seeds([s])
    assert seeds == ["Personal Data Protection Act 2010"]
    assert all(isinstance(x, str) for x in seeds)
    note = to_provenance_note(s)
    assert isinstance(note, str)
    assert "corroborated by secondary source" in note
    # no helper anywhere in the layer returns a Citation/EvidenceClaim
    for name in sb.__all__:
        obj = getattr(sb, name)
        if callable(obj):
            assert obj not in (Citation, EvidenceClaim)


# --- config catalogue -------------------------------------------------------

def test_config_loads_and_maps_real_in_scope_indicators():
    specs = load_secondary_sources()
    assert specs, "catalogue should not be empty"
    by_key = {s.key: s for s in specs}
    assert "unctad_data_protection" in by_key
    assert "P7-I1" in by_key["unctad_data_protection"].indicators
    assert by_key["unctad_data_protection"].tier == 1
    # every indicator referenced by any source must be a real submission_id
    valid = {i.submission_id for i in load_indicators(INDICATORS)}
    referenced = {ind for s in specs for ind in s.indicators}
    assert referenced <= valid, f"unknown indicator ids in catalogue: {referenced - valid}"
    # P6-I5 (trade agreements) is out of scope -> must never be referenced
    assert "P6-I5" not in referenced


def test_applicable_sources_filters_by_indicator():
    specs = load_secondary_sources()
    p7 = {s.key for s in applicable_sources("P7-I1", specs)}
    assert "unctad_data_protection" in p7
    assert "oecd_dstri" not in p7  # OECD STRI is a P6 source
    p6 = {s.key for s in applicable_sources("P6-I1", specs)}
    assert "oecd_dstri" in p6
    assert "unctad_data_protection" not in p6


# --- registry ---------------------------------------------------------------

def test_register_and_lookup_adapter():
    assert adapter_for("nonexistent") is None

    @register_source("fake_src")
    def _fake(economy, indicators, **kw):
        return [_signal(economy=economy, source_name="fake")]

    try:
        got = adapter_for("fake_src")
        assert got is _fake
        out = got("AU", [])
        assert len(out) == 1 and isinstance(out[0], SecondarySignal)
    finally:
        sb.SECONDARY_SOURCES.pop("fake_src", None)  # don't leak into other tests


def test_unctad_adapters_self_register():
    """Importing the package registers the S-1 UNCTAD adapters."""
    for key in ("unctad_data_protection", "unctad_cyberlaw", "unctad_cybercrime"):
        assert adapter_for(key) is not None


# --- disk cache -------------------------------------------------------------

def test_cached_json_caches_fresh_and_skips_producer(tmp_path):
    calls = {"n": 0}

    def produce():
        calls["n"] += 1
        return [{"economy": "MY", "law": "PDPA 2010"}]

    cache = tmp_path / "x.json"
    first = cached_json(cache, ttl_hours=24, producer=produce)
    second = cached_json(cache, ttl_hours=24, producer=produce)
    assert first == second == [{"economy": "MY", "law": "PDPA 2010"}]
    assert calls["n"] == 1  # second call served from disk


def test_cached_json_does_not_cache_empty(tmp_path):
    calls = {"n": 0}

    def produce_empty():
        calls["n"] += 1
        return []

    cache = tmp_path / "y.json"
    cached_json(cache, ttl_hours=24, producer=produce_empty)
    cached_json(cache, ttl_hours=24, producer=produce_empty)
    assert calls["n"] == 2  # empty never cached -> producer runs again
    assert not cache.exists()


# --- sanctioned exits -------------------------------------------------------

def test_to_discovery_seeds_keeps_yes_and_draft_dedup_drops_no():
    sigs = [
        _signal(presence=Presence.yes, primary_law_name="PDPA 2010"),
        _signal(presence=Presence.draft, primary_law_name="Cyber Bill"),
        _signal(presence=Presence.yes, primary_law_name="pdpa 2010"),  # dup (case)
        _signal(presence=Presence.no, primary_law_name="Ghost Act"),  # dropped
        _signal(presence=Presence.yes, primary_law_name="   "),  # blank dropped
    ]
    assert to_discovery_seeds(sigs) == ["PDPA 2010", "Cyber Bill"]


def test_coverage_gaps_flags_expected_but_not_found():
    sigs = [
        _signal(presence=Presence.yes, primary_law_name="Personal Data Protection Act 2010"),
        _signal(presence=Presence.yes, primary_law_name="Phantom Privacy Act"),
        _signal(presence=Presence.no, primary_law_name="Should Be Ignored"),
    ]
    discovered = ["Personal Data Protection Act 2010 (Act 709)"]
    gaps = coverage_gaps(sigs, discovered)
    assert len(gaps) == 1
    assert isinstance(gaps[0], CoverageGap)
    assert gaps[0].expected_law_name == "Phantom Privacy Act"


def test_signal_has_retrieved_at_default():
    assert isinstance(_signal().retrieved_at, datetime)
