"""UNCTAD secondary-source adapters (WS-S, S-1) — offline, fixture-driven.

The three trackers share one dataset (CyberlawData.js); these tests inject a
parsed fixture via ``data=`` so they never hit the network.
"""
from __future__ import annotations

from pathlib import Path

from lexora.collect.secondary import load_secondary_sources
from lexora.collect.secondary.unctad import (
    parse_cyberlaw_js,
    unctad_cybercrime,
    unctad_cyberlaw,
    unctad_data_protection,
)
from lexora.indicators import load_indicators
from lexora.models.secondary import Presence

REPO = Path(__file__).resolve().parent.parent
INDICATORS = REPO / "configs" / "rdtii_indicators.yaml"
INDS = load_indicators(INDICATORS)

# columns: [e-transactions, consumer, PRIVACY(2), CYBERCRIME(3), taxation]
FIX = {
    "SG": [1, 3, 1, 2, 1],  # privacy=yes,      cybercrime=draft
    "MY": [1, 1, 3, 1, 1],  # privacy=no,       cybercrime=yes
    "AU": [1, 1, 0, 1, 1],  # privacy=unknown,  cybercrime=yes
}


def test_parse_cyberlaw_js_extracts_iso_map():
    js = 'var currentData2 = { creditText: \'x\', countries: {"SG":[1,3,1,1,1],"MY":[1,1,1,1,1]} };let statistics=[];'
    data = parse_cyberlaw_js(js)
    assert data == {"SG": [1, 3, 1, 1, 1], "MY": [1, 1, 1, 1, 1]}


def test_parse_cyberlaw_js_returns_empty_on_garbage():
    assert parse_cyberlaw_js("no countries here") == {}


def _by_ind(signals):
    return {s.indicator_id: s for s in signals}


def test_data_protection_reads_privacy_column():
    sigs = _by_ind(unctad_data_protection("SG", INDS, data=FIX))
    assert set(sigs) == {"P7-I1", "P7-I4"}
    assert sigs["P7-I1"].presence is Presence.yes
    assert sigs["P7-I4"].presence is Presence.yes
    assert sigs["P7-I1"].source_name.startswith("UNCTAD Data Protection")
    assert sigs["P7-I1"].primary_law_name == ""  # UNCTAD gives presence, not a name


def test_cyberlaw_reads_both_columns():
    sigs = _by_ind(unctad_cyberlaw("MY", INDS, data=FIX))
    assert set(sigs) == {"P7-I1", "P7-I2"}
    assert sigs["P7-I1"].presence is Presence.no   # MY privacy = 3
    assert sigs["P7-I2"].presence is Presence.yes  # MY cybercrime = 1


def test_cybercrime_column_maps_to_cybersecurity_only():
    # Guide names UNCTAD Cybercrime Legislation Worldwide specifically under P7-I2;
    # the old P7-I3/P7-I5 inference was dropped 2026-06-22 (presence != evidence).
    sigs = _by_ind(unctad_cybercrime("AU", INDS, data=FIX))
    assert set(sigs) == {"P7-I2"}
    assert sigs["P7-I2"].presence is Presence.yes  # AU cybercrime = 1


def test_unknown_presence_code_maps_to_unknown():
    sigs = _by_ind(unctad_data_protection("AU", INDS, data=FIX))  # AU privacy = 0
    assert sigs["P7-I1"].presence is Presence.unknown


def test_respects_in_scope_indicator_filter():
    only_cyber = [i for i in INDS if i.submission_id == "P7-I2"]
    sigs = unctad_cybercrime("AU", only_cyber, data=FIX)
    assert {s.indicator_id for s in sigs} == {"P7-I2"}


def test_economy_absent_from_dataset_yields_nothing():
    assert unctad_data_protection("ZZ", INDS, data=FIX) == []


def test_adapter_indicators_are_subset_of_config():
    """Guard against drift: config `indicators` is the guide's attachment for the
    source; an adapter emits the SUBSET it can actually read from the dataset (e.g.
    the Cyberlaw Tracker is attached Pillar-7-wide but only carries privacy +
    cybercrime columns). So emitted must be a non-empty subset of config."""
    by_key = {s.key: s for s in load_secondary_sources()}
    adapters = {
        "unctad_data_protection": unctad_data_protection,
        "unctad_cyberlaw": unctad_cyberlaw,
        "unctad_cybercrime": unctad_cybercrime,
    }
    for key, adapter in adapters.items():
        emitted = {s.indicator_id for s in adapter("SG", INDS, data=FIX)}
        assert emitted and emitted <= set(by_key[key].indicators), key
