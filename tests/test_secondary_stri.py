"""STRI snapshot adapters (WS-S, S-3) — OECD Digital STRI + WB-WTO STRI.

The indices come from a curated, human-verified snapshot. Tests use a fixture
snapshot for emission; the shipped snapshot is currently unfilled (null), so the
live adapters emit nothing — verified here so we never fabricate index values.
"""
from __future__ import annotations

from pathlib import Path

from lexora.collect.secondary import load_secondary_sources
from lexora.collect.secondary.stri import load_stri_snapshot, oecd_dstri, worldbank_wto_stri
from lexora.indicators import load_indicators
from lexora.models.secondary import Presence

REPO = Path(__file__).resolve().parent.parent
INDS = load_indicators(REPO / "configs" / "rdtii_indicators.yaml")

FIX = {
    "oecd_dstri": {
        "name": "OECD Digital STRI", "scale": "0-1",
        "indicators": ("P6-I1", "P6-I2", "P6-I3", "P6-I4"),
        "economies": {"AU": {"value": 0.21, "as_of": "2024", "source_url": "u"}},
    },
    "worldbank_wto_stri": {
        "name": "WB-WTO STRI", "scale": "0-100",
        "indicators": ("P6-I1", "P6-I2", "P6-I4"),
        "economies": {
            "MY": {"value": 35, "as_of": "2022", "source_url": "u"},
            "SG": {"value": 0, "as_of": "2022", "source_url": "u"},  # explicit no-restriction
        },
    },
}


def test_oecd_emits_p6_presence_from_snapshot():
    sigs = oecd_dstri("AU", INDS, snapshot=FIX)
    assert {s.indicator_id for s in sigs} == {"P6-I1", "P6-I2", "P6-I3", "P6-I4"}
    assert all(s.presence is Presence.yes for s in sigs)  # 0.21 > 0
    assert all("OECD" in s.source_name and s.primary_law_name == "" for s in sigs)  # index, no law name


def test_wb_value_zero_is_presence_no():
    sigs = worldbank_wto_stri("SG", INDS, snapshot=FIX)
    assert sigs and all(s.presence is Presence.no for s in sigs)  # value 0 -> no


def test_wb_covers_three_economies_incl_my():
    assert worldbank_wto_stri("MY", INDS, snapshot=FIX)  # MY present (OECD wouldn't have it)
    assert {s.indicator_id for s in worldbank_wto_stri("MY", INDS, snapshot=FIX)} == {"P6-I1", "P6-I2", "P6-I4"}


def test_unfilled_or_absent_entry_emits_nothing():
    assert oecd_dstri("SG", INDS, snapshot=FIX) == []  # SG absent from OECD snapshot
    null_snap = {"oecd_dstri": {"name": "x", "scale": "", "indicators": ("P6-I1",),
                                "economies": {"AU": {"value": None}}}}
    assert oecd_dstri("AU", INDS, snapshot=null_snap) == []  # null -> skip, never fabricate


def test_respects_in_scope_filter():
    only = [i for i in INDS if i.submission_id == "P6-I2"]
    assert {s.indicator_id for s in oecd_dstri("AU", only, snapshot=FIX)} == {"P6-I2"}


def test_shipped_snapshot_loads_and_is_currently_unfilled():
    snap = load_stri_snapshot()
    assert "oecd_dstri" in snap and "worldbank_wto_stri" in snap
    assert "AU" in snap["oecd_dstri"]["economies"]
    assert set(snap["worldbank_wto_stri"]["economies"]) >= {"SG", "AU", "MY"}
    # honest current state: no fabricated values -> live adapters emit nothing
    assert oecd_dstri("AU", INDS) == []
    assert worldbank_wto_stri("MY", INDS) == []


def test_config_and_snapshot_indicators_agree():
    by_key = {s.key: s for s in load_secondary_sources()}
    snap = load_stri_snapshot()
    for key in ("oecd_dstri", "worldbank_wto_stri"):
        assert set(snap[key]["indicators"]) == set(by_key[key].indicators), key
