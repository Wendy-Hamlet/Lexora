"""Tests for the official RDTII indicator config."""
from __future__ import annotations

from pathlib import Path

from lexora.indicators import load_indicators

INDICATORS_PATH = Path(__file__).resolve().parent.parent / "configs" / "rdtii_indicators.yaml"


def test_loads_nine_regulatory_indicators():
    inds = load_indicators(INDICATORS_PATH)
    # Pillar 6 regulatory: 6.1-6.4 (6.5 is non-regulatory, out of scope); Pillar 7: 7.1-7.5
    assert len(inds) == 9
    assert {i.submission_id for i in inds} == {
        "P6-I1", "P6-I2", "P6-I3", "P6-I4",
        "P7-I1", "P7-I2", "P7-I3", "P7-I4", "P7-I5",
    }


def test_code_mapping_is_consistent():
    inds = {i.submission_id: i for i in load_indicators(INDICATORS_PATH)}
    # P6-I4 must be the conditional-flow indicator and map to rdtii 6.4
    assert inds["P6-I4"].rdtii_id == "6.4"
    assert "conditional flow" in inds["P6-I4"].name.lower()
    # P7-I2 must be the cybersecurity indicator and map to rdtii 7.2
    assert inds["P7-I2"].rdtii_id == "7.2"
    assert "cybersecurity" in inds["P7-I2"].name.lower()
    # the convenience .id property returns the canonical rdtii id
    assert inds["P7-I2"].id == "7.2"


def test_pillars_assigned():
    inds = load_indicators(INDICATORS_PATH)
    assert all(i.pillar == 6 for i in inds if i.submission_id.startswith("P6"))
    assert all(i.pillar == 7 for i in inds if i.submission_id.startswith("P7"))
