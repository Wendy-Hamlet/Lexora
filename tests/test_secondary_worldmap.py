"""World Map of Encryption adapter (WS-S) — offline, fixture-driven.

The map is one server-rendered page split into per-country sections. Tests use a
small HTML fixture for the splitter and inject section TEXT via ``data=`` for the
adapter (``use_llm=False`` exercises the deterministic regex fallback).
"""
from __future__ import annotations

from pathlib import Path

from lexora.collect.secondary import load_secondary_sources
from lexora.collect.secondary.worldmap import (
    _fetch_section_text,
    country_section_text,
    parse_country_sections,
    world_map_encryption,
)
from lexora.indicators import load_indicators
from lexora.models.secondary import Presence

REPO = Path(__file__).resolve().parent.parent
INDS = load_indicators(REPO / "configs" / "rdtii_indicators.yaml")

_HTML = """
<html><body>
<h4 class="h2">Singapore</h4>
<h5>General right to encryption</h5>
<p>There is no general prohibition on the use of encryption in Singapore.</p>
<h5>Other restrictions</h5>
<p>Under the Criminal Procedure Code 2010 and the Computer Misuse Act 1993 a police
officer may require a person to provide decryption information to access encrypted data.</p>
<h4 class="h2">Slovakia</h4>
<h5>General right to encryption</h5>
<p>Encryption is permitted under the Act on Electronic Communications of 2011.</p>
</body></html>
"""

_SG_SECTION = (
    "Under the Criminal Procedure Code 2010 and the Computer Misuse Act 1993 a police "
    "officer may require a person to provide decryption information to access encrypted data."
)


def test_parse_country_sections_splits_by_country():
    secs = parse_country_sections(_HTML)
    assert set(secs) == {"singapore", "slovakia"}
    assert "Computer Misuse Act 1993" in secs["singapore"]
    assert "Slovakia" not in secs["singapore"]  # block stops at the next country


def test_country_section_text_picks_one_country():
    assert "Computer Misuse Act" in country_section_text(_HTML, "Singapore")
    assert country_section_text(_HTML, "Nowhere") == ""


def test_adapter_emits_p7i5_with_law_names():
    sigs = world_map_encryption("SG", INDS, data=_SG_SECTION, use_llm=False)
    assert sigs and all(s.indicator_id == "P7-I5" for s in sigs)
    assert all(s.presence is Presence.yes for s in sigs)
    names = {s.primary_law_name for s in sigs}
    assert "Computer Misuse Act 1993" in names  # NAMES a law -> USE-1 seed
    assert all("Encryption" in s.source_name for s in sigs)


def test_adapter_respects_in_scope_filter():
    only_other = [i for i in INDS if i.submission_id == "P7-I1"]  # not P7-I5
    assert world_map_encryption("SG", only_other, data=_SG_SECTION, use_llm=False) == []


def test_empty_or_nameless_text_yields_nothing():
    assert world_map_encryption("SG", INDS, data="", use_llm=False) == []
    assert world_map_encryption("SG", INDS, data="No statute named here.", use_llm=False) == []


def test_unmapped_iso_fetches_nothing_offline():
    # An ISO with no country mapping returns "" before any network call.
    assert _fetch_section_text("ZZ") == ""


def test_config_lists_world_map_with_subset_indicators():
    by_key = {s.key: s for s in load_secondary_sources()}
    assert "world_map_encryption" in by_key
    sigs = world_map_encryption("SG", INDS, data=_SG_SECTION, use_llm=False)
    emitted = {s.indicator_id for s in sigs}
    assert emitted and emitted <= set(by_key["world_map_encryption"].indicators)
