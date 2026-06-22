"""Linklaters 'Data Protected' adapter (WS-S) — offline, fixture-driven.

Section TEXT is injected via ``data=`` (``use_llm=False`` exercises the regex +
foreign-marker fallback). The key behaviour under test is the jurisdiction
cross-check that drops Linklaters' mis-served bodies (the ---malaysia page renders
EU/Malta content).
"""
from __future__ import annotations

from pathlib import Path

from lexora.collect.secondary import load_secondary_sources
from lexora.collect.secondary.linklaters import (
    extract_laws_for_country,
    linklaters_data_protection,
    linklaters_law_text,
)
from lexora.indicators import load_indicators
from lexora.models.secondary import Presence

REPO = Path(__file__).resolve().parent.parent
INDS = load_indicators(REPO / "configs" / "rdtii_indicators.yaml")

_SG_HTML = """
<html><body>
<h3>National Legislation</h3>
<p><strong>General data protection laws</strong></p>
<p>The Personal Data Protection Act 2012 ("PDPA").</p>
<p>In addition, certain sector-specific laws such as the Banking Act 1970 and the
Securities and Futures Act 2001 include provisions relating to personal data.</p>
</body></html>
"""

_SG_SECTION = (
    'The Personal Data Protection Act 2012 ("PDPA"). In addition, certain sector-specific '
    "laws such as the Banking Act 1970 and the Securities and Futures Act 2001 include "
    "provisions relating to the protection of certain personal data."
)

# The mis-served Malaysia body: Linklaters renders the EU GDPR / a Maltese act.
_MY_MISSERVED = (
    "The General Data Protection Regulation (EU) (2016/679) (\"GDPR\"). The Maltese Data "
    "Protection Act 2018 implements the GDPR in Malta."
)


def test_law_text_pulls_national_legislation_prose():
    txt = linklaters_law_text(_SG_HTML)
    assert "Personal Data Protection Act 2012" in txt
    assert "Banking Act 1970" in txt


def test_adapter_emits_p7_signals_for_clean_jurisdiction():
    sigs = linklaters_data_protection("SG", INDS, data=_SG_SECTION, use_llm=False)
    assert {s.indicator_id for s in sigs} == {"P7-I1", "P7-I4"}
    assert all(s.presence is Presence.yes for s in sigs)
    names = {s.primary_law_name for s in sigs}
    assert "Personal Data Protection Act 2012" in names


def test_misserved_body_is_dropped_by_jurisdiction_guard():
    # EU-GDPR-led body for Malaysia -> nothing (accuracy over recall).
    assert extract_laws_for_country(_MY_MISSERVED, "Malaysia", use_llm=False) == []
    assert linklaters_data_protection("MY", INDS, data=_MY_MISSERVED, use_llm=False) == []


def test_foreign_marker_laws_are_filtered():
    laws = extract_laws_for_country(
        'The Privacy Act 2020 and the Maltese Data Protection Act 2018.',
        "New Zealand", use_llm=False,
    )
    assert "Privacy Act 2020" in laws
    assert all("Maltese" not in n for n in laws)  # foreign demonym dropped


def test_adapter_respects_in_scope_filter():
    only_i1 = [i for i in INDS if i.submission_id == "P7-I1"]
    sigs = linklaters_data_protection("SG", INDS, data=_SG_SECTION, use_llm=False)
    sigs1 = linklaters_data_protection("SG", only_i1, data=_SG_SECTION, use_llm=False)
    assert {s.indicator_id for s in sigs1} == {"P7-I1"}
    assert {s.indicator_id for s in sigs} == {"P7-I1", "P7-I4"}


def test_empty_text_yields_nothing():
    assert linklaters_data_protection("SG", INDS, data="", use_llm=False) == []


def test_config_lists_linklaters_with_subset_indicators():
    by_key = {s.key: s for s in load_secondary_sources()}
    assert "linklaters" in by_key
    sigs = linklaters_data_protection("SG", INDS, data=_SG_SECTION, use_llm=False)
    emitted = {s.indicator_id for s in sigs}
    assert emitted and emitted <= set(by_key["linklaters"].indicators)
