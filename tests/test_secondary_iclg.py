"""ICLG adapter (WS-S, S-4 second source) — offline. Reuses DLA's extract_laws,
so this focuses on ICLG-specific bits: statute-paragraph targeting, the ISO->slug
gate (no Malaysia chapter), and signal emission."""
from __future__ import annotations

from pathlib import Path

from lexora.collect.secondary import load_secondary_sources, to_discovery_seeds
from lexora.collect.secondary.iclg import iclg_data_protection, iclg_law_text
from lexora.indicators import load_indicators
from lexora.models.secondary import Presence

REPO = Path(__file__).resolve().parent.parent
INDS = load_indicators(REPO / "configs" / "rdtii_indicators.yaml")

_SG = ("The Personal Data Protection Act 2012 (PDPA) is the principal data protection "
       "legislation in Singapore. The PDPA was amended by the Personal Data Protection "
       "(Amendment) Act 2020 to strengthen accountability.")

_HTML = f"""<html><body>
<p>Data Protection Laws and Regulations 2025 covers common issues in 27 jurisdictions.</p>
<p>{_SG}</p>
</body></html>"""


def test_iclg_law_text_skips_preface_keeps_statute_para():
    txt = iclg_law_text(_HTML)
    assert "Personal Data Protection Act 2012" in txt
    assert "27 jurisdictions" not in txt  # generic preface (no statute year) skipped


def test_adapter_emits_named_signals():
    sigs = iclg_data_protection("SG", INDS, data=_SG, use_llm=False)
    assert {s.indicator_id for s in sigs} == {"P7-I1", "P7-I4"}
    assert all(s.presence is Presence.yes and s.primary_law_name for s in sigs)
    assert all(s.source_name.startswith("ICLG") for s in sigs)
    assert "Personal Data Protection Act 2012" in to_discovery_seeds(sigs)


def test_uncovered_iso_is_skipped_live_path():
    # An ISO with no slug never fetches and yields nothing (data not injected).
    assert iclg_data_protection("ZZ", INDS) == []


def test_empty_text_yields_nothing():
    assert iclg_data_protection("SG", INDS, data="", use_llm=False) == []


def test_config_lists_iclg_with_matching_indicators():
    by_key = {s.key: s for s in load_secondary_sources()}
    assert "iclg" in by_key
    sigs = iclg_data_protection("SG", INDS, data=_SG, use_llm=False)
    assert {s.indicator_id for s in sigs} == set(by_key["iclg"].indicators)
