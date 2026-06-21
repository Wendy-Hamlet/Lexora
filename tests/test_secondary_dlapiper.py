"""DLA Piper adapter (WS-S, S-4) — offline. LLM path uses a fake client; the
deterministic regex is exercised directly and as the fallback.
"""
from __future__ import annotations

from pathlib import Path

from lexora.collect.secondary import load_secondary_sources, to_discovery_seeds
from lexora.collect.secondary.dlapiper import (
    dla_piper_data_protection,
    extract_laws,
    extract_laws_regex,
    law_section_text,
)
from lexora.indicators import load_indicators
from lexora.models.secondary import Presence

REPO = Path(__file__).resolve().parent.parent
INDS = load_indicators(REPO / "configs" / "rdtii_indicators.yaml")

_SG_TEXT = ("Singapore enacted the Personal Data Protection Act of 2012 (No. 26 of 2012) "
            "on October 15, 2012, and it was subsequently amended via the Personal Data "
            "Protection (Amendment) Act 2020 (together, the Act). The Act has "
            "extraterritorial effect and applies to organizations in Singapore.")

_HTML = f"""<html><body>
<h2>Data protection laws in Singapore</h2>
<p>{_SG_TEXT}</p>
<p>short</p>
</body></html>"""


class _FakeClient:
    def __init__(self, laws=None, raise_it=False):
        self._laws = laws or []
        self._raise = raise_it

    def chat(self, system, user, json_schema=None):
        if self._raise:
            raise RuntimeError("backend down")
        return {"laws": self._laws}


def test_law_section_text_pulls_the_prose():
    txt = law_section_text(_HTML)
    assert "Personal Data Protection Act of 2012" in txt
    assert "short" not in txt  # the <120-char paragraph is skipped


def test_regex_extractor_gets_principal_and_amendment():
    laws = extract_laws_regex(_SG_TEXT)
    assert "Personal Data Protection Act of 2012 (No. 26 of 2012)" in laws
    assert "Personal Data Protection (Amendment) Act 2020" in laws


def test_regex_drops_bare_junk():
    laws = extract_laws_regex("Under the Act, the Amendment Act applies. The Privacy Act.")
    assert "Amendment Act" not in laws
    assert all(j not in [x.lower() for x in laws] for j in ("act", "the act"))


def test_llm_path_used_when_client_supplied():
    laws = extract_laws(_SG_TEXT, use_llm=True,
                        client=_FakeClient(laws=["Law No. 27 of 2022 concerning Personal Data Protection"]))
    assert laws == ["Law No. 27 of 2022 concerning Personal Data Protection"]  # non-"Act" name kept


def test_llm_failure_falls_back_to_regex():
    laws = extract_laws(_SG_TEXT, use_llm=True, client=_FakeClient(raise_it=True))
    assert "Personal Data Protection (Amendment) Act 2020" in laws  # regex result


def test_adapter_emits_named_signals_for_p7i1_and_p7i4():
    sigs = dla_piper_data_protection("SG", INDS, data=_SG_TEXT, use_llm=False)
    assert sigs
    assert {s.indicator_id for s in sigs} == {"P7-I1", "P7-I4"}
    assert all(s.presence is Presence.yes for s in sigs)
    assert all(s.primary_law_name for s in sigs)  # the seed is populated
    assert all(s.source_name.startswith("DLA Piper") for s in sigs)


def test_adapter_respects_in_scope_filter():
    only_i1 = [i for i in INDS if i.submission_id == "P7-I1"]
    sigs = dla_piper_data_protection("SG", only_i1, data=_SG_TEXT, use_llm=False)
    assert {s.indicator_id for s in sigs} == {"P7-I1"}


def test_adapter_activates_use1_discovery_seeds():
    sigs = dla_piper_data_protection("SG", INDS, data=_SG_TEXT, use_llm=False)
    seeds = to_discovery_seeds(sigs)
    # the named statutes become discovery seeds (deduped across the two indicators)
    assert "Personal Data Protection (Amendment) Act 2020" in seeds
    assert len(seeds) == len(set(s.lower() for s in seeds))  # deduped


def test_empty_text_yields_nothing():
    assert dla_piper_data_protection("SG", INDS, data="", use_llm=False) == []


def test_config_lists_dla_piper_with_matching_indicators():
    by_key = {s.key: s for s in load_secondary_sources()}
    assert "dla_piper" in by_key
    sigs = dla_piper_data_protection("SG", INDS, data=_SG_TEXT, use_llm=False)
    assert {s.indicator_id for s in sigs} == set(by_key["dla_piper"].indicators)
