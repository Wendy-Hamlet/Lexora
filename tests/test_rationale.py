"""Mapping Rationale generation — deterministic template + guarded LLM layer.

Offline: a fake LLM client stands in for the endpoint so these pin the contract
(template floor, ≤300 chars, verbatim-copy guard, graceful fallback) with no
network.
"""
from __future__ import annotations

from lexora.cite.rationale import (
    RATIONALE_MAX_CHARS,
    RationaleGenerator,
    copies_provision,
    make_rationale_generator,
    template_rationale,
)
from lexora.models.clause import CanonicalSpan, Clause
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import LegalSystem, SourceProfile


def _indicator() -> RDTIIIndicator:
    return RDTIIIndicator(
        rdtii_id="6.4", submission_id="P6-I4", pillar=6,
        name="Conditional flow regimes",
        description="Cross-border transfer permitted subject to conditions.",
    )


def _profile() -> SourceProfile:
    return SourceProfile(
        jurisdiction="Singapore", iso_code="SG", primary_language="en",
        legal_system=LegalSystem.common,
        keywords_by_indicator={"6.4": {"en": ["transfer personal data to a country or territory outside Singapore",
                                              "comparable to the protection"]}},
    )


_TEXT = ("An organisation must not transfer personal data to a country or territory "
         "outside Singapore except where the protection is comparable to the protection "
         "under this Act.")


def _clause(text: str = _TEXT) -> Clause:
    return Clause(clause_id="d::s26", document_id="d", structural_path="S. 26",
                  span=CanonicalSpan(span_id="d::s26.s", document_id="d",
                                     char_start=0, char_end=len(text), text=text))


class _FakeClient:
    def __init__(self, rationale: str) -> None:
        self._r = rationale
        self.calls = 0

    def chat(self, system, user, json_schema=None):  # noqa: ANN001
        self.calls += 1
        return {"rationale": self._r}


class _BoomClient:
    def chat(self, system, user, json_schema=None):  # noqa: ANN001
        raise RuntimeError("backend down")


def test_template_cites_provision_indicator_and_matched_phrases():
    r = template_rationale(_indicator(), _profile(), _clause(), "S. 26")
    assert r.startswith("S. 26 maps to P6-I4 (Conditional flow regimes).")
    assert "transfer personal data to a country or territory outside Singapore" in r
    assert len(r) <= RATIONALE_MAX_CHARS


def test_template_falls_back_to_base_when_no_phrase_matches():
    prof = _profile()
    r = template_rationale(_indicator(), prof, _clause("An unrelated provision with no concept terms."), "S. 99")
    assert r == "S. 99 maps to P6-I4 (Conditional flow regimes)."


def test_template_never_exceeds_limit_with_many_long_phrases():
    prof = SourceProfile(
        jurisdiction="X", iso_code="XX", primary_language="en", legal_system=LegalSystem.common,
        keywords_by_indicator={"6.4": {"en": [f"very long curated concept phrase number {i}" for i in range(40)]}},
    )
    text = " ".join(f"very long curated concept phrase number {i}" for i in range(40))
    r = template_rationale(_indicator(), prof, _clause(text), "S. 1")
    assert len(r) <= RATIONALE_MAX_CHARS
    assert r.endswith(".")  # whole-phrase assembly, not a mid-word cut


def test_inert_generator_returns_template():
    gen = RationaleGenerator(client=None)
    assert gen.generate(_indicator(), _profile(), _clause(), "S. 26") == \
        template_rationale(_indicator(), _profile(), _clause(), "S. 26")
    assert make_rationale_generator(use_llm=False)._client is None


def test_llm_rationale_used_when_clean():
    gen = RationaleGenerator(_FakeClient(
        "Section 26 conditions overseas personal-data transfers on comparable protection, mapping to P6-I4."))
    out = gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert out.startswith("Section 26 conditions overseas")
    assert gen.llm_used == 1 and gen.fallbacks == 0


def test_llm_output_copying_provision_is_rejected():
    # Reproduces a 6+ word run from the provision verbatim -> must fall back.
    leak = "It says an organisation must not transfer personal data to a country or territory outside Singapore."
    assert copies_provision(leak, _TEXT)
    gen = RationaleGenerator(_FakeClient(leak))
    out = gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert out == template_rationale(_indicator(), _profile(), _clause(), "S. 26")
    assert gen.fallbacks == 1 and gen.llm_used == 0


def test_llm_overlength_output_falls_back():
    gen = RationaleGenerator(_FakeClient("x " * 200))  # > 300 chars
    out = gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert len(out) <= RATIONALE_MAX_CHARS
    assert gen.fallbacks == 1


def test_llm_backend_error_falls_back_and_counts():
    gen = RationaleGenerator(_BoomClient())
    out = gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert out == template_rationale(_indicator(), _profile(), _clause(), "S. 26")
    assert gen.error_count == 1 and gen.last_error_type == "RuntimeError"
