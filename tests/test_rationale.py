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
    def __init__(self, rationale: str, notes: str = "") -> None:
        self._r = rationale
        self._notes = notes
        self.calls = 0

    def chat(self, system, user, json_schema=None):  # noqa: ANN001
        self.calls += 1
        return {"rationale": self._r, "notes": self._notes}


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
    text, note = gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert text == template_rationale(_indicator(), _profile(), _clause(), "S. 26")
    assert note == ""  # template path has no own-knowledge channel
    assert make_rationale_generator(use_llm=False)._client is None


def test_llm_rationale_used_when_clean():
    gen = RationaleGenerator(_FakeClient(
        "Section 26 conditions overseas personal-data transfers on comparable protection, mapping to P6-I4."))
    out, _note = gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert out.startswith("Section 26 conditions overseas")
    assert gen.llm_used == 1 and gen.fallbacks == 0


def test_own_knowledge_note_is_tagged_and_kept_out_of_rationale():
    gen = RationaleGenerator(_FakeClient(
        "Section 26 conditions overseas transfers on comparable protection, mapping to P6-I4.",
        notes="This section was amended in 2020 (not in the provided text).",
    ))
    out, note = gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert "amended in 2020" not in out                 # background stays out of the answer
    assert "amended in 2020" in note                     # ...but is preserved for review
    assert "own knowledge" in note.lower() and "NOT used as answer" in note


def test_llm_output_copying_provision_is_rejected():
    # Reproduces a 6+ word run from the provision verbatim -> must fall back.
    leak = "It says an organisation must not transfer personal data to a country or territory outside Singapore."
    assert copies_provision(leak, _TEXT)
    gen = RationaleGenerator(_FakeClient(leak))
    out, _note = gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert out == template_rationale(_indicator(), _profile(), _clause(), "S. 26")
    assert gen.fallbacks == 1 and gen.llm_used == 0


def test_llm_overlength_output_falls_back():
    # Length expressed relative to the constant, not hardcoded: the cap moved from 300 to
    # 1000 on 2026-08-09 and a test that pins a literal length silently stops testing the
    # guard it is named after.
    gen = RationaleGenerator(_FakeClient("x " * (RATIONALE_MAX_CHARS // 2 + 10)))
    out, _note = gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert len(out) <= RATIONALE_MAX_CHARS
    assert gen.fallbacks == 1


def test_llm_backend_error_falls_back_and_counts():
    gen = RationaleGenerator(_BoomClient())
    out, _note = gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert out == template_rationale(_indicator(), _profile(), _clause(), "S. 26")
    assert gen.error_count == 1 and gen.last_error_type == "RuntimeError"


def test_the_tool_does_not_score_and_neither_does_its_notes_column():
    """Lexora's stated scope is ESCAP's Step 1: find the provision and cite it.

    The round-1 submission nevertheless shipped 184 rows whose Notes column speculated
    about a score, 45 asserting a value outright, and one that managed "whether the
    framework is scored 0 or 0". A policy judge reads that column. The prompt now
    forbids it; this is the part that enforces it, because a prompt is a request.
    """
    from lexora.cite.rationale import strip_score_talk

    gen = RationaleGenerator(_FakeClient(
        "Section 26 conditions overseas transfers on comparable protection, mapping to P6-I4.",
        notes="This is from Singapore's PDPA, a horizontal regime. "
              "The indicator score for Singapore would be 0.",
    ))
    _out, note = gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert "horizontal regime" in note        # the useful half survives...
    assert "score" not in note.lower()        # ...the out-of-scope half does not
    assert gen.score_talk_stripped == 1

    # Sentence-granular, so what is left still reads as prose.
    assert strip_score_talk("Alpha holds. It would score 0. Beta holds.") == \
        "Alpha holds. Beta holds."
    # A note that is ONLY about scoring leaves nothing behind.
    assert strip_score_talk("The indicator scores 0.") == ""


def test_a_rationale_that_scores_falls_back_to_the_template():
    """Out of scope in the ANSWER column is worse than out of scope in a note."""
    gen = RationaleGenerator(_FakeClient(
        "Section 26 restricts overseas transfers, so Singapore should score 0 on P6-I4."))
    out, _note = gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert out == template_rationale(_indicator(), _profile(), _clause(), "S. 26")
    assert gen.llm_used == 0 and gen.fallbacks == 1


# --- which guard fired ----------------------------------------------------------------
#
# 29.1% of the Round-1 submission's rationales were template while the LLM layer was ON,
# and nobody could say why: one `fallbacks` counter was bumped from five places for five
# reasons. Four of the five are OUR guards rejecting the model's answer rather than the
# model failing to give one, and that distinction decides whether the fix is a better
# prompt or a less trigger-happy guard. These pin the attribution.

_COPY = "transfer personal data to a country or territory"   # 8 words lifted from _TEXT
# Comfortably over the cap, whatever the cap currently is. "Mechanism. " shares no 6-word
# run with _TEXT, so it trips `too_long` and nothing else.
_LONG = "Mechanism. " * (RATIONALE_MAX_CHARS // 11 + 4)


def test_each_fallback_reason_is_named():
    cases = {
        "empty": "",
        "score_talk": "Section 26 conditions outbound flows and should score 0.",
        "too_long": _LONG,                                    # over the cap, no copied run
        "copied_provision": f"Section 26 says an organisation must not {_COPY}.",
    }
    for expected, output in cases.items():
        gen = RationaleGenerator(_FakeClient(output))
        gen.generate(_indicator(), _profile(), _clause(), "S. 26")
        assert gen.fallbacks == 1
        assert gen.fallback_reasons == {expected: 1}, f"{expected} mis-attributed"

    gen = RationaleGenerator(_BoomClient())
    gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert gen.fallback_reasons == {"backend_error": 1}


def test_overlapping_reasons_are_all_recorded_not_just_the_first():
    """An `or` chain can only ever name the first reason, and the first reason is not
    the actionable one: a rationale that copies AND runs long is not fixed by relaxing
    the copy check."""
    gen = RationaleGenerator(_FakeClient(
        f"It provides that an organisation must not {_COPY}. " + _LONG))
    gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert gen.fallback_reasons == {"copied_provision+too_long": 1}


def test_reason_counts_sum_to_the_fallback_total():
    """Keyed by the whole set, so the two numbers can never drift apart — otherwise a
    percentage computed from either one is quietly wrong."""
    gen = RationaleGenerator(_FakeClient(""))
    for _ in range(3):
        gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    gen._client = _FakeClient(f"An organisation must not {_COPY}.")
    gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert sum(gen.fallback_reasons.values()) == gen.fallbacks == 4
    assert gen.fallback_summary() == "empty 3, copied_provision 1"
