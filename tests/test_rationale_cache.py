"""Rationale cache — the layer that removes the reason the pitch demo was crippled.

A full Malaysia run cost 2,545 s with ``--rationale-llm`` and 554 s without, because every
rationale was a fresh live call. The flag was dropped from the full-economy commands for
speed, and on 2026-08-03 a full Singapore run was shown to judges with a 100%-template
Mapping Rationale column. Caching removes the reason to ever drop the flag again.

These are offline: a fake client stands in for the endpoint and counts calls.
"""
from __future__ import annotations

from lexora.cite import rationale as R
from lexora.cite.rationale_cache import RationaleCache, system_fingerprint
from lexora.models.clause import CanonicalSpan, Clause
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import LegalSystem, SourceProfile

_TEXT = ("An organisation must not transfer personal data to a country or territory "
         "outside Singapore except where the protection is comparable.")


def _indicator(sid: str = "P6-I4") -> RDTIIIndicator:
    return RDTIIIndicator(rdtii_id="6.4", submission_id=sid, pillar=6,
                          name="Conditional flow regimes",
                          description="Transfer permitted subject to conditions.")


def _profile() -> SourceProfile:
    return SourceProfile(jurisdiction="Singapore", iso_code="SG", primary_language="en",
                         legal_system=LegalSystem.common, keywords_by_indicator={})


def _clause(text: str = _TEXT) -> Clause:
    return Clause(clause_id="d::s26", document_id="d", structural_path="S. 26",
                  span=CanonicalSpan(span_id="d::s26.s", document_id="d",
                                     char_start=0, char_end=len(text), text=text))


class _CountingClient:
    model = "glm-5.2"

    def __init__(self, rationale: str = "Section 26 conditions outbound transfers.",
                 notes: str = "") -> None:
        self._r, self._notes, self.calls = rationale, notes, 0

    def chat(self, system, user, json_schema=None):  # noqa: ANN001
        self.calls += 1
        return {"rationale": self._r, "notes": self._notes}


class _BoomClient:
    model = "glm-5.2"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, system, user, json_schema=None):  # noqa: ANN001
        self.calls += 1
        raise RuntimeError("backend down")


def _gen(tmp_path, client):
    return R.RationaleGenerator(client=client, cache=RationaleCache(tmp_path / "r.sqlite"))


def test_the_same_question_is_asked_once(tmp_path):
    gen = _gen(tmp_path, _CountingClient())
    first, _ = gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    second, _ = gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert first == second
    assert gen._client.calls == 1
    assert (gen._cache.hits, gen._cache.misses) == (1, 1)


def test_a_different_question_is_a_different_key(tmp_path):
    gen = _gen(tmp_path, _CountingClient())
    gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    gen.generate(_indicator("P7-I1"), _profile(), _clause(), "S. 26")   # other indicator
    gen.generate(_indicator(), _profile(), _clause(_TEXT + " Amended."), "S. 26")  # text
    gen.generate(_indicator(), _profile(), _clause(), "S. 27")          # other locator
    assert gen._client.calls == 4


def test_editing_the_system_prompt_voids_every_stored_answer(tmp_path, monkeypatch):
    """How you ask is part of what you ask. A reworded instruction can change the answer,
    so it must change the key -- otherwise the cache serves rationales written under
    instructions we no longer give."""
    gen = _gen(tmp_path, _CountingClient())
    gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert gen._client.calls == 1

    monkeypatch.setattr(R, "_SYSTEM", R._SYSTEM + "\nWrite in the present tense.")
    gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert gen._client.calls == 2


def test_the_raw_answer_is_stored_so_a_guard_change_is_measurable(tmp_path, monkeypatch):
    """The design decision this cache turns on.

    The guards (copy check, length, score talk) are OURS, not the model's, and whether the
    6-word copy check misfires on formulaic legal prose is open work. If the cache stored
    the ACCEPTED rationale, a relaxed guard would leave every cached row as a template and
    the experiment would silently measure nothing. Storing the raw response means the
    guards re-run on read -- so the same stored answer is rejected under the old guard and
    accepted under the new one, with no second call to the model.
    """
    monkeypatch.setattr(R, "copies_provision", lambda *a, **k: True)
    gen = _gen(tmp_path, _CountingClient())
    out, _ = gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert out == R.template_rationale(_indicator(), _profile(), _clause(), "S. 26")
    assert gen.fallback_reasons == {"copied_provision": 1}
    assert gen._client.calls == 1

    monkeypatch.setattr(R, "copies_provision", lambda *a, **k: False)
    out2, _ = gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert out2 == "Section 26 conditions outbound transfers."   # the model's words now ship
    assert gen.llm_used == 1
    assert gen._client.calls == 1                                # ...and it cost nothing


def test_a_backend_failure_is_never_stored(tmp_path):
    """Caching an outage would turn one bad minute into a permanently templated row."""
    gen = _gen(tmp_path, _BoomClient())
    gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert gen._client.calls == 2
    assert gen._cache.writes == 0
    assert gen.fallback_reasons == {"backend_error": 2}


def test_an_empty_answer_is_stored_because_it_is_an_answer(tmp_path):
    gen = _gen(tmp_path, _CountingClient(rationale=""))
    gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert gen._client.calls == 1
    assert gen.fallback_reasons == {"empty": 2}


def test_coverage_answers_the_question_before_the_run(tmp_path):
    cache = RationaleCache(tmp_path / "r.sqlite")
    gen = R.RationaleGenerator(client=_CountingClient(), cache=cache)
    gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert cache.coverage("glm-5.2", system_fingerprint(R._SYSTEM)) == (1, 1)
    assert cache.coverage("some-other-model", system_fingerprint(R._SYSTEM)) == (0, 1)


def test_no_cache_still_works(tmp_path):
    """The generator must never require a cache: an unwritable store costs money, not
    correctness."""
    gen = R.RationaleGenerator(client=_CountingClient(), cache=None)
    out, _ = gen.generate(_indicator(), _profile(), _clause(), "S. 26")
    assert out == "Section 26 conditions outbound transfers."
    assert gen._client.calls == 1
