"""Unit tests for the per-clause verdict cache (pure logic, no network).

The saving is not what these guard. What they guard is that a cached verdict can never
outlive the question that produced it: change the model, the indicator definitions, or the
clause text, and the model must be asked again.
"""
from __future__ import annotations

import pytest

from lexora.classify.judge_cache import JudgeCache, cache_enabled, prompt_fingerprint
from lexora.classify.verifier import _PER_CLAUSE_SYSTEM, Verifier
from lexora.models.clause import CanonicalSpan, Clause
from lexora.models.indicator import RDTIIIndicator


def catalogue_fingerprint(indicators, system: str = _PER_CLAUSE_SYSTEM) -> str:
    """Render the real template with a sentinel clause, exactly as the Verifier does."""
    return prompt_fingerprint(system, Verifier._clause_prompt(_clause(""), indicators))


def _ind(sub: str, long_def: str = "long definition") -> RDTIIIndicator:
    n = sub.split("-")[0][1]
    return RDTIIIndicator(
        rdtii_id=f"{n}.{sub[-1]}", submission_id=sub, pillar=int(n),
        name=f"name {sub}", description=f"desc {sub}", long_definition=long_def,
    )


def _clause(text: str = "A person shall not transfer personal data outside Malaysia.") -> Clause:
    return Clause(
        clause_id="c1", document_id="d1", structural_path="Section 129",
        section_number="129",
        span=CanonicalSpan(span_id="s1", document_id="d1", char_start=0,
                           char_end=len(text), text=text),
    )


INDS = [_ind("P6-I4"), _ind("P7-I5")]


@pytest.fixture()
def cache(tmp_path):
    c = JudgeCache(tmp_path / "judge.sqlite")
    yield c
    c.close()


def test_roundtrip(cache):
    fp = catalogue_fingerprint(INDS)
    k = cache.key("m1", fp, _clause())
    assert cache.get(k) is None          # miss
    cache.put(k, "m1", {"P6-I4"})
    assert cache.get(k) == {"P6-I4"}     # hit
    assert cache.hits == 1 and cache.misses == 1


def test_empty_verdict_is_a_real_answer(cache):
    """"supports nothing" is a verdict, not a miss -- it must not be re-asked forever."""
    fp = catalogue_fingerprint(INDS)
    k = cache.key("m1", fp, _clause())
    cache.put(k, "m1", set())
    assert cache.get(k) == set()
    assert cache.hits == 1


def test_editing_a_long_definition_invalidates(cache):
    """The long definition carries the boundary rules. Reword it and every verdict taken
    under the old wording is stale."""
    before = catalogue_fingerprint(INDS)
    after = catalogue_fingerprint([_ind("P6-I4"), _ind("P7-I5", "REWORDED boundary rule")])
    assert before != after

    k_before = cache.key("m1", before, _clause())
    cache.put(k_before, "m1", {"P6-I4"})
    assert cache.get(cache.key("m1", after, _clause())) is None


def test_model_and_clause_text_are_in_the_key(cache):
    fp = catalogue_fingerprint(INDS)
    cache.put(cache.key("m1", fp, _clause()), "m1", {"P6-I4"})

    assert cache.get(cache.key("m2", fp, _clause())) is None            # other model
    assert cache.get(cache.key("m1", fp, _clause("Other text."))) is None  # other clause


def test_indicator_order_is_part_of_the_prompt():
    """The catalogue is rendered in list order, so a reordering is a different prompt."""
    assert catalogue_fingerprint([_ind("P6-I4"), _ind("P7-I5")]) != \
        catalogue_fingerprint([_ind("P7-I5"), _ind("P6-I4")])


def test_changing_the_instructions_invalidates():
    """The system prompt is part of the question. Reword it and the old verdicts are stale."""
    assert catalogue_fingerprint(INDS) != \
        catalogue_fingerprint(INDS, system=_PER_CLAUSE_SYSTEM + " Prefer abstaining.")


def test_changing_the_template_invalidates():
    """The killer case. On 2026-07-14 the clause moved from the top of the prompt to the
    bottom (to expose a cacheable prefix). Same indicators, same clause -- a fingerprint
    over the indicator DATA alone would have been identical, and the cache would have
    answered the new prompt with verdicts taken under the old one. The fingerprint hashes
    the RENDERED template, so a reordering like that one changes the key."""
    rendered_now = Verifier._clause_prompt(_clause(), INDS)
    # the pre-2026-07-14 layout: clause first, catalogue after
    lines = [f"CLAUSE ({_clause().structural_path}):", _clause().span.text, "", "INDICATORS:"]
    for i in INDS:
        lines.append(f"- {i.submission_id} ({i.name}): {i.description}"
                     f"\n  Definition (scope + boundaries): {i.long_definition}")
    rendered_before = "\n".join(lines)

    assert rendered_now != rendered_before
    assert prompt_fingerprint(_PER_CLAUSE_SYSTEM, rendered_now) != \
        prompt_fingerprint(_PER_CLAUSE_SYSTEM, rendered_before)


def test_failures_are_never_cached(tmp_path, monkeypatch):
    """A backend error yields None. Storing it would turn one outage into a permanently
    missing citation on every future run."""
    class Boom:
        model = "m1"

        def chat(self, *a, **k):
            raise RuntimeError("backend down")

    c = JudgeCache(tmp_path / "judge.sqlite")
    v = Verifier(Boom(), mode="per_clause", cache=c)
    assert v.judge_clause(_clause(), INDS) is None
    assert v.error_count == 1
    assert c.writes == 0          # nothing stored

    # and a later healthy run still asks the model
    assert c.get(c.key("m1", catalogue_fingerprint(INDS), _clause())) is None
    c.close()


def test_hit_short_circuits_the_llm(tmp_path):
    calls = {"n": 0}

    class Counting:
        model = "m1"

        def chat(self, *a, **k):
            calls["n"] += 1
            return {"indicators": ["P6-I4"]}

    c = JudgeCache(tmp_path / "judge.sqlite")
    v = Verifier(Counting(), mode="per_clause", cache=c)

    assert v.judge_clause(_clause(), INDS) == {"P6-I4"}
    assert calls["n"] == 1
    # second ask: same question, no call
    assert v.judge_clause(_clause(), INDS) == {"P6-I4"}
    assert calls["n"] == 1
    assert c.hits == 1
    c.close()


def test_cache_enabled_switch(monkeypatch):
    monkeypatch.delenv("LEXORA_JUDGE_CACHE", raising=False)
    assert cache_enabled() is True
    monkeypatch.setenv("LEXORA_JUDGE_CACHE", "0")
    assert cache_enabled() is False
