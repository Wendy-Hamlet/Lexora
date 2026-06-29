"""Unit tests for the regime-2 brute relevance judge (pure logic, no network)."""
from __future__ import annotations

from lexora.classify.brute_judge import (
    BruteJudge,
    _parse,
    brute_enabled,
    make_brute_judge,
)
from lexora.models.indicator import RDTIIIndicator


def _ind(sub: str) -> RDTIIIndicator:
    n = sub.split("-")[0][1]  # "P6-I4" -> "6"
    return RDTIIIndicator(
        rdtii_id=f"{n}.{sub[-1]}", submission_id=sub, pillar=int(n),
        name=f"name {sub}", description=f"desc {sub}",
    )


INDS = [_ind(s) for s in ("P6-I1", "P7-I1", "P7-I5")]


def _judge() -> BruteJudge:
    return BruteJudge(base_url="https://x/v1", api_key="k", passes=2)


def test_parse_handles_plain_fenced_and_trailing_prose():
    assert _parse('{"verdicts":[]}') == {"verdicts": []}
    assert _parse('```json\n{"verdicts":[{"indicator_id":"P7-I5"}]}\n```')["verdicts"][0]["indicator_id"] == "P7-I5"
    assert _parse('here you go {"a":1} done')["a"] == 1
    assert _parse("") is None
    assert _parse("no json at all") is None


def test_make_brute_judge_inert_when_disabled(monkeypatch):
    monkeypatch.delenv("LEXORA_BRUTE_JUDGE", raising=False)
    assert make_brute_judge() is None  # disabled by default
    # enabled but no API config -> still inert (never raises)
    monkeypatch.setenv("LEXORA_BRUTE_JUDGE", "1")
    monkeypatch.delenv("LEXORA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LEXORA_LLM_API_KEY", raising=False)
    assert make_brute_judge() is None
    assert brute_enabled() is True


def test_make_brute_judge_uses_separate_brute_model(monkeypatch):
    monkeypatch.setenv("LEXORA_BRUTE_JUDGE", "1")
    monkeypatch.setenv("LEXORA_LLM_BASE_URL", "https://x/v1")
    monkeypatch.setenv("LEXORA_LLM_API_KEY", "k")
    monkeypatch.setenv("LEXORA_LLM_MODEL", "gpt-5.4")  # reasoning backend
    monkeypatch.delenv("LEXORA_BRUTE_MODEL", raising=False)
    j = make_brute_judge()
    assert j is not None and j.model == "deepseek-v4-flash"  # brute stays non-reasoning


def test_relevant_unions_passes(monkeypatch):
    j = _judge()
    seq = [{"P7-I5"}, {"P7-I1"}]  # two passes disagree -> union keeps both
    monkeypatch.setattr(j, "_one", lambda system, text, temp: seq.pop(0))
    assert j.relevant("x" * 500, INDS) == {"P7-I5", "P7-I1"}


def test_relevant_empty_on_short_text_or_total_failure(monkeypatch):
    j = _judge()
    assert j.relevant("short", INDS) == set()  # below length floor, no call
    monkeypatch.setattr(j, "_one", lambda system, text, temp: None)  # every pass fails
    assert j.relevant("x" * 500, INDS) == set()


def test_subset_returns_none_on_failure_else_filters(monkeypatch):
    j = _judge()
    monkeypatch.setattr(j, "_one", lambda system, text, temp: None)
    assert j.subset("x" * 500, INDS) is None  # caller falls back to attribution
    monkeypatch.setattr(j, "_one", lambda system, text, temp: {"P7-I5"})
    out = j.subset("x" * 500, INDS)
    assert [i.submission_id for i in out] == ["P7-I5"]
