"""Unit tests for AU catalogue enumeration as a discovery source (no network)."""
from __future__ import annotations

import json

import lexora.collect.au_enumerate as ae
from lexora.collect.au_enumerate import enumerate_au_candidates, enumerate_enabled
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import SourceType

INDS = [
    RDTIIIndicator(rdtii_id="7.5", submission_id="P7-I5", pillar=7,
                   name="govt access", description="d"),
    RDTIIIndicator(rdtii_id="6.4", submission_id="P6-I4", pillar=6,
                   name="cross-border", description="d"),
]


class _Judge:
    """Stand-in BruteJudge: flags by a fixed id->indicators map, and records the
    shell text it was given (keyed by the marker prefix) so a test can assert it
    judged the shell, and that cached ids are NOT re-judged."""

    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.seen: dict[str, str] = {}

    def relevant(self, text, indicators):
        tid = text.split("|", 1)[0]
        self.seen[tid] = text
        return set(self.verdicts.get(tid, []))


def _wire(monkeypatch, catalogue, shells, verdicts):
    monkeypatch.setattr(ae, "au_act_catalogue", lambda **k: catalogue)
    monkeypatch.setattr(ae, "_shell_text", lambda tid, **k: shells.get(tid, ""))
    monkeypatch.setattr(ae.time, "sleep", lambda *_: None)
    return _Judge(verdicts)


def test_enumerate_enabled(monkeypatch):
    monkeypatch.delenv("LEXORA_AU_ENUMERATE", raising=False)
    assert enumerate_enabled() is False
    monkeypatch.setenv("LEXORA_AU_ENUMERATE", "1")
    assert enumerate_enabled() is True


def test_shell_judge_gates_and_attributes(monkeypatch, tmp_path):
    catalogue = [
        {"id": "A1", "name": "Privacy Act 1988", "isPrincipal": True},
        {"id": "A2", "name": "ASIO Act 1979", "isPrincipal": True},
        {"id": "A3", "name": "Boring Act 1900", "isPrincipal": True},
        {"id": "X9", "name": "Some Regulation", "isPrincipal": False},
    ]
    shells = {"A1": "A1|titles", "A2": "A2|titles", "A3": "A3|titles"}
    verdicts = {"A1": ["P6-I4"], "A2": ["P7-I5"], "A3": []}  # A3 judged irrelevant
    judge = _wire(monkeypatch, catalogue, shells, verdicts)

    out = enumerate_au_candidates(
        INDS, judge=judge, source_type=SourceType.primary,
        known_instruments=["Privacy Act 1988"],
        verdict_cache=str(tmp_path / "v.jsonl"),
    )
    by_id = {r.url.split("/")[-2]: r for r in out}
    assert set(by_id) == {"A1", "A2"}                 # A3 (empty verdict) dropped
    assert by_id["A1"].indicator_hits == ["P6-I4"]
    assert by_id["A2"].indicator_hits == ["P7-I5"]
    assert by_id["A1"].discovery_tag == "KNOWN"       # name matches known instrument
    assert by_id["A2"].discovery_tag == "NEW"
    assert by_id["A1"].via == "enumerate"
    assert "X9" not in judge.seen                      # non-principal never enumerated


def test_verdict_cache_resumes_without_refetch(monkeypatch, tmp_path):
    catalogue = [
        {"id": "A1", "name": "Privacy Act 1988", "isPrincipal": True},
        {"id": "A2", "name": "ASIO Act 1979", "isPrincipal": True},
    ]
    shells = {"A2": "A2|titles"}  # A1 has no shell available -> must come from cache
    judge = _wire(monkeypatch, catalogue, shells, {"A2": ["P7-I5"]})
    vpath = tmp_path / "v.jsonl"
    vpath.write_text(json.dumps({"id": "A1", "name": "Privacy Act 1988",
                                 "relevant": ["P6-I4"]}) + "\n", encoding="utf-8")

    out = enumerate_au_candidates(
        INDS, judge=judge, verdict_cache=str(vpath),
    )
    by_id = {r.url.split("/")[-2]: r for r in out}
    assert set(by_id) == {"A1", "A2"}
    assert by_id["A1"].indicator_hits == ["P6-I4"]    # reused from cache
    assert "A1" not in judge.seen                      # cached id never re-judged
    assert "A2" in judge.seen                          # only the uncached id judged
