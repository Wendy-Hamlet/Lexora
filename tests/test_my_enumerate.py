"""Two-stage Malaysian enumeration — the shortlist, and what it refuses to throw away."""
from __future__ import annotations

import json

import pytest

from lexora.collect import my_inventory
from lexora.collect.my_enumerate import enumerate_my_candidates
from lexora.collect.my_inventory import InventoryEntry


class _Ind:
    def __init__(self, sid):
        self.submission_id = sid


INDICATORS = [_Ind(s) for s in
              ("P6-I1", "P6-I2", "P6-I3", "P6-I4", "P7-I1", "P7-I2", "P7-I3", "P7-I4", "P7-I5")]

INVENTORY = {
    # a framework law: stage 1 alone decides it
    "709": InventoryEntry(act_no="709", title_en="PERSONAL DATA PROTECTION ACT 2010",
                          commencement="15-11-2013", pdf_url="https://x/709.pdf"),
    # horizontal-only: stage 2 reads the contents and confirms
    "593": InventoryEntry(act_no="593", title_en="CRIMINAL PROCEDURE CODE",
                          commencement="1976", pdf_url="https://x/593.pdf"),
    # horizontal-only: stage 2 reads the contents and cuts it
    "874": InventoryEntry(act_no="874", title_en="FINANCE ACT 2025",
                          commencement="2025", pdf_url="https://x/874.pdf"),
    # horizontal-only but UNREADABLE: must survive on the stage-1 verdict
    "571": InventoryEntry(act_no="571", title_en="BANK SIMPANAN NASIONAL BERHAD ACT 1997",
                          commencement="1997", pdf_url="https://x/571.pdf"),
    # nothing at all
    "427": InventoryEntry(act_no="427", title_en="PINEAPPLE INDUSTRY ACT 1957",
                          commencement="1957", pdf_url="https://x/427.pdf"),
    # repealed, and still relevant on its title
    "762": InventoryEntry(act_no="762", title_en="GOODS AND SERVICES TAX ACT 2014",
                          commencement="2015", pdf_url="https://x/762.pdf",
                          repealed_by="805", repealed_by_title="GST (REPEAL) ACT 2018"),
}

TITLE_VERDICTS = {
    "PERSONAL DATA PROTECTION ACT 2010": {"P7-I1", "P6-I4"},
    "CRIMINAL PROCEDURE CODE": {"P7-I5"},
    "FINANCE ACT 2025": {"P7-I3"},
    "BANK SIMPANAN NASIONAL BERHAD ACT 1997": {"P7-I3"},
    "PINEAPPLE INDUSTRY ACT 1957": set(),
    "GOODS AND SERVICES TAX ACT 2014": {"P7-I3"},
}
TOC = {
    "593": "ARRANGEMENT OF SECTIONS\n116B. Access to computerized data\n",
    "874": "ARRANGEMENT OF SECTIONS\n1. Short title\n2. Amendment of Act 53\n",
    "762": "ARRANGEMENT OF SECTIONS\n36. Duty to keep records\n",
}


def _title_judge(prompt, _indicators):
    for name, hits in TITLE_VERDICTS.items():
        if name in prompt:
            return set(hits)
    return set()


def _toc_judge(toc, _indicators):
    if "Access to computerized data" in toc:
        return {"P7-I5"}
    if "Duty to keep records" in toc:
        return {"P7-I3"}
    return set()


@pytest.fixture
def _wired(monkeypatch, tmp_path):
    monkeypatch.setattr(my_inventory, "_CACHE", INVENTORY)
    # The repository ships 1,287 real verdicts so a fresh clone inherits the judged
    # statute book. They must not leak into a test whose whole world is six Acts.
    monkeypatch.setattr("lexora.collect.my_enumerate.SHIPPED_VERDICTS",
                        tmp_path / "no-such-reference.jsonl")

    def fake_toc(url, *, timeout, ocr_engine=None):
        act = url.rsplit("/", 1)[-1].removesuffix(".pdf")
        if act == "571":
            return "", "no-text-layer"
        return TOC.get(act, "ARRANGEMENT OF SECTIONS\n1. Short title\n"), ""

    monkeypatch.setattr("lexora.collect.my_enumerate._toc_from_pdf", fake_toc)
    return tmp_path / "verdicts.jsonl"


def _run(cache, **kw):
    return enumerate_my_candidates(
        INDICATORS, title_judge=_title_judge, toc_judge=_toc_judge,
        verdict_cache=str(cache), ocr=False, log=lambda *_: None, **kw)


def test_the_contents_overrule_the_title(_wired):
    by = {r.law_number: r for r in _run(_wired)}
    # read and confirmed
    assert by["Act 593"].indicator_hits == ["P7-I5"]
    # read and cut: the Finance Act's headings show nothing, so it is gone
    assert "Act 874" not in by
    # never flagged at all
    assert "Act 427" not in by


def test_a_framework_title_is_not_second_guessed(_wired):
    by = {r.law_number: r for r in _run(_wired)}
    assert set(by["Act 709"].indicator_hits) == {"P6-I4", "P7-I1"}


def test_an_act_we_could_not_read_is_kept(_wired):
    """Unreadable is not irrelevant. A filter that silently drops what it failed to open
    is worse than no filter -- the Bank Simpanan Nasional Berhad Act is 63 MB of scan."""
    by = {r.law_number: r for r in _run(_wired)}
    assert by["Act 571"].indicator_hits == ["P7-I3"]


def test_the_portals_own_lifecycle_words_are_carried(_wired):
    by = {r.law_number: r for r in _run(_wired)}
    assert by["Act 762"].status == "REPEALED"
    assert by["Act 709"].status == "IN_FORCE"


def test_known_instruments_are_tagged_not_rediscovered(_wired):
    by = {r.law_number: r for r in _run(_wired, known_instrument_ids={"709": "PDPA 2010"})}
    assert by["Act 709"].discovery_tag == "KNOWN"
    assert by["Act 709"].matched_instrument == "PDPA 2010"
    assert by["Act 593"].discovery_tag == "NEW"


def test_a_second_run_judges_nothing(_wired):
    calls = {"n": 0}

    def counting(prompt, indicators):
        calls["n"] += 1
        return _title_judge(prompt, indicators)

    enumerate_my_candidates(INDICATORS, title_judge=counting, toc_judge=_toc_judge,
                            verdict_cache=str(_wired), ocr=False, log=lambda *_: None)
    first = calls["n"]
    assert first == len(INVENTORY)
    enumerate_my_candidates(INDICATORS, title_judge=counting, toc_judge=_toc_judge,
                            verdict_cache=str(_wired), ocr=False, log=lambda *_: None)
    assert calls["n"] == first, "a cached statute book must cost nothing to re-read"


def test_the_shipped_verdicts_cover_the_statute_book(monkeypatch, tmp_path):
    """The pitch says an auditor can re-derive our numbers. That has to be true for
    someone who is not us, so the judged statute book ships WITH the repository: a fresh
    clone must reach the Malaysian working set without an API key and without paying to
    re-judge 1,287 titles."""
    from lexora.collect.my_enumerate import SHIPPED_VERDICTS, _load_jsonl

    assert SHIPPED_VERDICTS.exists(), "the reference verdict set is not in the repository"
    shipped = _load_jsonl(SHIPPED_VERDICTS)
    assert len(shipped) > 1200
    # Every gold statute must have a verdict on file, whatever that verdict is -- the
    # backstop covers a wrong one, but a MISSING one means the sweep never saw the Act.
    for act in ("53", "563", "588", "593", "709", "747", "807", "854", "A1727"):
        assert act in shipped, f"no shipped verdict for gold Act {act}"

    def _explode(*_a, **_k):
        raise AssertionError("the shipped verdicts did not cover the statute book")

    monkeypatch.setattr(my_inventory, "_CACHE",
                        {a: InventoryEntry(act_no=a, title_en=shipped[a]["title"],
                                           pdf_url=f"https://x/{a}.pdf")
                         for a in list(shipped)[:200]})
    results = enumerate_my_candidates(
        [_Ind(s) for s in ("P6-I1", "P7-I5")], title_judge=_explode, toc_judge=_explode,
        verdict_cache=str(tmp_path / "empty.jsonl"), ocr=False, log=lambda *_: None)
    assert results, "a shipped-cache run produced no candidates at all"


def test_the_cache_is_one_json_object_per_act(_wired):
    _run(_wired)
    lines = [json.loads(x) for x in _wired.read_text(encoding="utf-8").splitlines()]
    assert {ln["act"] for ln in lines} == set(INVENTORY)
    assert all("title_hits" in ln for ln in lines)
