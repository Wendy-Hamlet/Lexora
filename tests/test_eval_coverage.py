"""Coverage eval harness (P-0) + rank-before-truncation diagnostic (G-6.1) — core.

Loads ``scripts/eval_coverage.py`` (evals live in scripts/, outside the package)
and pins the pure matchers and the rank-bucketing logic. The live discovery path
is network-coupled and not unit-tested here; the diagnostic core that decides
kept / truncated / secondary / not_found is, since that is what a fix acts on.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_coverage.py"
_spec = importlib.util.spec_from_file_location("eval_coverage", _SCRIPT)
ec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ec)


def _hit(title: str, url: str = "https://e/x") -> dict:
    return {"title": title, "url": url}


def test_matches_by_fuzzy_name_and_by_act_number():
    pdpa = "Personal Data Protection Act 2012"
    assert ec._matches(pdpa, _hit("Personal Data Protection Act 2012"))
    # Filename title defeats name fuzz but the Act number carries identity.
    assert ec._matches("Computer Crimes Act (Act 563)", _hit("ACT 563.pdf"))
    assert not ec._matches(pdpa, _hit("Electronic Transactions Act 2010"))


def test_is_covered_any_hit():
    gold = "Personal Data Protection Act 2012"
    assert ec._is_covered(gold, [_hit("misc"), _hit("Personal Data Protection Act 2012")])
    assert not ec._is_covered(gold, [_hit("misc"), _hit("Spam Control Act")])


def test_bucket_gold_separates_truncation_from_recall():
    gold = ["Alpha Act 2001", "Bravo Act 2002", "Charlie Act 2003", "Delta Act 2004"]
    # Alpha at rank 1 (kept), Bravo at rank 5 (truncated under budget 3),
    # Charlie only on a regulator site (secondary), Delta nowhere (not_found).
    primary = [
        _hit("Alpha Act 2001"), _hit("x1"), _hit("x2"), _hit("x3"),
        _hit("Bravo Act 2002"),
    ]
    secondary = [_hit("Charlie Act 2003 guidance")]
    rows, buckets = ec._bucket_gold(gold, primary, secondary, budget=3)
    by_inst = {r["instrument"]: r for r in rows}
    assert by_inst["Alpha Act 2001"]["bucket"] == "kept"
    assert by_inst["Alpha Act 2001"]["rank"] == 1
    assert by_inst["Bravo Act 2002"]["bucket"] == "truncated"
    assert by_inst["Bravo Act 2002"]["rank"] == 5
    assert by_inst["Charlie Act 2003"]["bucket"] == "secondary"
    assert by_inst["Delta Act 2004"]["bucket"] == "not_found"
    assert buckets == {"kept": 1, "truncated": 1, "secondary": 1, "not_found": 1}


def test_bucket_gold_marks_agreement_rows():
    rows, _ = ec._bucket_gold(["Comprehensive and Progressive Agreement for TPP"], [], [], budget=20)
    assert rows[0]["is_agreement"] is True
    assert rows[0]["bucket"] == "not_found"
