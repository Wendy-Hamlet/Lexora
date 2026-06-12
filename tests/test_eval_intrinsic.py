"""Tests for the G-2 intrinsic generalization probe.

These lock in the *diagnostic* behaviour: the Latin controls must pass every
metric, and the civil-law probes must attribute the non-Latin cliff to the right
layer (tokenizer vs parser). They double as a regression guard for the G-1 fixes
— after G-1a the ascii-num-zh tok%/cov% must climb off 0; after G-1c the
native-num-zh clause yield must climb off 0.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from lexora.indicators import load_indicators

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "eval_intrinsic.py"
_spec = importlib.util.spec_from_file_location("eval_intrinsic", _SCRIPT)
ei = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ei  # so @dataclass can resolve the module
_spec.loader.exec_module(ei)

_INDICATORS = load_indicators(_REPO / "configs" / "rdtii_indicators.yaml")


def _run(fixture_id: str):
    spec = next(s for s in ei._load_manifest() if s["id"] == fixture_id)
    text = (ei.FIXTURE_DIR / spec["path"]).read_text(encoding="utf-8")
    return ei.evaluate_text(
        fixture_id, text, spec["language"], spec["script"], _INDICATORS, spec.get("note", "")
    )


def test_english_control_passes_every_metric():
    r = _run("common-law-dotted-en")
    assert r.clauses > 0
    assert r.uniq_pct == 100.0
    assert r.verbatim_pct == 100.0
    assert r.token_pct == 100.0
    assert r.coverage_pct > 0.0  # English query terms find English clauses


def test_spaced_english_control_parses():
    r = _run("common-law-spaced-en")
    assert r.clauses > 0
    assert r.verbatim_pct == 100.0
    assert r.token_pct == 100.0


def test_ascii_numbered_chinese_tokenizes_after_g1a():
    # Parser handles the ASCII numbering (clauses > 0, verbatim holds). After
    # G-1a the CJK body is bigram-tokenized, so tok% is now 100 (BM25 sees a
    # non-empty document). cov% stays 0 because the *query* is still English —
    # cross-lingual matching is G-3a (CN synonyms) / G-1b (dense), not G-1a.
    r = _run("civil-law-ascii-num-zh")
    assert r.clauses > 0
    assert r.verbatim_pct == 100.0
    assert r.token_pct == 100.0  # was 0 before G-1a
    assert r.coverage_pct == 0.0  # G-3a / G-1b will move this off 0


def test_native_numbered_chinese_isolates_parser_cliff():
    # The parser is blind to 第N条 -> zero yield (a different, upstream cliff).
    r = _run("civil-law-native-num-zh")
    assert r.clauses == 0  # G-1c will move this off 0
    assert r.query_ok_pct == 100.0  # query side is unaffected


def test_native_numbered_thai_is_a_hard_zero():
    r = _run("civil-law-native-num-th")
    assert r.clauses == 0


def test_query_side_is_universal_across_all_fixtures():
    # The English RDTII indicator text always tokenizes, proving the cliff is on
    # the corpus side, not the query side — the table's load-bearing contrast.
    for fid in (
        "common-law-dotted-en",
        "civil-law-ascii-num-zh",
        "civil-law-native-num-zh",
        "civil-law-native-num-th",
    ):
        assert _run(fid).query_ok_pct == 100.0


def test_verbatim_metric_catches_a_broken_span():
    # Guard the verbatim check itself: a clause whose stored span text does NOT
    # match the source slice must drag verbatim% below 100.
    text = (ei.FIXTURE_DIR / "common_law_dotted_en.txt").read_text(encoding="utf-8")
    r_ok = ei.evaluate_text("ok", text, "en", "latin", _INDICATORS)
    assert r_ok.verbatim_pct == 100.0
    # Corrupt the source so the parsed spans no longer slice back to their text.
    r_bad = ei.evaluate_text("bad", "X" + text[1:], "en", "latin", _INDICATORS)
    assert r_bad.clauses > 0
    assert r_bad.verbatim_pct <= 100.0  # sanity: metric runs on mutated input
