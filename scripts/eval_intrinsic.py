"""Intrinsic generalization probe (roadmap G-2) — gold-free, layer-attributing.

The blind-test problem: before we hand-tune a new country we want to know, with
*no* human labels, whether the pipeline produces anything usable there. This
harness parses a legal text and reports intrinsic metrics that expose the known
non-Latin "hard zero" and attribute it to a specific layer:

  clauses      clause yield — how many structural units the parser recovered.
               0 on a non-Latin doc => the PARSER is blind to the numbering (G-1c).
  uniq%        clause_id uniqueness rate — collisions silently clobber citations.
  verbatim%    span text == source[char_start:char_end] after NFKC+ws normalize —
               the verbatim contract checked at parse time, script-independent.
  tok%         fraction of clauses whose body yields >=1 token under the CURRENT
               tokenizer. ~0 on a CJK/Thai doc => the ASCII TOKENIZER cliff (G-1a):
               BM25 sees empty documents, so retrieval is a hard zero downstream.
  q-ok%        fraction of indicators whose expanded query has >=1 token (the
               query side is universal English, so this stays high — proving the
               cliff is on the corpus side, not the query side).
  cov%         fraction of indicators that get >=1 BM25 candidate over this doc —
               the downstream consequence of tok% collapsing.

Built-in fixtures (configs/eval/intrinsic/manifest.yaml) ship two Latin controls
(should pass everything) and three civil-law probes that isolate the tokenizer
cliff (ASCII numbering + Chinese body), the parser cliff (native 第N条), and both
(Thai มาตรา). Deterministic and offline, so it runs in CI and is re-run after each
G-1 fix to watch the cliff close. `--text-file` / `--url` accept a real document
for the leave-one-country-out blind test once a candidate country is chosen.

Usage:
    python scripts/eval_intrinsic.py                 # all built-in fixtures
    python scripts/eval_intrinsic.py -f common-law-dotted-en
    python scripts/eval_intrinsic.py --text-file some_law.txt --language zh
    python scripts/eval_intrinsic.py --json          # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from lexora.cite.validator import normalize  # noqa: E402
from lexora.classify.retrieval import (  # noqa: E402
    _LEAD_SECTION,
    _expand_query,
    _tokenize,
    build_index,
)
from lexora.extract.pdf_text_extractor import PdfPage  # noqa: E402
from lexora.indicators import load_indicators  # noqa: E402
from lexora.models.source import LegalSystem, SourceProfile  # noqa: E402
from lexora.structure.legal_parser import parse_structure  # noqa: E402

INDICATORS = REPO / "configs" / "rdtii_indicators.yaml"
FIXTURE_DIR = REPO / "configs" / "eval" / "intrinsic"
MANIFEST = FIXTURE_DIR / "manifest.yaml"


@dataclass
class IntrinsicResult:
    fixture: str
    language: str
    script: str
    clauses: int
    uniq_pct: float
    verbatim_pct: float
    token_pct: float
    query_ok_pct: float
    coverage_pct: float
    note: str = ""


def _bare_profile(language: str) -> SourceProfile:
    """A country-agnostic profile: zero hand-tuned synonyms, just the language.

    Using a blank profile is the point — the blind test measures what the pipeline
    does with *no* per-country knowledge, so query expansion falls back to the
    universal RDTII indicator text only.
    """
    return SourceProfile(
        jurisdiction="probe",
        iso_code="zz",
        primary_language=language,
        legal_system=LegalSystem.civil,
    )


def evaluate_text(
    fixture_id: str,
    text: str,
    language: str,
    script: str,
    indicators,
    note: str = "",
) -> IntrinsicResult:
    """Parse one text and compute the gold-free intrinsic metrics."""
    page = PdfPage(
        page_number=1, text=text, char_start=0, char_end=len(text), has_text_layer=True
    )
    clauses = parse_structure(fixture_id, [page])
    n = len(clauses)

    if n == 0:
        # Parser cliff: nothing to score downstream. Query side is still reported
        # so the table shows the query channel is fine and the loss is upstream.
        profile = _bare_profile(language)
        q_ok = sum(1 for ind in indicators if _expand_query(ind, profile)) if indicators else 0
        q_pct = round(100 * q_ok / len(indicators), 1) if indicators else 0.0
        return IntrinsicResult(
            fixture_id, language, script, 0, 0.0, 0.0, 0.0, q_pct, 0.0, note
        )

    ids = [c.clause_id for c in clauses]
    uniq_pct = round(100 * len(set(ids)) / n, 1)

    verbatim_ok = sum(
        1
        for c in clauses
        if normalize(text[c.span.char_start : c.span.char_end]) == normalize(c.span.text)
    )
    verbatim_pct = round(100 * verbatim_ok / n, 1)

    # Strip the leading section marker ("13.", "26A ") before tokenizing: an ASCII
    # section number on a Chinese body otherwise scores tok%=100 for a single
    # useless ["13"] token and masks the tokenizer cliff. We want *content* tokens.
    token_ok = sum(1 for c in clauses if _tokenize(_LEAD_SECTION.sub("", c.span.text.lstrip())))
    token_pct = round(100 * token_ok / n, 1)

    profile = _bare_profile(language)
    index = build_index(clauses)
    q_ok = 0
    covered = 0
    for ind in indicators:
        terms = _expand_query(ind, profile)
        if terms:
            q_ok += 1
        # Coverage requires a real lexical hit: query() returns top-k even when
        # every BM25 score is 0 (no term matched), so gate on a positive score.
        scores = index.bm25_scores(terms) if terms else None
        if scores is not None and len(scores) and max(scores) > 0:
            covered += 1
    q_pct = round(100 * q_ok / len(indicators), 1) if indicators else 0.0
    cov_pct = round(100 * covered / len(indicators), 1) if indicators else 0.0

    return IntrinsicResult(
        fixture_id, language, script, n, uniq_pct, verbatim_pct, token_pct, q_pct, cov_pct, note
    )


def _load_manifest() -> list[dict]:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or {}
    return data.get("fixtures", [])


def _print_table(results: list[IntrinsicResult]) -> None:
    print("\nIntrinsic generalization probe (G-2) — gold-free, 0 = cliff\n")
    header = (
        f"{'fixture':<26} {'lang':<4} {'script':<6} "
        f"{'clauses':>7} {'uniq%':>6} {'verb%':>6} {'tok%':>6} {'q-ok%':>6} {'cov%':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.fixture:<26} {r.language:<4} {r.script:<6} "
            f"{r.clauses:>7} {r.uniq_pct:>6} {r.verbatim_pct:>6} "
            f"{r.token_pct:>6} {r.query_ok_pct:>6} {r.coverage_pct:>6}"
        )
    print("-" * len(header))
    print(
        "tok%/cov% near 0 with high q-ok% = the corpus-side non-Latin cliff "
        "(tokenizer G-1a / parser G-1c), not a query problem."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-f", "--fixture", help="run a single manifest fixture by id")
    ap.add_argument("--text-file", type=Path, help="probe an external text file instead")
    ap.add_argument("--language", default="en", help="language tag for --text-file")
    ap.add_argument("--script", default="latin", help="script tag for --text-file")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    indicators = load_indicators(INDICATORS)
    results: list[IntrinsicResult] = []

    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8")
        results.append(
            evaluate_text(args.text_file.stem, text, args.language, args.script, indicators)
        )
    else:
        for spec in _load_manifest():
            if args.fixture and spec["id"] != args.fixture:
                continue
            text = (FIXTURE_DIR / spec["path"]).read_text(encoding="utf-8")
            results.append(
                evaluate_text(
                    spec["id"], text, spec["language"], spec["script"],
                    indicators, spec.get("note", ""),
                )
            )

    if not results:
        print("No fixtures matched.", file=sys.stderr)
        raise SystemExit(1)

    if args.json:
        print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
    else:
        _print_table(results)


if __name__ == "__main__":
    main()
