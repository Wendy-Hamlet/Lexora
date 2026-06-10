"""Coverage evaluation harness (P0 multi-instrument map).

Quantifies how much of the official RDTII master dataset's Pillar-6/7 instrument
*family* Lexora's discovery now reaches — the metric P0 is meant to move, since
one indicator maps to many instruments (flagship law + sectoral statutes +
subsidiary instruments + trade agreements), not just the top hit.

The gold fixture (`configs/eval/legal_inventory_p67.csv`) is the distilled P6/P7
slice of the official long-form "Legal Inventory" (a SECONDARY reference per the
9 June ESCAP email; the master dataset stays primary). The gold is used ONLY to
score recall — it is never fed into discovery (pure-discovery design).

For each jurisdiction we run `discover_for_indicators` live and fuzzy-match each
distinct gold instrument name against the discovered working set. We report
coverage overall and excluding the trade-agreement family (P6-I5, which the Guide
sources from third-party treaty databases and Lexora deliberately does not crawl).
NEW = a discovered instrument that matches no gold row (the 20/40-point capability).

Usage:
    LEXORA_LIVE=1 python scripts/eval_coverage.py -j sg
    python scripts/eval_coverage.py -j all --dry-run   # list gold, no network
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexora.collect.discovery import discover_for_indicators  # noqa: E402
from lexora.collect.profile_loader import load_profile  # noqa: E402
from lexora.indicators import load_indicators  # noqa: E402
from lexora.models.source import FetchMethod  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GOLD = REPO / "configs" / "eval" / "legal_inventory_p67.csv"
INDICATORS = REPO / "configs" / "rdtii_indicators.yaml"
OUT = REPO / "outputs" / "eval_coverage.json"

ISO_TO_COUNTRY = {"sg": "Singapore", "au": "Australia", "my": "Malaysia"}
_MATCH_THRESHOLD = 85  # token_set_ratio >= this -> the gold instrument was found
# Trade-agreement family (P6-I5) — out of scope by design (third-party sourced).
_AGREEMENT_MARKERS = ("agreement", "partnership", "cptpp", "rcep", "aanzfta",
                      "depa", "ksdpa", "free trade", "cbpr")


def _is_agreement(name: str) -> bool:
    n = name.lower()
    return any(m in n for m in _AGREEMENT_MARKERS)


def _load_gold() -> dict[str, list[str]]:
    """country -> distinct instrument names (P6 + P7)."""
    by_country: dict[str, set[str]] = defaultdict(set)
    with GOLD.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_country[row["country"]].add(row["instrument"].strip())
    return {c: sorted(v) for c, v in by_country.items()}


def _act_ids(text: str) -> set[str]:
    """Malaysian Act identifiers from a name/URL/filename, e.g. '(Act 854)',
    'act=854', 'Act 709.pdf', 'A1727'. 4-digit years (19xx/20xx) are excluded so
    'Personal Data Protection Act 2010' doesn't read 2010 as an Act number."""
    ids = set()
    for m in re.findall(r"(?:act[=\s]*)([a-z]?\d{1,4})", text.lower()):
        digits = m.lstrip("abcdefghijklmnopqrstuvwxyz")
        if not m[0].isalpha() and re.fullmatch(r"(?:19|20)\d{2}", digits):
            continue  # a year, not an Act number
        ids.add(m)
    return ids


def _is_covered(gold_name: str, discovered: list[dict]) -> bool:
    """A gold instrument is covered if any discovered hit matches it by fuzzy name
    OR by shared Act identifier (filename titles like 'Act 854.pdf' defeat name
    fuzz but carry the Act number)."""
    gold_l = gold_name.lower()
    gold_ids = _act_ids(gold_name)
    for d in discovered:
        hay = f"{d['title']} {d['url']}"
        if int(fuzz.token_set_ratio(gold_l, hay.lower())) >= _MATCH_THRESHOLD:
            return True
        if gold_ids and (gold_ids & _act_ids(hay)):
            return True
    return False


def _best_match(name: str, hay: list[str]) -> int:
    # Case-insensitive: discovered titles are often UPPERCASE / filenames while
    # the gold names are mixed-case.
    nl = name.lower()
    return max((int(fuzz.token_set_ratio(nl, h.lower())) for h in hay), default=0)


def _eval_one(iso: str, gold: list[str], *, budget: int, dry_run: bool,
              use_semantic: bool = True) -> dict:
    profile = load_profile(REPO / "configs" / "jurisdictions" / f"{iso}.yaml")
    indicators = load_indicators(INDICATORS)
    portal = profile.portals[0]

    discovered: list[dict] = []
    error = None
    if not dry_run:
        try:
            hits = discover_for_indicators(
                portal, indicators, budget=budget, timeout=60.0,
                force_browser=portal.fetch_method is FetchMethod.playwright,
                known_instruments=profile.known_instruments,
                known_instrument_ids=profile.known_instrument_ids,
                use_semantic=use_semantic,
            )
            discovered = [{"title": h.title, "url": h.url, "tag": h.discovery_tag,
                           "score": round(h.score, 2), "hits": h.indicator_hits} for h in hits]
        except Exception as exc:  # network/portal failure — report, don't crash
            error = f"{type(exc).__name__}: {exc}"

    covered, missed = [], []
    for inst in gold:
        (covered if _is_covered(inst, discovered) else missed).append(inst)

    statute_gold = [g for g in gold if not _is_agreement(g)]
    statute_covered = [g for g in covered if not _is_agreement(g)]
    # NEW = discovered instrument matching no gold row at all.
    new_found = [d for d in discovered
                 if _best_match(d["title"], gold) < _MATCH_THRESHOLD]

    return {
        "iso": iso, "error": error,
        "gold_total": len(gold), "covered": len(covered),
        "gold_statute_total": len(statute_gold), "statute_covered": len(statute_covered),
        "discovered_n": len(discovered), "new_found": len(new_found),
        "missed": missed, "discovered": discovered,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--jurisdiction", default="all", help="sg|au|my|all")
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true", help="list gold only, no network")
    ap.add_argument("--no-semantic", action="store_true",
                    help="disable the dense crosswalk/re-rank (keyword-only baseline)")
    args = ap.parse_args()

    if not args.dry_run and not os.environ.get("LEXORA_LIVE"):
        print("note: set LEXORA_LIVE=1 to run live discovery (or use --dry-run).")

    gold = _load_gold()
    isos = ["sg", "au", "my"] if args.jurisdiction == "all" else [args.jurisdiction.lower()]

    report = []
    for iso in isos:
        country = ISO_TO_COUNTRY.get(iso, iso)
        report.append(_eval_one(iso, gold.get(country, []), budget=args.budget,
                                dry_run=args.dry_run, use_semantic=not args.no_semantic))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def pct(a: int, b: int) -> str:
        return f"{a}/{b} ({100*a/b:.0f}%)" if b else "0/0"

    print(f"\nCoverage eval (budget {args.budget}) — discovery recall vs P6/P7 gold")
    print(f"{'iso':<5}{'coverage(all)':<18}{'coverage(statute)':<20}{'discovered':<12}{'NEW'}")
    print("-" * 70)
    for e in report:
        if e["error"]:
            print(f"{e['iso']:<5}ERROR: {e['error']}")
            continue
        print(f"{e['iso']:<5}{pct(e['covered'], e['gold_total']):<18}"
              f"{pct(e['statute_covered'], e['gold_statute_total']):<20}"
              f"{e['discovered_n']:<12}{e['new_found']}")
    print("-" * 70)
    print("coverage(statute) excludes the P6-I5 trade-agreement family (out of scope).")
    print(f"(full results -> {OUT.relative_to(REPO)})")


if __name__ == "__main__":
    main()
