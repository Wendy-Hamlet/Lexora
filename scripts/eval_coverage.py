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

from lexora.collect.discovery import discover_for_indicators, discover_secondary  # noqa: E402
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


def _matches(gold_name: str, d: dict) -> bool:
    """True if one discovered hit ``d`` matches a gold instrument by fuzzy name OR
    by shared Act identifier (filename titles like 'Act 854.pdf' defeat name fuzz
    but carry the Act number)."""
    hay = f"{d['title']} {d['url']}"
    if int(fuzz.token_set_ratio(gold_name.lower(), hay.lower())) >= _MATCH_THRESHOLD:
        return True
    gold_ids = _act_ids(gold_name)
    return bool(gold_ids and (gold_ids & _act_ids(hay)))


def _is_covered(gold_name: str, discovered: list[dict]) -> bool:
    """A gold instrument is covered if any discovered hit matches it."""
    return any(_matches(gold_name, d) for d in discovered)


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
            # Secondary portals (regulator sites) add subsidiary instruments —
            # guidance / codes / notices — that never appear on the statute portal.
            hits = hits + discover_secondary(profile, indicators, timeout=60.0)
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


def _rank_report_one(iso: str, gold: list[str], *, budget: int, rank_budget: int,
                     use_semantic: bool) -> dict:
    """G-6.1 rank-before-truncation diagnostic (discovery layer).

    Runs discovery ONCE with a large ``rank_budget`` so nothing is truncated, then
    records each gold instrument's actual rank in the primary ranked list (the same
    ``known + rest`` ordering the live budget cut uses). Each gold lands in one of
    four buckets, which point at different fixes:

      kept       rank <= budget                  — already returned live
      truncated  budget < rank <= rank_budget    — FOUND but the budget cut drops it
                                                    (fix: budget split / re-rank, NOT retrieval)
      secondary  only matched on a regulator site — subsidiary instrument, not a
                                                    primary-portal ranking problem
      not_found  matched nowhere                  — retrieval / connector / query gap

    Pure diagnostic: it never changes the live ``budget``, only measures where the
    gold sits relative to it.
    """
    profile = load_profile(REPO / "configs" / "jurisdictions" / f"{iso}.yaml")
    indicators = load_indicators(INDICATORS)
    portal = profile.portals[0]

    error = None
    primary: list[dict] = []
    secondary: list[dict] = []
    try:
        hits = discover_for_indicators(
            portal, indicators, budget=rank_budget, timeout=60.0,
            force_browser=portal.fetch_method is FetchMethod.playwright,
            known_instruments=profile.known_instruments,
            known_instrument_ids=profile.known_instrument_ids,
            use_semantic=use_semantic,
        )
        primary = [{"title": h.title, "url": h.url, "tag": h.discovery_tag,
                    "score": round(h.score, 3)} for h in hits]
        secondary = [{"title": h.title, "url": h.url}
                     for h in discover_secondary(profile, indicators, timeout=60.0)]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    rows, buckets = _bucket_gold(gold, primary, secondary, budget)
    return {"iso": iso, "error": error, "budget": budget, "rank_budget": rank_budget,
            "primary_n": len(primary), "secondary_n": len(secondary),
            "buckets": buckets, "rows": rows}


def _bucket_gold(gold: list[str], primary: list[dict], secondary: list[dict],
                 budget: int) -> tuple[list[dict], dict[str, int]]:
    """Pure core of the rank report: bucket each gold instrument against the live
    ``budget`` given the full ranked ``primary`` list and the ``secondary`` hits.

    ``primary`` must be in the live ``known + rest`` order so an index is the rank
    the live cut would apply. Returns (rows, bucket-counts)."""
    rows = []
    buckets = {"kept": 0, "truncated": 0, "secondary": 0, "not_found": 0}
    for inst in gold:
        rank = next((i + 1 for i, d in enumerate(primary) if _matches(inst, d)), None)
        if rank is not None and rank <= budget:
            bucket = "kept"
        elif rank is not None:
            bucket = "truncated"
        elif any(_matches(inst, d) for d in secondary):
            bucket = "secondary"
        else:
            bucket = "not_found"
        buckets[bucket] += 1
        rows.append({"instrument": inst, "rank": rank, "bucket": bucket,
                     "is_agreement": _is_agreement(inst)})
    return rows, buckets


def _print_rank_report(report: list[dict]) -> None:
    print("\nDiscovery rank-before-truncation (G-6.1) — gold position vs the budget cut")
    print("buckets: kept(<=budget)  truncated(found but cut)  secondary(regulator site)  not_found")
    for e in report:
        if e["error"]:
            print(f"\n[{e['iso']}] ERROR: {e['error']}")
            continue
        b = e["buckets"]
        print(f"\n[{e['iso']}] budget={e['budget']} rank_budget={e['rank_budget']} "
              f"primary={e['primary_n']} secondary={e['secondary_n']}")
        print(f"   kept {b['kept']}  truncated {b['truncated']}  "
              f"secondary {b['secondary']}  not_found {b['not_found']}")
        # The truncated bucket is the actionable headline: found-but-cut gold is
        # pure budget/ranking loss, recoverable without touching retrieval.
        for r in sorted(e["rows"], key=lambda r: (r["bucket"] != "truncated", r["rank"] or 1e9)):
            tag = " (agreement)" if r["is_agreement"] else ""
            rk = f"#{r['rank']}" if r["rank"] is not None else "--"
            print(f"   {r['bucket']:<10}{rk:>5}  {r['instrument'][:60]}{tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--jurisdiction", default="all", help="sg|au|my|all")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true", help="list gold only, no network")
    ap.add_argument("--no-semantic", action="store_true",
                    help="disable the dense crosswalk/re-rank (keyword-only baseline)")
    ap.add_argument("--rank-report", action="store_true",
                    help="G-6.1: run discovery with a large budget and bucket each "
                         "gold instrument as kept / truncated / secondary / not_found")
    ap.add_argument("--rank-budget", type=int, default=200,
                    help="diagnostic budget for --rank-report (large, ~no truncation)")
    args = ap.parse_args()

    if args.rank_report:
        if not os.environ.get("LEXORA_LIVE"):
            print("note: set LEXORA_LIVE=1 to run live discovery for the rank report.")
        gold = _load_gold()
        isos = ["sg", "au", "my"] if args.jurisdiction == "all" else [args.jurisdiction.lower()]
        report = [_rank_report_one(iso, gold.get(ISO_TO_COUNTRY.get(iso, iso), []),
                                   budget=args.budget, rank_budget=args.rank_budget,
                                   use_semantic=not args.no_semantic) for iso in isos]
        out = OUT.parent / "eval_coverage_rank.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _print_rank_report(report)
        print(f"\n(full results -> {out.relative_to(REPO)})")
        return

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
