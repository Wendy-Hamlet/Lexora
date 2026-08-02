"""Read a submission CSV the way an operator needs to read it, on stage, under time.

PowerShell has Import-Csv; bash has nothing equivalent, and a python one-liner long
enough to do the job is not something anyone types correctly while a judge watches. So
the three questions that actually come up get a command each:

    python scripts/show.py outputs/DEMO_demo.csv
        every row, one line each -- what the demo just produced

    python scripts/show.py outputs/DEMO_measure_all.csv --economy Malaysia --indicator P7-I5
        one cell of the country x indicator grid, which is what "specific case" means

    python scripts/show.py outputs/DEMO_demo.csv --quote P7-I3
        the verbatim text of one row, wrapped, for reading aloud or checking against
        the source

Rows the judge model never saw are printed with a DEGRADED marker rather than silently
looking like every other row: quoting one as a model finding would be the one claim the
verbatim design exists to make checkable.
"""
from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexora.console import console_safe  # noqa: E402


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _degraded(row: dict) -> bool:
    return "DEGRADED" in (row.get("Notes") or "").upper()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--economy", help="filter, e.g. Malaysia")
    ap.add_argument("--indicator", help="filter, e.g. P7-I5")
    ap.add_argument("--quote", metavar="INDICATOR",
                    help="print the verbatim text of the row(s) for this indicator")
    ap.add_argument("--width", type=int, default=96)
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"no such file: {args.csv}")
    rows = load(args.csv)
    total = len(rows)
    if args.economy:
        rows = [r for r in rows if r["Economy"].casefold() == args.economy.casefold()]
    if args.indicator:
        rows = [r for r in rows if r["Indicator ID"].casefold() == args.indicator.casefold()]

    if args.quote:
        hits = [r for r in rows if r["Indicator ID"].casefold() == args.quote.casefold()]
        if not hits:
            # An empty cell is a finding, not an error -- say which one it is.
            print(f"NO PROVISION FOUND for {args.quote}"
                  + (f" in {args.economy}" if args.economy else "")
                  + f"  ({total} rows in the file)")
            return
        for r in hits:
            print("=" * args.width)
            print(console_safe(f"{r['Law Name']}  |  {r['Law Number / Ref']}  |  "
                               f"{r['Article / Section']}  |  {r['Discovery Tag']}"))
            if _degraded(r):
                print("!! DEGRADED - the judge model never saw this clause; keyword "
                      "retrieval selected it. Do not present it as a model finding.")
            print("-" * args.width)
            text = " ".join((r["Verbatim Snippet"] or "").split())
            print(console_safe(textwrap.fill(text, args.width)))
            print(f"\nwhy: {console_safe(' '.join((r['Mapping Rationale'] or '').split()))}")
            print(f"source: {r['Source URL']}")
        return

    if not rows:
        print("NO PROVISION FOUND"
              + (f" for {args.indicator}" if args.indicator else "")
              + (f" in {args.economy}" if args.economy else "")
              + f"  ({total} rows in the file)")
        return

    print(f"{'IND':<7} {'SECTION':<22} {'TAG':<6} LAW")
    print("-" * args.width)
    for r in rows:
        mark = " !!DEGRADED" if _degraded(r) else ""
        print(console_safe(
            f"{r['Indicator ID']:<7} {r['Article / Section'][:22]:<22} "
            f"{r['Discovery Tag']:<6} {r['Law Name'][:40]}{mark}"))
    n_deg = sum(1 for r in rows if _degraded(r))
    print("-" * args.width)
    print(f"{len(rows)} row(s) of {total} in the file"
          + (f", {n_deg} DEGRADED" if n_deg else ""))


if __name__ == "__main__":
    main()
