"""Cell-level precision / coverage of a submission CSV against the gold inventory.

Scores a Round-1 submission CSV (``Law Name`` + ``Indicator ID`` columns) at the
(law x indicator) CELL level against ``configs/eval/legal_inventory_p67.csv``.

This is a calibration aid for the per-cell LLM verifier (``--verify-cells``): the
same scorer is applied to the baseline, the over-strict, and the softened run so
the deltas are comparable, even though law-name matching is necessarily fuzzy
(our ``Law Name`` is often a raw PDF filename; the gold carries clean titles).

Matching key is the Malaysian Act number (``Act 709``, ``Act 53``, ``Act A1727`` …),
which both sides almost always carry, with a small title-signature fallback for
the Codes of Practice / Standards / Bills that have no plain Act number.

A predicted cell is:
  * TP   — its law maps to a gold instrument AND that (instrument, indicator)
           pairing is in the gold;
  * FP   — its law maps to a gold instrument but NOT for this indicator;
  * n/j  — its law maps to no gold instrument (un-judgeable, excluded from
           precision, mirroring the gold-instrument-restricted §4 measurement).

Coverage = distinct in-scope gold cells hit / total in-scope gold cells
(P6-I5 is out of scope — a non-regulatory, third-party-sourced indicator).

Usage:
    python scripts/score_cells.py outputs/ab/my_v0.csv [more.csv ...] [--iso my]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "configs" / "eval" / "legal_inventory_p67.csv"

ISO2COUNTRY = {"my": "Malaysia", "sg": "Singapore", "au": "Australia"}

LABEL2ID = {
    "Ban & local processing requirements": "P6-I1",
    "Local storage requirements": "P6-I2",
    "Infrastructure requirements": "P6-I3",
    "Conditional flow regimes": "P6-I4",
    "Not in agreement with binding commitments on data transfer": "P6-I5",
    "Lack of comprehensive legal framework for data protection": "P7-I1",
    "Lack of dedicated legal framework for cybersecurity": "P7-I2",
    "Minimum period of data retention requirements": "P7-I3",
    "Data Protection Impact Assessment or Data Protection Officer requirements": "P7-I4",
    "Requirements to allow government access to personal data": "P7-I5",
}
OUT_OF_SCOPE = {"P6-I5"}

# Canonical gold-instrument catalogue (Malaysia). Our submission "Law Name" is
# usually a title (often a raw PDF filename) and almost never carries the gold
# Act number, so the PRIMARY match key is a title signature; the Act number is a
# secondary key for the rows that do carry it ("… (Act 53)", "Act 593 as at …").
#
# Each entry: canon id -> (act_tokens, require_all, exclude_any). A law name maps
# to this instrument if it carries one of act_tokens, OR contains every substring
# in require_all and none in exclude_any. Order matters: the first match wins, so
# the more specific variants (Amendment / Code / Standard) precede the base Act.
INSTRUMENTS = [
    # id            act tokens      require_all (lowercase)             exclude_any
    ("pdpa-amend",  {"acta1727"},   ["personal data protection", "amendment"], []),
    ("cop-banking", set(),          ["banking"],                        []),
    ("cop-licensee", set(),         ["licensee"],                       []),
    ("pdp-standard", set(),         ["personal data protection", "standard"], []),
    ("pdpa",        {"act709"},     ["personal data protection", "act"],
                                    ["amendment", "code", "standard"]),
    ("income-tax",  {"act53"},      ["income tax"],                     []),
    ("services-tax", {"act807"},    ["service tax"],                    []),  # matches service/services
    ("computer-crimes", {"act563"}, ["computer crimes"],                []),
    ("cyber-security", {"act854"},  ["cyber security"],                 []),
    ("cyber-security", {"act854"},  ["cybersecurity"],                  []),
    ("criminal-proc", {"act593"},   ["criminal procedure"],             []),
    ("security-offences", {"act747"}, ["security offences"],            []),
]

# gold instrument name (substring) -> canon id, to fold the gold rows onto the catalogue.
GOLD_NAME_TO_CANON = [
    ("amendment", "pdpa-amend"),
    ("banking", "cop-banking"),
    ("licensees", "cop-licensee"),
    ("standard", "pdp-standard"),
    ("income tax", "income-tax"),
    ("services tax", "services-tax"),
    ("computer crimes", "computer-crimes"),
    ("cyber security", "cyber-security"),
    ("criminal procedure", "criminal-proc"),
    ("security offences", "security-offences"),
    ("personal data protection act", "pdpa"),  # base PDPA last (after amendment/code/standard)
]


def _act_tokens(name: str) -> set[str]:
    """Act-number tokens carried in a name, e.g. 'act53', 'act593', 'acta1727'.

    Pure 4-digit years (1998, 1967, 2010 …) are years of enactment, not act
    numbers, and are dropped; the A-prefixed amendment codes (A1727) are kept."""
    n = name.lower()
    toks = set()
    for m in re.finditer(r"act\s*([a-z]?\d+)", n):
        num = m.group(1)
        if num[0].isdigit() and re.fullmatch(r"(19|20)\d{2}", num):
            continue
        toks.add("act" + num)
    for m in re.finditer(r"\b(a\d{3,4})\b", n):
        toks.add("act" + m.group(1))
    return toks


def classify_law(name: str) -> str | None:
    """Map a submission Law Name to a canonical gold-instrument id, or None."""
    n = name.lower()
    acts = _act_tokens(name)
    for canon, act_toks, require_all, exclude_any in INSTRUMENTS:
        if act_toks & acts:
            return canon
        if (require_all and all(s in n for s in require_all)
                and not any(s in n for s in exclude_any)):
            return canon
    return None


def _gold_canon(instrument: str) -> str | None:
    n = instrument.lower()
    for sub, canon in GOLD_NAME_TO_CANON:
        if sub in n:
            return canon
    return None


def load_gold(country: str) -> tuple[set[tuple[str, str]], dict[str, str]]:
    """Return (gold cells {(indicator_id, canon_instrument)}, canon -> sample name)."""
    cells: set[tuple[str, str]] = set()
    canon_name: dict[str, str] = {}
    with GOLD.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["country"].strip() != country:
                continue
            iid = LABEL2ID.get(r["indicator_label"].strip())
            if not iid or iid in OUT_OF_SCOPE:
                continue
            canon = _gold_canon(r["instrument"].strip())
            if canon is None:
                continue
            cells.add((iid, canon))
            canon_name.setdefault(canon, r["instrument"].strip())
    return cells, canon_name


def score(csv_path: Path, gold_cells: set, gold_keys: set) -> dict:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

    def col(r, name):  # tolerate the BOM on the first header
        return r.get(name) or r.get("﻿" + name) or ""

    pred_cells: set[tuple[str, str]] = set()  # (indicator, canon) on gold instruments
    pred_all: set[tuple[str, str]] = set()    # (indicator, raw law) every distinct cell
    inds = set()
    for r in rows:
        ind = col(r, "Indicator ID").strip()
        law = col(r, "Law Name").strip()
        if not ind or not law:
            continue
        inds.add(ind)
        pred_all.add((ind, law))
        canon = classify_law(law)
        if canon in gold_keys:
            pred_cells.add((ind, canon))

    tp = sorted(c for c in pred_cells if c in gold_cells)
    fp = sorted(c for c in pred_cells if c not in gold_cells)
    hit_gold = set(tp)
    return {
        "citations": len(rows),
        "cells_total": len(pred_all),
        "cells_on_gold_laws": len(pred_cells),
        "indicators_covered": sorted(inds),
        "n_indicators": len(inds),
        "TP": len(tp),
        "FP": len(fp),
        "precision": round(len(tp) / len(pred_cells), 3) if pred_cells else None,
        "gold_cells_total": len(gold_cells),
        "gold_cells_hit": len(hit_gold),
        "coverage": round(len(hit_gold) / len(gold_cells), 3) if gold_cells else None,
        "tp_cells": tp,
        "fp_cells": fp,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", type=Path)
    ap.add_argument("--iso", default="my")
    ap.add_argument("-v", "--verbose", action="store_true", help="list TP/FP cells")
    args = ap.parse_args(argv)

    country = ISO2COUNTRY[args.iso]
    gold_cells, key_name = load_gold(country)
    gold_keys = set(key_name)
    print(f"Gold: {len(gold_cells)} in-scope cells over {len(gold_keys)} matchable "
          f"instruments ({country}); keys={sorted(gold_keys)}\n")

    hdr = f"{'run':<24} {'cites':>6} {'cells':>6} {'onGold':>7} {'inds':>5} {'TP':>4} {'FP':>4} {'prec':>6} {'cov':>6}"
    print(hdr)
    print("-" * len(hdr))
    for p in args.csvs:
        s = score(p, gold_cells, gold_keys)
        print(f"{p.stem:<24} {s['citations']:>6} {s['cells_total']:>6} "
              f"{s['cells_on_gold_laws']:>7} {s['n_indicators']:>5} {s['TP']:>4} "
              f"{s['FP']:>4} {str(s['precision']):>6} {str(s['coverage']):>6}")
        if args.verbose:
            print(f"   TP: {s['tp_cells']}")
            print(f"   FP: {s['fp_cells']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
