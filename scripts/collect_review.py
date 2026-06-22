"""Collect the legal group's filled gold-review worksheets back into a validated
gold + an agreement report against our internal self-screen.

Input: the ``gold_review_{my,sg,au}.md`` files after a reviewer has filled the
``VERDICT=`` / ``NOTE=`` / ``ADD=`` lines (and optional ``DROP-ALL`` at the top of
an instrument block). Parsing is tolerant: it reads the token after ``VERDICT=``
up to the end of line or a closing backtick, upper-cases it, and accepts KEEP /
DROP (anything else, e.g. the unfilled template, is treated as "not answered").

Outputs (under outputs/review/):
  * validated_gold_cells.csv — the (economy, instrument, indicator) cells the
    reviewers confirmed KEEP plus any ADD= additions, i.e. the lawyer-validated
    instrument->indicator gold. DROP cells and DROP-ALL instruments are excluded.
  * review_agreement.csv + a printed summary — per cell, our self-screen verdict
    vs the reviewer's KEEP/DROP, and the agreement rate. This tells us how
    reliable the automated self-screen is for jurisdictions we cannot send to a
    lawyer (the generalisation goal). NB: §8 of the task sheet gives a category-
    level attention hint, so the DP-act×P6 cells are mildly anchored — read their
    agreement with that caveat.

Usage:
    python scripts/collect_review.py outputs/review/gold_review_*.md
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs" / "review"
SELF_SCREEN = OUTDIR / "self_screen.csv"

ISO2COUNTRY = {"my": "Malaysia", "sg": "Singapore", "au": "Australia"}
_INDICATOR_RE = re.compile(r"^-\s*\*\*(P[67]-I\d)\b")             # - **P6-I1 — ...**
_INSTRUMENT_RE = re.compile(r"^###\s+(.+?)\s*$")                   # ### <law name>
_VERDICT_RE = re.compile(r"VERDICT=\s*`?\s*([A-Za-z/ ]*)")
_NOTE_RE = re.compile(r"NOTE=\s*`?\s*(.*?)\s*`?\s*$")
_ADD_RE = re.compile(r"ADD=\s*`?\s*(.*?)\s*`?\s*$")
_ADD_ID_RE = re.compile(r"P[67]-I\d")


def _iso_of(path: Path) -> str:
    m = re.search(r"_(my|sg|au)\b", path.stem)
    return m.group(1) if m else "??"


def parse_worksheet(path: Path) -> tuple[list[dict], list[str]]:
    """Return (cells, warnings). Each cell: economy, instrument, indicator,
    verdict (KEEP/DROP/''/DROP-ALL), note, added(bool)."""
    iso = _iso_of(path)
    economy = ISO2COUNTRY.get(iso, iso)
    cells: list[dict] = []
    warnings: list[str] = []
    cur_inst: str | None = None
    cur_ind: str | None = None
    drop_all: set[str] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, raw in enumerate(lines):
        line = raw.strip()
        m_ind = _INDICATOR_RE.match(line)
        m_inst = _INSTRUMENT_RE.match(raw)
        if m_inst:
            cur_inst = m_inst.group(1).strip()
            cur_ind = None
            continue
        if m_ind:
            # ignore the rubric block at the top (before any instrument heading)
            if cur_inst is None:
                continue
            cur_ind = m_ind.group(1)
            cells.append({"economy": economy, "instrument": cur_inst,
                          "indicator": cur_ind, "verdict": "", "note": "",
                          "added": False})
            continue
        if "DROP-ALL" in raw and cur_inst:
            drop_all.add(cur_inst)
            continue
        mv = _VERDICT_RE.search(raw)
        if mv and cur_ind and cells:
            val = mv.group(1).strip().upper().replace(" ", "")
            if val in ("KEEP", "DROP"):
                cells[-1]["verdict"] = val
            elif "KEEP" in val and "DROP" in val:
                pass  # the unfilled template "(KEEP / DROP)" — leave unanswered
            elif re.search(r"[A-Z]", val):  # only flag letters, not "/" residue
                warnings.append(f"{path.name}:{i+1} unrecognised VERDICT='{mv.group(1).strip()}' "
                                f"({cur_inst} / {cur_ind})")
            continue
        mn = _NOTE_RE.search(raw)
        if mn and cur_ind and cells and mn.group(1):
            cells[-1]["note"] = mn.group(1)
            continue
        ma = _ADD_RE.search(raw)
        if ma and cur_inst and ma.group(1):
            for iid in _ADD_ID_RE.findall(ma.group(1)):
                cells.append({"economy": economy, "instrument": cur_inst,
                              "indicator": iid, "verdict": "KEEP",
                              "note": ma.group(1), "added": True})
    # apply DROP-ALL
    for c in cells:
        if c["instrument"] in drop_all and not c["added"]:
            c["verdict"] = "DROP-ALL"
    return cells, warnings


def load_self_screen() -> dict[tuple[str, str, str], str]:
    if not SELF_SCREEN.exists():
        return {}
    out = {}
    with SELF_SCREEN.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["country"], r["instrument"], r["indicator"])] = r["self_verdict"]
    return out


def _expected(self_verdict: str) -> str:
    return {"LIKELY-OK": "KEEP", "LIKELY-WRONG": "DROP"}.get(self_verdict, "EITHER")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("worksheets", nargs="+", type=Path)
    args = ap.parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    all_cells: list[dict] = []
    warnings: list[str] = []
    for p in args.worksheets:
        cells, warns = parse_worksheet(p)
        all_cells.extend(cells)
        warnings.extend(warns)

    answered = [c for c in all_cells if c["verdict"]]
    keeps = [c for c in all_cells if c["verdict"] == "KEEP"]
    unanswered = [c for c in all_cells if not c["verdict"]]

    # validated gold
    gpath = OUTDIR / "validated_gold_cells.csv"
    with gpath.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["economy", "instrument", "indicator", "note", "added"])
        for c in sorted(keeps, key=lambda x: (x["economy"], x["instrument"], x["indicator"])):
            w.writerow([c["economy"], c["instrument"], c["indicator"], c["note"],
                        "Y" if c["added"] else ""])

    # agreement vs self-screen
    self_screen = load_self_screen()
    apath = OUTDIR / "review_agreement.csv"
    agree = disagree = ambiguous = 0
    with apath.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["economy", "instrument", "indicator", "self_verdict",
                    "expected", "reviewer_verdict", "match"])
        for c in answered:
            if c["added"]:
                continue
            sv = self_screen.get((c["economy"], c["instrument"], c["indicator"]), "")
            exp = _expected(sv)
            rv = c["verdict"]
            if exp == "EITHER":
                match = "ambiguous"; ambiguous += 1
            elif exp == rv:
                match = "agree"; agree += 1
            else:
                match = "DISAGREE"; disagree += 1
            w.writerow([c["economy"], c["instrument"], c["indicator"], sv, exp, rv, match])

    print(f"parsed {len(all_cells)} cells | answered {len(answered)} "
          f"(KEEP {len(keeps)}, DROP {len(answered)-len(keeps)}) | unanswered {len(unanswered)}")
    print(f"validated gold -> {gpath.relative_to(ROOT)} ({len(keeps)} KEEP cells, "
          f"{sum(c['added'] for c in keeps)} added)")
    decisive = agree + disagree
    rate = f"{agree}/{decisive} = {agree/decisive:.0%}" if decisive else "n/a"
    print(f"self-screen agreement (decisive cells only): {rate}  "
          f"[+{ambiguous} ambiguous (SUSPECT)] -> {apath.relative_to(ROOT)}")
    if disagree:
        print(f"  DISAGREE cells (our screen vs reviewer) — review these:")
    if unanswered:
        print(f"NOTE: {len(unanswered)} cells still blank — worksheet incomplete?")
    for wmsg in warnings:
        print("  WARN", wmsg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
