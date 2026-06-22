"""Generate per-country gold-review worksheets for the legal group + an internal
self-screen of our own instrument->indicator labels.

Background. The official ESCAP Legal Inventory enumerates the instruments and
assigns each to a POLICY AREA = pillar (Cross-border data policies = P6,
Domestic data protection & privacy = P7). The finer split into the individual
indicators (P6-I1..I4, P7-I1..I5) is OUR OWN draft labelling in
``configs/eval/legal_inventory_p67.csv`` and has never been validated by a
lawyer. This script produces:

  outputs/review/gold_review_{my,sg,au}.md  — one neutral verification worksheet
      per economy (hand one to each reviewer). It lists, per instrument, the
      indicators WE currently tag, with a fillable KEEP/DROP verdict, a place to
      add missing indicators, and the official indicator rubric. It does NOT show
      our own suspicions, so the reviewer's judgement is not anchored.

  outputs/review/self_screen.csv + a printed summary — OUR provisional, rule-based
      screen of every in-scope cell (likely-ok / suspect / likely-wrong), so we
      have a baseline read on the gold (and on any tool run scored against it)
      BEFORE the legal group returns. This is explicitly NOT gold and NOT shown
      to reviewers.

P6-I5 ("Not in agreement with binding commitments on data transfer") is out of
scope (non-regulatory, third-party-sourced) and excluded.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "configs" / "eval" / "legal_inventory_p67.csv"
OUTDIR = ROOT / "outputs" / "review"

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
ISO = {"Malaysia": "my", "Singapore": "sg", "Australia": "au"}

# Official indicator rubric (condensed from configs/rdtii_indicators.yaml). The
# "scores positive when" line is the test a reviewer applies to KEEP a cell.
RUBRIC = [
    ("P6-I1", "Ban & local processing requirements",
     "The law BANS cross-border transfer, or REQUIRES data to be processed locally. "
     "(Mere conditions on transfer are NOT a ban — that is P6-I4.)"),
    ("P6-I2", "Local storage requirements",
     "The law REQUIRES data to be STORED within the territory (data localisation)."),
    ("P6-I3", "Infrastructure requirements",
     "Transfer abroad is conditioned on an INFRASTRUCTURE requirement / establishing a "
     "local data centre or server."),
    ("P6-I4", "Conditional flow regimes",
     "The law imposes CONDITIONS on cross-border transfer (adequacy, safeguards, "
     "comparable protection, or consent)."),
    ("P7-I1", "Comprehensive legal framework for data protection",
     "The instrument IS a personal-data-protection framework (horizontal = full; "
     "single-sector = partial)."),
    ("P7-I2", "Dedicated legal framework for cybersecurity",
     "The instrument IS a cybersecurity framework / computer-crime / critical-"
     "infrastructure law."),
    ("P7-I3", "Minimum period of data retention requirements",
     "The law imposes a MINIMUM period for RETAINING data. NB: a data-protection "
     "'retention limitation / do-not-keep-too-long' duty is the OPPOSITE and does "
     "NOT count."),
    ("P7-I4", "DPIA or DPO requirements",
     "The law REQUIRES appointing a Data Protection Officer or conducting a Data "
     "Protection Impact Assessment."),
    ("P7-I5", "Government access to personal data",
     "The law contains a MEASURE allowing government to ACCESS personal data "
     "(interception, production orders, search & seizure, lawful access)."),
]
RUBRIC_NAME = {iid: name for iid, name, _ in RUBRIC}
INSCOPE_IDS = [iid for iid, _, _ in RUBRIC]


def load():
    data = defaultdict(lambda: defaultdict(set))   # country -> instrument -> {iid}
    url = defaultdict(dict)
    with GOLD.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            iid = LABEL2ID.get(r["indicator_label"].strip())
            if not iid or iid in OUT_OF_SCOPE:
                continue
            c, name = r["country"].strip(), r["instrument"].strip()
            data[c][name].add(iid)
            url[c].setdefault(name, r["reference"].strip())
    return data, url


# ---------------------------------------------------------------------------
# Worksheet (neutral, for the legal group)
# ---------------------------------------------------------------------------
def rubric_block() -> str:
    lines = ["## In-scope indicator rubric (the standard to apply)\n",
             "Score a cell **KEEP** only if the instrument itself meets the test below.\n"]
    for iid, name, test in RUBRIC:
        lines.append(f"- **{iid} — {name}**: {test}")
    return "\n".join(lines) + "\n"


def worksheet(country: str, insts: dict, urls: dict) -> str:
    iso = ISO[country]
    n_cells = sum(len(v) for v in insts.values())
    out = []
    out.append(f"# Gold review — {country} (P6 & P7 instrument→indicator labels)\n")
    out.append(
        "**What this is.** The instrument list below and its pillar (P6 = cross-border "
        "data, P7 = domestic data protection) come from the official ESCAP Legal "
        "Inventory. The specific **indicator** each instrument is tagged with "
        "(P6-I1…P7-I5) is **our team's draft** and has NOT been checked by a lawyer. "
        "Please confirm or correct it on the law itself.\n")
    out.append(
        "**How to fill in.** For each tagged indicator, set `VERDICT=` to **KEEP** "
        "(the instrument really scores on that indicator) or **DROP** (it does not — "
        "shared vocabulary / wrong indicator). In `NOTE=` add the section/article that "
        "supports a KEEP, or one line on why DROP. If the instrument scores on an "
        "indicator we did **not** tag, list it in `ADD=` (with section). If the "
        "instrument is wholly irrelevant to P6/P7, write `DROP-ALL` at the top of its "
        "block.\n")
    out.append(
        f"_{country}: {len(insts)} instruments, {n_cells} tagged cells to review. "
        "Judge on the law as enforced; the 2024/2025 amendments may have changed "
        "cross-border rules._\n")
    out.append(rubric_block())
    out.append("---\n")
    out.append("## Instruments\n")
    # flagship (many tags) first — most worth a careful look
    for name, ids in sorted(insts.items(), key=lambda x: (-len(x[1]), x[0])):
        out.append(f"### {name}")
        ref = urls.get(name, "")
        if ref:
            out.append(f"- Official source: {ref}")
        out.append(f"- Our draft tags: {', '.join(sorted(ids))}\n")
        for iid in sorted(ids):
            out.append(f"- **{iid} — {RUBRIC_NAME[iid]}**")
            out.append("  - `VERDICT=` (KEEP / DROP)")
            out.append("  - `NOTE=`")
        out.append(f"- Missing indicators this instrument also scores on? `ADD=`\n")
    out.append("---\n")
    out.append(
        "When done, save this file with your entries and send it back. We re-collect "
        "the `VERDICT=` / `ADD=` lines programmatically into the validated gold.\n")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Self-screen (internal, rule-based — NOT shown to reviewers)
# ---------------------------------------------------------------------------
def _is_dp_framework(name: str) -> bool:
    n = name.lower()
    return ("personal data protection" in n or "privacy act" in n
            or "data protection" in n)


def self_screen(iid: str, name: str) -> tuple[str, str]:
    """Return (verdict, reason): LIKELY-OK / SUSPECT / LIKELY-WRONG."""
    n = name.lower()
    dp = _is_dp_framework(name)
    # P6-I1 ban / local processing — a DP framework imposes conditions, not a ban.
    if iid == "P6-I1":
        if dp:
            return "LIKELY-WRONG", "DP framework imposes transfer CONDITIONS (→P6-I4), not a ban/local-processing mandate"
        return "SUSPECT", "verify a genuine prohibition / local-processing mandate exists"
    # P6-I2 local storage
    if iid == "P6-I2":
        if dp:
            return "LIKELY-WRONG", "DP frameworks do not mandate in-territory storage"
        return "SUSPECT", "verify an explicit storage-in-country requirement (vs mere record-keeping)"
    # P6-I3 infrastructure / data centre
    if iid == "P6-I3":
        if dp:
            return "LIKELY-WRONG", "DP frameworks rarely impose a local-data-centre requirement"
        return "SUSPECT", "verify a data-centre / local-infrastructure condition"
    # P6-I4 conditional flow
    if iid == "P6-I4":
        if dp:
            return "LIKELY-OK", "DP framework's cross-border transfer provision = conditional flow"
        return "SUSPECT", "verify a cross-border transfer condition"
    # P7-I1 comprehensive DP framework
    if iid == "P7-I1":
        if dp:
            return "LIKELY-OK", "instrument is a data-protection framework"
        return "SUSPECT", "non-DP instrument tagged as DP framework — confirm sectoral relevance"
    # P7-I2 cybersecurity framework
    if iid == "P7-I2":
        if any(k in n for k in ("cyber", "computer crime", "computer misuse",
                                "critical infrastructure", "criminal code")):
            return "LIKELY-OK", "cybersecurity / computer-crime instrument"
        if dp:
            return "LIKELY-WRONG", "a data-protection act is not a cybersecurity framework"
        return "SUSPECT", "confirm it is a dedicated cybersecurity framework"
    # P7-I3 minimum retention — DP 'retention limitation' is the opposite
    if iid == "P7-I3":
        if dp:
            return "SUSPECT", "DP retention principle is a MAX/limitation duty, not a MINIMUM-retention mandate — confirm a true minimum exists"
        if any(k in n for k in ("tax", "companies", "employment", "telecommunic",
                                "income", "service")):
            return "LIKELY-OK", "sectoral record-keeping statute typically sets a minimum retention"
        return "SUSPECT", "confirm a minimum-retention period is imposed"
    # P7-I4 DPO / DPIA
    if iid == "P7-I4":
        if "amendment" in n or "2020" in n or "guide" in n or "impact assessment" in n:
            return "LIKELY-OK", "amendment/guide that introduces DPO/DPIA"
        if dp:
            return "SUSPECT", "base DP act may predate any DPO/DPIA duty — confirm the provision"
        return "SUSPECT", "confirm an explicit DPO or DPIA requirement"
    # P7-I5 government access
    if iid == "P7-I5":
        if any(k in n for k in ("criminal procedure", "security offences", "interception",
                                "surveillance", "intelligence", "production order",
                                "assistance and access", "cyber security", "data availability",
                                "telecommunic")):
            return "LIKELY-OK", "instrument carries a government-access / interception power"
        if dp:
            return "SUSPECT", "a DP act's law-enforcement exemption is not itself an access-granting measure — confirm"
        return "SUSPECT", "confirm a government-access measure"
    return "SUSPECT", "unclassified"


def main():
    data, url = load()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    for country in ("Malaysia", "Singapore", "Australia"):
        iso = ISO[country]
        path = OUTDIR / f"gold_review_{iso}.md"
        path.write_text(worksheet(country, data[country], url[country]), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")

    # self-screen
    rows = []
    for country in ("Malaysia", "Singapore", "Australia"):
        for name, ids in data[country].items():
            for iid in sorted(ids):
                v, why = self_screen(iid, name)
                rows.append((country, name, iid, RUBRIC_NAME[iid], v, why))
    csv_path = OUTDIR / "self_screen.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["country", "instrument", "indicator", "indicator_name",
                    "self_verdict", "reason"])
        w.writerows(rows)
    print(f"wrote {csv_path.relative_to(ROOT)}\n")

    from collections import Counter
    overall = Counter(r[4] for r in rows)
    print(f"SELF-SCREEN — {len(rows)} in-scope cells: "
          f"{dict(overall)}")
    for country in ("Malaysia", "Singapore", "Australia"):
        c = Counter(r[4] for r in rows if r[0] == country)
        n = sum(c.values())
        print(f"  {country:<10} n={n:<3} OK={c['LIKELY-OK']:<3} "
              f"SUSPECT={c['SUSPECT']:<3} WRONG={c['LIKELY-WRONG']}")
    print("\nLIKELY-WRONG cells (highest-priority for the reviewer):")
    for r in rows:
        if r[4] == "LIKELY-WRONG":
            print(f"  [{ISO[r[0]]}] {r[2]}  {r[1][:48]:<48}  {r[5]}")


if __name__ == "__main__":
    main()
