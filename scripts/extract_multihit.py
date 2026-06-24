"""Extract provisions that the pipeline mapped to MULTIPLE indicators.

The legal group's hypothesis: within Pillar 6 the indicators are (largely)
mutually exclusive at the PROVISION grain — one statutory provision is unlikely
to genuinely satisfy several P6 indicators at once (ban vs. conditional flow are
opposite modalities). The same broadly holds for P7 (different subject matter),
with a couple of legitimate exceptions (retention-for-access = P7-I3 + P7-I5).

So a provision our BEST run still maps to >=2 indicators is exactly the cell to
hand to a lawyer: it is either (a) a genuinely multi-modal provision or (b) a
residual over-mapping false positive that survived the per-cell verifier. This
script pulls those out of a submission CSV and writes a neutral review worksheet.

Provision identity = (Economy, Law Name, Article / Section) — that column is
100% populated in our output and is the finest stable provision locator. Each
indicator row keeps its own verbatim snippet + rationale so the reviewer can see
what each mapping actually rests on.

Outputs (outputs/review/):
  * multihit_provisions.csv — one row per (provision x indicator) inside a
    multi-hit group, with a `group_kind` tag (P6_internal / P7_internal /
    cross_pillar) and a stable `provision_id`.
  * multihit_for_legal.md — per-economy worksheet, one block per multi-hit
    provision, listing every indicator it hit with the verbatim + rationale and
    a fillable KEEP-WHICH= / DROP= line. Suspicions are NOT pre-filled (blind).

Usage:
    python scripts/extract_multihit.py outputs/submission_best.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs" / "review"

# tolerate the BOM the exporter writes (utf-8-sig) on the Economy header.
COL = {"economy": "Economy", "law": "Law Name", "art": "Article / Section",
       "ind": "Indicator ID", "quote": "Verbatim Snippet",
       "rat": "Mapping Rationale", "loc": "Location Reference", "url": "Source URL"}


def _pillar(ind: str) -> str:
    return ind.split("-")[0] if "-" in ind else ind  # "P6-I2" -> "P6"


def _group_kind(indicators: set[str]) -> str:
    pillars = {_pillar(i) for i in indicators}
    if pillars == {"P6"}:
        return "P6_internal"
    if pillars == {"P7"}:
        return "P7_internal"
    return "cross_pillar"


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="submission CSV (best-version run)")
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.csv)
    # group provision -> list of (indicator, row)
    groups: dict[tuple[str, str, str], list[dict]] = collections.defaultdict(list)
    for r in rows:
        key = (r.get(COL["economy"], "").strip(),
               r.get(COL["law"], "").strip(),
               r.get(COL["art"], "").strip())
        groups[key].append(r)

    multi = {}
    for key, members in groups.items():
        inds = {m.get(COL["ind"], "").strip() for m in members if m.get(COL["ind"], "").strip()}
        if len(inds) >= 2:
            multi[key] = members

    # stable provision ids
    pid_of = {key: f"M{idx+1:03d}" for idx, key in enumerate(sorted(multi))}

    # ---- machine CSV ----
    csv_out = args.outdir / "multihit_provisions.csv"
    with csv_out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["provision_id", "group_kind", "n_indicators", "economy",
                    "law_name", "article_section", "indicator_id",
                    "location_reference", "verbatim_snippet", "mapping_rationale",
                    "source_url"])
        for key in sorted(multi):
            economy, law, art = key
            members = multi[key]
            inds = {m.get(COL["ind"], "").strip() for m in members}
            kind = _group_kind(inds)
            for m in sorted(members, key=lambda x: x.get(COL["ind"], "")):
                w.writerow([pid_of[key], kind, len(inds), economy, law, art,
                            m.get(COL["ind"], "").strip(),
                            m.get(COL["loc"], "").strip(),
                            m.get(COL["quote"], "").strip(),
                            m.get(COL["rat"], "").strip(),
                            m.get(COL["url"], "").strip()])

    # ---- per-economy markdown worksheet ----
    by_econ: dict[str, list] = collections.defaultdict(list)
    for key in sorted(multi):
        by_econ[key[0]].append(key)

    kinds_count = collections.Counter(_group_kind({m.get(COL["ind"], "").strip()
                                      for m in multi[k]}) for k in multi)

    md = args.outdir / "multihit_for_legal.md"
    lines: list[str] = []
    lines.append("# Multi-indicator provisions — legal review worksheet")
    lines.append("")
    lines.append(f"Source run: `{args.csv.name}` (best version: BM25 + per-cell "
                 "verifier + LLM rationale/metadata + secondary seeds, OCR on).")
    lines.append("")
    lines.append(f"**{len(multi)} provisions** were mapped to 2+ indicators after "
                 "per-cell verification: "
                 f"{kinds_count.get('P6_internal', 0)} P6-internal, "
                 f"{kinds_count.get('P7_internal', 0)} P7-internal, "
                 f"{kinds_count.get('cross_pillar', 0)} cross-pillar.")
    lines.append("")
    lines.append("## What to decide")
    lines.append("For each provision below, the tool claims the SAME article/section "
                 "satisfies several indicators. Judge whether that is correct:")
    lines.append("")
    lines.append("- `KEEP-WHICH=` list the indicator IDs that the quoted text "
                 "*genuinely* satisfies (e.g. `P6-I2, P6-I4`).")
    lines.append("- `DROP=` list the indicator IDs that are wrong for this text.")
    lines.append("- `NOTE=` free text (e.g. which indicator the text really belongs "
                 "to, or that two are duplicative).")
    lines.append("")
    lines.append("Indicator reference (in scope): P6-I1 ban/local-processing · "
                 "P6-I2 local storage · P6-I3 infrastructure/data-centre · "
                 "P6-I4 conditional flow · P7-I1 comprehensive DP framework · "
                 "P7-I2 cybersecurity framework · P7-I3 minimum retention · "
                 "P7-I4 DPO/DPIA · P7-I5 government access.")
    lines.append("")

    for econ in sorted(by_econ):
        lines.append(f"## {econ}")
        lines.append("")
        for key in by_econ[econ]:
            economy, law, art = key
            members = multi[key]
            inds = sorted({m.get(COL["ind"], "").strip() for m in members})
            kind = _group_kind(set(inds))
            pid = pid_of[key]
            lines.append(f"### {pid} · {law} — {art}")
            lines.append(f"*{kind}* · hits: {', '.join(inds)}")
            url = next((m.get(COL["url"], "").strip() for m in members
                        if m.get(COL["url"], "").strip()), "")
            loc = next((m.get(COL["loc"], "").strip() for m in members
                        if m.get(COL["loc"], "").strip()), "")
            if url:
                lines.append(f"Source: {url}" + (f" · loc {loc}" if loc else ""))
            lines.append("")
            for m in sorted(members, key=lambda x: x.get(COL["ind"], "")):
                ind = m.get(COL["ind"], "").strip()
                quote = " ".join(m.get(COL["quote"], "").split())
                rat = " ".join(m.get(COL["rat"], "").split())
                lines.append(f"- **{ind}**")
                lines.append(f"  - verbatim: “{quote}”")
                if rat:
                    lines.append(f"  - tool rationale: {rat}")
            lines.append("")
            lines.append("  ```")
            lines.append("  KEEP-WHICH= ")
            lines.append("  DROP= ")
            lines.append("  NOTE= ")
            lines.append("  ```")
            lines.append("")

    md.write_text("\n".join(lines), encoding="utf-8")

    print(f"multi-hit provisions: {len(multi)} "
          f"(P6_internal {kinds_count.get('P6_internal', 0)}, "
          f"P7_internal {kinds_count.get('P7_internal', 0)}, "
          f"cross_pillar {kinds_count.get('cross_pillar', 0)})")
    print(f"  -> {csv_out.relative_to(ROOT)}")
    print(f"  -> {md.relative_to(ROOT)}")
    # quick breakdown of which indicator pairs co-occur (P6 and P7 separately)
    pair = collections.Counter()
    for key in multi:
        inds = sorted({m.get(COL["ind"], "").strip() for m in multi[key]})
        for i in range(len(inds)):
            for j in range(i + 1, len(inds)):
                pair[(inds[i], inds[j])] += 1
    print("  top co-occurring indicator pairs:")
    for (a, b), n in pair.most_common(12):
        print(f"    {a} + {b}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
