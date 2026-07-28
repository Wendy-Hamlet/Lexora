"""Score the KEY-FREE lane -- BM25 + boundary rules, no LLM anywhere.

This lane used to matter only as the `--no-llm` mode. Since the degrade gate
(`pipeline._judge_is_dead`) it is also what ships when the judge is unreachable, so its
quality is now a live-run property and not just a development convenience. It costs nothing
to measure: every knob here (dense fusion, cross-encoder rerank, top_k, rel_floor) is a
local model or pure arithmetic.

    python scripts/bench_fallback.py                    # the grid, on MY PDPA
    python scripts/bench_fallback.py --top-k 5 --dense  # one configuration
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location("bj", REPO / "scripts" / "bench_judge.py")
bj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bj)


def score(cites, gold) -> dict:
    pred: dict[str, set[str]] = defaultdict(set)
    for c in cites:
        # The citation's own section label ("S. 26(1)"), folded to gold's token ("26").
        sec = (c.article_path or "").strip()
        for prefix in ("Section ", "section ", "S. ", "s. ", "S.", "s."):
            if sec.startswith(prefix):
                sec = sec[len(prefix):]
                break
        pred[c.indicator_id].add(sec.split("(")[0].strip())
    tp = n_gold = n_pred = na = 0
    for ind, want in gold.items():
        got = pred.get(ind, set())
        if not want:
            na += len(got)
            continue
        tp += len(want & got)
        n_gold += len(want)
        n_pred += len(got)
    return {"recall": tp / n_gold if n_gold else 0.0,
            "precision": tp / n_pred if n_pred else 0.0,
            "hit": tp, "gold": n_gold, "citations": len(cites), "na_violations": na}


def run_one(art, profile, inds, gold, *, top_k, min_score, rel_floor, dense, rerank) -> dict:
    os.environ["LEXORA_MAP_DENSE"] = "1" if dense else "0"
    os.environ["LEXORA_MAP_RERANK"] = "1" if rerank else "0"
    from lexora.pipeline import _citations_from_clauses

    t0 = time.perf_counter()
    cites = _citations_from_clauses(
        art.clauses, art.document, profile, inds, "statute",
        top_k=top_k, min_score=min_score, rel_floor=rel_floor, verifier=None,
    )
    out = score(cites, gold)
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso", default="my")
    ap.add_argument("--gold", type=Path, default=None)
    ap.add_argument("--match", default=None)
    ap.add_argument("--top-k", type=int, default=0, help="0 = sweep the grid")
    ap.add_argument("--min-score", type=float, default=0.05)
    ap.add_argument("--rel-floor", type=float, default=0.0)
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()

    from lexora.collect.profile_loader import load_profile
    from lexora.pipeline import run_demo_pipeline

    profile = load_profile(REPO / "configs" / "jurisdictions" / f"{args.iso}.yaml")
    inds = bj.load_indicators(bj.INDICATORS, pillars=[6, 7])
    needles = tuple(x.strip() for x in args.match.split(",")) if args.match else (
        "PERSONAL DATA PROTECTION ACT 2010", "Act 709")
    pdf = bj.find_doc(args.iso, needles)
    art = run_demo_pipeline(pdf_path=pdf, profile=profile, indicators=inds,
                            source_url=str(profile.portals[0].url), portal_name="local-pdf")
    gold = bj.load_gold(args.gold or bj.GOLD_MY, args.iso.upper())
    print(f"document: {pdf.name}  ({len(art.clauses)} clauses)")
    print(f"{'config':34s} {'recall':>8s} {'prec':>7s} {'cites':>6s} {'N/A':>4s} {'wall':>7s}")

    if args.top_k:
        grid = [(args.top_k, args.rel_floor, args.dense, args.rerank)]
    else:
        grid = [(k, rf, d, rr)
                for k, rf in ((1, 0.0), (3, 0.0), (3, 0.6), (5, 0.0), (8, 0.0))
                for d, rr in ((False, False), (True, False), (False, True))]
    for k, rf, d, rr in grid:
        m = run_one(art, profile, inds, gold, top_k=k, min_score=args.min_score,
                    rel_floor=rf, dense=d, rerank=rr)
        name = (f"top_k={k} rel_floor={rf}"
                + (" +dense" if d else "") + (" +rerank" if rr else ""))
        print(f"{name:34s} {m['recall']:7.0%} {m['precision']:6.0%} "
              f"{m['citations']:6d} {m['na_violations']:4d} {m['wall_s']:6.1f}s")


if __name__ == "__main__":
    main()
