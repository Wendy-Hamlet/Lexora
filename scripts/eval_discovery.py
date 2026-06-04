"""Discovery evaluation harness (multi-query, mode-aware).

Each gold target carries several queries in two modes: `name` (query contains the
law name) and `indicator` (RDTII-concept phrasing with no law name — the real
test of full-text discovery). A hit = a candidate whose URL/title/full-text
matches the target's `match_re`. Queries flagged `expect_miss` pass when nothing
matches (a portal that genuinely cannot serve that query, e.g. AU has no
full-text search).

Reports hit@k per query and aggregates split by mode, so name-fitting can no
longer hide as generalization. Writes outputs/eval_discovery.json.

Usage:
    python scripts/eval_discovery.py
    python scripts/eval_discovery.py --mode indicator   # only indicator queries
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexora.collect.discovery import discover  # noqa: E402
from lexora.collect.profile_loader import load_profile  # noqa: E402
from lexora.models.source import FetchMethod  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GOLD = REPO / "configs" / "eval" / "discovery_gold.yaml"
OUT = REPO / "outputs" / "eval_discovery.json"


def _rank_of(results, pattern: re.Pattern) -> int | None:
    for i, r in enumerate(results):
        hay = f"{r.url} {r.title} {r.fulltext_url or ''}"
        if pattern.search(hay):
            return i + 1
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["name", "indicator"], help="only this query mode")
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    gold = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    k = args.k or gold.get("k", 5)

    profiles: dict[str, object] = {}
    report = []
    # aggregates: mode -> [hits, total]
    agg = {"name": [0, 0], "indicator": [0, 0]}
    miss_ok = miss_total = 0

    for t in gold["targets"]:
        iso = t["jurisdiction"].lower()
        if iso not in profiles:
            profiles[iso] = load_profile(REPO / "configs" / "jurisdictions" / f"{iso}.yaml")
        profile = profiles[iso]
        portal = profile.portals[0]
        force_browser = portal.fetch_method is FetchMethod.playwright
        pattern = re.compile(t["match_re"], re.I)

        for qspec in t["queries"]:
            mode = qspec["mode"]
            if args.mode and mode != args.mode:
                continue
            q = qspec["q"]
            expect_miss = bool(qspec.get("expect_miss"))
            t0 = time.time()
            try:
                results = discover(
                    portal, query=q, limit=k, force_browser=force_browser,
                    timeout=60.0, known_instruments=profile.known_instruments,
                    known_instrument_ids=profile.known_instrument_ids,
                )
                rank = _rank_of(results, pattern)
                err = None
            except Exception as exc:
                results, rank, err = [], None, f"{type(exc).__name__}: {exc}"

            hit = rank is not None
            if expect_miss:
                ok = not hit
                miss_total += 1
                miss_ok += int(ok)
            else:
                ok = hit
                agg[mode][1] += 1
                agg[mode][0] += int(ok)

            report.append({
                "id": t["id"], "mode": mode, "query": q, "expect_miss": expect_miss,
                "hit": hit, "rank": rank, "ok": ok, "n": len(results),
                "elapsed_s": round(time.time() - t0, 1), "error": err,
                "top": [{"score": round(r.score, 2), "via": r.via, "tag": r.discovery_tag,
                         "title": r.title, "url": r.url} for r in results[:3]],
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDiscovery eval (top-{k})  -  hit = identity matched within top-k")
    print(f"{'target':<18}{'mode':<11}{'ok':<5}{'rank':<6}{'query'}")
    print("-" * 78)
    for e in report:
        flag = ("MISS-OK" if e["expect_miss"] and e["ok"]
                else ("exp-miss!" if e["expect_miss"] else ("YES" if e["ok"] else "no")))
        print(f"{e['id']:<18}{e['mode']:<11}{flag:<5}{str(e.get('rank') or '-'):<6}{e['query'][:38]}")
    print("-" * 78)

    def pct(h, n):
        return f"{h}/{n}  ({100*h/n:.0f}%)" if n else "0/0"

    print(f"NAME      mode hit@{k}: {pct(*agg['name'])}")
    print(f"INDICATOR mode hit@{k}: {pct(*agg['indicator'])}   <- generalization (no law name in query)")
    if miss_total:
        print(f"expected-miss honored: {miss_ok}/{miss_total}")
    print(f"(full results -> {OUT.relative_to(REPO)})")


if __name__ == "__main__":
    main()
