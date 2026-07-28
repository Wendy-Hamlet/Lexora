"""Small-batch bench for the per-clause 9-in-1 judge.

The judge is the whole LLM bill: one call per pooled clause, ~22k calls in a full run.
Every question about it -- which model, how wide a pool, is the prefix cache actually
hitting, is the verdict stable across identical runs -- is answerable on ONE document for
a few yuan. This bench answers them on the MY PDPA flagship, the one document the legal
group annotated at provision level (``configs/eval/mapping_sections_legal_my.csv``).

It reproduces the production pool exactly (``pipeline._specs_per_clause``): per-indicator
BM25 retrieval to ``pool_k``, ``min_score`` gate, union, then one 9-in-1 call per clause.
The boundary rule is applied to the verdict, as in production.

    # free: pool size only, no LLM calls at all
    python scripts/bench_judge.py --dry-run --pool-k 40

    # one model, cold (bypasses the verdict cache so it really asks)
    python scripts/bench_judge.py --model GLM-5.2 --out outputs/_bench/glm52.json

    # reproducibility: three identical cold passes
    python scripts/bench_judge.py --model GLM-5.2 --repeat 3

Reported per run: gold recall / precision / N-A violations, token split (incl. the
provider's cached-prefix share), wall time, and cost under a per-model rate table.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from lexora.classify.boundaries import admits_clause  # noqa: E402
from lexora.classify.retrieval import build_index, retrieve_candidates  # noqa: E402
from lexora.classify.verifier import _CLAUSE_TEXT_CAP, Verifier  # noqa: E402
from lexora.indicators import load_indicators  # noqa: E402
from lexora.pipeline import _normalize_score  # noqa: E402

INDICATORS = REPO / "configs" / "rdtii_indicators.yaml"
GOLD_MY = REPO / "configs" / "eval" / "mapping_sections_legal_my.csv"

# CNY per 1M tokens: (fresh input, cached input, output). DERIVED FROM THE INVOICE
# (2026-07-28), not from a price page: every rate below is a billed line divided by its
# billed token count, so the numbers reconcile with what we were actually charged.
# A model missing here still reports tokens; only the cost line is suppressed.
RATES = {
    "GLM-5.2": (8.0, 2.0, 28.0),          # confirmed to the cent on all three lines
    "GLM-4.5-Flash": (0.0, 0.0, 0.0),     # genuinely free, all three lines billed 0
    "DeepSeek-V4-Flash": (1.0, 0.2, 2.0),  # was guessed at 0.5/0.1; the invoice says 1.0/0.2
    # Qwen has NO cached-input line on the invoice at all, which corroborates the measured
    # 0% prefix-cache hit rate: Paratera does not cache this family. Priced as input-only.
    "Qwen3.5-35B-A3B": (1.6, 1.6, 12.8),
}


def _clause_key(clause) -> str:
    """Section token for gold comparison (mirrors scripts/eval_mapping.py)."""
    if "APP" in (clause.structural_path or "") or "Australian Privacy Principle" in (
        clause.structural_path or ""
    ):
        return f"APP{clause.section_number}"
    return str(clause.section_number or "").strip()


def load_gold(path: Path = GOLD_MY, iso: str = "MY") -> dict[str, set[str]]:
    """indicator -> set of gold section tokens for one economy; ``N/A`` becomes an empty set.

    An empty set is load-bearing: it is the legal group asserting the document has NOTHING
    for that indicator, so any prediction on it is a provable false positive (the bench
    reports these as ``na_violations``). Dropping such a row would silently retire the
    check, which is why a doubtful cell is deleted deliberately, never blanked."""
    import csv

    gold: dict[str, set[str]] = {}
    with path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["iso"].strip().upper() != iso.upper():
                continue
            raw = (row["gold_sections"] or "").strip()
            gold[row["indicator"]] = (
                set() if raw.upper() in ("", "N/A") else {s.strip() for s in raw.split(";") if s.strip()}
            )
    return gold


def find_doc(iso: str, needles: tuple[str, ...]) -> Path:
    """The cached principal PDF, chosen by CONTENT (store filenames are content hashes)."""
    import fitz

    best: tuple[int, Path] | None = None
    for p in sorted((REPO / "data" / "raw" / iso).glob("*.pdf")):
        try:
            doc = fitz.open(p)
            head = "".join(doc[i].get_text() for i in range(min(2, doc.page_count)))
            n_pages = doc.page_count
            doc.close()
        except Exception:
            continue
        if all(n.upper() in head.upper() for n in needles) and (
            best is None or n_pages > best[0]
        ):
            best = (n_pages, p)
    if best is None:
        raise SystemExit(f"no {iso} doc matching {needles} in data/raw/{iso}")
    return best[1]


def clauses_from_pdf(pdf: Path, profile, indicators):
    """Same ingest path as scripts/eval_mapping.py, so pools are comparable."""
    from lexora.pipeline import run_demo_pipeline

    art = run_demo_pipeline(
        pdf_path=pdf, profile=profile, indicators=indicators,
        source_url=str(profile.portals[0].url), portal_name="local-pdf",
    )
    return art.clauses


def build_pool(clauses, indicators, profile, *, pool_k: int, min_score: float):
    """The production pool: per-indicator retrieval, min_score gate, union."""
    index = build_index(clauses)
    pool_score: dict[str, float] = {}
    pair_rank: dict[str, dict[str, int]] = defaultdict(dict)
    for indicator in indicators:
        for rank, hit in enumerate(
            retrieve_candidates(indicator, profile, index, top_k=pool_k, pool_k=max(pool_k, 20))
        ):
            if _normalize_score(hit.score) < min_score:
                continue
            pool_score[hit.clause_id] = max(pool_score.get(hit.clause_id, 0.0), hit.score)
            pair_rank[indicator.submission_id][hit.clause_id] = rank
    return index, pool_score, pair_rank


def score_verdicts(verdicts, clause_by_id, gold) -> dict:
    """Judge-level gold metrics: per-indicator predicted section set vs the legal group's."""
    pred: dict[str, set[str]] = defaultdict(set)
    for cid, sids in verdicts.items():
        if not sids:
            continue
        clause = clause_by_id[cid]
        for sid in sids:
            pred[sid].add(_clause_key(clause))
    graded, na_violations = [], 0
    for ind, want in sorted(gold.items()):
        got = pred.get(ind, set())
        if not want:
            na_violations += len(got)
            continue
        graded.append({
            "indicator": ind,
            "gold": sorted(want),
            "pred": sorted(got),
            "hit": sorted(want & got),
            "recall": len(want & got) / len(want),
        })
    tp = sum(len(set(r["hit"])) for r in graded)
    n_gold = sum(len(r["gold"]) for r in graded)
    n_pred_graded = sum(len(r["pred"]) for r in graded)
    return {
        "per_indicator": graded,
        "recall": tp / n_gold if n_gold else 0.0,
        "precision": tp / n_pred_graded if n_pred_graded else 0.0,
        "gold_sections": n_gold,
        "predicted_sections": n_pred_graded,
        "na_violations": na_violations,
        "citations": sum(len(v) for v in verdicts.values() if v),
    }


def cost_cny(model: str, prompt: int, cached: int, completion: int) -> float | None:
    rate = RATES.get(model)
    if rate is None:
        return None
    fresh_in, cached_in, out = rate
    return (
        (prompt - cached) / 1e6 * fresh_in + cached / 1e6 * cached_in + completion / 1e6 * out
    )


_BINARY_SYSTEM = (
    "You are a legal-mapping auditor for the UN ESCAP RDTII framework. You are given ONE "
    "indicator definition and ONE statutory clause. Answer a single question: does this "
    "clause's operative provision fall within THIS indicator's definition, as direct "
    "primary-source evidence?\n"
    "- An incidental mention, a bare definition with no operative rule, or a pure "
    "cross-reference to another Act is NOT support.\n"
    "- For an indicator phrased as 'Lack of <framework>', a clause that enacts a core "
    "obligation or data-subject right of that framework IS support, on its own -- but the "
    "statute's machinery (registration, appeals, penalties, transitional provisions) is not.\n"
    "- Judge only the text given, not the rest of the Act and not outside knowledge.\n"
    'Respond with a single JSON object: {"supports": true|false}.'
)
_BINARY_SCHEMA = {"type": "object", "properties": {"supports": {"type": "boolean"}},
                  "required": ["supports"]}


def judge_binary(client, clause, indicators) -> set[str] | None:
    """One yes/no question per indicator instead of one 9-in-1 subset question.

    The 9-in-1 asks a model to hold nine long definitions at once and emit the right
    SUBSET -- a multi-label task over a ~3.5k-token context. Frontier models do it well;
    small ones collapse to one or two answers regardless of the clause. Decomposed, each
    call carries a single definition and asks a question with two possible answers, which
    is squarely inside a 7B model's competence.

    The trade is 9x the calls. For a metered frontier model that is the wrong trade. For a
    self-hosted open-weight model -- the No-Vendor-Lock-in lane -- calls are free and only
    wall-clock is spent, so it is exactly the right one.
    """
    out: set[str] = set()
    text = clause.span.text.strip().replace("\n", " ")[:_CLAUSE_TEXT_CAP]
    for ind in indicators:
        user = (f"INDICATOR {ind.submission_id} ({ind.name}): {ind.description}\n"
                f"{ind.long_definition or ''}\n\nCLAUSE:\n{text}")
        try:
            data = client.chat(_BINARY_SYSTEM, user, json_schema=_BINARY_SCHEMA)
        except Exception:
            return None
        if isinstance(data, dict) and data.get("supports") is True:
            out.add(ind.submission_id)
    return out


def run_pass(verifier, pool_ids, clause_by_id, indicators, workers: int,
             samples: int = 1, binary: bool = False) -> tuple[dict, float]:
    def _judge(cid):
        if binary:
            verdict = judge_binary(verifier._client, clause_by_id[cid], indicators)
            if verdict:
                verdict = {s for s in verdict
                           if admits_clause(_RDTII_BY_SID[s], clause_by_id[cid].span.text)}
            return cid, verdict
        verdict = verifier.judge_clause(clause_by_id[cid], indicators)
        for _ in range(samples - 1):
            # Self-consistency: the model is not deterministic even at temperature 0, so
            # a second independent look surfaces indicators the first pass happened to
            # miss. UNION (not intersection) because the failure we are chasing is recall.
            more = verifier.judge_clause(clause_by_id[cid], indicators)
            if more:
                verdict = (verdict or set()) | more
        if verdict:
            # Production applies the boundary rule to the verdict, so the bench must too.
            verdict = {
                sid for sid in verdict
                if admits_clause(_RDTII_BY_SID[sid], clause_by_id[cid].span.text)
            }
        return cid, verdict

    t0 = time.perf_counter()
    if workers > 1 and len(pool_ids) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(workers, len(pool_ids))) as ex:
            verdicts = dict(ex.map(_judge, pool_ids))
    else:
        verdicts = dict(_judge(cid) for cid in pool_ids)
    return verdicts, time.perf_counter() - t0


_RDTII_BY_SID: dict[str, str] = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso", default="my")
    ap.add_argument("--gold", type=Path, default=None, help="gold CSV (default: MY legal-group)")
    ap.add_argument("--match", default=None, help="comma-separated strings identifying the doc")
    ap.add_argument("--pdf", type=Path, default=None)
    ap.add_argument("--pool-k", type=int, default=int(os.environ.get("LEXORA_MAP_POOL_K", "40")))
    ap.add_argument("--min-score", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="cap the pool (0 = full pool)")
    ap.add_argument("--model", default=None, help="override LEXORA_BRUTE_MODEL")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--system-file", type=Path, default=None,
                    help="replace the per-clause system prompt with this file (A/B a wording)")
    ap.add_argument("--thinking", choices=["on", "off"], default=None,
                    help="override LEXORA_LLM_DISABLE_THINKING for this bench")
    ap.add_argument("--max-tokens", type=int, default=0, help="override LEXORA_LLM_MAX_TOKENS")
    ap.add_argument("--binary", action="store_true",
                    help="ask one yes/no per indicator instead of the 9-in-1 subset")
    ap.add_argument("--samples", type=int, default=1,
                    help="self-consistency: union N independent verdicts per clause")
    ap.add_argument("--dry-run", action="store_true", help="pool stats only, no LLM calls")
    ap.add_argument("--use-cache", action="store_true", help="allow the verdict cache (default: cold)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from lexora.collect.profile_loader import load_profile

    profile = load_profile(REPO / "configs" / "jurisdictions" / f"{args.iso}.yaml")
    indicators = load_indicators(INDICATORS, pillars=[6, 7])
    _RDTII_BY_SID.update({i.submission_id: i.rdtii_id for i in indicators})

    gold_path = args.gold or GOLD_MY
    needles = tuple(x.strip() for x in args.match.split(",")) if args.match else (
        "PERSONAL DATA PROTECTION ACT 2010", "Act 709")
    pdf = args.pdf or find_doc(args.iso, needles)
    clauses = clauses_from_pdf(pdf, profile, indicators)
    clause_by_id = {c.clause_id: c for c in clauses}
    index, pool_score, pair_rank = build_pool(
        clauses, indicators, profile, pool_k=args.pool_k, min_score=args.min_score
    )
    pool_ids = sorted(pool_score, key=lambda c: pool_score[c], reverse=True)
    if args.limit:
        pool_ids = pool_ids[: args.limit]

    print(f"document : {pdf.name}  ({len(clauses)} clauses parsed)")
    print(f"pool_k   : {args.pool_k}  min_score {args.min_score}")
    print(f"pool     : {len(pool_ids)} clause(s) -> {len(pool_ids)} judge call(s) per pass")
    est = sum(len(clause_by_id[c].span.text) for c in pool_ids) // 4
    print(f"est clause tokens in pool: ~{est}  (+ ~3.1k catalogue per call)")

    if args.dry_run:
        # What a narrower pool would cost, and which gold sections it would lose.
        gold = load_gold(gold_path, args.iso)
        want = {s for v in gold.values() for s in v}
        for k in (10, 15, 20, 25, 30, 40):
            _, ps, _ = build_pool(clauses, indicators, profile, pool_k=k, min_score=args.min_score)
            keys = {_clause_key(clause_by_id[c]) for c in ps}
            print(f"  pool_k={k:3d} -> {len(ps):4d} calls, gold sections reachable "
                  f"{len(keys & want)}/{len(want)}")
        return

    if not args.use_cache:
        os.environ["LEXORA_JUDGE_CACHE"] = "0"
    if args.model:
        os.environ["LEXORA_BRUTE_MODEL"] = args.model
    if args.thinking:
        os.environ["LEXORA_LLM_DISABLE_THINKING"] = "0" if args.thinking == "on" else "1"
    if args.max_tokens:
        os.environ["LEXORA_LLM_MAX_TOKENS"] = str(args.max_tokens)

    from lexora.classify.llm_client import LlmClient

    if args.system_file:
        from lexora.classify import verifier as _v

        _v._PER_CLAUSE_SYSTEM = args.system_file.read_text(encoding="utf-8")
        print(f"system prompt: {args.system_file} ({len(_v._PER_CLAUSE_SYSTEM)} chars)")

    runs = []
    for n in range(args.repeat):
        client = LlmClient(model=args.model or os.environ.get("LEXORA_BRUTE_MODEL"))
        verifier = Verifier(client, mode="per_clause", cache=None)
        verdicts, wall = run_pass(verifier, pool_ids, clause_by_id, indicators, args.workers,
                                  samples=args.samples, binary=args.binary)
        gold = load_gold(gold_path, args.iso)
        metrics = score_verdicts(verdicts, clause_by_id, gold)
        cached_share = client.cached_prompt_tokens / client.prompt_tokens if client.prompt_tokens else 0.0
        cost = cost_cny(client.model, client.prompt_tokens, client.cached_prompt_tokens,
                        client.completion_tokens)
        runs.append({
            "model": client.model, "wall_s": round(wall, 1),
            "calls": client.calls, "failed": client.failed_calls,
            "prompt": client.prompt_tokens, "cached_prompt": client.cached_prompt_tokens,
            "completion": client.completion_tokens,
            "cached_share": round(cached_share, 3),
            "cost_cny": round(cost, 4) if cost is not None else None,
            "verdicts": {c: sorted(v or ()) for c, v in verdicts.items()},
            **{k: v for k, v in metrics.items() if k != "per_indicator"},
            "per_indicator": metrics["per_indicator"],
        })
        r = runs[-1]
        print(f"\n--- pass {n + 1}/{args.repeat}  [{r['model']}] ---")
        print(f"  wall {r['wall_s']}s  calls {r['calls']} ({r['failed']} failed)  "
              f"{r['wall_s'] / max(1, r['calls']):.2f}s/call")
        print(f"  tokens: {r['prompt']} prompt ({r['cached_share']:.0%} cached) "
              f"+ {r['completion']} completion"
              + (f"   cost CNY {r['cost_cny']:.3f}" if r["cost_cny"] is not None else ""))
        print(f"  gold: recall {r['recall']:.0%} ({sum(len(x['hit']) for x in r['per_indicator'])}"
              f"/{r['gold_sections']})  precision {r['precision']:.0%}  "
              f"N/A-violations {r['na_violations']}  citations {r['citations']}")

    if args.repeat > 1:
        stable = sum(
            1 for cid in pool_ids
            if len({tuple(r["verdicts"].get(cid, ())) for r in runs}) == 1
        )
        print(f"\nreproducibility over {args.repeat} cold passes:")
        print(f"  identical verdict on {stable}/{len(pool_ids)} clauses "
              f"({stable / len(pool_ids):.1%})")
        print(f"  citations per pass: {[r['citations'] for r in runs]} "
              f"(stdev {statistics.pstdev([r['citations'] for r in runs]):.1f})")
        print(f"  recall per pass:    {[round(r['recall'], 3) for r in runs]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "document": pdf.name, "pool_k": args.pool_k, "pool": len(pool_ids),
            "runs": runs,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
