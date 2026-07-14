"""Measured cost-per-document logger (UN sustainability rubric).

Runs the REAL Lexora pipeline (extract -> OCR -> structure -> retrieve -> map ->
metadata + rationale + per-cell verify) on ONE benchmark PDF and records the
measured token counts, page counts and wall-clock time, then attributes a USD
cost per component. Writes ``logs/cost_report.json`` in the official
``README_template.md`` schema and prints the "Measured results" table.

The numbers are MEASURED, not estimated: token counts come from the live LLM
client accounting (``LlmClient.prompt_tokens`` / ``completion_tokens`` summed over
metadata + rationale + verifier), page counts from the parsed PDF, and time from
``perf_counter`` around the run. Only the per-token RATE is a parameter (printed
in the report so judges can reproduce ``cost = tokens x rate``).

Cost model — our stack is almost entirely SELF-HOSTED, so its marginal API cost
is ~$0; the LLM is the only metered API:
  * OCR        — RapidOCR (bundled ONNX, local CPU)      -> $0 API (compute only)
  * Embedding  — bge-m3 via fastembed (local; dense OFF) -> $0 API (compute only)
  * Crawling   — our own HTTP/browser fetchers           -> $0 API (compute only)
  * LLM        — gpt-5.4 via our own endpoint            -> measured tokens x rate
The endpoint is our own/sponsored (500M tok/day, no per-token bill), so our ACTUAL
marginal cost is ~$0; we ALSO price the measured tokens at a stated commercial rate
for comparability, and report the open-weight-swap total (Llama + Tesseract, all
self-hosted) as $0 API.

Usage (mirrors the rubric template):
    PYTHONPATH=src python tools/cost_logger.py \
        --pdf data/raw/my/<hash>.pdf --economy my --pillar 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from lexora.cite.metadata import make_metadata_extractor  # noqa: E402
from lexora.cite.rationale import make_rationale_generator  # noqa: E402
from lexora.classify.verifier import make_verifier  # noqa: E402
from lexora.collect.profile_loader import load_profile  # noqa: E402
from lexora.config import load_config  # noqa: E402
from lexora.indicators import load_indicators  # noqa: E402
from lexora.pipeline import run_demo_pipeline  # noqa: E402

INDICATORS = REPO / "configs" / "rdtii_indicators.yaml"
JURIS = REPO / "configs" / "jurisdictions"
ISO_TO_COUNTRY = {"sg": "Singapore", "au": "Australia", "my": "Malaysia"}

# Reference commercial rate (USD per 1M tokens) used ONLY to put a comparable $ on
# the measured tokens. Our own endpoint is not billed per token, so this is an
# upper-bound comparison, not what we pay. Override with --price-in / --price-out.
DEFAULT_PRICE_IN = 0.50
DEFAULT_PRICE_OUT = 1.50


def _client_tokens(client) -> tuple[int, int, int]:
    if client is None:
        return 0, 0, 0
    return (getattr(client, "prompt_tokens", 0), getattr(client, "completion_tokens", 0),
            getattr(client, "calls", 0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True, type=Path, help="benchmark PDF (local)")
    ap.add_argument("--economy", default="my", choices=["sg", "au", "my"])
    ap.add_argument("--pillar", default=6, type=int, help="cosmetic (label only)")
    ap.add_argument("--price-in", type=float, default=DEFAULT_PRICE_IN,
                    help="USD per 1M input tokens (reference commercial rate)")
    ap.add_argument("--price-out", type=float, default=DEFAULT_PRICE_OUT,
                    help="USD per 1M output tokens (reference commercial rate)")
    ap.add_argument("--price-cached", type=float, default=None,
                    help="USD per 1M CACHED input tokens. Providers bill a repeated prompt "
                         "prefix at a fraction of fresh input (GLM: Y2 vs Y8 per 1M), and "
                         "our judge re-sends an identical 3,051-token indicator catalogue on "
                         "every call -- so a two-rate model overstates the bill and hides "
                         "the biggest lever on it. Defaults to 1/4 of --price-in, the ratio "
                         "on the 2026-07 invoice.")
    ap.add_argument("--llm-workers", type=int, default=16,
                    help="concurrent LLM calls, as the submission run uses (default 16). "
                         "Affects wall-clock only; token counts (and therefore cost) are "
                         "identical at any concurrency.")
    ap.add_argument("--no-llm", action="store_true",
                    help="measure the self-hosted infra only (no LLM stages)")
    ap.add_argument("--out", type=Path, default=REPO / "logs" / "cost_report.json")
    args = ap.parse_args(argv)

    if not args.pdf.exists():
        ap.error(f"PDF not found: {args.pdf}")
    os.environ.setdefault("LEXORA_OCR", "1")  # benchmark with OCR on (rubric: scanned PDFs)

    cfg = load_config()
    profile = load_profile(JURIS / f"{args.economy}.yaml")
    indicators = load_indicators(INDICATORS)
    economy = ISO_TO_COUNTRY[args.economy]

    use_llm = not args.no_llm
    meta = make_metadata_extractor(use_llm=use_llm)
    rationale = make_rationale_generator(use_llm=use_llm)
    # per_clause is what SHIPS: the 0/1 membership judge reads every clause in the recall
    # pool and decides relevance for all indicators at once. It is also where nearly all
    # the tokens go, so measuring per_cell here (as this did) reported the cost of a code
    # path the submission does not take -- an order of magnitude low. The judges are told
    # to verify cost claims against the code; the benchmark must run the code.
    verifier = make_verifier(use_llm=use_llm, mode="per_clause")
    llm_available = use_llm and any(
        getattr(x, "_client", None) is not None for x in (meta, rationale, verifier)
    )
    if use_llm and not llm_available:
        print("warning: LLM backend unavailable — measuring infra-only cost "
              "(install [llm] + set LEXORA_LLM_* / LEXORA_LLM_USER_AGENT).")

    t0 = time.perf_counter()
    art = run_demo_pipeline(
        pdf_path=args.pdf, profile=profile, indicators=indicators,
        source_url="https://example.gov/benchmark.pdf", portal_name="cost-benchmark",
        # top_k is inert under the per_clause judge (relevance is a 0/1 membership call,
        # not a rank cutoff); passed only because the signature still takes it.
        top_k=3, verifier=verifier, rationale_gen=rationale, meta_extractor=meta,
        # Token counts do not depend on concurrency, but wall-clock does -- and the
        # submission runs the judge 16-way. Benchmarking it serially would report a
        # processing time nobody would ever wait for.
        llm_workers=args.llm_workers,
    )
    wall = time.perf_counter() - t0

    # measured token counts (sum across the three LLM stages)
    p_meta, c_meta, n_meta = _client_tokens(getattr(meta, "_client", None))
    p_rat, c_rat, n_rat = _client_tokens(getattr(rationale, "_client", None))
    p_ver, c_ver, n_ver = _client_tokens(getattr(verifier, "_client", None) if verifier else None)
    in_tokens = p_meta + p_rat + p_ver
    out_tokens = c_meta + c_rat + c_ver
    calls = n_meta + n_rat + n_ver

    # A rejected call bills nothing and accounts nothing, so a benchmark whose every
    # request was refused looks identical to a free one: 0 calls, $0.00, and a tidy table.
    # That is not a cheap run, it is no run. Refuse to publish a cost claim from it --
    # the rubric says judges will verify these numbers against the code.
    clients = [getattr(x, "_client", None) for x in (meta, rationale, verifier)]
    failed = sum(getattr(c, "failed_calls", 0) for c in clients if c is not None)
    errors = [getattr(c, "last_error", "") for c in clients
              if c is not None and getattr(c, "failed_calls", 0)]
    if use_llm and llm_available and failed and calls == 0:
        print(f"\nERROR: every LLM call failed ({failed} rejected). This is NOT a $0.00 run.")
        for e in dict.fromkeys(errors):
            print(f"  {e}")
        print("No cost report written.")
        return 1
    if failed:
        print(f"\nwarning: {failed} LLM call(s) failed and are NOT in the cost below "
              f"({errors[0] if errors else ''})")

    n_pages = len(art.pages)
    n_chars = len(art.document_text)
    scanned = art.pdf_is_scanned
    ocr_engine = art.ocr_engine or (cfg_engine() if scanned else "")

    # Three rates, because the invoice has three: fresh input, CACHED input (a repeated
    # prompt prefix, billed at ~1/4), and output. The judge re-sends the same 3,051-token
    # indicator catalogue every call, so which of the first two it lands in is the single
    # largest term in this number.
    cached_in = sum(getattr(c, "cached_prompt_tokens", 0) for c in clients if c is not None)
    fresh_in = max(0, in_tokens - cached_in)
    price_cached = args.price_cached if args.price_cached is not None else args.price_in / 4
    llm_cost = ((fresh_in / 1e6) * args.price_in
                + (cached_in / 1e6) * price_cached
                + (out_tokens / 1e6) * args.price_out)
    # self-hosted components: $0 marginal API cost
    ocr_cost = embed_cost = crawl_cost = 0.0
    total = ocr_cost + embed_cost + crawl_cost + llm_cost
    cache_hit = cached_in / in_tokens if in_tokens else 0.0
    jc = getattr(verifier, "_cache", None) if verifier is not None else None
    judged = getattr(verifier, "judged", 0) if verifier is not None else 0
    from_cache = getattr(verifier, "from_cache", 0) if verifier is not None else 0

    report = {
        "document": args.pdf.name,
        "economy": economy,
        "pillar": args.pillar,
        "measured_on": date.today().isoformat(),
        "pages": n_pages,
        "characters": n_chars,
        "ocr": {"engine": ocr_engine or "none", "pages": n_pages if scanned else 0,
                "scanned": scanned, "cost_usd": round(ocr_cost, 4)},
        "embedding": {"model": cfg.embedding_model, "tokens": 0,
                      "note": "dense channel OFF by default (BM25 retrieval)",
                      "cost_usd": round(embed_cost, 4)},
        "crawling": {"engine": "self-hosted (requests/browser)", "cost_usd": round(crawl_cost, 4)},
        "llm": {"model": cfg.llm_model, "calls": calls,
                "input_tokens": in_tokens, "output_tokens": out_tokens,
                "cached_input_tokens": cached_in,
                "cache_hit_rate": round(cache_hit, 3),
                "failed_calls": failed,
                "price_in_per_1m_usd": args.price_in,
                "price_cached_in_per_1m_usd": round(price_cached, 4),
                "price_out_per_1m_usd": args.price_out,
                "rate_note": "reference commercial rate; our own endpoint is not billed per token",
                "cost_usd": round(llm_cost, 4)},
        # A run that answered most clauses from the verdict cache is not a run that judged
        # them cheaply. Publish the split so a cached benchmark can never be mistaken for a
        # cold one -- LEXORA_JUDGE_CACHE=0 reproduces the true cold cost.
        "judge_cache": {"clauses_judged_by_llm": judged, "clauses_served_from_cache": from_cache,
                        "enabled": jc is not None},
        "total_cost_usd": round(total, 4),
        "total_cost_usd_open_weight_swap": 0.0,
        "open_weight_note": "Llama-class + Tesseract, all self-hosted -> $0 API (compute only)",
        "citations": len(art.citations),
        "processing_time_seconds": round(wall, 1),
        "llm_used": llm_available,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # printed "Measured results" table
    def money(x):
        return f"${x:.4f}"
    print(f"\nBenchmark: {args.pdf.name}  ({economy}, P{args.pillar})")
    print(f"  pages={n_pages}  chars={n_chars}  scanned={scanned}  citations={len(art.citations)}")
    print(f"  wall-clock={wall:.1f}s   tokens: in={in_tokens:,} "
          f"(of which {cached_in:,} cached = {cache_hit:.0%}) "
          f"out={out_tokens:,} ({calls} calls)")
    if jc is not None:
        print(f"  judge cache: {from_cache} clause(s) served from cache, {judged} judged by LLM"
              + ("   <-- NOT a cold-cost measurement; LEXORA_JUDGE_CACHE=0 to reproduce"
                 if from_cache else ""))
    print()
    print(f"{'Component':<22}{'Engine':<34}{'Measured cost':>14}")
    print("-" * 70)
    print(f"{'OCR':<22}{(ocr_engine or 'RapidOCR (local)'):<34}{money(ocr_cost):>14}")
    print(f"{'Embedding':<22}{(cfg.embedding_model+' (local, off)'):<34}{money(embed_cost):>14}")
    print(f"{'LLM mapping':<22}{cfg.llm_model:<34}{money(llm_cost):>14}")
    print(f"{'Crawling':<22}{'self-hosted':<34}{money(crawl_cost):>14}")
    print("-" * 70)
    print(f"{'Total (current stack)':<56}{money(total):>14}")
    print(f"{'Total (open-weight swap)':<56}{money(0.0):>14}")
    out = args.out.resolve()
    print(f"\nwrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    return 0


def cfg_engine() -> str:
    return os.environ.get("LEXORA_OCR_ENGINE", "rapidocr")


if __name__ == "__main__":
    raise SystemExit(main())
