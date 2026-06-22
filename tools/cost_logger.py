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
    verifier = make_verifier(use_llm=use_llm, mode="per_cell")
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
        top_k=3, verifier=verifier, rationale_gen=rationale, meta_extractor=meta,
    )
    wall = time.perf_counter() - t0

    # measured token counts (sum across the three LLM stages)
    p_meta, c_meta, n_meta = _client_tokens(getattr(meta, "_client", None))
    p_rat, c_rat, n_rat = _client_tokens(getattr(rationale, "_client", None))
    p_ver, c_ver, n_ver = _client_tokens(getattr(verifier, "_client", None) if verifier else None)
    in_tokens = p_meta + p_rat + p_ver
    out_tokens = c_meta + c_rat + c_ver
    calls = n_meta + n_rat + n_ver

    n_pages = len(art.pages)
    n_chars = len(art.document_text)
    scanned = art.pdf_is_scanned
    ocr_engine = art.ocr_engine or (cfg_engine() if scanned else "")

    llm_cost = (in_tokens / 1e6) * args.price_in + (out_tokens / 1e6) * args.price_out
    # self-hosted components: $0 marginal API cost
    ocr_cost = embed_cost = crawl_cost = 0.0
    total = ocr_cost + embed_cost + crawl_cost + llm_cost

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
                "price_in_per_1m_usd": args.price_in, "price_out_per_1m_usd": args.price_out,
                "rate_note": "reference commercial rate; our own endpoint is not billed per token",
                "cost_usd": round(llm_cost, 4)},
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
    print(f"  wall-clock={wall:.1f}s   tokens: in={in_tokens:,} out={out_tokens:,} ({calls} calls)\n")
    print(f"{'Component':<22}{'Engine':<34}{'Measured cost':>14}")
    print("-" * 70)
    print(f"{'OCR':<22}{(ocr_engine or 'RapidOCR (local)'):<34}{money(ocr_cost):>14}")
    print(f"{'Embedding':<22}{(cfg.embedding_model+' (local, off)'):<34}{money(embed_cost):>14}")
    print(f"{'LLM mapping':<22}{cfg.llm_model:<34}{money(llm_cost):>14}")
    print(f"{'Crawling':<22}{'self-hosted':<34}{money(crawl_cost):>14}")
    print("-" * 70)
    print(f"{'Total (current stack)':<56}{money(total):>14}")
    print(f"{'Total (open-weight swap)':<56}{money(0.0):>14}")
    print(f"\nwrote {args.out.relative_to(REPO)}")
    return 0


def cfg_engine() -> str:
    return os.environ.get("LEXORA_OCR_ENGINE", "rapidocr")


if __name__ == "__main__":
    raise SystemExit(main())
