"""Full Round-1 submission run (P-5).

Runs the autonomous multi-instrument map for every mandatory economy
(SG / AU / MY) end to end and concatenates the verbatim-validated citations into
ONE official 13-column submission CSV (+ JSON-LD), each row already carrying its
NEW/KNOWN discovery tag. This is the Round-1 deliverable prototype: one command,
one file, no URL handed in.

It reuses the exact production path (`run_pipeline_map` per economy, the same
exporter as `lexora map`), so what it emits is what a judge would score. A
per-economy summary (instruments discovered, full texts fetched, citations,
NEW/KNOWN split, indicators covered, rows routed to review) is printed and
written alongside the CSV.

Usage:
    LEXORA_LIVE=1 python scripts/run_submission.py            # SG + AU + MY
    LEXORA_LIVE=1 python scripts/run_submission.py -j sg --verify
    python scripts/run_submission.py --dry-run                # plan only, no network
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexora.cite.metadata import make_metadata_extractor  # noqa: E402
from lexora.cite.rationale import make_rationale_generator  # noqa: E402
from lexora.classify.verifier import make_verifier  # noqa: E402
from lexora.collect.profile_loader import load_profile  # noqa: E402
from lexora.config import load_config  # noqa: E402
from lexora.export.csv_exporter import to_csv  # noqa: E402
from lexora.export.jsonld_exporter import to_jsonld  # noqa: E402
from lexora.indicators import load_indicators  # noqa: E402
from lexora.pipeline import MapResult, run_pipeline_map  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
INDICATORS = REPO / "configs" / "rdtii_indicators.yaml"
JURIS = REPO / "configs" / "jurisdictions"
OUT_CSV = REPO / "outputs" / "submission_round1.csv"

ISO_TO_COUNTRY = {"sg": "Singapore", "au": "Australia", "my": "Malaysia"}


def run_one(
    iso: str,
    *,
    budget: int,
    verify: bool,
    rationale_llm: bool = False,
    metadata_llm: bool = False,
    timeout: float,
    llm_workers: int = 1,
    doc_workers: int = 1,
    fetch_min_interval: float = 0.0,
    serial_fetch: bool = False,
    use_secondary: bool = False,
    verify_cells: bool = False,
) -> MapResult:
    """Run the production multi-instrument map for one economy."""
    profile = load_profile(JURIS / f"{iso.lower()}.yaml")
    indicators = load_indicators(INDICATORS)
    secondary = []
    if use_secondary:
        from lexora.collect.secondary import gather_signals

        secondary = gather_signals(iso, indicators)
        print(f"  secondary sources [{iso}]: {len(secondary)} signal(s) from "
              f"{len({s.source_name for s in secondary})} tracker(s)")
    # --verify-cells (per-cell universal precision lane) takes precedence over the
    # legacy pick-one --verify.
    verifier = make_verifier(use_llm=verify or verify_cells,
                             mode="per_cell" if verify_cells else "pick_one")
    if (verify or verify_cells) and verifier is None:
        print("warning: --verify requested but the LLM verifier is unavailable "
              "(install the [llm] extra); continuing with BM25 + verbatim only.")
    rationale_gen = make_rationale_generator(use_llm=rationale_llm)
    if rationale_llm and rationale_gen._client is None:
        print("warning: --rationale-llm requested but the LLM backend is unavailable; "
              "using the deterministic template rationale.")
    meta_extractor = make_metadata_extractor(use_llm=metadata_llm)
    if metadata_llm and meta_extractor._client is None:
        print("warning: --metadata-llm requested but the LLM backend is unavailable; "
              "using portal structured metadata only.")
    result = run_pipeline_map(
        portal=profile.portals[0], profile=profile, indicators=indicators,
        budget=budget, timeout=timeout, verifier=verifier, rationale_gen=rationale_gen,
        meta_extractor=meta_extractor, llm_workers=llm_workers, doc_workers=doc_workers,
        fetch_min_interval=fetch_min_interval, serial_fetch=serial_fetch,
        secondary_signals=secondary,
    )
    tokens = {"calls": 0, "prompt": 0, "completion": 0, "total": 0}

    def _account(label: str, client) -> None:
        """Print per-channel token usage and fold it into this economy's total."""
        if client is None or not getattr(client, "calls", 0):
            return
        print(
            f"  LLM {label} tokens [{iso}]: {client.calls} call(s), "
            f"{client.total_tokens} tokens "
            f"({client.prompt_tokens} prompt + {client.completion_tokens} completion)"
        )
        tokens["calls"] += client.calls
        tokens["prompt"] += client.prompt_tokens
        tokens["completion"] += client.completion_tokens
        tokens["total"] += client.total_tokens

    if meta_extractor._client is not None:
        print(
            f"  LLM metadata usage [{iso}]: {meta_extractor.extracted} doc(s) extracted, "
            f"{meta_extractor.rejected} field(s) rejected by source-check"
            + (f" ({meta_extractor.error_count} backend error(s))"
               if meta_extractor.error_count else "")
        )
        _account("metadata", meta_extractor._client)
    if rationale_gen._client is not None:
        print(
            f"  LLM rationale usage [{iso}]: {rationale_gen.llm_used} authored, "
            f"{rationale_gen.fallbacks} fell back to template"
            + (f" ({rationale_gen.error_count} backend error(s))"
               if rationale_gen.error_count else "")
        )
        _account("rationale", rationale_gen._client)
    if verifier is not None and getattr(verifier, "error_count", 0):
        print(
            f"warning: LLM verifier backend errors for {iso}: "
            f"{verifier.error_count} judgement(s) failed "
            f"(last error: {verifier.last_error_type or 'unknown'}); "
            "run continued without fabricating citations."
        )
    _account("verifier", getattr(verifier, "_client", None) if verifier is not None else None)
    return result, tokens


def summarize(iso: str, result: MapResult) -> dict:
    """Per-economy counts for the run summary (pure — unit-tested offline).

    NEW/KNOWN is reported on the *instruments* discovered (the 20/40-point NEW
    capability is about finding laws), while the citation counts describe the CSV
    rows actually emitted and how many indicators they cover."""
    fetched_ok = sum(1 for d in result.documents if 200 <= d.document.http_status < 300)
    # Real yield: documents that actually parsed into clauses. `fetched_ok` (2xx count)
    # is MISLEADING under anti-bot — AU serves an HTTP-200 HTML challenge impersonating
    # the PDF (0 clauses), so the 2xx count overstates what the run can map. Track the
    # metric that reflects real output to judge the serial-fetch / throttle fix.
    docs_with_clauses = sum(1 for d in result.documents if d.clauses)
    new_instruments = sum(1 for r in result.discovered if r.discovery_tag == "NEW")
    known_instruments = sum(1 for r in result.discovered if r.discovery_tag == "KNOWN")
    indicators_covered = sorted({c.indicator_id for c in result.citations})
    review_rows = sum(
        1 for c in result.citations if c.review_status.value == "CONFLICT_REVIEW"
    )
    return {
        "iso": iso,
        "economy": ISO_TO_COUNTRY.get(iso, iso),
        "instruments": len(result.discovered),
        "new_instruments": new_instruments,
        "known_instruments": known_instruments,
        "fetched_ok": fetched_ok,
        "docs_with_clauses": docs_with_clauses,
        "citations": len(result.citations),
        "indicators_covered": indicators_covered,
        "n_indicators_covered": len(indicators_covered),
        "review_rows": review_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--jurisdiction", default="all", help="sg|au|my|all")
    ap.add_argument("--budget", type=int, default=20, help="Max instruments per economy")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--verify", action="store_true",
                    help="Tighten mappings with the legacy pick-one LLM verifier "
                         "(≤1 clause/indicator/doc; needs LEXORA_LLM_* endpoint)")
    ap.add_argument("--verify-cells", action="store_true",
                    help="Universal per-cell LLM verifier: judge EVERY (clause × "
                         "indicator) keep/drop to kill the broad-statute-floods-all-9 "
                         "false positives. On endpoint error a cell is KEPT (never "
                         "worse than baseline). Needs LEXORA_LLM_* endpoint.")
    ap.add_argument("--rationale-llm", action="store_true",
                    help="Author the Mapping Rationale column with the LLM (template fallback "
                         "+ verbatim-copy guard; needs LEXORA_LLM_* endpoint)")
    ap.add_argument("--metadata-llm", action="store_true",
                    help="Extract Law Number / Last Amended from document text with the LLM "
                         "(source-verified) when portal channel + curated anchor don't supply them")
    ap.add_argument("--check-links", action="store_true",
                    help="Probe each citation's Source URL for reachability and annotate "
                         "dead links in Notes (extra network I/O; off by default)")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--jobs", type=int, default=0,
                    help="Economies to run in parallel (country-level parallelism). "
                         "0 = auto (all requested economies at once); 1 = serial. "
                         "Each economy is independent (own portal / dest_dir / LLM clients); "
                         "OCR (onnxruntime) and network/LLM I/O release the GIL, so threads "
                         "give real speedup. SG is the only browser portal, so no cross-economy "
                         "browser contention.")
    ap.add_argument("--llm-workers", type=int, default=1,
                    help="Threads for the per-citation Mapping Rationale LLM calls within a "
                         "document (LLM-call-layer parallelism). 1 = serial. Runtime-adjustable "
                         "per run. Stacks with --jobs (e.g. --jobs 3 --llm-workers 8 = up to 24 "
                         "concurrent requests; the endpoint handles >=32 with no rate limit).")
    ap.add_argument("--doc-workers", type=int, default=1,
                    help="Threads for processing instruments within an economy concurrently "
                         "(document-level parallelism). This is what parallelizes the per-document "
                         "metadata extraction (the serial floor of an LLM run), plus fetch/OCR/"
                         "rationale across documents. 1 = serial. Stacks with --jobs and "
                         "--llm-workers.")
    ap.add_argument("--fetch-min-interval", type=float, default=0.0,
                    help="Minimum seconds between fetch starts to the SAME host (per-host "
                         "rate-spacing, thread-enforced). Dodges request-rate anti-bot under "
                         "doc-level concurrency (e.g. AU serving an HTML challenge instead of "
                         "the PDF); post-fetch OCR/LLM still parallelize. 0 = off.")
    ap.add_argument("--secondary", action="store_true",
                    help="Use RDTII secondary sources (UNCTAD etc.) as a DISCOVERY AID: seed "
                         "discovery with the laws they point to (recall), stamp matching "
                         "citations with a 'corroborated by <source>' note (provenance), and "
                         "print a coverage cross-check. Never cited as evidence.")
    ap.add_argument("--serial-fetch", action="store_true",
                    help="Fully serialize same-host DOWNLOADS (one request in flight per host) "
                         "while OCR/parse/map/LLM still run parallel across documents. Stronger "
                         "than --fetch-min-interval against cumulative anti-bot (no request burst "
                         "at all); different economies (hosts) stay parallel. Use with --doc-workers.")
    ap.add_argument("--dry-run", action="store_true", help="plan only, no network")
    args = ap.parse_args()

    isos = ["sg", "au", "my"] if args.jurisdiction == "all" else [args.jurisdiction.lower()]

    if args.dry_run:
        print("Submission run plan (no network):")
        for iso in isos:
            print(f"  - {ISO_TO_COUNTRY.get(iso, iso)} ({iso}) "
                  f"-> map budget {args.budget}, verify={args.verify}, "
                  f"rationale_llm={args.rationale_llm}")
        if args.verify:
            verifier = make_verifier(use_llm=True)
            if verifier is None:
                print("  -> LLM verifier unavailable (install the [llm] extra)")
            else:
                cfg = load_config()
                print(f"  -> LLM verifier ON (model: {cfg.llm_model})")
        print(f"  -> would write {args.out} (+ .jsonld) and {args.out.with_suffix('.summary.json')}")
        return

    if not os.environ.get("LEXORA_LIVE"):
        print("note: set LEXORA_LIVE=1 to run the live submission crawl (or use --dry-run).")

    all_citations = []
    summaries = []
    results_by_iso = {}
    tokens_total = {"calls": 0, "prompt": 0, "completion": 0, "total": 0}

    def _run(iso: str):
        return run_one(iso, budget=args.budget, verify=args.verify,
                       rationale_llm=args.rationale_llm, metadata_llm=args.metadata_llm,
                       timeout=args.timeout, llm_workers=args.llm_workers,
                       doc_workers=args.doc_workers, fetch_min_interval=args.fetch_min_interval,
                       serial_fetch=args.serial_fetch, use_secondary=args.secondary,
                       verify_cells=args.verify_cells)

    # Country-level parallelism: economies are independent, so run them concurrently.
    # Threads (not processes) because the heavy stages — network fetch, OCR
    # (onnxruntime releases the GIL), LLM calls — are I/O- or C++-bound; this dodges
    # Windows spawn/pickling and lets workers share one process. Results collected
    # per-iso so one economy failing can't lose the others, then walked in input
    # order for a stable summary table.
    jobs = args.jobs if args.jobs > 0 else len(isos)
    outcomes: dict[str, tuple[str, object]] = {}
    if jobs > 1 and len(isos) > 1:
        from concurrent.futures import ThreadPoolExecutor

        print(f"Running {len(isos)} economies with {min(jobs, len(isos))} parallel worker(s)...")
        with ThreadPoolExecutor(max_workers=min(jobs, len(isos))) as ex:
            futs = {ex.submit(_run, iso): iso for iso in isos}
            for fut, iso in futs.items():
                try:
                    outcomes[iso] = ("ok", fut.result())
                except Exception as exc:  # one economy failing must not lose the others
                    outcomes[iso] = ("err", exc)
    else:
        for iso in isos:
            try:
                outcomes[iso] = ("ok", _run(iso))
            except Exception as exc:
                outcomes[iso] = ("err", exc)

    for iso in isos:  # input order -> stable summary table
        kind, payload = outcomes[iso]
        if kind == "err":
            summaries.append({"iso": iso, "economy": ISO_TO_COUNTRY.get(iso, iso),
                              "error": f"{type(payload).__name__}: {payload}"})
            continue
        result, tokens = payload
        all_citations.extend(result.citations)
        results_by_iso[iso] = result
        summaries.append(summarize(iso, result))
        for k in tokens_total:
            tokens_total[k] += tokens[k]

    dead_links = 0
    if args.check_links and all_citations:
        from lexora.collect.liveness import annotate_dead_links, check_urls

        checks = check_urls(str(c.source_url) for c in all_citations)
        all_citations, dead_links = annotate_dead_links(all_citations, checks)
        print(f"\nLink check: probed {len(checks)} distinct URL(s), "
              f"{dead_links} citation row(s) carry a dead-link note.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = to_csv(all_citations, args.out)
    jsonld_out = args.out.with_suffix(".jsonld")
    to_jsonld(all_citations, jsonld_out)
    summary_out = args.out.with_suffix(".summary.json")
    summary_out.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nRound-1 submission run (budget {args.budget}"
          f"{', verifier ON' if args.verify else ''})")
    print(f"{'economy':<12}{'instr':<7}{'NEW':<5}{'KNOWN':<7}{'fetched':<9}{'real':<6}"
          f"{'cites':<7}{'inds':<6}{'review'}")
    print("-" * 70)
    for s in summaries:
        if s.get("error"):
            print(f"{s['economy']:<12}ERROR: {s['error']}")
            continue
        print(f"{s['economy']:<12}{s['instruments']:<7}{s['new_instruments']:<5}"
              f"{s['known_instruments']:<7}{s['fetched_ok']:<9}{s['docs_with_clauses']:<6}"
              f"{s['citations']:<7}{s['n_indicators_covered']:<6}{s['review_rows']}")
    print("-" * 70)
    if tokens_total["calls"]:
        print(
            f"LLM token total (all economies): {tokens_total['total']} tokens across "
            f"{tokens_total['calls']} call(s) "
            f"({tokens_total['prompt']} prompt + {tokens_total['completion']} completion)"
        )
    if args.secondary:
        from lexora.collect.secondary import indicator_gaps

        print("\nSecondary-source coverage cross-check (tracker says a law exists, "
              "we cited none):")
        any_gap = False
        for iso in isos:
            result = results_by_iso.get(iso)
            if result is None:
                continue
            covered = {c.indicator_id for c in result.citations}
            gaps = indicator_gaps(result.secondary_signals, covered)
            for g in gaps:
                any_gap = True
                print(f"  GAP {g.economy} {g.indicator_id}: {g.source_name}")
        if not any_gap:
            print("  none — every indicator a secondary source flags is also cited.")

    print(f"Wrote {n} citation row(s) -> {args.out} (submission CSV)")
    print(f"                          -> {jsonld_out} (JSON-LD)")
    print(f"                          -> {summary_out} (run summary)")


if __name__ == "__main__":
    main()
