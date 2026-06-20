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
) -> MapResult:
    """Run the production multi-instrument map for one economy."""
    profile = load_profile(JURIS / f"{iso.lower()}.yaml")
    indicators = load_indicators(INDICATORS)
    verifier = make_verifier(use_llm=verify)
    if verify and verifier is None:
        print("warning: --verify requested but the LLM verifier is unavailable "
              "(install the [llm] extra); continuing with BM25 + verbatim only.")
    rationale_gen = make_rationale_generator(use_llm=rationale_llm)
    if rationale_llm and rationale_gen._client is None:
        print("warning: --rationale-llm requested but the LLM backend is unavailable; "
              "using the deterministic template rationale.")
    meta_extractor = make_metadata_extractor(use_llm=metadata_llm)
    if metadata_llm and meta_extractor._client is None:
        print("warning: --metadata-llm requested but the LLM backend is unavailable; "
              "using portal metadata + curated anchor only.")
    result = run_pipeline_map(
        portal=profile.portals[0], profile=profile, indicators=indicators,
        budget=budget, timeout=timeout, verifier=verifier, rationale_gen=rationale_gen,
        meta_extractor=meta_extractor,
    )
    if meta_extractor._client is not None:
        print(
            f"  LLM metadata usage [{iso}]: {meta_extractor.extracted} doc(s) extracted, "
            f"{meta_extractor.rejected} field(s) rejected by source-check"
            + (f" ({meta_extractor.error_count} backend error(s))"
               if meta_extractor.error_count else "")
        )
    if rationale_gen._client is not None:
        print(
            f"  LLM rationale usage [{iso}]: {rationale_gen.llm_used} authored, "
            f"{rationale_gen.fallbacks} fell back to template"
            + (f" ({rationale_gen.error_count} backend error(s))"
               if rationale_gen.error_count else "")
        )
    if verifier is not None and getattr(verifier, "error_count", 0):
        print(
            f"warning: LLM verifier backend errors for {iso}: "
            f"{verifier.error_count} judgement(s) failed "
            f"(last error: {verifier.last_error_type or 'unknown'}); "
            "run continued without fabricating citations."
        )
    client = getattr(verifier, "_client", None) if verifier is not None else None
    if client is not None and getattr(client, "calls", 0):
        print(
            f"  LLM verifier usage [{iso}]: {client.calls} call(s), "
            f"{client.total_tokens} tokens "
            f"({client.prompt_tokens} prompt + {client.completion_tokens} completion)"
        )
    return result


def summarize(iso: str, result: MapResult) -> dict:
    """Per-economy counts for the run summary (pure — unit-tested offline).

    NEW/KNOWN is reported on the *instruments* discovered (the 20/40-point NEW
    capability is about finding laws), while the citation counts describe the CSV
    rows actually emitted and how many indicators they cover."""
    fetched_ok = sum(1 for d in result.documents if 200 <= d.document.http_status < 300)
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
                    help="Tighten mappings with the LLM verifier (needs LEXORA_LLM_* endpoint)")
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
    for iso in isos:
        try:
            result = run_one(iso, budget=args.budget, verify=args.verify,
                             rationale_llm=args.rationale_llm, metadata_llm=args.metadata_llm,
                             timeout=args.timeout)
        except Exception as exc:  # one economy failing must not lose the others
            summaries.append({"iso": iso, "economy": ISO_TO_COUNTRY.get(iso, iso),
                              "error": f"{type(exc).__name__}: {exc}"})
            continue
        all_citations.extend(result.citations)
        summaries.append(summarize(iso, result))

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
    print(f"{'economy':<12}{'instr':<7}{'NEW':<5}{'KNOWN':<7}{'fetched':<9}"
          f"{'cites':<7}{'inds':<6}{'review'}")
    print("-" * 64)
    for s in summaries:
        if s.get("error"):
            print(f"{s['economy']:<12}ERROR: {s['error']}")
            continue
        print(f"{s['economy']:<12}{s['instruments']:<7}{s['new_instruments']:<5}"
              f"{s['known_instruments']:<7}{s['fetched_ok']:<9}{s['citations']:<7}"
              f"{s['n_indicators_covered']:<6}{s['review_rows']}")
    print("-" * 64)
    print(f"Wrote {n} citation row(s) -> {args.out} (submission CSV)")
    print(f"                          -> {jsonld_out} (JSON-LD)")
    print(f"                          -> {summary_out} (run summary)")


if __name__ == "__main__":
    main()
