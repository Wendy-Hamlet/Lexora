"""Lexora — one-command entry point for the RDTII hackathon reviewer.

    python main.py --economy Singapore --pillar 6

A reviewer hands the engine an ECONOMY and a PILLAR (the topic). The engine then does
the whole job with no manual steps in between: it crawls the economy's official legal
portal, retrieves the instruments (text PDFs *and* scanned/image PDFs, which are OCR'd),
parses them to article level, maps each provision to an RDTII indicator, and writes the
13-column submission CSV plus the richer JSON sidecar.

Inputs are forgiving on purpose (the brief calls for "inputs you did not anticipate"):
the economy may be given as a name, an ISO code, in any case, or misspelt — it is fuzzy
matched against the economies this build supports, and an unrecognisable one exits with
the list of valid choices rather than a stack trace.

Outputs (per the submission template):
    outputs/<Economy>_P<pillar>_<timestamp>.csv    13 official columns, one row per provision
    outputs/<Economy>_P<pillar>_<timestamp>.json   same rows + OCR/timing/context metadata

The LLM lanes (relevance judging, rationale authoring, metadata extraction) need an
OpenAI-compatible endpoint configured in .env — see .env.example. Without one the engine
still runs and still emits both files, degrading to BM25 retrieval with template
rationales, so a reviewer with no API key can always get output.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from rapidfuzz import fuzz, process  # noqa: E402

from scripts.run_submission import ISO_TO_COUNTRY, run_one, write_outputs  # noqa: E402

# Every spelling a reviewer might reasonably type, mapped to the ISO code the
# jurisdiction profiles are keyed by.
_ALIASES: dict[str, str] = {
    "singapore": "sg", "sg": "sg", "sgp": "sg",
    "australia": "au", "au": "au", "aus": "au",
    "malaysia": "my", "my": "my", "mys": "my",
}
_MIN_MATCH = 70  # rapidfuzz score below which we refuse to guess


def resolve_economy(raw: str) -> str:
    """Map whatever the reviewer typed to an ISO code, tolerating case and typos.

    Exact alias first, then a fuzzy match ("Singapre" -> sg). Below ``_MIN_MATCH`` we
    refuse rather than silently run the wrong country."""
    key = raw.strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    match = process.extractOne(key, _ALIASES.keys(), scorer=fuzz.WRatio)
    if match and match[1] >= _MIN_MATCH:
        iso = _ALIASES[match[0]]
        print(f"note: interpreting economy {raw!r} as {ISO_TO_COUNTRY[iso]} "
              f"(fuzzy match, score {match[1]:.0f})")
        return iso
    supported = ", ".join(sorted({ISO_TO_COUNTRY[i] for i in _ALIASES.values()}))
    raise SystemExit(f"error: unrecognised economy {raw!r}. Supported: {supported}")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="main.py",
        description="Map an economy's laws to RDTII Pillar 6 / 7 indicators, end to end.")
    ap.add_argument("--economy", required=True,
                    help="Economy name or ISO code, e.g. Singapore | SG | Malaysia")
    ap.add_argument("--pillar", default="all", choices=["6", "7", "all"],
                    help="RDTII pillar: 6 (cross-border data flows), 7 (data protection), "
                         "or all (default)")
    ap.add_argument("--output-dir", type=Path, default=REPO / "outputs",
                    help="Where the CSV/JSON land (default: outputs/)")
    ap.add_argument("--budget", type=int, default=20,
                    help="Max instruments to map for this economy (default: 20)")
    ap.add_argument("--llm-workers", type=int, default=8,
                    help="Parallel LLM calls (default: 8)")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip every LLM lane — BM25 retrieval + template rationales only. "
                         "Use when you have no API key; output is still complete.")
    args = ap.parse_args()

    iso = resolve_economy(args.economy)
    economy = ISO_TO_COUNTRY[iso]
    pillars = None if args.pillar == "all" else [int(args.pillar)]
    ptag = "6-7" if args.pillar == "all" else args.pillar

    # The crawl is live by design (Task 1 is "no manual steps"), so opt the run in
    # rather than making the reviewer discover an env var.
    os.environ.setdefault("LEXORA_LIVE", "1")
    # OCR on by default: the brief requires scanned/image PDFs to work out of the box.
    os.environ.setdefault("LEXORA_OCR", "1")

    use_llm = not args.no_llm
    print(f"Lexora — {economy} | RDTII Pillar {ptag} | budget {args.budget} instruments")
    print(f"  LLM lanes: {'ON' if use_llm else 'OFF (--no-llm)'}   OCR: ON")
    print("  crawling official portal -> extracting -> parsing -> mapping ...\n")

    result = run_one(
        iso,
        budget=args.budget,
        verify=False,
        verify_clauses=use_llm,   # 0/1 membership judge — the relevance decision
        rationale_llm=use_llm,
        metadata_llm=use_llm,
        amendment_llm=use_llm,
        timeout=60.0,
        llm_workers=args.llm_workers if use_llm else 1,
        pillars=pillars,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = args.output_dir / f"{economy}_P{ptag}_{stamp}.csv"
    write_outputs(result, out_csv)


if __name__ == "__main__":
    main()
