"""Mapping-quality evaluation harness (P-4).

Coverage (P-0/P-2 eval) asks *which instruments* discovery reaches. This harness
asks the next question: once we have the right document, does retrieval map each
indicator to the *right section* — or to a section that merely shares vocabulary
(the wrong-indicator failure mode, e.g. P6-I4 landing on a generic "protection"
clause instead of the cross-border-transfer section)?

For one document we parse it into clauses, run per-indicator retrieval, and check
whether the gold section(s) appear in the top-1 / top-3 retrieved clauses. We run
it twice — BM25-only vs BM25+dense fusion — so the numbers directly test the P-2
claim that the dense channel reorders the pool toward the on-point section.

The gold (`configs/eval/mapping_sections.csv`) is a small hand-labelled set of
``indicator -> correct section number`` for the flagship statutes (SG PDPA, AU
Privacy Act). Section numbers are matched at the top level (s.26 covers 26(1)),
so subsection splitting does not affect a hit.

Document source, in priority order:
    --pdf PATH   local PDF (deterministic; works on the cached data/raw store)
    --url  URL   live-fetch one full-text URL (--browser to escalate to Chromium)
    (default)    live discovery-by-name of the profile's flagship instrument

Usage:
    python scripts/eval_mapping.py --iso sg --pdf data/raw/sg/<hash>.pdf
    LEXORA_LIVE=1 python scripts/eval_mapping.py --iso sg            # live discover
    python scripts/eval_mapping.py --iso au --url <privacy-act-pdf> --browser
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexora.classify.retrieval import build_index, retrieve_candidates  # noqa: E402
from lexora.indicators import load_indicators  # noqa: E402
from lexora.models.clause import Clause  # noqa: E402
from lexora.models.indicator import RDTIIIndicator  # noqa: E402
from lexora.models.source import SourceProfile  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GOLD = REPO / "configs" / "eval" / "mapping_sections.csv"
INDICATORS = REPO / "configs" / "rdtii_indicators.yaml"
ISO_TO_COUNTRY = {"sg": "Singapore", "au": "Australia", "my": "Malaysia"}
MAX_K = 3


# --- pure, offline-testable core -------------------------------------------

def load_gold(path: Path = GOLD) -> dict[str, dict[str, dict[str, set[str]]]]:
    """iso (lower) -> {document name -> {submission_id -> set of section numbers}}.

    Document-aware so several flagship statutes per economy can each carry their
    own section gold; one eval run scores exactly one document's block."""
    by_iso: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(lambda: defaultdict(dict))
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sections = {s.strip() for s in row["gold_sections"].split(";") if s.strip()}
            by_iso[row["iso"].lower()][row["document"].strip()][row["indicator"].strip()] = sections
    return {iso: dict(docs) for iso, docs in by_iso.items()}


def select_gold(
    by_doc: dict[str, dict[str, set[str]]], doc: str | None
) -> tuple[str, dict[str, set[str]]]:
    """Pick the gold block for ``doc`` (case-insensitive substring), else the only
    block, else raise so the caller can list the choices."""
    if not by_doc:
        raise KeyError("no gold documents for this jurisdiction")
    if doc:
        matches = [name for name in by_doc if doc.lower() in name.lower()]
        if len(matches) != 1:
            raise KeyError(f"--doc {doc!r} matched {matches or 'nothing'}; choose one of {list(by_doc)}")
        return matches[0], by_doc[matches[0]]
    if len(by_doc) == 1:
        name = next(iter(by_doc))
        return name, by_doc[name]
    raise KeyError(f"multiple gold documents — pass --doc one of {list(by_doc)}")


def evaluate(
    clauses: list[Clause],
    profile: SourceProfile,
    indicators: list[RDTIIIndicator],
    gold: dict[str, set[str]],
    *,
    use_semantic: bool,
    embedder=None,
    **retrieval_kwargs,
) -> list[dict]:
    """Per gold indicator, retrieve top-``MAX_K`` clauses and record whether a
    gold section is hit at rank 1 and within the top 3.

    Returns one row per gold indicator with the retrieved section sequence and
    the two boolean hits, so the caller can aggregate or print a diff."""
    index = build_index(clauses)
    clause_by_id = {c.clause_id: c for c in clauses}
    by_submission = {i.submission_id: i for i in indicators}

    rows: list[dict] = []
    for submission_id, gold_sections in gold.items():
        indicator = by_submission.get(submission_id)
        if indicator is None:
            continue
        hits = retrieve_candidates(
            indicator, profile, index, top_k=MAX_K,
            use_semantic=use_semantic, embedder=embedder, **retrieval_kwargs,
        )
        retrieved = [clause_by_id[h.clause_id].section_number or "?" for h in hits]
        hit1 = bool(retrieved[:1]) and retrieved[0] in gold_sections
        hit3 = any(sec in gold_sections for sec in retrieved[:3])
        rows.append({
            "indicator": submission_id,
            "gold": sorted(gold_sections),
            "retrieved": retrieved,
            "hit1": hit1,
            "hit3": hit3,
        })
    return rows


def summarize(rows: list[dict]) -> tuple[int, int, int]:
    """(#hit@1, #hit@3, #indicators)."""
    return (sum(r["hit1"] for r in rows), sum(r["hit3"] for r in rows), len(rows))


# --- document acquisition (I/O) --------------------------------------------

def _clauses_from_pdf(pdf: Path, profile: SourceProfile, indicators) -> list[Clause]:
    from lexora.pipeline import run_demo_pipeline

    # source_url is metadata only here; the canonical portal URL keeps it a valid
    # https URL (RawDocument requires http/https) without implying a live fetch.
    art = run_demo_pipeline(
        pdf_path=pdf, profile=profile, indicators=indicators,
        source_url=str(profile.portals[0].url), portal_name="local-pdf",
    )
    return art.clauses


def _clauses_from_url(url: str, profile: SourceProfile, indicators, *, browser: bool) -> list[Clause]:
    from lexora.collect.browser import DEFAULT_UA as UA
    from lexora.pipeline import run_pipeline_from_url

    art = run_pipeline_from_url(
        url=url, profile=profile, indicators=indicators, portal_name="live-fetch",
        browser_fallback=browser, user_agent=UA if browser else None,
    )
    return art.clauses


def _clauses_from_discovery(profile: SourceProfile, indicators) -> list[Clause]:
    from lexora.pipeline import run_pipeline_autodiscover

    _, art = run_pipeline_autodiscover(
        portal=profile.portals[0], profile=profile, indicators=indicators,
        top_k=1, min_score=0.0,
    )
    return art.clauses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso", required=True, help="sg|au|my")
    ap.add_argument("--doc", help="gold document name (substring) when an economy has several")
    ap.add_argument("--pdf", type=Path, help="local PDF (deterministic)")
    ap.add_argument("--url", help="live-fetch one full-text URL")
    ap.add_argument("--browser", action="store_true", help="escalate --url to Chromium")
    args = ap.parse_args()

    from lexora.collect.profile_loader import load_profile

    iso = args.iso.lower()
    profile = load_profile(REPO / "configs" / "jurisdictions" / f"{iso}.yaml")
    indicators = load_indicators(INDICATORS)
    by_doc = load_gold().get(iso, {})
    if not by_doc:
        print(f"No mapping gold for {iso} in {GOLD.relative_to(REPO)}.")
        return
    try:
        doc_name, gold = select_gold(by_doc, args.doc)
    except KeyError as exc:
        print(exc)
        return

    if args.pdf:
        clauses = _clauses_from_pdf(args.pdf, profile, indicators)
        src = str(args.pdf)
    elif args.url:
        clauses = _clauses_from_url(args.url, profile, indicators, browser=args.browser)
        src = args.url
    else:
        clauses = _clauses_from_discovery(profile, indicators)
        src = f"{profile.portals[0].name} (discovery-by-name)"

    print(f"\nMapping-quality eval -- {ISO_TO_COUNTRY.get(iso, iso)} / {doc_name}")
    print(f"document: {src}")
    print(f"parsed clauses: {len(clauses)}\n")
    if not clauses:
        print("No clauses parsed — check the document source.")
        return

    # A/B: BM25-only vs BM25+dense fusion on the SAME parsed document.
    from lexora.classify.retrieval import _maybe_embedder

    embedder = _maybe_embedder(True)
    runs = [("bm25", False, None)]
    if embedder is not None:
        runs.append(("fused", True, embedder))

    for label, use_sem, emb in runs:
        rows = evaluate(clauses, profile, indicators, gold, use_semantic=use_sem, embedder=emb)
        h1, h3, n = summarize(rows)
        print(f"[{label}]  hit@1 {h1}/{n}   hit@3 {h3}/{n}")
        for r in rows:
            mark1 = "x" if r["hit1"] else " "
            mark3 = "x" if r["hit3"] else " "
            print(f"   {r['indicator']:<7} gold={','.join(r['gold']):<8} "
                  f"top3={r['retrieved']}  @1[{mark1}] @3[{mark3}]")
        print()

    if embedder is None:
        print("note: dense backend unavailable — install .[embeddings] for the A/B.")


if __name__ == "__main__":
    main()
