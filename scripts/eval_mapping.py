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
so subsection splitting does not affect a hit. Australian Privacy Principles
share numbers with the main body (APP 8 vs s.8), so they are keyed ``APP8`` —
write the gold for an APP-targeting indicator as ``APP8`` (see ``_clause_key``).

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
import re
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

def document_identity_ok(clauses: list[Clause], doc_name: str) -> bool:
    """Guard against scoring section gold on the WRONG document.

    Gold is matched by section NUMBER, so a different statute sharing section numbers
    scores silent false hits (this masked a Criminal Procedure Code file standing in
    for the SG PDPA). Check that the gold ``document`` name's core phrase (its words
    minus the year) appears verbatim in the parsed text — measured to discriminate
    the SG PDPA from the CPC perfectly (the phrase 'personal data protection act' is
    in one and not the other). Heuristic + lenient: it only flags a gross mismatch."""
    core = " ".join(re.findall(r"[a-z]+", doc_name.lower()))  # drops the year (digits)
    if len(core.split()) < 2:
        return True
    full = " ".join(" ".join(c.span.text for c in clauses).lower().split())
    return core in full


def _clause_key(clause: Clause) -> str:
    """The token a retrieved clause is matched against in the gold. An Australian
    Privacy Principle shares its number with a main-body section (APP 8 vs s.8),
    so it is keyed ``APP8`` to stay distinct; every other clause keys on its
    section number."""
    if "Australian Privacy Principle" in clause.structural_path:
        return f"APP{clause.section_number}"
    return clause.section_number or "?"


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
        retrieved = [_clause_key(clause_by_id[h.clause_id]) for h in hits]
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


def dump_candidates(
    clauses: list[Clause],
    profile: SourceProfile,
    indicators: list[RDTIIIndicator],
    *,
    top_k: int,
    use_semantic: bool,
    embedder=None,
) -> list[dict]:
    """G-6.4 gold-expansion aid: for EVERY indicator (not just labelled ones),
    return the top-``top_k`` candidate clauses with a text snippet, so the law
    group can verify/correct them into section gold. This produces a draft for
    human review — it never writes gold itself (gold stays eval-only, hand-checked).
    """
    index = build_index(clauses)
    clause_by_id = {c.clause_id: c for c in clauses}
    rows: list[dict] = []
    for indicator in indicators:
        hits = retrieve_candidates(
            indicator, profile, index, top_k=top_k,
            use_semantic=use_semantic, embedder=embedder,
        )
        cands = []
        for h in hits:
            c = clause_by_id[h.clause_id]
            text = " ".join(c.span.text.split())
            cands.append({"key": _clause_key(c), "path": c.structural_path,
                          "page": c.span.page_number, "text": text})
        rows.append({
            "indicator": indicator.submission_id,
            "name": indicator.name,
            "candidates": cands,
        })
    return rows


def evaluate_rank(
    clauses: list[Clause],
    profile: SourceProfile,
    indicators: list[RDTIIIndicator],
    gold: dict[str, set[str]],
    *,
    rank_k: int,
    use_semantic: bool,
    embedder=None,
    **retrieval_kwargs,
) -> list[dict]:
    """G-6.1 rank-before-truncation diagnostic (mapping layer).

    Retrieves ``rank_k`` clauses per gold indicator and records the rank of the
    FIRST gold section in that list (None if outside the top ``rank_k``). hit@1
    only tells us rank==1; the rank distribution separates a *ranking* problem
    (gold at rank 5/10 — fixable by re-weighting) from a *recall* problem (gold
    absent even at rank_k — a retrieval/parse gap). Pure diagnostic, larger top_k.
    """
    index = build_index(clauses)
    clause_by_id = {c.clause_id: c for c in clauses}
    by_submission = {i.submission_id: i for i in indicators}

    rows: list[dict] = []
    for submission_id, gold_sections in gold.items():
        indicator = by_submission.get(submission_id)
        if indicator is None:
            continue
        hits = retrieve_candidates(
            indicator, profile, index, top_k=rank_k, pool_k=max(20, rank_k),
            use_semantic=use_semantic, embedder=embedder, **retrieval_kwargs,
        )
        retrieved = [_clause_key(clause_by_id[h.clause_id]) for h in hits]
        rank = next((i + 1 for i, sec in enumerate(retrieved) if sec in gold_sections), None)
        rows.append({
            "indicator": submission_id,
            "gold": sorted(gold_sections),
            "rank": rank,
            "retrieved": retrieved[:rank_k],
        })
    return rows


def summarize_rank(rows: list[dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """MRR + recall@k over a rank-report (rows from ``evaluate_rank``)."""
    n = len(rows) or 1
    mrr = sum(1.0 / r["rank"] for r in rows if r["rank"]) / n
    recall = {k: sum(1 for r in rows if r["rank"] and r["rank"] <= k) for k in ks}
    return {"n": len(rows), "mrr": mrr, "recall": recall}


# G-6.2 ablation grid. Each entry toggles ONE general scoring knob (never an
# answer-specific change); the rank distribution (MRR/recall@k), not a single
# hit@1, tells whether that signal helps or hurts. ``use_semantic`` selects the
# channel; the kwargs are forwarded to ``retrieve_candidates``. Conclusions are
# only valid across all three economies (leave-one-country-out) — one doc's grid
# is a diagnostic, not a tuning decision.
def ablation_grid() -> list[tuple[str, bool, dict]]:
    return [
        ("bm25-only", False, {}),
        ("fused (default)", True, {}),
        ("fused -anchor", True, {"anchor_bm25_top1": False}),
        ("fused -boilerplate_drop", True, {"drop_boilerplate": False}),
        ("fused dense*2", True, {"dense_weight": 2.0}),
        ("fused bm25*2", True, {"bm25_weight": 2.0}),
    ]


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

    # Always send a real UA (a None header crashes httpx); --browser only decides
    # whether to escalate to Chromium when the plain fetch is blocked.
    art = run_pipeline_from_url(
        url=url, profile=profile, indicators=indicators, portal_name="live-fetch",
        browser_fallback=browser, user_agent=UA,
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
    ap.add_argument("--rank-report", action="store_true",
                    help="G-6.1: record each gold section's rank (top --rank-k), "
                         "report MRR + recall@k instead of just hit@1/hit@3")
    ap.add_argument("--rank-k", type=int, default=20,
                    help="how deep to look for the gold section in --rank-report")
    ap.add_argument("--ablate", action="store_true",
                    help="G-6.2: sweep single general scoring knobs (anchor / "
                         "boilerplate-drop / channel weights) and report MRR + recall@k")
    ap.add_argument("--dump", action="store_true",
                    help="G-6.4: dump top candidate sections for EVERY in-scope "
                         "indicator (draft for law-group gold verification, not gold)")
    ap.add_argument("--dump-k", type=int, default=5, help="candidates per indicator in --dump")
    args = ap.parse_args()

    from lexora.collect.profile_loader import load_profile

    iso = args.iso.lower()
    profile = load_profile(REPO / "configs" / "jurisdictions" / f"{iso}.yaml")
    indicators = load_indicators(INDICATORS)
    by_doc = load_gold().get(iso, {})
    if not by_doc and not args.dump:
        print(f"No mapping gold for {iso} in {GOLD.relative_to(REPO)}.")
        return
    if by_doc:
        try:
            doc_name, gold = select_gold(by_doc, args.doc)
        except KeyError as exc:
            if not args.dump:
                print(exc)
                return
            doc_name, gold = (args.doc or "(document)"), {}
    else:
        doc_name, gold = (args.doc or "(document)"), {}

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
    if doc_name and not doc_name.startswith("(") and not document_identity_ok(clauses, doc_name):
        print(f"!! WARNING: parsed document does not look like '{doc_name}'.\n"
              "!! Gold is matched by section NUMBER, so a wrong statute with the same\n"
              "!! numbers would score FALSE hits. Check the --pdf/--url source.\n")

    if args.dump:
        # G-6.4: draft candidate sections for law-group gold verification. BM25-only
        # (the live mapping default after G-6.3); writes a review markdown.
        official = {
            "sg": "https://sso.agc.gov.sg/Act/PDPA2012",
            "au": "https://www.legislation.gov.au/C2004A03712",
            "my": "https://mohre.um.edu.my/img/files/Personal%20Data%20Protection%20(PDPA)%20Act%202010.pdf",
        }
        in_scope = [i for i in indicators if i.submission_id != "P6-I5"]
        rows = dump_candidates(clauses, profile, in_scope, top_k=args.dump_k,
                               use_semantic=False)
        out = REPO / "outputs" / f"mapping_candidates_{iso}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# Mapping gold candidates — {ISO_TO_COUNTRY.get(iso, iso)} / {doc_name}",
                 f"\n- Official text (authoritative, read the full section here): "
                 f"{official.get(iso, '(see jurisdiction profile)')}",
                 f"- Parsed from: `{src}`  ·  clauses parsed: {len(clauses)}  ·  ranking: BM25-only, top-{args.dump_k}",
                 "\n**Draft for law-group verification — NOT gold.** Each candidate below shows the "
                 "**full provision text** plus its page in the source PDF and the official URL above, so "
                 "you can read the whole section and judge. Mark the on-point section(s) per indicator; "
                 "they then go into `mapping_sections.csv`.\n"]
        for r in rows:
            lines.append(f"\n## {r['indicator']} — {r['name']}")
            for i, c in enumerate(r["candidates"], 1):
                pg = f"p.{c['page']}" if c["page"] else "p.?"
                body = c["text"] if len(c["text"]) <= 3500 else \
                    c["text"][:3500] + f" …[truncated — read the full section at {pg} / official URL]"
                lines.append(f"\n**{i}. `{c['key']}`** — {c['path']} ({pg})\n\n> {body}")
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nMapping candidate dump -- {ISO_TO_COUNTRY.get(iso, iso)} / {doc_name}")
        print(f"parsed clauses: {len(clauses)}; {len(rows)} indicators")
        print(f"review file -> {out.relative_to(REPO)}")
        return

    # A/B: BM25-only vs BM25+dense fusion on the SAME parsed document.
    from lexora.classify.retrieval import _maybe_embedder

    embedder = _maybe_embedder(True)

    if args.ablate:
        # G-6.2: one knob at a time, judged on MRR/recall@k (rank distribution),
        # not hit@1 (too gameable on a handful of gold points).
        print(f"[ablation] single-knob sweep, rank_k={args.rank_k}, MRR + recall@k\n")
        for label, use_sem, kw in ablation_grid():
            emb = embedder if use_sem else None
            if use_sem and emb is None:
                continue
            rows = evaluate_rank(clauses, profile, indicators, gold,
                                 rank_k=args.rank_k, use_semantic=use_sem, embedder=emb, **kw)
            s = summarize_rank(rows)
            rec = "  ".join(f"r@{k} {v}/{s['n']}" for k, v in s["recall"].items())
            print(f"   {label:<26} MRR {s['mrr']:.3f}   {rec}")
        if embedder is None:
            print("\nnote: dense backend unavailable — only the bm25-only row ran.")
        return

    runs = [("bm25", False, None)]
    if embedder is not None:
        runs.append(("fused", True, embedder))

    for label, use_sem, emb in runs:
        if args.rank_report:
            rows = evaluate_rank(clauses, profile, indicators, gold,
                                 rank_k=args.rank_k, use_semantic=use_sem, embedder=emb)
            s = summarize_rank(rows)
            rec = "  ".join(f"r@{k} {v}/{s['n']}" for k, v in s["recall"].items())
            print(f"[{label}]  MRR {s['mrr']:.3f}   {rec}")
            for r in sorted(rows, key=lambda r: r["rank"] or 1e9):
                rk = f"#{r['rank']}" if r["rank"] else "miss"
                print(f"   {r['indicator']:<7} gold={','.join(r['gold']):<8} "
                      f"rank={rk:<5} top={r['retrieved'][:8]}")
            print()
            continue
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
