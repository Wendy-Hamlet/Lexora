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
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexora.classify.retrieval import (  # noqa: E402
    _concept_text,
    _is_boilerplate,
    build_index,
    retrieve_candidates,
)
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
            # "N/A" marks an indicator the Act does not address — a real annotation,
            # but not a retrievable section, so it carries no eval gold. Drop it (and
            # any row left empty) so it never pollutes hit@k / MRR.
            sections = {s for s in sections if s.upper() != "N/A"}
            if not sections:
                continue
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


# --- multi-method candidate pool (independent-gold aid) ---------------------
# Single-system top-k labelling makes gold ⊆ the system's own output, so hit@k is
# tautological and a truly-missed provision can never be detected. Pooling the
# UNION of several DIFFERENT methods' top-k (the standard IR fix) lets gold come
# from outside any one system, so the production system's misses become visible.
# Methods: BM25-only, dense-only, and an LLM asked which sections fit the
# indicator (a non-retrieval mechanism that can surface what retrieval misses).
# The LLM only feeds the pool for human review — it never writes gold.

def _section_index_lines(clauses_by_key: dict[str, list[Clause]]) -> str:
    """One compact "key: heading…" line per section for the LLM channel (ToC mode)."""
    lines = []
    for key, group in clauses_by_key.items():
        head = " ".join(group[0].span.text.split())[:90]
        lines.append(f"{key}: {head}")
    return "\n".join(lines)


def _section_fulltext_lines(clauses_by_key: dict[str, list[Clause]]) -> str:
    """One "key: <full provision text>" block per section (full-text LLM mode), so
    the LLM judges on the same text BM25/dense see, not just the heading."""
    lines = []
    for key, group in clauses_by_key.items():
        body = " ".join(" ".join(c.span.text.split()) for c in group)
        lines.append(f"{key}: {body}")
    return "\n".join(lines)


def _est_tokens(text: str) -> int:
    """Rough token estimate (~chars/4 for English legal text). Used only to keep a
    full-text LLM prompt under the model's context window; an over-estimate just
    triggers the ToC fallback, which is safe."""
    return len(text) // 4


def _blind_order(cands: list[dict], seed: str) -> list[dict]:
    """A reproducible random ordering for the law-group copy. The pool is built by
    several methods; presenting it in method-confidence (consensus) order, or with
    "found by" tags, would anchor the reviewer onto the system's guess — exactly
    what an independent gold must avoid. So the handed-off markdown shuffles the
    candidates (seeded per indicator so re-runs are stable) and drops the tags."""
    out = list(cands)
    random.Random(seed).shuffle(out)
    return out


def _norm_section(raw: str) -> str:
    """Fold an LLM-returned section label to a `_clause_key` token (APP8 / 13 / 26WR)."""
    s = str(raw).strip().upper()
    for noise in ("SECTION", "CLAUSE", "S.", "SEC.", "ART.", "ARTICLE"):
        s = s.replace(noise, "")
    return s.replace(" ", "")


def llm_suggest_sections(
    llm, indicator: RDTIIIndicator, index_lines: str, k: int
) -> list[str]:
    """Ask the LLM which sections address the indicator (ToC-only, returns keys).

    A mechanism independent of BM25/dense: it reads the section headings and reasons
    about which provision fits, so it can nominate sections retrieval ranked low or
    missed. Hallucinated numbers are filtered by the caller against the real index.
    """
    system = (
        "You are a legal analyst mapping a data-protection statute to RDTII "
        "indicators. Given an indicator and the Act's sections (each with its "
        "number and text), return the section numbers whose provision most likely "
        "addresses that indicator. Use ONLY numbers from the list, ordered "
        "best-first, at most the requested count; return an empty list if none "
        'fit. Respond as json: {"sections": ["13", "24"]}.'
    )
    user = (
        f"Indicator {indicator.submission_id}: {indicator.name}\n"
        f"Definition: {indicator.description}\n\n"
        f"Sections:\n{index_lines}\n\nReturn at most {k} section numbers."
    )
    data = llm.chat(system, user, json_schema={"type": "object"})
    out = data.get("sections", []) if isinstance(data, dict) else []
    return [_norm_section(s) for s in out][:k]


def pool_candidates(
    clauses: list[Clause],
    profile: SourceProfile,
    indicators: list[RDTIIIndicator],
    *,
    pool_k: int,
    embedder=None,
    llm=None,
    llm_fulltext: bool = True,
    llm_context_tokens: int = 120_000,
) -> list[dict]:
    """For every indicator, pool the UNION of BM25-only, dense-only and LLM section
    suggestions (each top ``pool_k``), keyed by section. Each candidate records
    which methods found it and at what rank, plus the full provision text (all the
    section's clauses joined). A draft for independent human gold labelling.

    The LLM channel reads the whole statute (all section texts) so it judges on the
    same evidence as BM25/dense — UNLESS that prompt would exceed
    ``llm_context_tokens`` (the model's usable context), in which case it falls back
    to the compact heading-only ToC for this statute. Set ``llm_fulltext=False`` to
    force ToC. Decided once per statute and announced on stdout.
    """
    index = build_index(clauses)
    clause_by_id = {c.clause_id: c for c in clauses}

    clauses_by_key: dict[str, list[Clause]] = defaultdict(list)
    for c in clauses:
        clauses_by_key[_clause_key(c)].append(c)
    for group in clauses_by_key.values():
        group.sort(key=lambda c: c.span.char_start)

    index_lines = _section_index_lines(clauses_by_key)
    if llm is not None:
        if llm_fulltext:
            full = _section_fulltext_lines(clauses_by_key)
            est = _est_tokens(full)
            if est <= llm_context_tokens:
                index_lines = full
                print(f"   LLM channel: full-text (~{est:,} tok, budget {llm_context_tokens:,})")
            else:
                print(f"   LLM channel: ToC fallback (full ~{est:,} tok > budget "
                      f"{llm_context_tokens:,}); raise --llm-context-tokens if the model allows")
        else:
            print("   LLM channel: ToC (forced --llm-toc)")

    rows: list[dict] = []
    for indicator in indicators:
        methods: dict[str, dict[str, int]] = defaultdict(dict)
        # BM25-only
        for rank, h in enumerate(retrieve_candidates(
            indicator, profile, index, top_k=pool_k, use_semantic=False), 1):
            methods[_clause_key(clause_by_id[h.clause_id])].setdefault("bm25", rank)
        # dense-only (boilerplate dropped, mirroring retrieval's pool)
        if embedder is not None:
            skip = {i for i, c in enumerate(index.clauses) if _is_boilerplate(c)}
            dense_idx = [i for i in index.dense_ranking(
                _concept_text(indicator, profile), embedder, pool_k + len(skip))
                if i not in skip][:pool_k]
            for rank, i in enumerate(dense_idx, 1):
                methods[_clause_key(index.clauses[i])].setdefault("dense", rank)
        # LLM section suggestions (filtered to real sections)
        if llm is not None:
            for rank, key in enumerate(
                llm_suggest_sections(llm, indicator, index_lines, pool_k), 1):
                if key in clauses_by_key:
                    methods[key].setdefault("llm", rank)

        cands = []
        for key, found in methods.items():
            group = clauses_by_key[key]
            text = " ".join(" ".join(c.span.text.split()) for c in group)
            cands.append({
                "key": key,
                "path": group[0].structural_path,
                "page": group[0].span.page_number,
                "found": found,           # {method: rank}
                "text": text,
            })
        # consensus first (more methods, then best single rank), so we can SEE the
        # overlap; the law-group copy can drop these tags to avoid anchoring.
        cands.sort(key=lambda c: (-len(c["found"]), min(c["found"].values())))
        rows.append({
            "indicator": indicator.submission_id,
            "name": indicator.name,
            "candidates": cands,
        })
    return rows


def section_index_rows(clauses: list[Clause]) -> list[dict]:
    """A whole-statute section index (one row per section, document order) for the
    law group to scan ALL sections by topic and locate gold INDEPENDENTLY of our
    scored pool — the channel that lets gold come from outside any system's output."""
    by_key: dict[str, list[Clause]] = defaultdict(list)
    for c in clauses:
        by_key[_clause_key(c)].append(c)
    for group in by_key.values():
        group.sort(key=lambda c: c.span.char_start)
    rows = []
    for key, group in by_key.items():
        head = " ".join(group[0].span.text.split())[:130]
        rows.append({"key": key, "path": group[0].structural_path,
                     "page": group[0].span.page_number, "head": head,
                     "start": group[0].span.char_start})
    rows.sort(key=lambda r: r["start"])
    return rows


def pool_gold_recall(rows: list[dict], gold: dict[str, set[str]]) -> dict:
    """Honesty / recall check for the pool: for each known gold section, is it in the
    pool, by which methods, at what best rank — and would a single-method top-5 have
    caught it? Aggregates the *out-of-pool rate* (gold a pool of this depth still
    misses) and the *hidden-by-top5 rate* (gold no single method ranked in its top 5,
    i.e. exactly what the old top-5 labelling protocol would have buried)."""
    per: list[dict] = []
    for r in rows:
        g = gold.get(r["indicator"])
        if not g:
            continue
        found = {c["key"]: c["found"] for c in r["candidates"]}
        for sec in sorted(g):
            f = found.get(sec)
            best = min(f.values()) if f else None
            in_top5 = bool(f) and any(rk <= 5 for rk in f.values())
            per.append({"indicator": r["indicator"], "gold": sec, "found": f,
                        "best_rank": best, "in_some_top5": in_top5})
    n = len(per)
    out_of_pool = sum(1 for p in per if not p["found"])
    hidden_by_top5 = sum(1 for p in per if not p["in_some_top5"])
    return {"n": n, "out_of_pool": out_of_pool, "hidden_by_top5": hidden_by_top5, "per": per}


# --- fillable answer block in the pool md + its collector --------------------
# The law group fills GOLD=/NOTE= between these per-indicator markers; the markers
# are machine-extractable so the filled md round-trips straight into gold CSV rows.
def _answer_block(indicator_id: str) -> str:
    return (f"\n**✍️ 答案 {indicator_id}** — 在 `GOLD=` 后填正确条文号（多条用 `;`；不涉及填 "
            f"`N/A`）。**勿改动 `<!-- -->` 标记。**\n\n"
            f"```\n<!--GOLD:{indicator_id}:START-->\nGOLD=\nNOTE=\n<!--GOLD:{indicator_id}:END-->\n```")


_ANSWER_RE = re.compile(
    r"<!--GOLD:(?P<ind>[^:>]+):START-->(?P<body>.*?)<!--GOLD:(?P=ind):END-->", re.S)


def collect_gold(md_text: str, document: str) -> list[dict]:
    """Extract filled answer blocks from a pool markdown into gold rows. An empty
    or N/A-less ``GOLD=`` is skipped; separators (; ， ；) normalise to ``;``."""
    rows: list[dict] = []
    for m in _ANSWER_RE.finditer(md_text):
        body = m.group("body")
        gm = re.search(r"GOLD[ \t]*=[ \t]*(.*)", body)
        nm = re.search(r"NOTE[ \t]*=[ \t]*(.*)", body)
        gold_raw = (gm.group(1).strip() if gm else "")
        if not gold_raw:
            continue  # unfilled
        gold = ";".join(s.strip() for s in re.split(r"[;,；，]", gold_raw) if s.strip())
        rows.append({"indicator": m.group("ind").strip(), "document": document,
                     "gold_sections": gold, "note": (nm.group(1).strip() if nm else "")})
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
    ap.add_argument("--iso", help="sg|au|my (not needed with --collect)")
    ap.add_argument("--gold", type=Path, default=GOLD,
                    help="gold CSV to score against (default the legal-group file; pass "
                         "configs/eval/mapping_sections_llm.csv for the LLM-drafted gold)")
    ap.add_argument("--doc", help="gold document name (substring) when an economy has several")
    ap.add_argument("--pdf", type=Path, help="local PDF (deterministic)")
    ap.add_argument("--url", help="live-fetch one full-text URL")
    ap.add_argument("--browser", action="store_true", help="escalate --url to Chromium")
    ap.add_argument("--rerank", action="store_true",
                    help="add a cross-encoder rerank run to the A/B: reorder the BM25 "
                         "recall pool with fastembed's TextCrossEncoder "
                         "(LEXORA_RERANK_MODEL, default BAAI/bge-reranker-base)")
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
    ap.add_argument("--pool", action="store_true",
                    help="pool the UNION of BM25 / dense / LLM top-k per indicator "
                         "(independent-gold aid; breaks single-system top-k circularity)")
    ap.add_argument("--pool-k", type=int, default=20, help="top-k per method in --pool")
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM channel in --pool")
    ap.add_argument("--llm-toc", action="store_true",
                    help="force the LLM channel to read the heading-only ToC instead of "
                         "full section text (default: full text, auto-falls-back to ToC "
                         "when it would exceed --llm-context-tokens)")
    ap.add_argument("--llm-context-tokens", type=int, default=120_000,
                    help="usable model context for the full-text LLM channel (else ToC)")
    ap.add_argument("--provenance", action="store_true",
                    help="(analysis only) keep consensus order + 'found by' method tags "
                         "in the pool markdown; default is a blind, shuffled, tag-free copy")
    ap.add_argument("--toc", action="store_true",
                    help="write a whole-statute section index (every section, document "
                         "order) for the law group to locate gold independently of the pool")
    ap.add_argument("--collect", action="store_true",
                    help="harvest filled answer blocks from outputs/mapping_pool_*.md "
                         "into outputs/collected_gold.csv (no PDF needed)")
    args = ap.parse_args()

    if args.collect:
        iso_to_name = {"sg": "Personal Data Protection Act 2012",
                       "au": "Privacy Act 1988", "my": "Personal Data Protection Act 2010"}
        out_rows: list[tuple[str, str, str, str, str]] = []
        for iso2 in ("sg", "au", "my"):
            p = REPO / "outputs" / f"mapping_pool_{iso2}.md"
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8")
            first = text.splitlines()[0] if text else ""
            mdoc = re.search(r"/\s*(.+?)\s*$", first)
            doc = mdoc.group(1) if mdoc else iso_to_name.get(iso2, iso2)
            for r in collect_gold(text, doc):
                out_rows.append((iso2.upper(), r["document"], r["indicator"],
                                 r["gold_sections"], r["note"]))
        out = REPO / "outputs" / "collected_gold.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iso", "document", "indicator", "gold_sections", "note"])
            w.writerows(out_rows)
        print(f"collected {len(out_rows)} filled gold rows -> {out.relative_to(REPO)}")
        return

    if not args.iso:
        ap.error("--iso is required (except with --collect)")

    from lexora.collect.profile_loader import load_profile

    iso = args.iso.lower()
    profile = load_profile(REPO / "configs" / "jurisdictions" / f"{iso}.yaml")
    indicators = load_indicators(INDICATORS)
    by_doc = load_gold(args.gold).get(iso, {})
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

    official = {
        "sg": "https://sso.agc.gov.sg/Act/PDPA2012",
        "au": "https://www.legislation.gov.au/C2004A03712",
        "my": "https://mohre.um.edu.my/img/files/Personal%20Data%20Protection%20(PDPA)%20Act%202010.pdf",
    }
    in_scope = [i for i in indicators if i.submission_id != "P6-I5"]

    if args.toc:
        rows = section_index_rows(clauses)
        out = REPO / "outputs" / f"section_index_{iso}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# Section index — {ISO_TO_COUNTRY.get(iso, iso)} / {doc_name}",
                 f"\n- Official text (authoritative): {official.get(iso, '(see profile)')}",
                 f"- Parsed from: `{src}`  ·  {len(rows)} sections, document order",
                 "\n**Scan every section by topic and locate the right provision for each "
                 "indicator independently — this is NOT filtered by our retrieval, so gold can "
                 "come from anywhere in the Act.** Jump to a section by its page; read the full "
                 "text in the official source above.\n"]
        for r in rows:
            pg = f"p.{r['page']}" if r["page"] else "p.?"
            lines.append(f"- **`{r['key']}`** ({pg}) — {r['head']}")
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nSection index -- {ISO_TO_COUNTRY.get(iso, iso)} / {doc_name}: "
              f"{len(rows)} sections -> {out.relative_to(REPO)}")
        return

    if args.pool:
        from lexora.classify.retrieval import _maybe_embedder

        embedder = _maybe_embedder(True)
        llm = None
        if not args.no_llm:
            from lexora.classify import llm_client
            if llm_client.is_available():
                llm = llm_client.LlmClient()
        rows = pool_candidates(clauses, profile, in_scope, pool_k=args.pool_k,
                               embedder=embedder, llm=llm,
                               llm_fulltext=not args.llm_toc,
                               llm_context_tokens=args.llm_context_tokens)
        llm_tag = "" if llm is None else (
            " (LLM reads ToC headings)" if args.llm_toc
            else " (LLM reads full section text, ToC fallback past --llm-context-tokens)")
        chans = ["bm25"] + (["dense"] if embedder is not None else []) + (["llm"] if llm else [])
        out = REPO / "outputs" / f"mapping_pool_{iso}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# Mapping candidate POOL — {ISO_TO_COUNTRY.get(iso, iso)} / {doc_name}",
                 f"\n- Official text (authoritative): {official.get(iso, '(see profile)')}",
                 f"- Parsed from: `{src}`  ·  clauses: {len(clauses)}  ·  "
                 f"pool = UNION of {' ∪ '.join(chans)}, each top-{args.pool_k}{llm_tag}",
                 "\n**Draft for INDEPENDENT law-group gold labelling — NOT gold.** Each candidate is a "
                 "whole section (full provision text), pooled from several methods and presented in "
                 "**random order with no scores or rankings**, so the system's guess does not anchor "
                 "your judgement. Pick the on-point section(s) on the law alone, **add any correct "
                 "section that is missing** (use the official text), and mark `N/A` if the Act does not "
                 "cover the indicator.\n"]
        # overlap stats for "see the effect"
        print(f"\nCandidate pool -- {ISO_TO_COUNTRY.get(iso, iso)} / {doc_name}")
        print(f"channels: {' ∪ '.join(chans)} (each top-{args.pool_k}); clauses {len(clauses)}\n")
        for r in rows:
            cs = r["candidates"]
            by_n = {n: sum(1 for c in cs if len(c["found"]) == n) for n in (3, 2, 1)}
            llm_only = sum(1 for c in cs if set(c["found"]) == {"llm"})
            print(f"   {r['indicator']:<7} union={len(cs):<3} "
                  f"all3={by_n.get(3,0)} two={by_n.get(2,0)} one={by_n.get(1,0)} "
                  f"llm-only={llm_only}")
            lines.append(f"\n## {r['indicator']} — {r['name']}  ·  {len(cs)} candidates")
            # default: blind copy for the law group — shuffle + no provenance tags so
            # the system's guess does not anchor the judgement. --provenance restores
            # the consensus order + "found by" tags for our own analysis.
            render = cs if args.provenance else _blind_order(cs, r["indicator"])
            for c in render:
                pg = f"p.{c['page']}" if c["page"] else "p.?"
                tag = ""
                if args.provenance:
                    found = ", ".join(f"{m}#{rk}" for m, rk in sorted(c["found"].items()))
                    tag = f"  ·  found by: {found}"
                # full provision text, no truncation — the law group needs the whole
                # section to judge (see gold-handoff full-text rule).
                lines.append(f"\n**`{c['key']}`** — {c['path']} ({pg}){tag}\n\n> {c['text']}")
            lines.append(_answer_block(r["indicator"]))
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nreview file -> {out.relative_to(REPO)}")
        # honesty / recall check against current gold (stdout only — never in the
        # blind law-group file): does the pool actually contain the known gold, and
        # how much would the old single-method top-5 protocol have buried?
        if gold:
            rec = pool_gold_recall(rows, gold)
            print(f"\ngold-in-pool check ({rec['n']} gold sections):"
                  f" out-of-pool {rec['out_of_pool']}/{rec['n']},"
                  f" hidden-by-top5 {rec['hidden_by_top5']}/{rec['n']}")
            for p in rec["per"]:
                where = ", ".join(f"{m}#{rk}" for m, rk in sorted(p["found"].items())) \
                    if p["found"] else "MISSED by all methods"
                print(f"   {p['indicator']:<7} gold {p['gold']:<5} -> {where}"
                      f"{'' if p['in_some_top5'] else '   <- hidden by single-method top-5'}")
        return

    if args.dump:
        # G-6.4: draft candidate sections for law-group gold verification. BM25-only
        # (the live mapping default after G-6.3); writes a review markdown.
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

    # A/B: BM25-only vs BM25+dense fusion (vs +cross-encoder rerank) on the SAME
    # parsed document.
    from lexora.classify.retrieval import _maybe_embedder, _maybe_reranker

    embedder = _maybe_embedder(True)
    reranker = _maybe_reranker(True) if args.rerank else None
    if args.rerank and reranker is None:
        print("note: --rerank requested but the cross-encoder backend is unavailable.\n")

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

    runs: list[tuple[str, bool, object, dict]] = [("bm25", False, None, {})]
    if embedder is not None:
        runs.append(("fused", True, embedder, {}))
    if reranker is not None:
        runs.append(("reranked", False, None, {"reranker": reranker}))

    for label, use_sem, emb, kw in runs:
        if args.rank_report:
            rows = evaluate_rank(clauses, profile, indicators, gold,
                                 rank_k=args.rank_k, use_semantic=use_sem, embedder=emb, **kw)
            s = summarize_rank(rows)
            rec = "  ".join(f"r@{k} {v}/{s['n']}" for k, v in s["recall"].items())
            print(f"[{label}]  MRR {s['mrr']:.3f}   {rec}")
            for r in sorted(rows, key=lambda r: r["rank"] or 1e9):
                rk = f"#{r['rank']}" if r["rank"] else "miss"
                print(f"   {r['indicator']:<7} gold={','.join(r['gold']):<8} "
                      f"rank={rk:<5} top={r['retrieved'][:8]}")
            print()
            continue
        rows = evaluate(clauses, profile, indicators, gold, use_semantic=use_sem, embedder=emb, **kw)
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
