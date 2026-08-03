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
import contextlib
import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexora.cite.amendments_llm import llm_enabled as amend_llm_env  # noqa: E402
from lexora.cite.amendments_llm import make_amendment_extractor  # noqa: E402
from lexora.cite.metadata import make_metadata_extractor  # noqa: E402
from lexora.cite.rationale import make_rationale_generator  # noqa: E402
from lexora.classify.verifier import make_verifier  # noqa: E402
from lexora.collect.profile_loader import load_profile  # noqa: E402
from lexora.config import load_config  # noqa: E402
from lexora.export.csv_exporter import to_csv  # noqa: E402
from lexora.export.json_exporter import to_submission_json  # noqa: E402
from lexora.export.jsonld_exporter import to_jsonld  # noqa: E402
from lexora.indicators import load_indicators  # noqa: E402
from lexora.pipeline import MapResult, documents_completed, run_pipeline_map  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
INDICATORS = REPO / "configs" / "rdtii_indicators.yaml"
JURIS = REPO / "configs" / "jurisdictions"
OUT_CSV = REPO / "outputs" / "submission_round1.csv"

ISO_TO_COUNTRY = {"sg": "Singapore", "au": "Australia", "my": "Malaysia"}

log = logging.getLogger("lexora.submission")


def configure_logging(verbose: bool = False) -> None:
    """Send the pipeline's progress logs to stdout.

    Without this the run is SILENT for hours: `discovery sweep: 39 quer(ies)`,
    `[ 3/39] …`, `working set: N instrument(s)` and the per-document `mapped 12/38 …`
    line all go to a logger with no handler. `main.py` has configured logging since the
    day that was found; this entry point — the one the README gives for the full run —
    never did, so the command most likely to be left running unattended was the one that
    showed nothing while it ran.

    Line buffering is part of the fix, not a detail. Python block-buffers stdout the
    moment it is not a terminal, so `run_submission.py > run.log` — how anyone actually
    runs something for two hours — holds every progress line and every heartbeat in an
    8 KB buffer. The run then looks just as dead as it did with no handler at all, which
    is how the first launch of the paid run was flying blind three minutes in.
    """
    with contextlib.suppress(AttributeError, ValueError):  # not a real text stream
        sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if not verbose:
        for noisy in ("httpx", "httpcore", "urllib3", "openai", "PIL", "fontTools"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


# Live LLM clients, registered as each economy builds them, so one meter can report the
# whole run rather than each lane reporting only after its economy has finished.
_LIVE_CLIENTS: list[tuple[str, str, object]] = []
_LIVE_LOCK = threading.Lock()


def _register_client(iso: str, lane: str, client) -> None:
    if client is None:
        return
    with _LIVE_LOCK:
        _LIVE_CLIENTS.append((iso, lane, client))


def _live_totals() -> dict:
    """Snapshot every registered client. Counters are updated under each client's own
    lock, so reading them from another thread is safe and at worst one call stale."""
    with _LIVE_LOCK:
        clients = list(_LIVE_CLIENTS)
    t = {"calls": 0, "prompt": 0, "cached": 0, "completion": 0, "failed": 0}
    cost: float | None = 0.0
    for _iso, _lane, c in clients:
        t["calls"] += getattr(c, "calls", 0)
        t["prompt"] += getattr(c, "prompt_tokens", 0)
        t["cached"] += getattr(c, "cached_prompt_tokens", 0)
        t["completion"] += getattr(c, "completion_tokens", 0)
        t["failed"] += getattr(c, "failed_calls", 0)
        if cost is not None:
            from lexora.classify.pricing import cost_cny

            one = cost_cny(getattr(c, "model", ""), getattr(c, "prompt_tokens", 0),
                           getattr(c, "cached_prompt_tokens", 0),
                           getattr(c, "completion_tokens", 0))
            cost = None if one is None else cost + one
    t["cost_cny"] = cost
    return t


class LiveMeter(threading.Thread):
    """Prints what the run has spent, while it is still spending it.

    A full run is hours of judge time, and until now every token, cost and failure
    counter was printed only once its ECONOMY had finished — so a wall of 403s, or a
    prefix cache that stopped hitting and tripled the bill, stayed invisible for an hour
    or more. This reports the same numbers on a clock instead.

    ``budget_cny`` is an ALARM, not a brake: it cannot stop work already dispatched
    across parallel economies, and pretending otherwise would be worse than saying so.
    It is still worth having, because stopping by hand is nearly free — the verdict cache
    commits per verdict, so a Ctrl-C loses no judged clause and a resumed run reads them
    back at zero cost.
    """

    def __init__(self, every: float, budget_cny: float = 0.0) -> None:
        super().__init__(daemon=True, name="lexora-live-meter")
        self._every = every
        self._budget = budget_cny
        self._stopped = threading.Event()
        self._t0 = time.monotonic()
        self._over_budget = False
        self._last_counters: tuple | None = None
        self._still_since: float | None = None
        self._said_stalled_at = 0.0

    def stop(self) -> None:
        self._stopped.set()

    def run(self) -> None:
        while not self._stopped.wait(self._every):
            try:
                self._emit()
            except Exception as exc:  # noqa: BLE001
                # A monitor that dies of its own output is worse than no monitor: the run
                # keeps spending and the last thing on screen is a stale number. This
                # happened for real — the line carried a ¥ sign and this machine's console
                # is GBK, so the thread died on its FIRST emit while the unit test (UTF-8
                # capsys) stayed green. Everything printed is ASCII now; this is the belt.
                print(f"  ~ (meter: {type(exc).__name__}, continuing)", flush=True)

    # How long the counters may stand still before that itself is the news. Long enough
    # to sit through a big scan's OCR and a slow download (one Malaysian document
    # legitimately took 21 minutes), short enough to beat the body deadline.
    STILL_SECONDS = 480.0

    def _report_if_nothing_moved(self, totals: dict, elapsed: float) -> None:
        """Say when the numbers stop changing, because the heartbeat itself will not.

        Measured 2026-08-02: a Malaysian run stood still for 27 minutes -- same call
        count, same cost, no document finished -- while this meter printed a cheerful,
        identical line every 60 seconds. Three worker threads were blocked on portal
        sockets that had been accepted and were delivering nothing. Every layer was
        "working": the process was alive, the heartbeat was on time, the log was
        growing. Reading it required noticing that two numbers a minute apart were the
        same, which is exactly the kind of noticing a monitor exists to do for you.

        This does not decide anything is wrong -- OCR and a slow fetch are silent here
        too, and saying "hung" about a healthy run is how a warning gets ignored. It
        reports the fact and how long it has held.
        """
        # Documents finished belongs in here with the token counters, and is the only one
        # of the five that moves on a fully cached run -- the configuration a replayed
        # demo runs in. Watching the LLM alone, "every clause was already judged" and
        # "three workers are blocked on dead sockets" look exactly the same.
        counters = (totals["calls"], totals["prompt"], totals["completion"],
                    totals["failed"], documents_completed())
        now = time.monotonic()
        if counters != self._last_counters:
            self._last_counters = counters
            self._still_since = None
            self._said_stalled_at = 0.0
            return
        if self._still_since is None:
            self._still_since = now
            return
        still_for = now - self._still_since
        if still_for < self.STILL_SECONDS:
            return
        # Repeat on the same cadence rather than once: a stall that is still there ten
        # minutes later is a different decision from one that has just started.
        if self._said_stalled_at and (now - self._said_stalled_at) < self.STILL_SECONDS:
            return
        self._said_stalled_at = now
        print(f"  !! NOTHING has moved for {int(still_for // 60)}m - same call count, same "
              f"tokens, same cost, after {int(elapsed // 60)}m of run. OCR and a slow fetch "
              "look like this too, so it is not proof of a hang; but if it holds, workers "
              "are blocked on sockets that are delivering nothing. Ctrl-C loses no judged "
              "clause.", flush=True)

    def _emit(self) -> None:
        t = _live_totals()
        elapsed = time.monotonic() - self._t0
        # BEFORE the early return, not after. The stall check used to sit at the bottom
        # of this method, below a `return` taken whenever nothing had reached an LLM --
        # so on a run whose judge cache answers everything, the one monitor that notices
        # a hang was silently switched off. That is the replayed-demo configuration.
        self._report_if_nothing_moved(t, elapsed)
        if not t["calls"] and not t["failed"]:
            return  # nothing has reached an LLM yet; the log lines carry the progress
        cached_share = f" {t['cached'] / t['prompt']:.0%} cached" if t["prompt"] else ""
        cost = t["cost_cny"]
        money = (f"CNY {cost:.2f}" if cost is not None
                 else "CNY ? (model not on the invoice)")
        print(
            f"  ~ {int(elapsed // 60):3d}m{int(elapsed % 60):02d}s | "
            f"{t['calls']:,} call(s) | "
            f"{(t['prompt'] + t['completion']) / 1e6:.2f}M tok{cached_share} | {money}"
            + (f" | {t['failed']} FAILED" if t["failed"] else ""),
            flush=True,
        )
        # A run whose calls are all failing bills nothing and accounts nothing, so the
        # token line alone looks like a run that is simply quiet. Name it.
        if t["failed"] and not t["calls"]:
            print("  !! every LLM call so far has FAILED - check the endpoint/key before "
                  "this burns the wall clock. Ctrl-C is free: verdicts commit per clause.",
                  flush=True)
        if self._budget and cost is not None and cost > self._budget and not self._over_budget:
            self._over_budget = True
            print(f"  !! BUDGET CNY {self._budget:.2f} EXCEEDED (CNY {cost:.2f}). An alarm, "
                  "not a brake - work already dispatched keeps running. Ctrl-C loses no "
                  "judged clause (the verdict cache commits per verdict).", flush=True)


def _report_replay_readiness(iso: str, verifier, indicators: list) -> None:
    """Under ``--offline``, say up front whether the judge can actually answer.

    Replay intercepts httpx at the transport, and the OpenAI SDK builds an httpx client
    like everything else — so in replay the judge's calls are intercepted too, and a clause
    with no cached verdict gets the synthetic 504. Every judgement then fails, the breaker
    opens and the degrade gate drops the document to the BM25 lane.

    That is the correct behaviour and it is loudly marked, but it is not a demonstration of
    the engine, and finding out costs a full run. The judge's replay layer is the VERDICT
    CACHE, not the HTTP recording: a cache hit makes no HTTP call at all, which is exactly
    why a recording taken with a warm cache contains no LLM traffic to replay. So the
    question worth asking before the run is about the cache, not the recording.
    """
    from lexora.collect import http_cache

    if http_cache.mode() != http_cache.REPLAY or verifier is None:
        return
    coverage = getattr(verifier, "cache_coverage", lambda _inds: None)(indicators)
    if coverage is None:
        print(f"  !! judge cache [{iso}]: DISABLED — in replay every judgement will fail "
              "and every document will degrade to the BM25 lane.")
        return
    live, total = coverage
    if live:
        print(f"  judge cache [{iso}]: {live} verdict(s) stored for this exact prompt "
              f"({total} in the store) — replay can answer from cache.")
        return
    print(f"  !! judge cache [{iso}]: NO verdict answers the prompt this run will send "
          f"({total} stored under an older prompt or model).")
    print("     In replay an unanswered judgement is a synthetic 504, so every document "
          "will degrade to the BM25 lane and say so.")
    print("     Fix: re-run once LIVE with --record to refill both data/http_cache/ and "
          "data/cache/judge.sqlite, then replay.")


def run_one(
    iso: str,
    *,
    budget: int,
    verify: bool,
    rationale_llm: bool = False,
    metadata_llm: bool = False,
    amendment_llm: bool = False,
    timeout: float,
    llm_workers: int = 1,
    doc_workers: int | None = None,
    fetch_min_interval: float | None = None,
    serial_fetch: bool | None = None,
    use_secondary: bool = False,
    verify_cells: bool = False,
    verify_clauses: bool = False,
    pillars: list[int] | None = None,
    discover_amendments: bool = True,
    discover_child_regulations: bool = True,
    discover_regulator_instruments: bool = True,
) -> MapResult:
    """Run the production multi-instrument map for one economy.

    ``pillars`` restricts the indicator set (e.g. ``[6]`` for cross-border data flows
    only); ``None`` means all mandatory pillars — the Round-1 default.
    """
    profile = load_profile(JURIS / f"{iso.lower()}.yaml")
    # The safe amount of concurrency is a property of the portal, so it comes from the
    # jurisdiction profile; an explicit argument still wins, for measuring a change.
    policy = profile.fetch_policy
    doc_workers = policy.doc_workers if doc_workers is None else doc_workers
    fetch_min_interval = (policy.min_interval if fetch_min_interval is None
                          else fetch_min_interval)
    serial_fetch = policy.serial_fetch if serial_fetch is None else serial_fetch
    # None -> load_indicators' default, the mandatory scope (pillars 6 + 7).
    indicators = load_indicators(INDICATORS, pillars=pillars)
    if pillars and not indicators:
        raise SystemExit(f"no RDTII indicators for pillar(s) {pillars}")
    secondary = []
    if use_secondary:
        from lexora.collect.secondary import gather_signals

        secondary = gather_signals(iso, indicators)
        print(f"  secondary sources [{iso}]: {len(secondary)} signal(s) from "
              f"{len({s.source_name for s in secondary})} tracker(s)")
    # Verifier mode, most capable first. Only "per_clause" lifts the discovery-attribution
    # ceiling (pipeline passes ALL indicators to each document, so a law surfaced by one
    # indicator's query is still judged for the other eight); "per_cell" is a precision
    # lane INSIDE that ceiling and "pick_one" is the legacy single-pick.
    mode = ("per_clause" if verify_clauses
            else "per_cell" if verify_cells
            else "pick_one")
    verifier = make_verifier(use_llm=verify or verify_cells or verify_clauses, mode=mode)
    if (verify or verify_cells or verify_clauses) and verifier is None:
        print("warning: --verify requested but the LLM verifier is unavailable "
              "(install the [llm] extra); continuing with BM25 + verbatim only.")
    _report_replay_readiness(iso, verifier, indicators)
    _register_client(iso, "verifier",
                     getattr(verifier, "_client", None) if verifier is not None else None)
    rationale_gen = make_rationale_generator(use_llm=rationale_llm)
    if rationale_llm and rationale_gen._client is None:
        print("warning: --rationale-llm requested but the LLM backend is unavailable; "
              "using the deterministic template rationale.")
    meta_extractor = make_metadata_extractor(use_llm=metadata_llm)
    if metadata_llm and meta_extractor._client is None:
        print("warning: --metadata-llm requested but the LLM backend is unavailable; "
              "using portal structured metadata only.")
    amendment_extractor = make_amendment_extractor(use_llm=amendment_llm or amend_llm_env())
    if amendment_llm and amendment_extractor._client is None:
        print("warning: --amendment-llm requested but the LLM backend is unavailable; "
              "using the regex amendment parser only.")
    result = run_pipeline_map(
        portal=profile.portals[0], profile=profile, indicators=indicators,
        budget=budget, timeout=timeout, verifier=verifier, rationale_gen=rationale_gen,
        meta_extractor=meta_extractor, llm_workers=llm_workers, doc_workers=doc_workers,
        fetch_min_interval=fetch_min_interval, serial_fetch=serial_fetch,
        secondary_signals=secondary, amendment_extractor=amendment_extractor,
        discover_amendments=discover_amendments,
        discover_child_regulations=discover_child_regulations,
        discover_regulator_instruments=discover_regulator_instruments,
    )
    tokens = {"calls": 0, "prompt": 0, "completion": 0, "total": 0, "cached_prompt": 0,
              "failed": 0}
    _register_client(iso, "rationale", rationale_gen._client)
    _register_client(iso, "metadata", meta_extractor._client)
    _register_client(iso, "amendment", amendment_extractor._client)

    def _account(label: str, client) -> None:
        """Print per-channel token usage and fold it into this economy's total."""
        if client is None:
            return
        failed = getattr(client, "failed_calls", 0)
        if not getattr(client, "calls", 0):
            if failed:
                # Not a free channel -- a broken one. Say so; a silent zero here is how a
                # wall of 403s once passed for a $0.00 run.
                print(f"  LLM {label} [{iso}]: 0 successful call(s), {failed} FAILED "
                      f"({getattr(client, 'last_error', '') [:90]})")
                tokens["failed"] += failed
            return
        cached = getattr(client, "cached_prompt_tokens", 0)
        share = f", {cached / client.prompt_tokens:.0%} of prompt cached" if cached else ""
        print(
            f"  LLM {label} tokens [{iso}]: {client.calls} call(s), "
            f"{client.total_tokens} tokens "
            f"({client.prompt_tokens} prompt + {client.completion_tokens} completion"
            f"{share})"
            + (f"  [{failed} failed]" if failed else "")
        )
        tokens["calls"] += client.calls
        tokens["prompt"] += client.prompt_tokens
        tokens["completion"] += client.completion_tokens
        tokens["total"] += client.total_tokens
        tokens["cached_prompt"] += cached
        tokens["failed"] += failed

    if meta_extractor._client is not None:
        print(
            f"  LLM metadata usage [{iso}]: {meta_extractor.extracted} doc(s) extracted, "
            f"{meta_extractor.rejected} field(s) rejected by source-check"
            + (f" ({meta_extractor.error_count} backend error(s))"
               if meta_extractor.error_count else "")
        )
        _account("metadata", meta_extractor._client)
    if amendment_extractor._client is not None:
        print(
            f"  LLM amendment usage [{iso}]: {amendment_extractor.extracted} amending Act(s) "
            f"extracted, {amendment_extractor.rejected} instruction(s) rejected by source-check"
            + (f" ({amendment_extractor.error_count} backend error(s))"
               if amendment_extractor.error_count else "")
        )
        _account("amendment", amendment_extractor._client)
    # Always report, including when the layer is OFF. At the 2026-08-03 live pitch the
    # judges asked for a full Singapore run and read a Mapping Rationale column that was
    # 181/181 deterministic template, because the command omitted --rationale-llm. This
    # line used to be printed only when the client existed, so a run with the layer off
    # said nothing about rationales at all and the omission had no output signature --
    # every rehearsal artifact was 0% LLM and nobody noticed for two weeks.
    if rationale_gen._client is None:
        print(
            f"  LLM rationale [{iso}]: OFF -- the whole Mapping Rationale column is the "
            "deterministic template (it restates the section number and indicator name). "
            "Pass --rationale-llm to have the model author it."
        )
    else:
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

    # Verdict cache: a run that answered most clauses from cache is NOT a run that judged
    # them cheaply, and a reader must be able to tell the two apart. Report the split.
    judged = getattr(verifier, "judged", 0) if verifier is not None else 0
    from_cache = getattr(verifier, "from_cache", 0) if verifier is not None else 0
    if judged or from_cache:
        total = judged + from_cache
        print(
            f"  judge cache [{iso}]: {from_cache}/{total} clause(s) served from cache "
            f"({from_cache / total:.0%}), {judged} judged by the LLM"
        )
        tokens["clauses_judged"] = judged
        tokens["clauses_from_cache"] = from_cache
    return result, tokens


def write_artifacts(citations, documents, out_csv: Path, *, judge: str = "") -> dict:
    """Write one run's four artifacts and return where they went.

    THE single writer. ``write_outputs`` (one economy, ``main.py``'s entry point) and
    the multi-economy run at the bottom of this file both go through it, because while
    they were separate they drifted: the multi-economy path — the one the README gives
    for the actual submission run — quietly skipped BOTH the ``DEMO_`` rename and the
    reviewer console, so a replayed full run wrote a file named exactly like a
    submission and no page to review it on.

    ``judge`` names the verifier lane that decided inclusion (e.g. ``"per_clause"``),
    for the sidecar's ``retrieval_method``. Empty means no verifier ran.
    """
    from lexora.export import provenance
    from lexora.export.html_exporter import to_html

    # A replayed run renames its artifacts. Rocky's engine does the same thing for the same
    # reason: a demonstration CSV that is byte-shaped like a submission CSV will eventually
    # be filed as one, and the filename is the one label that survives being emailed,
    # renamed by a download folder, or opened in Excel with the header row collapsed.
    out_csv = out_csv.with_name(provenance.label(out_csv.name))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = to_csv(citations, out_csv)
    jsonld_out = out_csv.with_suffix(".jsonld")
    to_jsonld(citations, jsonld_out)
    html_out = out_csv.with_suffix(".html")
    to_html(citations, html_out, title=f"Lexora — {out_csv.stem}")
    cfg = load_config()
    use_dense = os.environ.get("LEXORA_MAP_DENSE", "").lower() in ("1", "true", "yes", "on")
    json_out = out_csv.with_suffix(".json")
    n_json = to_submission_json(
        documents, json_out, model_version=f"llm:{cfg.llm_model}",
        use_dense=use_dense, judge=judge,
    )
    return {"csv": out_csv, "rows": n, "json": json_out, "provisions": n_json,
            "jsonld": jsonld_out, "html": html_out}


def write_outputs(result: MapResult, out_csv: Path, *, judge: str = "") -> int:
    """Write one economy's run (``main.py``'s path) and print its summary.

    Returns the number of CSV rows."""
    from lexora.export import provenance

    written = write_artifacts(result.citations, result.documents, out_csv, judge=judge)
    out_csv, n = written["csv"], written["rows"]
    json_out, html_out = written["json"], written["html"]
    tagged = {}
    for c in result.citations:
        tag = getattr(c.discovery_tag, "value", str(c.discovery_tag))
        tagged[tag] = tagged.get(tag, 0) + 1
    inds = len({c.indicator_id for c in result.citations})
    if provenance.is_demonstration():
        print(f"\n!! {provenance.BANNER}")
        print("   Artifacts are prefixed "
              f"{provenance.PREFIX!r} and must not be submitted as results.")
    print(f"\nWrote {n} provision(s) -> {out_csv}")
    print(f"                        -> {json_out}")
    print(f"                        -> {html_out}  (open in a browser to review)")
    print(f"  indicators covered: {inds}/9   discovery tags: {tagged or '{}'}")
    return n


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
    # An indicator the run ASKED ABOUT and found nothing for is a result, not an absence
    # of one, and until now it left no trace anywhere: the row simply was not written.
    # The judges may ask us to run one indicator in one country, so "we returned nothing
    # for that cell" has to be something the engine SAYS, not something we explain over a
    # blank screen. Singapore has no data-localisation requirement; the honest output for
    # SG P6-I1 is an explicit miss, and a tool that produced a provision there would be
    # inventing one.
    in_scope = sorted({i.submission_id for i in (result.indicators or [])})
    indicators_empty = [i for i in in_scope if i not in set(indicators_covered)]
    review_rows = sum(
        1 for c in result.citations if c.review_status.value == "CONFLICT_REVIEW"
    )
    # Amendment-currency activity (Tier-2): rows routed to AMENDMENT_REVIEW and the
    # spread of currency verdicts. CONFLICT_REVIEW (mapping doubt) and AMENDMENT_REVIEW
    # (currency doubt) are distinct queues, so the latter needs its own counter — else
    # a run that correctly flags amended/repealed provisions still reports 0 review rows.
    amendment_review_rows = sum(
        1 for c in result.citations if c.review_status.value == "AMENDMENT_REVIEW"
    )
    currency_breakdown: dict[str, int] = {}
    for c in result.citations:
        st = getattr(c, "currency_status", None) or "UNKNOWN"
        currency_breakdown[st] = currency_breakdown.get(st, 0) + 1
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
        "indicators_in_scope": in_scope,
        "indicators_empty": indicators_empty,
        "working_set_filter": getattr(result, "working_set_filter", ""),
        "citations_by_indicator": {
            ind: sum(1 for c in result.citations if c.indicator_id == ind)
            for ind in indicators_covered
        },
        "review_rows": review_rows,
        "amendment_review_rows": amendment_review_rows,
        "currency_breakdown": currency_breakdown,
    }


def _narrow_output_to_indicator(citations: list, summaries: list[dict]) -> list:
    """LIVE-PITCH ONLY. Narrow the WRITTEN OUTPUT to one indicator, after the run.

    A judge names an economy and an indicator and expects to watch the engine answer THAT.
    The obvious way -- load only that indicator -- is the wrong one: the per-clause judge
    puts all nine indicators in one prompt, and the verdict cache is keyed on the RENDERED
    prompt, so dropping eight of them changes every key. In replay an unanswered judgement
    is a synthetic 504, the breaker opens, and every document degrades to the keyword lane.
    The run would appear to work while every row said the model never saw it.

    So nothing about the run changes: all nine indicators are discovered, mapped and judged
    exactly as always, the cache still hits, and only the file we write is filtered. That
    also keeps the empty answer meaningful -- the indicator really was judged across the
    whole economy, so nothing found is a finding rather than an artefact of the filter,
    which is the opposite of what LEXORA_ONLY_LAW can promise.

    The printed indicator grid deliberately still shows all nine: the narrowing is a
    presentation choice and the screen should say so.
    """
    want = os.environ.get("LEXORA_ONLY_INDICATOR", "").strip().upper()
    if not want:
        return citations
    in_scope = sorted({i for s in summaries for i in (s.get("indicators_in_scope") or [])})
    if in_scope and want not in in_scope:
        # A typo would silently write an empty CSV, which on stage is indistinguishable
        # from "this economy has no such provision" -- the one confusion we never allow.
        print(f"\n!! LEXORA_ONLY_INDICATOR={want!r} is not one of the indicators this run "
              f"asked about ({', '.join(in_scope)}). Writing the FULL output instead.")
        return citations
    kept = [c for c in citations if c.indicator_id.upper() == want]
    print(f"\nLEXORA_ONLY_INDICATOR={want}: the run judged every in-scope indicator as "
          f"usual; only the written file is narrowed, to {len(kept)} of {len(citations)} "
          f"row(s). The grid below still shows the whole run.")
    return kept


def _print_indicator_grid(summaries: list[dict]) -> None:
    """One line per economy per indicator, INCLUDING the ones that found nothing.

    The judges have confirmed that a "specific case" may be one indicator in one country.
    Until now the answer to a cell we found nothing for was a blank screen, which is the
    same thing the screen shows when a run is broken -- so the operator had to talk over
    silence and ask to be believed.

    An empty cell is a finding. Singapore has no data-localisation requirement, so
    SG / P6-I1 SHOULD come back empty, and a tool that produced a provision for it would
    be inventing one. That is a good answer, and it is only a good answer if the engine
    is the thing that says it.

    NO PROVISION FOUND is not the same as NOT ASKED: the scope line above it names every
    indicator this run put to the judge, so a reader can tell a real miss from a pillar
    nobody enumerated.
    """
    rows = [s for s in summaries if not s.get("error") and s.get("indicators_in_scope")]
    if not rows:
        return
    scope = sorted({i for s in rows for i in s["indicators_in_scope"]})
    print("\nIndicator coverage (every indicator this run asked about, hit or miss):")
    print(f"  in scope: {', '.join(scope)}")
    # A filtered run must never let "no provision found" be read as a fact about the
    # country. On a full run the sentence means the ECONOMY has nothing for that
    # indicator; on a run narrowed to one Act it means only that THIS ACT does not -- and
    # Malaysia plainly does have cross-border provisions. Presenting the second as the
    # first would turn a demo's convenience into a false statement about a legal system,
    # on the one screen a judge is watching.
    for s in rows:
        if s.get("working_set_filter"):
            print(f"  !! {s['economy']}: working set was FILTERED to "
                  f"{s['working_set_filter']!r}. Below, 'NO PROVISION FOUND' means THAT "
                  "ACT has none - it says nothing about the economy. Re-run without "
                  "LEXORA_ONLY_LAW before quoting any of it.")
    for s in rows:
        hits = dict(s.get("citations_by_indicator") or {})
        empty = set(s.get("indicators_empty") or [])
        for ind in sorted(s["indicators_in_scope"]):
            if ind in empty:
                print(f"  {s['economy']:<11} {ind:<7} NO PROVISION FOUND "
                      "- judged and returned nothing (not skipped)")
            else:
                print(f"  {s['economy']:<11} {ind:<7} {hits.get(ind, 0)} citation(s)")


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
    ap.add_argument("--verify-clauses", action="store_true",
                    help="Per-clause 9-in-1 LLM verifier: judge each candidate clause against "
                         "ALL indicators at once. Unlike --verify-cells this LIFTS the "
                         "discovery-attribution ceiling — a law surfaced by one indicator's "
                         "query is still mapped for the other eight (recovers e.g. MY PDPA "
                         "s.129 for P6-I4, SG PDPA s.26/s.11). Needs LEXORA_LLM_* endpoint.")
    ap.add_argument("--rationale-llm", action="store_true",
                    help="Author the Mapping Rationale column with the LLM (template fallback "
                         "+ verbatim-copy guard; needs LEXORA_LLM_* endpoint)")
    ap.add_argument("--metadata-llm", action="store_true",
                    help="Extract Law Number / Last Amended from document text with the LLM "
                         "(source-verified) when portal channel + curated anchor don't supply them")
    ap.add_argument("--amendment-llm", action="store_true",
                    help="Extract amendment instructions (Tier-2 provision adjudication) with the "
                         "LLM, source-verified, regex fallback; runs concurrently under "
                         "--llm-workers. Also enabled by LEXORA_AMENDMENT_LLM=1")
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
    ap.add_argument("--doc-workers", type=int, default=None,
                    help="Threads for processing instruments within an economy concurrently "
                         "(document-level parallelism). This is what parallelizes the per-document "
                         "metadata extraction (the serial floor of an LLM run), plus fetch/OCR/"
                         "rationale across documents. 1 = serial. Stacks with --jobs and "
                         "--llm-workers. Unset = the jurisdiction profile's fetch_policy.")
    ap.add_argument("--fetch-min-interval", type=float, default=None,
                    help="Minimum seconds between fetch starts to the SAME host (per-host "
                         "rate-spacing, thread-enforced). Dodges request-rate anti-bot under "
                         "doc-level concurrency (e.g. AU serving an HTML challenge instead of "
                         "the PDF); post-fetch OCR/LLM still parallelize. 0 = off. Unset = the jurisdiction profile's fetch_policy.")
    ap.add_argument("--secondary", action="store_true",
                    help="Use RDTII secondary sources (UNCTAD etc.) as a DISCOVERY AID: seed "
                         "discovery with the laws they point to (recall), stamp matching "
                         "citations with a 'corroborated by <source>' note (provenance), and "
                         "print a coverage cross-check. Never cited as evidence.")
    ap.add_argument("--serial-fetch", action="store_true", default=None,
                    help="Fully serialize same-host DOWNLOADS (one request in flight per host) "
                         "while OCR/parse/map/LLM still run parallel across documents. Stronger "
                         "than --fetch-min-interval against cumulative anti-bot (no request burst "
                         "at all); different economies (hosts) stay parallel. Use with --doc-workers.")
    ap.add_argument("--no-ocr", action="store_true",
                    help="disable OCR. OCR is ON by default for submission runs so "
                         "scanned-only statutes (e.g. MY gazette PDFs) are not silently "
                         "dropped to 0 clauses; pass this only to reproduce the text-layer-"
                         "only behaviour.")
    # The three follow-on passes, separately. They shared one switch until 2026-08-02,
    # which meant the only way to skip the expensive one was to skip the cheap one too.
    # Cost is wildly asymmetric: amendment discovery issues a title search PER ACT (110
    # on Malaysia, hours), child regulations is AU-only, and regulator soft law fetches
    # three known URLs (minutes) while carrying 84 of Malaysia's 183 rows -- and the ONLY
    # evidence for MY / P7-I4.
    ap.add_argument("--no-amendment-discovery", action="store_true",
                    help="skip the per-Act amendment search. By far the most expensive "
                         "follow-on pass; skipping it costs amendment currency, not "
                         "instruments.")
    ap.add_argument("--no-child-regulations", action="store_true",
                    help="skip delegated legislation (AU only).")
    ap.add_argument("--no-regulator-instruments", action="store_true",
                    help="skip regulator soft law (codes of practice, standards). "
                         "CHEAP and high-yield -- skipping it removes gold instruments "
                         "the statute portal does not index, and can empty a whole "
                         "indicator. Rarely what you want.")
    ap.add_argument("--dry-run", action="store_true", help="plan only, no network")
    ap.add_argument("--heartbeat", type=float, default=60.0, metavar="SECONDS",
                    help="How often to print live call/token/cost totals (0 = off). A "
                         "full run is hours long and every cost counter used to appear "
                         "only after its economy finished.")
    ap.add_argument("--max-cost", type=float, default=0.0, metavar="CNY",
                    help="Print a loud alarm once the run passes this spend (CNY). An alarm, "
                         "not a brake — but Ctrl-C is nearly free, because the verdict "
                         "cache commits per verdict.")
    ap.add_argument("--verbose", action="store_true",
                    help="Do not quieten httpx/openai/PIL loggers.")
    args = ap.parse_args()

    # OCR defaults ON for submission (kill the silent-scan-drop footgun). An explicit
    # LEXORA_OCR in the environment still wins; --no-ocr forces it off.
    if args.no_ocr:
        os.environ.pop("LEXORA_OCR", None)
    elif "LEXORA_OCR" not in os.environ:
        os.environ["LEXORA_OCR"] = "1"

    configure_logging(args.verbose)
    isos = ["sg", "au", "my"] if args.jurisdiction == "all" else [args.jurisdiction.lower()]

    if args.dry_run:
        # This is the pre-flight for a run that costs real money, so it has to describe
        # the run that would ACTUALLY happen. It used to print `verify=False` whenever
        # the judge was selected as --verify-clauses rather than the legacy --verify,
        # i.e. it reported "no LLM verifier" for the exact configuration the README
        # recommends and which produces essentially all of our output.
        judge = ("per_clause" if args.verify_clauses else "per_cell" if args.verify_cells
                 else "pick_one" if args.verify else "")
        print("Submission run plan (no network):")
        for iso in isos:
            print(f"  - {ISO_TO_COUNTRY.get(iso, iso)} ({iso}) "
                  f"-> map budget {args.budget}, judge={judge or 'OFF (BM25 lane)'}, "
                  f"rationale_llm={args.rationale_llm}, metadata_llm={args.metadata_llm}, "
                  f"amendment_llm={args.amendment_llm or amend_llm_env()}")
        if judge:
            verifier = make_verifier(use_llm=True, mode=judge)
            if verifier is None:
                print("  -> LLM verifier UNAVAILABLE (install the [llm] extra) — the run "
                      "would degrade to the key-free BM25 lane")
            else:
                cfg = load_config()
                print(f"  -> LLM judge ON, mode {judge} (model: {cfg.llm_model})")
        concurrent = (args.jobs if args.jobs > 0 else len(isos)) * max(1, args.llm_workers)
        print(f"  -> up to {concurrent} concurrent LLM call(s) "
              f"({args.jobs or len(isos)} econom(ies) x {args.llm_workers} worker(s))"
              + ("   !! >32 collapses this endpoint to ~2.7x SLOWER with ZERO errors"
                 if concurrent > 32 else ""))
        print(f"  -> OCR {'OFF (--no-ocr)' if args.no_ocr else 'ON (default)'}")
        print(f"  -> heartbeat every {args.heartbeat:g}s"
              if args.heartbeat > 0 else "  -> heartbeat OFF (run will be silent on cost)")
        if args.max_cost:
            print(f"  -> cost alarm at CNY {args.max_cost:g} (an alarm, not a brake)")
        print(f"  -> would write {args.out} "
              "(+ .json sidecar, .jsonld, .html console, .summary.json)")
        return

    if not os.environ.get("LEXORA_LIVE"):
        print("note: set LEXORA_LIVE=1 to run the live submission crawl (or use --dry-run).")

    from lexora.collect import http_cache

    cache_mode = http_cache.install()
    if cache_mode != http_cache.OFF:
        n = http_cache.store().stats()
        print(f"  -> HTTP cache {cache_mode}: {n['responses']} responses, "
              f"{n['renders']} rendered pages on disk")

    all_citations = []
    all_documents = []
    summaries = []
    results_by_iso = {}
    tokens_total = {"calls": 0, "prompt": 0, "completion": 0, "total": 0}

    def _run(iso: str):
        return run_one(iso, budget=args.budget, verify=args.verify,
                       rationale_llm=args.rationale_llm, metadata_llm=args.metadata_llm,
                       amendment_llm=args.amendment_llm,
                       timeout=args.timeout, llm_workers=args.llm_workers,
                       doc_workers=args.doc_workers, fetch_min_interval=args.fetch_min_interval,
                       serial_fetch=args.serial_fetch, use_secondary=args.secondary,
                       verify_cells=args.verify_cells, verify_clauses=args.verify_clauses,
                       discover_amendments=not args.no_amendment_discovery,
                       discover_child_regulations=not args.no_child_regulations,
                       discover_regulator_instruments=not args.no_regulator_instruments)

    # Country-level parallelism: economies are independent, so run them concurrently.
    # Threads (not processes) because the heavy stages — network fetch, OCR
    # (onnxruntime releases the GIL), LLM calls — are I/O- or C++-bound; this dodges
    # Windows spawn/pickling and lets workers share one process. Results collected
    # per-iso so one economy failing can't lose the others, then walked in input
    # order for a stable summary table.
    jobs = args.jobs if args.jobs > 0 else len(isos)
    outcomes: dict[str, tuple[str, object]] = {}
    meter = LiveMeter(args.heartbeat, args.max_cost) if args.heartbeat > 0 else None
    if meter is not None:
        meter.start()
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
    if meter is not None:
        meter.stop()

    for iso in isos:  # input order -> stable summary table
        kind, payload = outcomes[iso]
        if kind == "err":
            # The summary table gets one line, which is right for a table and useless
            # for a diagnosis: an economy died mid-run and the traceback -- the only
            # thing that says WHERE -- was held in the exception object and dropped.
            # A paid run does not get repeated to find that out.
            log.error("%s failed, full traceback follows:", ISO_TO_COUNTRY.get(iso, iso))
            log.error("".join(traceback.format_exception(
                type(payload), payload, payload.__traceback__)).rstrip())
            summaries.append({"iso": iso, "economy": ISO_TO_COUNTRY.get(iso, iso),
                              "error": f"{type(payload).__name__}: {payload}"})
            continue
        result, tokens = payload
        all_citations.extend(result.citations)
        all_documents.extend(result.documents)
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

    all_citations = _narrow_output_to_indicator(all_citations, summaries)

    # Same writer as main.py: the CSV, the JSON sidecar (per-provision OCR audit, timing,
    # model version, raw context), the JSON-LD dump and the reviewer console — and the
    # DEMO_ rename when the run was replayed.
    written = write_artifacts(
        all_citations, all_documents, args.out,
        judge=("per_clause" if args.verify_clauses
               else "per_cell" if args.verify_cells
               else "pick_one" if args.verify else ""),
    )
    out_path, n = written["csv"], written["rows"]
    json_out, n_json = written["json"], written["provisions"]
    jsonld_out, html_out = written["jsonld"], written["html"]
    summary_out = out_path.with_suffix(".summary.json")
    summary_out.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nRound-1 submission run (budget {args.budget}"
          f"{', verifier ON' if args.verify else ''})")
    print(f"{'economy':<12}{'instr':<7}{'NEW':<5}{'KNOWN':<7}{'fetched':<9}{'real':<6}"
          f"{'cites':<7}{'inds':<6}{'review':<8}{'amend?'}")
    print("-" * 78)
    for s in summaries:
        if s.get("error"):
            print(f"{s['economy']:<12}ERROR: {s['error']}")
            continue
        print(f"{s['economy']:<12}{s['instruments']:<7}{s['new_instruments']:<5}"
              f"{s['known_instruments']:<7}{s['fetched_ok']:<9}{s['docs_with_clauses']:<6}"
              f"{s['citations']:<7}{s['n_indicators_covered']:<6}{s['review_rows']:<8}"
              f"{s.get('amendment_review_rows', 0)}")
    print("-" * 78)
    _print_indicator_grid(summaries)
    # Amendment-currency roll-up across economies (Tier-2 activity at a glance).
    _curr: dict[str, int] = {}
    for s in summaries:
        for k, v in (s.get("currency_breakdown") or {}).items():
            _curr[k] = _curr.get(k, 0) + v
    if _curr:
        print("currency: " + "  ".join(f"{k}={v}" for k, v in sorted(_curr.items())))
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

    from lexora.export import provenance

    if provenance.is_demonstration():
        print(f"\n!! {provenance.BANNER}")
        print("   Artifacts are prefixed "
              f"{provenance.PREFIX!r} and must not be submitted as results.")
    print(f"Wrote {n} citation row(s) -> {out_path} (submission CSV)")
    print(f"                          -> {json_out} ({n_json}-provision JSON sidecar)")
    print(f"                          -> {jsonld_out} (JSON-LD)")
    print(f"                          -> {html_out} (open in a browser to review)")
    print(f"                          -> {summary_out} (run summary)")


if __name__ == "__main__":
    main()
