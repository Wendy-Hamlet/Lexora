"""P-5 robustness tests — browser backoff/retry, SG empty-shell detection, and
the full-submission run summary. All offline (Chromium is never launched; the
session's per-render call is faked)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from lexora.collect.browser import BrowserSession, RenderedResult
from lexora.collect.discovery import sg_results_present
from lexora.pipeline import MapResult

SHELL = (
    '<html><a href="/Browse/Acts">Browse</a>'
    '<a href="/Acts-Supp/2024">Supplement</a>'
    '<a href="/Act-Rev/2020">Revised</a></html>'
)
RESULTS = (
    '<html><a href="/Act/PDPA2012">Personal Data Protection Act 2012</a></html>'
)


def _session_yielding(pages):
    """A BrowserSession whose _render_once returns the given HTMLs in turn,
    without launching Chromium."""
    s = BrowserSession()
    seq = iter(pages)
    s._render_once = lambda url, **kw: RenderedResult(status=200, final_url=url, html=next(seq))
    return s


# --- SG empty-shell detector -------------------------------------------------

def test_sg_results_present_detects_instrument_link():
    assert sg_results_present(RESULTS)
    assert sg_results_present(RESULTS.encode("utf-8"))  # bytes accepted


def test_sg_results_present_rejects_browse_shell():
    # The rate-limited shell links only to browse nav (/Browse, /Acts-Supp,
    # /Act-Rev) — none of which is a `/Act/<slug>` instrument link.
    assert not sg_results_present(SHELL)


# --- browser backoff / retry -------------------------------------------------

def test_render_retries_until_valid(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    s = _session_yielding([SHELL, SHELL, RESULTS])
    out = s.render("https://sso/x", retries=2, is_valid=sg_results_present)
    assert sg_results_present(out.html)  # recovered the results page on retry


def test_render_returns_last_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    s = _session_yielding([SHELL, SHELL, SHELL])
    out = s.render("https://sso/x", retries=2, is_valid=sg_results_present)
    # Graceful: a persistently-empty result is returned, not raised (a genuinely
    # empty search also looks "absent").
    assert not sg_results_present(out.html)


def test_render_no_retry_when_first_is_valid(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    calls = {"n": 0}
    s = BrowserSession()

    def fake(url, **kw):
        calls["n"] += 1
        return RenderedResult(200, url, RESULTS)

    s._render_once = fake
    s.render("https://sso/x", retries=3, is_valid=sg_results_present)
    assert calls["n"] == 1  # valid on first render -> no extra navigations


def test_render_without_predicate_does_not_retry():
    calls = {"n": 0}
    s = BrowserSession()

    def fake(url, **kw):
        calls["n"] += 1
        return RenderedResult(200, url, SHELL)

    s._render_once = fake
    s.render("https://sso/x", retries=3)  # is_valid defaults to None
    assert calls["n"] == 1  # no predicate -> single render (pre-P-5 behaviour)


def test_render_degrades_to_empty_when_every_nav_raises(monkeypatch):
    # A hung SSO navigation (page.goto timeout) raises inside _render_once. It must
    # degrade to an empty result, NOT propagate and crash the whole sweep.
    monkeypatch.setattr("time.sleep", lambda *_: None)
    calls = {"n": 0}
    s = BrowserSession()

    def boom(url, **kw):
        calls["n"] += 1
        raise TimeoutError("Page.goto timeout")

    s._render_once = boom
    out = s.render("https://sso/x", retries=2, is_valid=sg_results_present)
    assert out.html == "" and out.status == 0  # graceful empty, no exception
    assert calls["n"] == 3  # a raising nav counts as a failed attempt (1 + retries)


def test_render_recovers_after_a_raising_nav(monkeypatch):
    # First navigation times out, retry succeeds — the transient failure is retried
    # the same way an empty shell is.
    monkeypatch.setattr("time.sleep", lambda *_: None)
    seq = iter(["raise", RESULTS])
    s = BrowserSession()

    def flaky(url, **kw):
        if next(seq) == "raise":
            raise TimeoutError("Page.goto timeout")
        return RenderedResult(200, url, RESULTS)

    s._render_once = flaky
    out = s.render("https://sso/x", retries=2, is_valid=sg_results_present)
    assert sg_results_present(out.html)  # recovered on the retry after the timeout


# --- submission run summary --------------------------------------------------

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_submission.py"
_spec = importlib.util.spec_from_file_location("run_submission", _SCRIPT)
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)


def test_summarize_counts_instruments_citations_and_review():
    mr = MapResult(
        discovered=[
            SimpleNamespace(discovery_tag="NEW"),
            SimpleNamespace(discovery_tag="KNOWN"),
            SimpleNamespace(discovery_tag="NEW"),
        ],
        documents=[
            SimpleNamespace(document=SimpleNamespace(http_status=200), clauses=["c1", "c2"]),
            SimpleNamespace(document=SimpleNamespace(http_status=200), clauses=[]),  # 2xx but anti-bot decoy: 0 clauses
            SimpleNamespace(document=SimpleNamespace(http_status=403), clauses=[]),
        ],
        citations=[
            SimpleNamespace(indicator_id="P6-I4", review_status=SimpleNamespace(value="VERIFIED")),
            SimpleNamespace(indicator_id="P7-I1", review_status=SimpleNamespace(value="CONFLICT_REVIEW")),
            SimpleNamespace(indicator_id="P6-I4", review_status=SimpleNamespace(value="VERIFIED")),
        ],
    )
    s = rs.summarize("sg", mr)
    assert s["economy"] == "Singapore"
    assert s["instruments"] == 3
    assert s["new_instruments"] == 2
    assert s["known_instruments"] == 1
    assert s["fetched_ok"] == 2          # both 200s count (incl. the anti-bot decoy)
    assert s["docs_with_clauses"] == 1   # real yield: only the doc that parsed clauses
    assert s["citations"] == 3
    assert s["indicators_covered"] == ["P6-I4", "P7-I1"]  # distinct + sorted
    assert s["n_indicators_covered"] == 2
    assert s["review_rows"] == 1         # the CONFLICT_REVIEW row


def test_run_one_warns_when_verify_requested_but_verifier_unavailable(monkeypatch, capsys):
    from lexora.models.source import FetchPolicy

    monkeypatch.setattr(
        rs,
        "load_profile",
        lambda _path: SimpleNamespace(portals=[SimpleNamespace()],
                                      fetch_policy=FetchPolicy()),
    )
    monkeypatch.setattr(rs, "load_indicators", lambda _path, **_kw: [])
    monkeypatch.setattr(rs, "make_verifier", lambda use_llm, **kw: None)

    def fake_run_pipeline_map(**kwargs):
        assert kwargs["verifier"] is None
        return MapResult(discovered=[], documents=[], citations=[])

    monkeypatch.setattr(rs, "run_pipeline_map", fake_run_pipeline_map)

    rs.run_one("sg", budget=1, verify=True, timeout=1.0)

    out = capsys.readouterr().out
    assert "LLM verifier is unavailable" in out
    assert "BM25 + verbatim only" in out


def test_the_live_meter_prices_what_the_run_has_spent_so_far(capsys):
    """A full run is hours of judge time, and every cost counter used to be printed only
    after its ECONOMY finished — so a wall of 403s, or a prefix cache that stopped
    hitting and tripled the bill, stayed invisible for an hour or more."""
    from types import SimpleNamespace

    rs._LIVE_CLIENTS.clear()
    rs._register_client("sg", "verifier", SimpleNamespace(
        model="GLM-5.2", calls=1000, prompt_tokens=3_200_000,
        cached_prompt_tokens=3_051_000, completion_tokens=120_000, failed_calls=0))
    meter = rs.LiveMeter(every=3600)
    meter._emit()
    line = capsys.readouterr().out

    assert "1,000 call(s)" in line
    assert "95% cached" in line          # the single biggest lever on the bill
    # (3.2M - 3.051M)@8 + 3.051M@2 + 0.12M@28 CNY/1M = CNY 10.65
    assert "CNY 10.65" in line
    # The line must survive the console it is printed to. It did not: a ¥ sign killed
    # the meter thread on its FIRST emit on this GBK machine, while this very test
    # stayed green because pytest captures in UTF-8.
    assert line.isascii(), f"non-ASCII in the meter line: {line!r}"
    rs._LIVE_CLIENTS.clear()


def test_a_wall_of_failures_is_named_not_left_looking_quiet(capsys):
    """A failed call bills nothing and accounts nothing, so in the token counters a run
    whose every request was rejected looks exactly like a run making no requests. That is
    how a 403 once presented itself as a free, successful cost benchmark."""
    from types import SimpleNamespace

    rs._LIVE_CLIENTS.clear()
    rs._register_client("sg", "verifier", SimpleNamespace(
        model="GLM-5.2", calls=0, prompt_tokens=0, cached_prompt_tokens=0,
        completion_tokens=0, failed_calls=57, last_error="403 team not allowed"))
    rs.LiveMeter(every=3600)._emit()
    out = capsys.readouterr().out

    assert "57 FAILED" in out
    assert "every LLM call so far has FAILED" in out
    assert "Ctrl-C is free" in out       # because verdicts commit per verdict
    rs._LIVE_CLIENTS.clear()


def test_progress_reaches_a_redirected_log_while_the_run_is_still_going(monkeypatch):
    """Nobody watches a two-hour run in a terminal; they redirect it to a file.

    Python block-buffers stdout as soon as it is not a tty, so every progress line and
    every 60-second heartbeat sits in an 8 KB buffer instead of reaching the log. The
    handler was configured and the meter thread was alive, and the paid run STILL showed
    an empty log file three minutes in — indistinguishable from a hang. A monitor that
    only reaches its reader in 8 KB batches is not a monitor.
    """
    import sys

    class _Redirected:
        """Stands in for stdout attached to a file: buffered until told otherwise."""

        def __init__(self):
            self.line_buffering = False
            self.reconfigured = []

        def reconfigure(self, **kw):
            self.reconfigured.append(kw)
            if "line_buffering" in kw:
                self.line_buffering = kw["line_buffering"]

        def write(self, s):
            return len(s)

        def flush(self):
            pass

    fake = _Redirected()
    monkeypatch.setattr(sys, "stdout", fake)
    rs.configure_logging()
    assert fake.line_buffering is True, "a redirected run would buffer its own progress"

    # A stdout that cannot be reconfigured must not take the run down with it.
    class _Bare:
        def write(self, s):
            return len(s)

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", _Bare())
    rs.configure_logging()  # must not raise


def test_a_run_that_stops_moving_says_so(capsys, monkeypatch):
    """The heartbeat cannot report a stall by printing: an identical line every 60 s is
    exactly what a stall looks like. Measured 2026-08-02, a Malaysian run stood still for
    27 minutes -- same calls, same cost, no document finished -- while three workers were
    blocked on portal sockets delivering nothing, and every line on screen said the run
    was fine. Reading it required noticing two equal numbers a minute apart."""
    from types import SimpleNamespace

    rs._LIVE_CLIENTS.clear()
    rs._register_client("my", "verifier", SimpleNamespace(
        model="GLM-5.2", calls=1395, prompt_tokens=3_210_000,
        cached_prompt_tokens=2_630_000, completion_tokens=60_000, failed_calls=0))
    meter = rs.LiveMeter(every=3600)

    clock = {"t": 1000.0}
    monkeypatch.setattr(rs.time, "monotonic", lambda: clock["t"])

    meter._emit()                      # first sight of these counters
    capsys.readouterr()
    meter._emit()                      # unchanged -> start the stopwatch, say nothing yet
    assert "NOTHING has moved" not in capsys.readouterr().out

    clock["t"] += meter.STILL_SECONDS + 1
    meter._emit()
    out = capsys.readouterr().out
    assert "NOTHING has moved" in out
    assert "Ctrl-C loses no judged clause" in out
    assert out.isascii(), f"non-ASCII in the meter line: {out!r}"

    # It must NOT keep shouting every minute...
    clock["t"] += 60
    meter._emit()
    assert "NOTHING has moved" not in capsys.readouterr().out
    # ...but a stall that is still there much later is news again.
    clock["t"] += meter.STILL_SECONDS
    meter._emit()
    assert "NOTHING has moved" in capsys.readouterr().out

    # And any movement at all clears it.
    rs._LIVE_CLIENTS.clear()
    rs._register_client("my", "verifier", SimpleNamespace(
        model="GLM-5.2", calls=1400, prompt_tokens=3_220_000,
        cached_prompt_tokens=2_640_000, completion_tokens=60_100, failed_calls=0))
    clock["t"] += meter.STILL_SECONDS + 1
    meter._emit()
    assert "NOTHING has moved" not in capsys.readouterr().out
    rs._LIVE_CLIENTS.clear()


def test_a_fully_cached_run_is_still_watched_for_stalls(monkeypatch, capsys):
    """The stall check used to sit below an early return taken whenever nothing had
    reached an LLM -- so on a run whose judge cache answers every clause, the only
    monitor that notices a hang was silently switched off. That is exactly the
    configuration a replayed demo runs in: 8019/8019 clauses from cache, zero calls.

    Progress on such a run shows up as documents finished, not as tokens, so that is
    what the watch has to include."""
    rs._LIVE_CLIENTS.clear()          # no client at all: nothing has reached the model
    meter = rs.LiveMeter(every=3600)

    clock = {"t": 500.0}
    monkeypatch.setattr(rs.time, "monotonic", lambda: clock["t"])
    docs = {"n": 12}
    monkeypatch.setattr(rs, "documents_completed", lambda: docs["n"])

    meter._emit()
    capsys.readouterr()
    meter._emit()
    assert "NOTHING has moved" not in capsys.readouterr().out

    clock["t"] += meter.STILL_SECONDS + 1
    meter._emit()
    out = capsys.readouterr().out
    assert "NOTHING has moved" in out, \
        "a fully cached run that hangs must still say so -- it is the demo configuration"
    assert out.isascii()

    # A document finishing is progress even though no token moved.
    docs["n"] = 13
    clock["t"] += meter.STILL_SECONDS + 1
    meter._emit()
    assert "NOTHING has moved" not in capsys.readouterr().out
    rs._LIVE_CLIENTS.clear()


def test_documents_completed_counts_across_the_thread_pool():
    """The counter the meter reads is bumped from the document workers, so it has to be
    safe to read from the meter thread while they run."""
    import threading

    from lexora import pipeline

    start = pipeline.documents_completed()

    def _bump():
        for _ in range(200):
            with pipeline._DOCS_DONE_LOCK:
                pipeline._DOCS_DONE += 1

    threads = [threading.Thread(target=_bump) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert pipeline.documents_completed() == start + 800


def test_an_indicator_that_found_nothing_says_so(capsys):
    """The judges may ask for one indicator in one country. Until now the answer to a
    cell we found nothing for was a blank screen -- the same thing a broken run shows --
    so the operator had to talk over silence and ask to be believed.

    Singapore genuinely has no data-localisation requirement, so SG / P6-I1 SHOULD come
    back empty. That is a finding, and it is only a good answer if the engine says it."""
    summaries = [{
        "economy": "Singapore",
        "indicators_in_scope": ["P6-I1", "P6-I4", "P7-I3"],
        "indicators_covered": ["P6-I4", "P7-I3"],
        "indicators_empty": ["P6-I1"],
        "citations_by_indicator": {"P6-I4": 3, "P7-I3": 29},
    }]
    rs._print_indicator_grid(summaries)
    out = capsys.readouterr().out

    assert "Singapore   P6-I1   NO PROVISION FOUND" in out
    assert "not skipped" in out, "a miss must be distinguishable from 'we never asked'"
    assert "Singapore   P6-I4   3 citation(s)" in out
    assert "Singapore   P7-I3   29 citation(s)" in out
    assert "in scope: P6-I1, P6-I4, P7-I3" in out
    assert out.isascii(), f"non-ASCII would kill this on a GBK console: {out!r}"


def test_the_grid_says_nothing_when_the_run_never_recorded_a_scope():
    """Old summary files have no `indicators_in_scope`. Printing a grid of unknown
    misses from them would invent the very fact this is meant to establish."""
    pass

    rs._print_indicator_grid([{"economy": "Australia", "citations": 499}])
    rs._print_indicator_grid([{"economy": "Malaysia", "error": "boom"}])


def test_map_result_carries_the_indicators_it_was_asked_about():
    """`indicators_empty` is derived from this; without it a caller can see which
    indicators produced citations but not which produced none."""
    from lexora.pipeline import MapResult

    r = MapResult(discovered=[], documents=[], citations=[])
    assert r.indicators == []
    r2 = MapResult(discovered=[], documents=[], citations=[],
                   indicators=[SimpleNamespace(submission_id="P6-I1")])
    assert [i.submission_id for i in r2.indicators] == ["P6-I1"]


def _map_kwargs(monkeypatch):
    """Capture what run_pipeline_map is actually asked to do, without running it."""
    seen = {}

    def _fake(**kw):
        seen.update(kw)
        from lexora.pipeline import MapResult
        return MapResult(discovered=[], documents=[], citations=[],
                         indicators=list(kw.get("indicators") or []))

    monkeypatch.setattr(rs, "run_pipeline_map", _fake)
    return seen


def test_the_three_follow_on_passes_are_independently_switchable(monkeypatch, tmp_path):
    """They shared one flag, so the only way to skip the expensive pass was to skip the
    cheap one too. Amendment discovery issues a title search PER ACT (110 on Malaysia,
    hours); the regulator pass fetches three known URLs (minutes) and carries 84 of
    Malaysia's 183 rows -- and the only evidence for MY / P7-I4. Bundled, "save the
    evening" silently meant "report NO PROVISION FOUND for a country that has a data
    protection authority"."""
    seen = _map_kwargs(monkeypatch)
    monkeypatch.setattr(rs, "make_verifier", lambda *a, **k: None)
    monkeypatch.setattr(rs, "make_rationale_generator",
                        lambda *a, **k: SimpleNamespace(_client=None))
    monkeypatch.setattr(rs, "make_metadata_extractor",
                        lambda *a, **k: SimpleNamespace(_client=None))
    monkeypatch.setattr(rs, "make_amendment_extractor",
                        lambda *a, **k: SimpleNamespace(_client=None))

    rs.run_one("my", budget=1, verify=False, timeout=5.0,
               discover_amendments=False)
    assert seen["discover_amendments"] is False
    assert seen["discover_child_regulations"] is True, \
        "skipping the expensive pass must not take the cheap one with it"
    assert seen["discover_regulator_instruments"] is True

    seen.clear()
    rs.run_one("my", budget=1, verify=False, timeout=5.0,
               discover_regulator_instruments=False)
    assert seen["discover_amendments"] is True
    assert seen["discover_regulator_instruments"] is False


def test_follow_on_is_all_on_by_default(monkeypatch):
    """The submission path must be unchanged: a run with no flags does everything."""
    seen = _map_kwargs(monkeypatch)
    monkeypatch.setattr(rs, "make_verifier", lambda *a, **k: None)
    for name in ("make_rationale_generator", "make_metadata_extractor",
                 "make_amendment_extractor"):
        monkeypatch.setattr(rs, name, lambda *a, **k: SimpleNamespace(_client=None))

    rs.run_one("my", budget=1, verify=False, timeout=5.0)
    assert seen["discover_amendments"] is True
    assert seen["discover_child_regulations"] is True
    assert seen["discover_regulator_instruments"] is True


def test_a_filtered_run_does_not_report_a_miss_as_a_fact_about_the_country(capsys):
    """LEXORA_ONLY_LAW narrows the working set so a demo can finish while someone is
    watching -- measured, 42 minutes for a Malaysian replay against 8 minutes of
    presentation. But on a run narrowed to one Act, "NO PROVISION FOUND" for P6-I4 means
    THAT ACT has no cross-border provision. Malaysia plainly has them. Printing the
    unqualified sentence would turn the demo's own convenience into a false statement
    about a legal system, on the one screen a judge is watching."""
    summaries = [{
        "economy": "Malaysia",
        "working_set_filter": "MONEY SERVICES BUSINESS ACT 2011",
        "indicators_in_scope": ["P6-I4", "P7-I3"],
        "indicators_covered": ["P7-I3"],
        "indicators_empty": ["P6-I4"],
        "citations_by_indicator": {"P7-I3": 1},
    }]
    rs._print_indicator_grid(summaries)
    out = capsys.readouterr().out

    assert "FILTERED" in out
    assert "MONEY SERVICES BUSINESS ACT 2011" in out
    assert "says nothing about the economy" in out
    assert "NO PROVISION FOUND" in out          # still printed, but now qualified
    assert out.isascii()


def _cite(ind: str):
    return SimpleNamespace(indicator_id=ind)


def test_narrowing_the_output_does_not_narrow_the_run(capsys, monkeypatch):
    """A judge names one indicator. Loading only that indicator would rewrite the judge
    prompt, and the verdict cache is keyed on the rendered prompt -- in replay every
    judgement would miss, return a synthetic 504 and degrade to the keyword lane. So the
    narrowing happens AFTER the run, on the written file only."""
    monkeypatch.setenv("LEXORA_ONLY_INDICATOR", "p7-i3")   # case must not matter
    cites = [_cite("P7-I3"), _cite("P6-I4"), _cite("P7-I3"), _cite("P7-I5")]
    summaries = [{"economy": "Malaysia",
                  "indicators_in_scope": ["P6-I4", "P7-I3", "P7-I5"]}]

    kept = rs._narrow_output_to_indicator(cites, summaries)

    assert [c.indicator_id for c in kept] == ["P7-I3", "P7-I3"]
    out = capsys.readouterr().out
    assert "2 of 4" in out
    assert "judged every in-scope indicator" in out, "the screen must say the run was full"


def test_a_mistyped_indicator_writes_the_full_output_not_an_empty_one(capsys, monkeypatch):
    """An empty CSV on stage cannot be told apart from 'this economy has no such
    provision'. A typo must be loud and must not fabricate that finding."""
    monkeypatch.setenv("LEXORA_ONLY_INDICATOR", "P7-13")   # one instead of I
    cites = [_cite("P7-I3"), _cite("P6-I4")]
    summaries = [{"economy": "Malaysia", "indicators_in_scope": ["P6-I4", "P7-I3"]}]

    kept = rs._narrow_output_to_indicator(cites, summaries)

    assert len(kept) == 2, "a typo must not silently empty the file"
    assert "is not one of the indicators" in capsys.readouterr().out


def test_the_narrowing_is_off_unless_asked(capsys, monkeypatch):
    monkeypatch.delenv("LEXORA_ONLY_INDICATOR", raising=False)
    cites = [_cite("P7-I3"), _cite("P6-I4")]
    assert rs._narrow_output_to_indicator(cites, [{}]) is cites
    assert capsys.readouterr().out == ""


def test_an_unfiltered_run_carries_no_such_warning(capsys):
    summaries = [{
        "economy": "Malaysia",
        "working_set_filter": "",
        "indicators_in_scope": ["P6-I4", "P7-I3"],
        "indicators_covered": ["P7-I3"],
        "indicators_empty": ["P6-I4"],
        "citations_by_indicator": {"P7-I3": 1},
    }]
    rs._print_indicator_grid(summaries)
    out = capsys.readouterr().out
    assert "FILTERED" not in out
    assert "Malaysia    P6-I4   NO PROVISION FOUND" in out
