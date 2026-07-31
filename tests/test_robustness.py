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
