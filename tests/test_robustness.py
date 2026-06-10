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
            SimpleNamespace(document=SimpleNamespace(http_status=200)),
            SimpleNamespace(document=SimpleNamespace(http_status=403)),
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
    assert s["fetched_ok"] == 1          # only the 200 counts
    assert s["citations"] == 3
    assert s["indicators_covered"] == ["P6-I4", "P7-I1"]  # distinct + sorted
    assert s["n_indicators_covered"] == 2
    assert s["review_rows"] == 1         # the CONFLICT_REVIEW row
