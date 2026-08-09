"""Unit tests for the HTTP record/replay layer (pure logic, no network).

What these guard is not the speed-up. It is the two promises the layer makes to a judge
watching an offline run: a replayed response is byte-identical to what the portal really
sent, and a request that was never recorded is reported as unreachable rather than
answered with something plausible.
"""
from __future__ import annotations

import httpx
import pytest

from lexora.collect import http_cache
from lexora.collect.http_cache import RecordReplayTransport, Store


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LEXORA_HTTP_CACHE_DIR", str(tmp_path / "http_cache"))
    s = Store(tmp_path / "http_cache")
    monkeypatch.setattr(http_cache, "_STORE", s)
    yield s
    s.close()


def _upstream(body: bytes = b"<html>statute</html>", status: int = 200, calls=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        return httpx.Response(status, content=body,
                              headers={"content-type": "text/html"})

    return httpx.MockTransport(handler)


def test_record_then_replay_returns_the_same_bytes(store):
    url = "https://sso.agc.gov.sg/Act/PDPA2012"
    calls = []
    with httpx.Client(transport=RecordReplayTransport(
            _upstream(calls=calls), http_cache.RECORD)) as c:
        live = c.get(url)
    assert live.status_code == 200
    assert calls == [url]

    with httpx.Client(transport=RecordReplayTransport(
            _upstream(b"DIFFERENT", calls=calls), http_cache.REPLAY)) as c:
        replayed = c.get(url)

    # The upstream was never consulted, and the body is the recorded one.
    assert calls == [url]
    assert replayed.content == live.content
    assert replayed.headers["x-lexora-cache"] == "hit"


def test_replay_reports_when_the_bytes_were_really_retrieved(store):
    """A snapshot replayed next week is still a snapshot from the day it was taken."""
    from datetime import datetime, timezone

    url = "https://legislation.gov.au/C2004A03712/latest/text"
    with httpx.Client(transport=RecordReplayTransport(
            _upstream(), http_cache.RECORD)) as c:
        c.get(url)

    with httpx.Client(transport=RecordReplayTransport(
            _upstream(), http_cache.REPLAY)) as c:
        stamp = c.get(url).headers["x-lexora-recorded-at"]

    recorded = datetime.fromisoformat(stamp)
    assert recorded.tzinfo is not None
    # Recorded seconds ago, not at some default epoch and not in the future.
    assert 0 <= (datetime.now(timezone.utc) - recorded).total_seconds() < 120


def test_a_live_fetch_carries_no_recorded_at(store):
    """Only a replay may override the retrieval timestamp."""
    with httpx.Client(transport=RecordReplayTransport(
            _upstream(), http_cache.RECORD)) as c:
        assert "x-lexora-recorded-at" not in c.get("https://example.gov/x").headers


def test_replay_miss_is_reported_not_invented(store):
    with httpx.Client(transport=RecordReplayTransport(
            _upstream(), http_cache.REPLAY)) as c:
        response = c.get("https://lom.agc.gov.my/never-recorded")

    assert response.status_code == 504
    assert response.headers["x-lexora-cache"] == "miss"
    assert b"no recording" in response.content


@pytest.mark.parametrize(
    "value,expected",
    [("replay", True), ("offline", True), ("record", False), ("", False)],
)
def test_politeness_is_only_owed_to_a_real_server(monkeypatch, value, expected):
    monkeypatch.setenv("LEXORA_HTTP_CACHE", value)
    assert http_cache.serving_from_recording() is expected


def test_a_post_body_is_part_of_the_key(store):
    """Malaysia's Fess back-end is a search POST: two queries share a URL."""
    url = "https://lom.agc.gov.my/search"
    with httpx.Client(transport=RecordReplayTransport(
            _upstream(b"hits for data protection"), http_cache.RECORD)) as c:
        c.post(url, content=b"q=data+protection")

    with httpx.Client(transport=RecordReplayTransport(
            _upstream(), http_cache.REPLAY)) as c:
        same = c.post(url, content=b"q=data+protection")
        other = c.post(url, content=b"q=cybersecurity")

    assert same.content == b"hits for data protection"
    assert other.status_code == 504  # a different question was never asked live


def test_a_non_200_replays_as_itself(store):
    """SG SSO answers a plain client with 403; the browser leg keys off that status."""
    url = "https://sso.agc.gov.sg/blocked"
    with httpx.Client(transport=RecordReplayTransport(
            _upstream(b"denied", status=403), http_cache.RECORD)) as c:
        c.get(url)

    with httpx.Client(transport=RecordReplayTransport(
            _upstream(), http_cache.REPLAY)) as c:
        assert c.get(url).status_code == 403


def test_rendered_pages_round_trip(store):
    store.put_render("https://sso.agc.gov.sg/search?q=data",
                     200, "https://sso.agc.gov.sg/search?q=data",
                     "text/html", "<html>rendered hits</html>")
    got = store.get_render("https://sso.agc.gov.sg/search?q=data")
    assert got is not None
    status, final_url, content_type, html = got
    assert (status, content_type) == (200, "text/html")
    assert html == "<html>rendered hits</html>"
    assert store.get_render("https://sso.agc.gov.sg/search?q=other") is None


def test_bodies_are_deduplicated_by_content(store):
    """The same statute reached by two URLs is one blob on disk."""
    body = b"%PDF-1.4 identical bytes"
    store.put("GET", "https://a.example/act.pdf", b"", 200, {}, body)
    store.put("GET", "https://b.example/act.pdf", b"", 200, {}, body)
    assert len(list(store.blobs.glob("*.bin"))) == 1
    assert store.get("GET", "https://b.example/act.pdf", b"")[2] == body


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", http_cache.OFF), ("record", http_cache.RECORD), ("REPLAY", http_cache.REPLAY),
     ("offline", http_cache.REPLAY), ("nonsense", http_cache.OFF)],
)
def test_mode_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("LEXORA_HTTP_CACHE", value)
    assert http_cache.mode() == expected


def test_off_mode_installs_nothing(monkeypatch):
    """The default live run must be the same code path it has always been."""
    monkeypatch.delenv("LEXORA_HTTP_CACHE", raising=False)
    monkeypatch.setattr(http_cache, "_INSTALLED", False)
    before = httpx.Client.__init__
    assert http_cache.install() == http_cache.OFF
    assert httpx.Client.__init__ is before


class _CompressedUpstream(httpx.BaseTransport):
    """A portal that gzips, like Malaysia's. Streams the body, as a real one does."""

    def __init__(self, encoded: bytes):
        self.encoded = encoded

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        class _Stream(httpx.SyncByteStream):
            def __init__(self, data: bytes) -> None:
                self._data = data

            def __iter__(self):
                yield self._data

        return httpx.Response(
            200,
            headers={"content-type": "text/html", "content-encoding": "gzip"},
            stream=_Stream(self.encoded),
        )


def test_a_gzipped_portal_is_stored_and_served_as_the_document_it_sent(store):
    """The layer's promise is "byte-identical to what the portal really sent".

    At the transport layer httpx has NOT yet run the content-encoding decoder -- that
    happens in Response.read(). Draining `response.stream` therefore captured raw gzip,
    and stripping the `content-encoding` header on the way out published those bytes as
    if they were the document. Malaysia's portal gzips: a recorded live run fed
    `\x1f\x8b...` to the HTML parser and returned 0 hits on all 40 discovery queries,
    with no error anywhere -- while Singapore, which reaches its portal through the
    headless browser and never through this transport, looked perfect.

    Note what could NOT have caught this: comparing a replay against its own recording.
    Both sides were corrupt in exactly the same way, so that assertion stayed green.
    The baseline has to be the plaintext the portal sent.
    """
    import gzip

    url = "https://lom.agc.gov.my/search"
    plain = b"<html><body>Laws of Malaysia: Act 709</body></html>"
    upstream = _CompressedUpstream(gzip.compress(plain))

    with httpx.Client(transport=RecordReplayTransport(
            upstream, http_cache.RECORD)) as c:
        live = c.get(url)
    assert live.content == plain, "a recorded run got gzip bytes, not the document"

    with httpx.Client(transport=RecordReplayTransport(
            upstream, http_cache.REPLAY)) as c:
        replayed = c.get(url)
    assert replayed.headers["x-lexora-cache"] == "hit"
    assert replayed.content == plain, "the store kept gzip bytes, so replay serves them"


def test_a_dribbling_body_is_cut_off_by_the_wall_clock():
    """httpx timeouts are per socket read, not per request.

    A peer that sends a few bytes every couple of seconds resets the read timeout
    forever: the request never completes and never fails, and nothing in the pipeline
    capped how long a single document may take.

    The cap is a BACKSTOP and the default is deliberately hours, because the first
    version -- 300s -- cut two real Malaysian Acts at 5.0 MB and 6.8 MB while they
    were still arriving at 16-22 kB/s. What the pipeline was missing was never the
    cap; it was the measured rate, which is why the message carries it.
    """
    import time as _time

    class _Dribble(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(10_000):
                _time.sleep(0.005)
                yield b"x"

    response = httpx.Response(200, headers={"content-type": "text/html"},
                              stream=_Dribble())
    with pytest.raises(http_cache.BodyDeadlineExceeded) as caught:
        http_cache.read_body_within(response, deadline=0.05)
    assert "B/s" in str(caught.value)  # says how slow, not just that it was slow

    # Must NOT be a TransportError: the crawler RETRIES those, and retrying a portal
    # that dribbles just buys three more deadlines. It has to propagate to the
    # per-document isolation and cost exactly one document.
    assert not isinstance(caught.value, httpx.TransportError)


def test_a_body_that_arrives_in_time_is_returned_whole():
    """The cap must not truncate a slow-but-finishing download."""
    class _Chunked(httpx.SyncByteStream):
        def __iter__(self):
            yield b"<html>"
            yield b"statute"
            yield b"</html>"

    response = httpx.Response(200, headers={"content-type": "text/html"},
                              stream=_Chunked())
    assert http_cache.read_body_within(response, deadline=30.0) == b"<html>statute</html>"


def test_the_deadline_can_be_switched_off():
    """`LEXORA_BODY_DEADLINE=0` restores the old unbounded read, for anyone who
    genuinely wants to sit through a 500 MB consolidated statute."""
    class _Small(httpx.SyncByteStream):
        def __iter__(self):
            yield b"body"

    response = httpx.Response(200, stream=_Small())
    assert http_cache.read_body_within(response, deadline=0) == b"body"


def test_a_slow_but_moving_body_is_never_cut():
    """The cap that fires in practice is IDLE time, not total time, because the two
    failures look nothing alike and only one is a failure.

    Measured 2026-08-02: a 600s TOTAL budget cut the Communications and Multimedia Act
    1998 -- a gold instrument -- at 32.8 MB arriving steadily at 54 kB/s. Nothing was
    wrong with that download except its size. Meanwhile a socket that had been accepted
    and abandoned held three workers for 27 minutes and the total budget, set to two
    hours precisely so real Acts survive, never noticed.
    """
    import time as _time

    class _SlowButSteady(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(40):
                _time.sleep(0.005)
                yield b"statute-bytes"

    response = httpx.Response(200, stream=_SlowButSteady())
    body = http_cache.read_body_within(response, deadline=0)   # total cap off, idle on
    assert body == b"statute-bytes" * 40


def test_a_body_that_stops_producing_bytes_is_cut(monkeypatch):
    monkeypatch.setenv("LEXORA_BODY_IDLE", "0.05")

    import time as _time

    class _StallsMidway(httpx.SyncByteStream):
        def __iter__(self):
            yield b"header"
            _time.sleep(0.2)          # the gap between bytes is the signal
            yield b"one late byte"

    response = httpx.Response(200, stream=_StallsMidway())
    with pytest.raises(http_cache.BodyDeadlineExceeded) as caught:
        http_cache.read_body_within(response, deadline=0)
    message = str(caught.value)
    assert "no body byte for" in message
    assert "LEXORA_BODY_IDLE" in message
    # Same contract as the total cap: NOT a TransportError, so the crawler does not
    # retry it into three more deadlines; it costs exactly one document.
    assert not isinstance(caught.value, httpx.TransportError)


def test_both_caps_off_is_the_old_unbounded_read(monkeypatch):
    monkeypatch.setenv("LEXORA_BODY_IDLE", "0")

    class _Small(httpx.SyncByteStream):
        def __iter__(self):
            yield b"body"

    response = httpx.Response(200, stream=_Small())
    assert http_cache.read_body_within(response, deadline=0) == b"body"


def test_replay_can_be_told_to_let_the_model_out(store, monkeypatch, capsys):
    """Replay pins the CORPUS. Pinning the model too makes the layer unusable for the one
    job it is most needed for: iterating on the LLM over a fixed corpus. Every model call
    became a synthetic 504, so a run measuring how often the model declines would read
    100% backend error -- which is why the rationale layer could never be exercised offline
    at all. Opt-in, and it announces itself, because a replay that quietly reaches the
    network is worth less than one that refuses to."""
    monkeypatch.setenv("LEXORA_REPLAY_LIVE_LLM", "1")
    monkeypatch.setenv("LEXORA_LLM_BASE_URL", "https://llmapi.example.com/v1")
    calls = []
    tr = RecordReplayTransport(_upstream(b'{"ok":1}', calls=calls), http_cache.REPLAY)

    with httpx.Client(transport=tr) as c:
        # The model host goes out live, twice, even though nothing was ever recorded...
        assert c.post("https://llmapi.example.com/v1/chat").status_code == 200
        assert c.post("https://llmapi.example.com/v1/chat").status_code == 200
        # ...while an unrecorded PORTAL is still the honest 504.
        assert c.get("https://sso.agc.gov.sg/Act/Nope").status_code == 504

    assert len(calls) == 2
    assert "letting LLM traffic out" in capsys.readouterr().out


def test_the_model_is_pinned_by_default(store, monkeypatch):
    """--offline promises no socket is opened. The escape hatch must stay shut unless
    someone asks for it by name."""
    monkeypatch.delenv("LEXORA_REPLAY_LIVE_LLM", raising=False)
    monkeypatch.setenv("LEXORA_LLM_BASE_URL", "https://llmapi.example.com/v1")
    calls = []
    tr = RecordReplayTransport(_upstream(calls=calls), http_cache.REPLAY)
    with httpx.Client(transport=tr) as c:
        assert c.post("https://llmapi.example.com/v1/chat").status_code == 504
    assert calls == []
