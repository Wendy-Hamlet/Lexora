"""Record / replay of the network layer, so a whole run can be reproduced offline.

Task 1 is a *live* crawl, and that is how the engine runs by default: nothing here is on
unless asked for. But a live crawl has one property that is fine in a build loop and fatal
on a stage — it needs the internet, and it needs the portals to behave the same way twice.
The 3 August pitch is a live demo in a room whose network we do not control, in front of
judges who may ask to see the engine run again.

So this adds a third position between "crawl live" and "hand the judge a pre-baked CSV":

    LEXORA_HTTP_CACHE=record   crawl live, and keep every response
    LEXORA_HTTP_CACHE=replay   crawl the recording — no socket is opened at all

Replay is not a mock. Every byte the pipeline sees is the byte the government portal
actually sent, on a date we can name; discovery, OCR, parsing, retrieval and the judge all
run exactly as they do live. What changes is only where the bytes come from. That also
makes it the honest way to demo: the run really is the pipeline, not a replayed log of it.

WHAT IT IS NOT: it is not a fallback that silently kicks in when the network is bad. In
``replay`` a URL that was never recorded returns a synthetic 504 with a reason phrase
saying so — the pipeline treats it as an unreachable portal and degrades, and the run
summary shows the gap. A recording is a snapshot with a date, not a claim about today.

Interception is at the httpx *transport*, one layer below every client in the codebase, so
the portal adapters, the search back-ends (including Malaysia's Fess POSTs), the liveness
prober and the secondary-source readers are all covered without any of them knowing. The
key is method + URL + a hash of the request body; headers are deliberately excluded, so a
rotated User-Agent still hits.

The headless-browser leg (Singapore SSO answers a plain client with 403) does not go
through httpx at all, so ``render_cache`` below records the rendered DOM separately, keyed
by URL. Both stores live under ``data/http_cache/``.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key         TEXT PRIMARY KEY,
    method      TEXT NOT NULL,
    url         TEXT NOT NULL,
    status      INTEGER NOT NULL,
    headers     TEXT NOT NULL,
    body_sha    TEXT NOT NULL,
    recorded_at REAL NOT NULL DEFAULT (julianday('now'))
);
CREATE TABLE IF NOT EXISTS renders (
    url          TEXT PRIMARY KEY,
    status       INTEGER,
    final_url    TEXT NOT NULL,
    content_type TEXT NOT NULL,
    html_sha     TEXT NOT NULL,
    recorded_at  REAL NOT NULL DEFAULT (julianday('now'))
);
"""

OFF, RECORD, REPLAY = "off", "record", "replay"


def mode() -> str:
    """Current mode from ``LEXORA_HTTP_CACHE``. Anything unrecognised means off."""
    raw = os.environ.get("LEXORA_HTTP_CACHE", "").strip().lower()
    if raw in ("record", "rec", "1", "on", "true"):
        return RECORD
    if raw in ("replay", "offline", "play"):
        return REPLAY
    return OFF


def serving_from_recording() -> bool:
    """True when every HTTP byte comes from the store on this disk.

    Several layers below sleep on behalf of a server: the crawler spaces same-host
    requests, the discovery sweep spaces SSO navigations under its cumulative per-IP
    limit, the AU enumerator and EPUB assembler space their sequential fetches. Under
    replay there is no server on the other end, so those delays buy nothing and cost
    the one budget a live demo is short of — measured on Singapore, 84% of an offline
    replay was spent sleeping. Stated here once so every throttle site agrees on what
    "no server" means.
    """
    return mode() == REPLAY


def clamp_read_timeout(request: httpx.Request, seconds: float) -> httpx.Request:
    """Make ``seconds`` the ceiling on how long one socket read may block.

    This is the only mechanism that actually interrupts a stalled download, and finding
    that out cost two dead runs. The watchdog below closes the response when a budget is
    blown, which works over plain HTTP and does NOT work over TLS: py-spy on the second
    stalled run showed three workers parked in ``ssl.read`` inside ``read_body_within``,
    twenty-five minutes after ``response.close()`` had been called on them. Closing an
    httpx response returns a connection to the pool; it does not tear a socket out from
    under a thread already blocked in ``recv``.

    httpcore ends every read with ``self._sock.settimeout(timeout); self._sock.recv(...)``
    where ``timeout`` comes from ``request.extensions["timeout"]["read"]``. When that is
    ``None`` the socket blocks forever, which is what a portal that accepts a connection
    and then says nothing relies on. Setting it here puts the deadline in the socket,
    where it runs without a thread and cannot be ignored.

    httpx's read timeout is an IDLE timeout -- time between reads, not total -- so the
    idle budget is exactly the right number for it, and a slow-but-moving 33 MB download
    is untouched. Only lowered, never raised: a caller that asked for something stricter
    keeps it.
    """
    if seconds <= 0:
        return request
    extensions = dict(request.extensions or {})
    timeout = dict(extensions.get("timeout") or {})
    current = timeout.get("read")
    timeout["read"] = seconds if current is None else min(float(current), seconds)
    extensions["timeout"] = timeout
    request.extensions = extensions
    return request


def disk_cache_still_valid(path: str | os.PathLike, ttl_hours: float) -> bool:
    """Whether a TTL'd side-cache file on disk may still be used.

    Several layers keep their own JSON cache next to the recording: the AU Act catalogue
    (~48 requests) and six secondary-source adapters. Each expires after a few hours,
    which is right for a live run — the point of a TTL is to notice that the world moved.

    Under replay the world cannot move. Every byte comes from a store with a date on it,
    so an expiry does not fetch a fresher answer, it fetches a synthetic 504: the refetch
    finds no recording, the paging loop stops on page 0, and the caller gets an EMPTY
    catalogue. On 2026-08-01 that is exactly what happened — ``au_act_catalogue.json`` was
    ten hours past its 24-hour TTL, the semantic crosswalk (the only source of Australia's
    NEW instruments) received zero entries, and the run finished with 463 rows instead of
    499 and no error anywhere. The demo would have lost a third of Australia on stage,
    silently, and the only visible symptom is a row count nobody memorises.

    So under replay a side-cache never expires: what is on disk is what that run recorded.
    """
    if not os.path.exists(path):
        return False
    if serving_from_recording():
        return True
    return (time.time() - os.path.getmtime(path)) < ttl_hours * 3600


# A BACKSTOP, not a routine cutter. httpx's per-read timeout already fails a peer that
# sends nothing at all, so the only case left for a total budget is one that dribbles
# without ever stopping -- and against that, waiting an hour costs nothing that matters.
# Set to 300s first, which promptly cut two REAL Malaysian Acts at 5.0 MB and 6.8 MB
# while they were arriving steadily at 16-22 kB/s. This corpus contains a 98 MB scan;
# at that portal's speed it needs about 1.7 hours, so any "reasonable" budget is a
# document shredder. Errors here must cost a run time, never its content.
DEFAULT_BODY_DEADLINE = 7200.0
DEFAULT_BODY_IDLE = 180.0


class BodyDeadlineExceeded(httpx.HTTPError):
    """One response body took longer than the whole-request budget allows.

    Deliberately NOT an ``httpx.TransportError``: the crawler retries those, and a
    portal that dribbles is not having a bad second -- retrying buys three more
    deadlines. This propagates instead, and the per-document isolation in
    ``run_pipeline_map`` turns it into one SKIPPED line.
    """


def body_deadline() -> float:
    """Seconds a single response body may take. ``<= 0`` disables the cap."""
    raw = os.environ.get("LEXORA_BODY_DEADLINE", "").strip()
    if not raw:
        return DEFAULT_BODY_DEADLINE
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_BODY_DEADLINE


def body_idle_deadline() -> float:
    """Seconds the body may produce NO new byte before we give up. ``<= 0`` disables it.

    This is the cap that should fire in practice; ``body_deadline`` is the far backstop.
    180s is generous against a portal that pauses mid-transfer and still an order of
    magnitude below the 27-minute silence measured on 2026-08-02.
    """
    raw = os.environ.get("LEXORA_BODY_IDLE", "").strip()
    if not raw:
        return DEFAULT_BODY_IDLE
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_BODY_IDLE


def read_body_within(response: httpx.Response, deadline: float | None = None) -> bytes:
    """Drain a response body under a WALL-CLOCK budget, decoding as httpx would.

    httpx timeouts are per socket operation, not per request, so a peer that sends a
    few bytes every couple of seconds resets the read timeout forever and the request
    never completes and never fails. This is the backstop for that one case.

    It is NOT the answer to a slow portal, and the first version of it got that wrong.
    On 2026-07-31 a Malaysian fetch appeared to hang for 25 minutes; the throughput
    figure behind that call -- ~107 B/s -- came from a Windows process I/O counter that
    does not reliably reflect socket traffic, and was simply wrong. Counting the bytes
    HERE, where they actually arrive, the same documents download at 16-22 kB/s: slow,
    but progressing, and they would have finished. Hence a two-hour backstop rather
    than a minutes-long deadline, and hence the measured rate in the error message --
    the missing ingredient was never a cap, it was being able to see the rate at all.
    """
    budget = body_deadline() if deadline is None else deadline
    idle_budget = body_idle_deadline()
    if budget <= 0 and idle_budget <= 0:
        response.read()
        return response.content

    started = time.monotonic()
    chunks: list[bytes] = []
    got = 0
    state = _BodyClock(started)

    # The budgets are enforced by a WATCHDOG THREAD, not by a check inside the read loop.
    #
    # The check used to live in the loop, which cannot work and shipped anyway: the loop
    # body only runs when a chunk ARRIVES, so a stream that stops dead never evaluates it.
    # It could catch a trickle that resumed after too long a gap; it could never catch the
    # thing it was written for. Measured 2026-08-02, four hours after that guard shipped:
    # three sockets to the Malaysian portal sat ESTABLISHED for 27 minutes, zero bytes in
    # any 60-second window, process CPU flat at +0.016 s/30 s -- and the idle budget, set
    # to 180 s, was never once evaluated. httpx's own read timeout (60 s here) did not
    # fire either; rather than reason about why, the deadline is now kept by a clock that
    # runs whether or not bytes arrive.
    #
    # Shutting the socket down from the watchdog is what unblocks the read: the reader gets
    # EOF and either raises or returns short. Closing it is NOT enough -- that works on
    # Windows and does nothing on POSIX, which is why this asserted itself green here for a
    # day while hanging forever on Linux. See `_interrupt`.
    watchdog = threading.Thread(
        target=_watch_body, args=(response, state, budget, idle_budget),
        name="lexora-body-watchdog", daemon=True,
    )
    watchdog.start()
    try:
        for chunk in response.iter_bytes():
            chunks.append(chunk)
            got += len(chunk)
            state.saw_bytes(got)
    except Exception:
        # A read that fails because the watchdog pulled the socket is a deadline, not a
        # network fault: callers treat the two very differently (one is a refusing portal
        # to retry, the other is a document to give up on and say so).
        if state.expired is not None:
            raise BodyDeadlineExceeded(state.expired) from None
        raise
    finally:
        state.stop.set()
        watchdog.join(timeout=1.0)
    if state.expired is not None:
        raise BodyDeadlineExceeded(state.expired)
    return b"".join(chunks)


class _BodyClock:
    """When the last byte landed, readable from the watchdog thread."""

    def __init__(self, started: float) -> None:
        self.started = started
        self._last = started
        self._got = 0
        self._lock = threading.Lock()
        self.stop = threading.Event()
        self.expired: str | None = None

    def saw_bytes(self, total: int) -> None:
        with self._lock:
            self._last = time.monotonic()
            self._got = total

    def snapshot(self) -> tuple[float, int]:
        with self._lock:
            return self._last, self._got


def _watch_body(
    response: httpx.Response, state: _BodyClock, budget: float, idle_budget: float
) -> None:
    """Close ``response`` once a budget is blown, so the blocked read gives up.

    An IDLE budget as well as a total one, because the two failures look nothing alike and
    only one of them is a failure. A body arriving steadily at 54 kB/s is a 33 MB Act
    downloading; a body that has produced no new byte in minutes is a socket that was
    accepted and abandoned. Capping TOTAL time cannot tell them apart, and cost us both
    ways: a 300 s cap once cut two real Malaysian Acts, and a 600 s cap cut the
    Communications and Multimedia Act 1998 -- a GOLD instrument -- at 32.8 MB and 54 kB/s.
    """
    tick = min(1.0, idle_budget / 4 if idle_budget > 0 else 1.0)
    while not state.stop.wait(tick):
        now = time.monotonic()
        last, got = state.snapshot()
        elapsed = now - state.started
        if idle_budget > 0 and (now - last) > idle_budget:
            state.expired = (
                f"no body byte for {now - last:.0f}s "
                f"({got} bytes in {elapsed:.0f}s so far); "
                f"idle budget is {idle_budget:.0f}s (LEXORA_BODY_IDLE)"
            )
        elif 0 < budget < elapsed:
            state.expired = (
                f"body still arriving after {elapsed:.0f}s "
                f"({got} bytes at {got / max(elapsed, 1e-9):.0f} B/s); "
                f"budget is {budget:.0f}s (LEXORA_BODY_DEADLINE)"
            )
        else:
            continue
        _interrupt(response)


def _interrupt(response: httpx.Response) -> None:
    """End a read that is already blocked in ``recv``, on this platform and the other one.

    ``response.close()`` alone is enough on Windows, where closing a socket wakes a recv
    that is already waiting on it. On POSIX it is not: the kernel keeps the underlying file
    description alive for the blocked call, so closing the descriptor changes nothing and
    the reader waits forever. Measured 2026-08-03 -- the same probe returns in 1.0s on
    win32 and is still blocked after 20s on Linux.

    That difference cost a CI job six hours on all three Python versions, with no output at
    all: the suite hung in the very test that asserts this deadline works, and because
    stdout is block-buffered when it is not a terminal, a process that never exits never
    flushes a single character. The watchdog was decorative on every machine but ours.

    ``shutdown`` is the portable interrupt -- it delivers EOF to the waiting reader rather
    than trying to take the descriptor away from it. Closing afterwards still matters: it
    releases the connection.
    """
    stream = response.extensions.get("network_stream")
    sock = stream.get_extra_info("socket") if stream is not None else None
    if sock is not None:
        with contextlib.suppress(OSError, AttributeError):
            sock.shutdown(socket.SHUT_RDWR)
    with contextlib.suppress(Exception):
        response.close()
        return


def default_dir() -> Path:
    p = os.environ.get("LEXORA_HTTP_CACHE_DIR")
    if p:
        return Path(p)
    return Path(__file__).resolve().parents[3] / "data" / "http_cache"


def _julian_to_iso(julian_day: float) -> str:
    """SQLite's ``julianday('now')`` (UTC) as an ISO-8601 instant."""
    seconds = (float(julian_day) - 2440587.5) * 86400.0
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


def _key(method: str, url: str, body: bytes) -> str:
    h = hashlib.sha256()
    h.update(method.upper().encode())
    h.update(b"\0")
    h.update(str(url).encode())
    h.update(b"\0")
    h.update(hashlib.sha256(body or b"").hexdigest().encode())
    return h.hexdigest()


class Store:
    """SQLite index plus content-addressed blobs, safe under the crawler's thread pool.

    Bodies go to files rather than into the database: one Malaysian scan is 98 MB, and a
    BLOB that size turns every index read into a page-cache eviction. Addressing them by
    content hash also means the same statute fetched from two URLs is stored once.
    """

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = Path(directory) if directory else default_dir()
        self.blobs = self.dir / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.dir / "index.sqlite"), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # -- blobs ---------------------------------------------------------------
    def _write_blob(self, body: bytes) -> str:
        sha = hashlib.sha256(body).hexdigest()
        p = self.blobs / f"{sha}.bin"
        if not p.exists():
            p.write_bytes(body)
        return sha

    def _read_blob(self, sha: str) -> bytes | None:
        p = self.blobs / f"{sha}.bin"
        return p.read_bytes() if p.exists() else None

    # -- httpx responses -----------------------------------------------------
    def get(self, method: str, url: str, body: bytes) -> tuple[int, dict, bytes] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT status, headers, body_sha, recorded_at FROM responses WHERE key = ?",
                (_key(method, url, body),),
            ).fetchone()
        payload = self._read_blob(row[2]) if row is not None else None
        # Counted under the lock: the crawler reads this from its thread pool, and a
        # read-modify-write outside it loses updates. Replay coverage is reported from
        # these, so an undercount reads as a recording with holes it does not have.
        with self._lock:
            if row is None or payload is None:  # no row, or the blob went missing
                self.misses += 1
                return None
            self.hits += 1
        headers = json.loads(row[1])
        # When these bytes were really retrieved. The citation's retrieval_timestamp is
        # part of the audit trail, so a replay must not backdate a week-old snapshot to
        # "now" — the honest answer is the date the portal actually served it.
        headers["x-lexora-recorded-at"] = _julian_to_iso(row[3])
        return int(row[0]), headers, payload

    def put(self, method: str, url: str, req_body: bytes, status: int,
            headers: dict, body: bytes) -> None:
        sha = self._write_blob(body)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses "
                "(key, method, url, status, headers, body_sha) VALUES (?, ?, ?, ?, ?, ?)",
                (_key(method, url, req_body), method.upper(), str(url), int(status),
                 json.dumps(headers), sha),
            )
            self._db.commit()
            self.writes += 1

    # -- browser renders -----------------------------------------------------
    def get_render(self, url: str) -> tuple[int | None, str, str, str] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT status, final_url, content_type, html_sha FROM renders WHERE url = ?",
                (str(url),),
            ).fetchone()
        if row is None:
            return None
        html = self._read_blob(row[3])
        if html is None:
            return None
        return (row[0], row[1], row[2], html.decode("utf-8", "replace"))

    def put_render(self, url: str, status: int | None, final_url: str,
                   content_type: str, html: str) -> None:
        sha = self._write_blob(html.encode("utf-8"))
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO renders "
                "(url, status, final_url, content_type, html_sha) VALUES (?, ?, ?, ?, ?)",
                (str(url), status, str(final_url), content_type, sha),
            )
            self._db.commit()

    def stats(self) -> dict:
        with self._lock:
            n_resp = self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
            n_rend = self._db.execute("SELECT COUNT(*) FROM renders").fetchone()[0]
        return {"responses": n_resp, "renders": n_rend,
                "hits": self.hits, "misses": self.misses, "writes": self.writes}

    def close(self) -> None:
        with self._lock:
            self._db.close()


_STORE: Store | None = None
_STORE_LOCK = threading.Lock()


def store() -> Store:
    """The process-wide store. Created on first use so ``off`` runs touch no disk."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = Store()
        return _STORE


def llm_passthrough_host() -> str | None:
    """Host that REPLAY should let out to the network, or ``None`` (the default).

    Replay exists to pin the CORPUS. It should not have to pin the model as well. When the
    point of a run is to iterate on the LLM layer over a fixed corpus -- retune a guard,
    fill the rationale cache, measure how often the model declines -- intercepting the model
    turns every call into a synthetic 504, and the measurement reads 100% backend error.
    That is not a reproducible run, it is an unusable one, and it is why the rationale layer
    could never be exercised offline at all.

    Off by default, because ``--offline`` promises that no socket is opened and this breaks
    that promise for exactly one host. Opt in with ``LEXORA_REPLAY_LIVE_LLM=1``; every run
    that uses it says so, loudly, once -- a replay that quietly reached the network would be
    worth less than no replay at all.
    """
    if os.environ.get("LEXORA_REPLAY_LIVE_LLM", "").lower() not in (
        "1", "true", "yes", "on",
    ):
        return None
    from lexora.config import env_value

    base = env_value("LEXORA_LLM_BASE_URL", "", "OPENAI_BASE_URL")
    host = httpx.URL(base).host if base else ""
    return host or None


class RecordReplayTransport(httpx.BaseTransport):
    """Wraps a real transport: records what it returns, or answers from the store."""

    def __init__(self, inner: httpx.BaseTransport, cache_mode: str) -> None:
        self._inner = inner
        self._mode = cache_mode
        self._llm_host = llm_passthrough_host() if cache_mode == REPLAY else None
        self._passed_through = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = request.content or b""
        method, url = request.method, str(request.url)
        if self._llm_host and request.url.host == self._llm_host:
            # Announce it once per transport. A replay that reaches the network without
            # saying so is a worse artifact than one that refuses to.
            if not self._passed_through:
                print(f"  !! replay: letting LLM traffic out to {self._llm_host} live "
                      "(LEXORA_REPLAY_LIVE_LLM=1). The corpus is replayed; the model is not.")
            self._passed_through += 1
            clamp_read_timeout(request, body_idle_deadline())
            return self._inner.handle_request(request)
        # Every client in the process comes through here while recording, which makes it
        # the one place that can put a socket-level deadline on all of them -- the same
        # reason `install()` patches the constructor instead of the call sites.
        clamp_read_timeout(request, body_idle_deadline())

        if self._mode == REPLAY:
            found = store().get(method, url, body)
            if found is None:
                # Not a crash and not a lie: the pipeline already knows how to carry on
                # past a portal that will not answer, and the row count shows the gap.
                return httpx.Response(
                    504, request=request,
                    headers={"content-type": "text/plain", "x-lexora-cache": "miss"},
                    content=b"lexora: no recording for this request (LEXORA_HTTP_CACHE=replay)",
                )
            status, headers, payload = found
            headers = {k: v for k, v in headers.items()
                       if k.lower() not in ("content-encoding", "content-length",
                                            "transfer-encoding")}
            headers["x-lexora-cache"] = "hit"
            return httpx.Response(status, request=request, headers=headers, content=payload)

        response = self._inner.handle_request(request)
        # `response.stream` is the RAW body: at the transport layer httpx has not applied
        # the content-encoding decoder yet, it does that in Response.read(). Joining the
        # stream and then stripping `content-encoding` below therefore published gzip
        # bytes as if they were the decoded document -- and every consumer believed it.
        # Malaysia's portal gzips, so a recorded run fed `\x1f\x8b...` straight to the
        # HTML parser and scored 0 hits on all 40 discovery queries, silently, while
        # Singapore (headless browser, never through this transport) looked perfect.
        # read() runs the decoder, so what is stored and served is the real document --
        # here under a wall-clock budget, because this drain is what a dribbling portal
        # hangs on and recording moves the drain INSIDE the transport, out of reach of
        # any deadline the caller might apply to its own streaming.
        payload = read_body_within(response)
        response.close()
        headers = dict(response.headers)
        if self._mode == RECORD:
            store().put(method, url, body, response.status_code, headers, payload)
        headers = {k: v for k, v in headers.items()
                   if k.lower() not in ("content-encoding", "content-length",
                                        "transfer-encoding")}
        return httpx.Response(response.status_code, request=request,
                              headers=headers, content=payload)

    def close(self) -> None:
        self._inner.close()


_INSTALLED = False


def install() -> str:
    """Route every ``httpx.Client`` in this process through the store.

    Called once at start-up. In ``off`` mode this returns immediately and httpx is left
    untouched, so the default live run is the same code path it has always been.

    Patching the client constructor rather than each call site is deliberate: there are a
    dozen places that build their own client (portal adapters, Fess search, the liveness
    prober, six secondary-source readers), and a record/replay layer that covers only the
    ones someone remembered to wire up would record an incomplete run — which is worse
    than no recording, because the gap only shows up on stage.
    """
    global _INSTALLED
    current = mode()
    if current == OFF or _INSTALLED:
        return current

    original_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        if kwargs.get("transport") is None:
            kwargs["transport"] = RecordReplayTransport(httpx.HTTPTransport(), current)
        original_init(self, *args, **kwargs)

    httpx.Client.__init__ = patched_init  # type: ignore[method-assign]
    _INSTALLED = True
    return current
