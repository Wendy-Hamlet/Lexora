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

import hashlib
import json
import os
import sqlite3
import threading
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


class RecordReplayTransport(httpx.BaseTransport):
    """Wraps a real transport: records what it returns, or answers from the store."""

    def __init__(self, inner: httpx.BaseTransport, cache_mode: str) -> None:
        self._inner = inner
        self._mode = cache_mode

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = request.content or b""
        method, url = request.method, str(request.url)

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
        payload = b"".join(response.stream)  # drain before the connection is released
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
