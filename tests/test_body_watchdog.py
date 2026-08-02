"""A stalled download must fail on a clock, against a server that really goes silent.

The previous guard checked the idle budget INSIDE the read loop, which cannot work: the
loop body only runs when a chunk arrives, so a stream that stops dead never evaluates it.
Measured 2026-08-02, four hours after that guard shipped -- three sockets to the Malaysian
portal ESTABLISHED for 27 minutes, zero bytes in any 60-second window, process CPU flat,
and a 180-second idle budget never once evaluated.

So these tests do not mock the transport. They stand up a socket server that accepts the
connection, sends real HTTP headers, and then sends nothing at all -- the exact shape of
the failure -- and assert that the read gives up on schedule.
"""
from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest

from lexora.collect.http_cache import BodyDeadlineExceeded, read_body_within


class _SilentServer:
    """Accepts, sends headers promising a body, then never sends the body."""

    def __init__(self, *, dribble: float | None = None, chunks: int = 0) -> None:
        self.dribble = dribble          # seconds between one-byte writes, None = silence
        self.chunks = chunks
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(4)
        self.port = self._sock.getsockname()[1]
        self.stop = threading.Event()
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def _serve(self) -> None:
        while not self.stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            try:
                conn.recv(65536)
                # Content-Length promises far more than we will ever send.
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/pdf\r\n"
                    b"Content-Length: 1000000\r\n\r\n"
                )
                sent = 0
                while not self.stop.is_set():
                    if self.dribble is not None and sent < self.chunks:
                        time.sleep(self.dribble)
                        conn.sendall(b"x")
                        sent += 1
                    else:
                        time.sleep(0.05)
            except OSError:
                return

    def close(self) -> None:
        self.stop.set()
        with __import__("contextlib").suppress(OSError):
            self._sock.close()


@pytest.fixture
def silent():
    s = _SilentServer()
    yield s
    s.close()


def _stream(port: int) -> httpx.Response:
    client = httpx.Client(timeout=httpx.Timeout(connect=5.0, read=None,
                                                write=5.0, pool=5.0))
    return client.send(client.build_request("GET", f"http://127.0.0.1:{port}/act.pdf"),
                       stream=True)


def test_a_socket_that_sends_nothing_gives_up_on_the_idle_budget(silent, monkeypatch):
    """The 27-minute stall, in three seconds. `read=None` on the client is deliberate:
    it removes httpx's own read timeout, so this asserts OUR deadline and nothing else."""
    monkeypatch.setenv("LEXORA_BODY_IDLE", "2")
    monkeypatch.setenv("LEXORA_BODY_DEADLINE", "0")
    response = _stream(silent.port)
    t0 = time.monotonic()
    with pytest.raises(BodyDeadlineExceeded) as err:
        read_body_within(response)
    elapsed = time.monotonic() - t0
    assert elapsed < 15, f"took {elapsed:.1f}s -- the deadline did not fire"
    assert "no body byte for" in str(err.value)
    assert "LEXORA_BODY_IDLE" in str(err.value)


def test_a_dribble_that_never_ends_gives_up_on_the_total_budget(monkeypatch):
    """The other failure, and the reason a total budget survives: bytes keep arriving, so
    the idle clock is reset forever, and only the wall-clock cap stops it."""
    server = _SilentServer(dribble=0.05, chunks=100000)
    try:
        monkeypatch.setenv("LEXORA_BODY_IDLE", "30")
        monkeypatch.setenv("LEXORA_BODY_DEADLINE", "3")
        response = _stream(server.port)
        t0 = time.monotonic()
        with pytest.raises(BodyDeadlineExceeded) as err:
            read_body_within(response)
        elapsed = time.monotonic() - t0
        assert elapsed < 20, f"took {elapsed:.1f}s"
        assert "body still arriving after" in str(err.value)
    finally:
        server.close()


def test_a_body_that_arrives_is_returned_whole(monkeypatch):
    """The guard must not become the thing that eats documents. A 300s cap once cut two
    real Malaysian Acts and a 600s cap cut a GOLD instrument at 32.8 MB."""
    monkeypatch.setenv("LEXORA_BODY_IDLE", "10")
    monkeypatch.setenv("LEXORA_BODY_DEADLINE", "0")
    payload = b"%PDF-1.4 " + b"A" * 200_000

    class _Server(_SilentServer):
        def _handle(self, conn):  # noqa: ANN001
            with conn:
                try:
                    conn.recv(65536)
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/pdf\r\n"
                        b"Content-Length: %d\r\n\r\n" % len(payload)
                    )
                    for i in range(0, len(payload), 4096):
                        conn.sendall(payload[i:i + 4096])
                except OSError:
                    return

    server = _Server()
    try:
        got = read_body_within(_stream(server.port))
        assert got == payload
    finally:
        server.close()


def test_both_budgets_off_reads_normally(monkeypatch):
    monkeypatch.setenv("LEXORA_BODY_IDLE", "0")
    monkeypatch.setenv("LEXORA_BODY_DEADLINE", "0")
    payload = b"hello"

    class _Server(_SilentServer):
        def _handle(self, conn):  # noqa: ANN001
            with conn:
                try:
                    conn.recv(65536)
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(payload)
                        + payload
                    )
                except OSError:
                    return

    server = _Server()
    try:
        assert read_body_within(_stream(server.port)) == payload
    finally:
        server.close()


def test_the_watchdog_thread_does_not_outlive_the_read(monkeypatch):
    """One thread per document times a corpus of hundreds; they must all be gone."""
    monkeypatch.setenv("LEXORA_BODY_IDLE", "10")
    monkeypatch.setenv("LEXORA_BODY_DEADLINE", "0")
    payload = b"x" * 1024

    class _Server(_SilentServer):
        def _handle(self, conn):  # noqa: ANN001
            with conn:
                try:
                    conn.recv(65536)
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(payload)
                        + payload
                    )
                except OSError:
                    return

    server = _Server()
    try:
        before = {t.name for t in threading.enumerate()}
        for _ in range(5):
            read_body_within(_stream(server.port))
        time.sleep(0.3)
        left = [t.name for t in threading.enumerate()
                if t.name == "lexora-body-watchdog" and t.name not in before]
        assert left == [], f"watchdog threads left running: {left}"
    finally:
        server.close()


# --- the socket deadline, which is what actually interrupts a stalled read ----------
#
# These matter more than the watchdog tests above, and exist because those tests passed
# while production hung. They ran over plain HTTP, where closing the response happens to
# unblock the reader. Production is HTTPS: py-spy on the second stalled record run showed
# three workers parked in `ssl.read` inside `read_body_within`, twenty-five minutes after
# the watchdog had called `response.close()` on them.
#
# So these tests DISABLE THE WATCHDOG (the env var drives both mechanisms, so switching
# budgets off is not enough -- `_watch_body` is stubbed out) and leave the socket timeout
# as the only thing that can end the read. Otherwise they would prove, once again, that
# the mechanism which does not work in production works in a test.

@pytest.fixture
def no_watchdog(monkeypatch):
    from lexora.collect import http_cache

    monkeypatch.setattr(http_cache, "_watch_body", lambda *a, **k: None)


def test_the_read_deadline_reaches_the_socket_not_just_a_thread(
    silent, no_watchdog, monkeypatch
):
    """If this returns, it is because httpcore called `sock.settimeout()` with our
    number -- nothing else is left that could end the read."""
    monkeypatch.setenv("LEXORA_BODY_IDLE", "2")
    monkeypatch.setenv("LEXORA_BODY_DEADLINE", "0")
    from lexora.collect.http_cache import clamp_read_timeout

    client = httpx.Client(timeout=httpx.Timeout(connect=5.0, read=None,
                                                write=5.0, pool=5.0))
    request = clamp_read_timeout(
        client.build_request("GET", f"http://127.0.0.1:{silent.port}/act.pdf"), 2.0)
    t0 = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        response = client.send(request, stream=True)
        for _ in response.iter_bytes():
            pass
    assert time.monotonic() - t0 < 15


def test_clamp_only_lowers_never_raises():
    """A caller that asked for something stricter keeps it; and a request with no timeout
    at all -- the case that blocks forever, because settimeout(None) never returns --
    gets one."""
    from lexora.collect.http_cache import clamp_read_timeout

    client = httpx.Client()
    req = client.build_request("GET", "http://example.invalid/")
    req.extensions = {"timeout": {"connect": 5.0, "read": None}}
    clamp_read_timeout(req, 120.0)
    assert req.extensions["timeout"]["read"] == 120.0
    assert req.extensions["timeout"]["connect"] == 5.0, "other timeouts must survive"

    req.extensions = {"timeout": {"read": 30.0}}
    clamp_read_timeout(req, 120.0)
    assert req.extensions["timeout"]["read"] == 30.0, "a stricter caller keeps its number"

    req.extensions = {"timeout": {"read": 300.0}}
    clamp_read_timeout(req, 120.0)
    assert req.extensions["timeout"]["read"] == 120.0

    req.extensions = {}
    clamp_read_timeout(req, 0.0)
    assert req.extensions.get("timeout", {}).get("read") is None, "0 disables, as elsewhere"


def test_a_document_fetch_gives_up_instead_of_blocking_forever(
    silent, no_watchdog, monkeypatch
):
    """End to end through the real fetch path, with the watchdog stubbed out. This is the
    call chain that hung twice: `crawler.fetch` -> `_fetch_once` -> `read_body_within`."""
    monkeypatch.setenv("LEXORA_BODY_IDLE", "2")
    monkeypatch.setenv("LEXORA_BODY_DEADLINE", "0")
    from lexora.collect import crawler

    t0 = time.monotonic()
    with pytest.raises(Exception) as err:
        crawler.fetch(
            f"http://127.0.0.1:{silent.port}/act.pdf",
            jurisdiction="MY", portal_name="test", source_type="legislation",
            dest_dir=None, retries=0, timeout=None,
        )
    elapsed = time.monotonic() - t0
    assert elapsed < 30, f"fetch blocked for {elapsed:.0f}s -- the deadline never landed"
    assert err.type is not AssertionError
