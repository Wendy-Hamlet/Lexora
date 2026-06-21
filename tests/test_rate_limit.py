"""Per-host fetch throttle (crawler._rate_limit) — thread-enforced rate-spacing.

The throttle dodges request-rate anti-bot (e.g. AU serving an HTML challenge page
instead of the PDF under concurrent bursts): concurrent fetches to the SAME host
serialize through a per-host lock and start >= min_interval apart, while different
hosts keep independent locks and stay parallel.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from lexora.collect import crawler


def _reset() -> None:
    crawler._LAST_HIT.clear()
    crawler._HOST_LOCKS.clear()


def test_same_host_serializes_with_spacing_under_threads():
    _reset()
    n, gap = 5, 0.1
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(lambda _: crawler._rate_limit("au.example", gap), range(n)))
    elapsed = time.monotonic() - t0
    # n gated starts spaced by `gap` -> at least (n-1)*gap even with n threads.
    assert elapsed >= (n - 1) * gap * 0.9


def test_different_hosts_do_not_block_each_other():
    _reset()
    gap = 0.3
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(lambda h: crawler._rate_limit(h, gap),
                    ["a.example", "b.example", "c.example"]))
    # Distinct hosts -> independent locks, first hit of each waits 0 -> ~instant.
    assert (time.monotonic() - t0) < gap


def test_zero_interval_is_a_noop():
    _reset()
    t0 = time.monotonic()
    crawler._rate_limit("x.example", 0.0)
    assert (time.monotonic() - t0) < 0.05
    assert "x.example" not in crawler._LAST_HIT  # nothing recorded when off


class _CountingClient:
    """Fake httpx client that tracks how many GETs to one host run concurrently."""

    def __init__(self, registry):
        self._reg = registry

    def get(self, url):
        reg = self._reg
        with reg["guard"]:
            reg["in_flight"] += 1
            reg["max"] = max(reg["max"], reg["in_flight"])
        time.sleep(0.05)  # simulate a download in progress
        with reg["guard"]:
            reg["in_flight"] -= 1

        class _Resp:
            status_code = 200
            headers = {"content-type": "application/pdf"}
            content = b"%PDF-1.4 body"

        _Resp.url = url
        return _Resp()

    def close(self):
        pass


def _hammer(serial: bool):
    import threading

    reg = {"in_flight": 0, "max": 0, "guard": threading.Lock()}

    def _one(_):
        crawler.fetch(
            "https://au.example/doc.pdf", jurisdiction="AU", portal_name="p",
            source_type=crawler.SourceType.primary, serial_fetch=serial,
            client=_CountingClient(reg),
        )

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(_one, range(4)))
    return reg["max"]


def test_serial_fetch_serializes_the_download_itself():
    """serial_fetch=True holds the per-host lock across the HTTP GET, so concurrent
    same-host downloads never overlap (no request burst for anti-bot to react to)."""
    _reset()
    assert _hammer(serial=True) == 1


def test_non_serial_fetch_allows_concurrent_downloads():
    """Default does NOT gate the download — concurrent same-host GETs overlap (the
    regime that bursts AU into its anti-bot challenge)."""
    _reset()
    assert _hammer(serial=False) > 1
