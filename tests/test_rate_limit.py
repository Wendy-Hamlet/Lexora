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
