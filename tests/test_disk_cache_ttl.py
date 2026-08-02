"""A side-cache that expires under replay refetches nothing and reports success.

Not hypothetical. 2026-08-01: `au_act_catalogue.json` was ten hours past its 24-hour TTL
when a replay ran. The refetch found no recording, got the synthetic 504, the paging loop
stopped on page 0, and Australia's semantic crosswalk -- the only route by which its NEW
instruments are found -- received an empty catalogue. The run produced 463 rows instead of
499 and raised nothing. On stage that is a third of one economy disappearing between the
rehearsal and the demo, with a row count as the only symptom.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from lexora.collect import http_cache
from lexora.collect.http_cache import disk_cache_still_valid
from lexora.collect.secondary.base import cached_json


@pytest.fixture
def stale_file(tmp_path):
    p = tmp_path / "au_act_catalogue.json"
    p.write_text(json.dumps([{"id": "C2004A03712", "name": "Privacy Act 1988"}]),
                 encoding="utf-8")
    old = time.time() - 40 * 3600          # 40 h old against a 24 h TTL
    os.utime(p, (old, old))
    return p


def test_live_still_expires_a_stale_cache(stale_file, monkeypatch):
    """The TTL must keep doing its job when there IS a server to ask."""
    monkeypatch.delenv("LEXORA_HTTP_CACHE", raising=False)
    assert disk_cache_still_valid(stale_file, 24.0) is False


def test_replay_never_expires_a_side_cache(stale_file, monkeypatch):
    """Under replay the world cannot have moved: every byte is a dated snapshot."""
    monkeypatch.setenv("LEXORA_HTTP_CACHE", "replay")
    assert http_cache.serving_from_recording() is True
    assert disk_cache_still_valid(stale_file, 24.0) is True


def test_recording_still_expires(stale_file, monkeypatch):
    """`record` is a LIVE crawl that keeps a copy. Freshness still matters."""
    monkeypatch.setenv("LEXORA_HTTP_CACHE", "record")
    assert disk_cache_still_valid(stale_file, 24.0) is False


def test_a_missing_file_is_never_valid(tmp_path, monkeypatch):
    monkeypatch.setenv("LEXORA_HTTP_CACHE", "replay")
    assert disk_cache_still_valid(tmp_path / "not_here.json", 24.0) is False


def test_a_fresh_file_is_valid_in_every_mode(tmp_path, monkeypatch):
    p = tmp_path / "fresh.json"
    p.write_text("[1]", encoding="utf-8")
    for mode in ("", "record", "replay"):
        monkeypatch.setenv("LEXORA_HTTP_CACHE", mode)
        assert disk_cache_still_valid(p, 24.0) is True


def test_the_secondary_adapters_get_the_same_rule(stale_file, monkeypatch):
    """Six secondary-source readers share `cached_json`, so the fix has to land there
    too -- fixing only the AU catalogue would leave five of the same trap in place."""
    calls = []

    def _producer():
        calls.append(1)
        return [{"fetched": "live"}]

    monkeypatch.setenv("LEXORA_HTTP_CACHE", "replay")
    got = cached_json(stale_file, 24.0, _producer)
    assert calls == [], "replay refetched a stale side-cache; live there is no server"
    assert got[0]["name"] == "Privacy Act 1988"

    monkeypatch.delenv("LEXORA_HTTP_CACHE", raising=False)
    got = cached_json(stale_file, 24.0, _producer)
    assert calls == [1], "a live run must still notice the cache went stale"
    assert got[0]["fetched"] == "live"
