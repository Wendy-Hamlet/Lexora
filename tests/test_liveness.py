"""Source-URL liveness / dead-link detection (collect/liveness.py).

check_url is exercised against an httpx MockTransport (no real network); the
distinct-once probing and the Notes annotation are pure and tested offline.
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx

from lexora.collect.liveness import annotate_dead_links, check_url, check_urls
from lexora.models.citation import Citation, DiscoveryTag


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_check_url_live_200():
    client = _client(lambda req: httpx.Response(200))
    res = check_url("https://e.gov/live", client=client)
    assert res.ok and res.status_code == 200


def test_check_url_dead_404():
    client = _client(lambda req: httpx.Response(404))
    res = check_url("https://e.gov/gone", client=client)
    assert res.ok is False and res.status_code == 404
    assert res.note() == "[dead link: HTTP 404]"


def test_check_url_head_rejected_falls_back_to_get():
    def handler(req: httpx.Request) -> httpx.Response:
        # Server rejects HEAD with 405 but serves GET.
        return httpx.Response(405) if req.method == "HEAD" else httpx.Response(206)
    res = check_url("https://e.gov/nohead", client=_client(handler))
    assert res.ok and res.status_code == 206


def test_check_url_transport_error_is_not_raised():
    def handler(req):
        raise httpx.ConnectError("boom")
    res = check_url("https://nope.invalid/x", client=_client(handler))
    assert res.ok is False and res.status_code == 0
    assert "dead link" in res.note()


def test_check_urls_probes_each_distinct_once():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(200)

    urls = ["https://e.gov/a", "https://e.gov/a", "https://e.gov/b"]
    out = check_urls(urls, client=_client(handler))
    assert set(out) == {"https://e.gov/a", "https://e.gov/b"}
    assert calls["n"] == 2  # the duplicate was not re-probed


def _cite(url: str, notes: str = "") -> Citation:
    return Citation(
        title="Act", indicator_id="P6-I4", article_path="S. 1",
        discovery_tag=DiscoveryTag.known, page_or_dom_anchor="1", quote="verbatim text",
        source_url=url, confidence=0.9, clause_id="c1",
        retrieval_timestamp=datetime.now(timezone.utc), document_hash="sha256:abc",
        jurisdiction="SG", legal_form="statute", char_start=0, char_end=4, notes=notes,
    )


def test_annotate_dead_links_only_touches_dead_rows():
    cites = [_cite("https://e.gov/live"), _cite("https://e.gov/gone", notes="KNOWN")]
    checks = check_urls(
        ["https://e.gov/live", "https://e.gov/gone"],
        client=_client(lambda req: httpx.Response(200 if "live" in str(req.url) else 404)),
    )
    out, dead = annotate_dead_links(cites, checks)
    assert dead == 1
    assert out[0].notes == ""  # live row untouched
    assert out[1].notes == "KNOWN [dead link: HTTP 404]"  # appended, original kept
    # verbatim contract is never disturbed by annotation
    assert out[1].quote == "verbatim text"
