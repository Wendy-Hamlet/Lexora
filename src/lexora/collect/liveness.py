"""Source-URL liveness — dead-link detection for the submission (WS-4).

A citation's ``Source URL`` is the judge's gateway to verifying the verbatim
snippet against the official portal; a broken link silently sinks an otherwise
correct row. This module checks each distinct source URL and annotates the
citations whose link is dead, so a human can refresh it before submission.

It is **network I/O and therefore opt-in** (mirrors the verifier / LLM lanes):
the pipeline never calls it implicitly. The submission script exposes it behind
``--check-links``. The pure URL-set grouping and annotation logic are unit-tested
offline with a stub client; only :func:`check_url` touches the network.

Honest scope: liveness is a *reachability* check (does the URL resolve to a 2xx
after redirects), not a content check — it cannot tell that a live URL now serves
a different document. That deeper "still the same instrument" check belongs to the
hash/verbatim trail, not here.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import httpx

from lexora.collect.crawler import DEFAULT_UA


@dataclass(frozen=True)
class LinkCheck:
    """Outcome of probing one URL."""

    url: str
    ok: bool  # resolved to a final 2xx (after redirects)
    status_code: int  # final HTTP status; 0 on a transport error
    final_url: str = ""
    redirected: bool = False
    error: str = ""

    def note(self) -> str:
        """A short, submission-ready annotation when the link is NOT ok ("" when ok)."""
        if self.ok:
            return ""
        if self.status_code:
            return f"[dead link: HTTP {self.status_code}]"
        return f"[dead link: {self.error or 'unreachable'}]"


def check_url(
    url: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = 10.0,
    user_agent: str = DEFAULT_UA,
) -> LinkCheck:
    """Probe a single URL for reachability.

    Tries a cheap ``HEAD`` first and falls back to a ranged ``GET`` when the
    server rejects HEAD (405/501) or anti-bot blocks it (403) — many legal
    portals do. Redirects are followed; the final status decides ``ok``. A
    transport error (DNS / connection / timeout) is reported as ``ok=False`` with
    ``status_code=0`` rather than raised, so a batch check never crashes on one
    bad link.
    """
    owns = client is None
    client = client or httpx.Client(
        follow_redirects=True, timeout=timeout, headers={"User-Agent": user_agent}
    )
    try:
        try:
            resp = client.head(url)
            if resp.status_code in (403, 405, 501):
                # Server dislikes HEAD — retry with a light GET before judging.
                resp = client.get(url, headers={"Range": "bytes=0-0"})
        except httpx.TransportError as exc:
            return LinkCheck(url=url, ok=False, status_code=0, error=type(exc).__name__)
        final_url = str(resp.url)
        return LinkCheck(
            url=url,
            ok=200 <= resp.status_code < 300,
            status_code=resp.status_code,
            final_url=final_url,
            redirected=final_url != url,
        )
    finally:
        if owns:
            client.close()


def check_urls(
    urls: Iterable[str],
    *,
    client: httpx.Client | None = None,
    timeout: float = 10.0,
    user_agent: str = DEFAULT_UA,
) -> dict[str, LinkCheck]:
    """Probe each DISTINCT url once, returning ``{url: LinkCheck}``.

    Distinct-once is the whole point: a submission has many citations per
    instrument sharing one Source URL, so we never re-probe the same link."""
    seen: dict[str, LinkCheck] = {}
    for url in urls:
        if url in seen:
            continue
        seen[url] = check_url(url, client=client, timeout=timeout, user_agent=user_agent)
    return seen


def annotate_dead_links(citations: list, checks: dict[str, LinkCheck]) -> tuple[list, int]:
    """Append a dead-link note to every citation whose Source URL failed.

    Pure (no network): takes the citations and a precomputed ``{url: LinkCheck}``
    map. Returns ``(annotated_citations, n_dead_rows)``. A live (or unchecked)
    link leaves the citation untouched; the note is appended to ``notes`` so the
    verbatim/mapping fields are never disturbed."""
    out, dead = [], 0
    for c in citations:
        check = checks.get(str(c.source_url))
        note = check.note() if check is not None else ""
        if note:
            dead += 1
            new_notes = f"{c.notes} {note}".strip() if c.notes else note
            out.append(c.model_copy(update={"notes": new_notes}))
        else:
            out.append(c)
    return out, dead


__all__ = ["LinkCheck", "check_url", "check_urls", "annotate_dead_links"]
