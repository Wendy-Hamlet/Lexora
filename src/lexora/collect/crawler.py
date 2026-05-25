"""Portal crawler.

Strategy per portal:
    http        → requests via httpx, follow sitemap.xml if present
    sitemap     → walk sitemap.xml and fetch each <loc>
    playwright  → headless browser for JS-rendered pages
    api         → portal-specific REST/SOAP client (handled by subclass)

Each fetched response becomes a RawDocument with SHA-256 + retrieval timestamp.
ETag-aware re-crawls skip unchanged documents.
"""
from __future__ import annotations

from collections.abc import Iterable

from lexora.models.source import RawDocument, SourceProfile


def crawl(profile: SourceProfile) -> Iterable[RawDocument]:  # pragma: no cover
    """Yield RawDocument records for every fetch in the given profile.

    TODO: implement per-fetch-method handlers.
    """
    raise NotImplementedError(
        "Implement crawl(): see docs/architecture.md §1. Start with the `http` "
        "method and httpx, then add sitemap and playwright as needed."
    )
