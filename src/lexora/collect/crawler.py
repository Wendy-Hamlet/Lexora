"""Portal crawler.

Slice 0: single-URL fetch via httpx. Multi-portal walk, sitemap, and
playwright handlers are reserved for later slices and live behind the
`fetch_method` dispatch in `crawl()`.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import httpx

from lexora.collect.hasher import sha256_bytes
from lexora.models.source import (
    FetchMethod,
    PortalSpec,
    RawDocument,
    SourceProfile,
    SourceType,
)


def _document_id(jurisdiction: str, sha256_uri: str) -> str:
    """Stable, content-addressed ID. `sha256_uri` is the full `sha256:<hex>` string."""
    return f"{jurisdiction.lower()}:{sha256_uri.split(':', 1)[1][:16]}"


def fetch_url(
    url: str,
    *,
    jurisdiction: str,
    portal_name: str,
    source_type: SourceType,
    dest_dir: Path,
    title: str | None = None,
    timeout: float = 30.0,
    user_agent: str = "Lexora/0.1 (+https://github.com/Wendy-Hamlet/Lexora)",
) -> RawDocument:
    """Fetch a single URL and persist its bytes content-addressed under `dest_dir`.

    Returns a RawDocument record. Re-fetching the same URL that yields the same
    bytes is idempotent: the file already exists and is left untouched.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": user_agent}) as client:
        response = client.get(url)

    body = response.content
    sha = sha256_bytes(body)
    bytes_path = dest_dir / f"{sha.split(':', 1)[1]}.bin"
    if not bytes_path.exists():
        bytes_path.write_bytes(body)

    return RawDocument(
        document_id=_document_id(jurisdiction, sha),
        source_url=url,
        retrieval_timestamp=datetime.now(timezone.utc),
        http_status=response.status_code,
        sha256=sha,
        content_type=response.headers.get("content-type", "application/octet-stream").split(";")[0].strip(),
        bytes_path=str(bytes_path),
        portal_name=portal_name,
        jurisdiction=jurisdiction,
        source_type=source_type,
        title=title,
    )


def ingest_local_file(
    path: Path,
    *,
    jurisdiction: str,
    portal_name: str,
    source_url: str,
    source_type: SourceType,
    dest_dir: Path,
    content_type: str = "application/pdf",
    title: str | None = None,
) -> RawDocument:
    """Register a local file as if it had been crawled.

    Useful for demos where the team manually downloaded the gazette PDF. The
    file is copied into the content-addressed store so downstream stages see a
    uniform layout.
    """
    path = Path(path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    body = path.read_bytes()
    sha = sha256_bytes(body)
    bytes_path = dest_dir / f"{sha.split(':', 1)[1]}.bin"
    if not bytes_path.exists():
        bytes_path.write_bytes(body)

    return RawDocument(
        document_id=_document_id(jurisdiction, sha),
        source_url=source_url,
        retrieval_timestamp=datetime.now(timezone.utc),
        http_status=200,
        sha256=sha,
        content_type=content_type,
        bytes_path=str(bytes_path),
        portal_name=portal_name,
        jurisdiction=jurisdiction,
        source_type=source_type,
        title=title or path.stem,
    )


def crawl(profile: SourceProfile, *, dest_dir: Path) -> Iterable[RawDocument]:
    """Walk every portal in the profile and yield RawDocument records.

    Slice 0 only handles the `http` method and only fetches the portal's
    landing URL — discovery of individual instrument URLs is left to later
    slices that add sitemap walking and search-query expansion.
    """
    dest_dir = Path(dest_dir)
    for portal in profile.portals:
        if portal.fetch_method is not FetchMethod.http:
            continue
        yield fetch_url(
            str(portal.url),
            jurisdiction=profile.iso_code,
            portal_name=portal.name,
            source_type=portal.source_type,
            dest_dir=dest_dir,
            title=portal.name,
        )


__all__ = ["fetch_url", "ingest_local_file", "crawl"]


def _portal_from_profile(profile: SourceProfile, name: str) -> PortalSpec:  # pragma: no cover
    for p in profile.portals:
        if p.name == name:
            return p
    raise KeyError(name)
