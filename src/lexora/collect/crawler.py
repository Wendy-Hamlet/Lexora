"""Portal crawler.

Slice 1: live single-URL fetching via httpx with content-type routing
(PDF vs HTML), polite headers, retries, optional rate limiting and robots
checks. A 4xx/5xx is *captured* in the RawDocument (e.g. Singapore SSO returns
403 to bots) rather than crashing the pipeline, so the orchestrator can decide
to fall back to a browser engine in a later slice.

Per-portal discovery (search -> instrument URL), sitemap walking and a
Playwright fallback for JS/anti-bot portals are reserved for the next slice.
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlparse

import httpx

from lexora.collect.hasher import sha256_bytes
from lexora.models.source import (
    FetchMethod,
    RawDocument,
    SourceProfile,
    SourceType,
)

DEFAULT_UA = "Lexora/0.1 (+https://github.com/Wendy-Hamlet/Lexora)"

# crude per-host politeness clock
_LAST_HIT: dict[str, float] = {}


@dataclass
class FetchResult:
    document: RawDocument
    body: bytes

    @property
    def content_type(self) -> str:
        return self.document.content_type

    def is_pdf(self) -> bool:
        return is_pdf(self.document.content_type, self.body)

    def is_html(self) -> bool:
        return "html" in self.document.content_type.lower()


def is_pdf(content_type: str, body: bytes) -> bool:
    """A response is a PDF if the content-type says so or the bytes start with
    the PDF magic number (some portals serve PDFs as octet-stream)."""
    if "pdf" in content_type.lower():
        return True
    return body[:5] == b"%PDF-"


def _document_id(jurisdiction: str, sha256_uri: str) -> str:
    """Stable, content-addressed ID. `sha256_uri` is the full `sha256:<hex>`."""
    return f"{jurisdiction.lower()}:{sha256_uri.split(':', 1)[1][:16]}"


def robots_allows(url: str, user_agent: str = DEFAULT_UA) -> bool:
    """Best-effort robots.txt check. Fails OPEN (returns True) if robots cannot
    be fetched or parsed — we never block a fetch on an unreachable robots."""
    try:
        parts = urlparse(url)
        rp = robotparser.RobotFileParser()
        rp.set_url(f"{parts.scheme}://{parts.netloc}/robots.txt")
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True


def _browser_render(url: str, *, timeout: float):
    """Render `url` with a headless browser, or return None if Playwright is
    unavailable / rendering fails (caller keeps the original HTTP response).

    The browser uses its own realistic UA — forwarding the polite HTTP bot UA
    would just re-trigger the anti-bot 403 we are escalating past.
    """
    try:
        from lexora.collect.browser import is_available, render

        if not is_available():
            return None
        return render(url, timeout=timeout)
    except Exception:
        return None


def _rate_limit(host: str, min_interval: float) -> None:
    if min_interval <= 0:
        return
    now = time.monotonic()
    last = _LAST_HIT.get(host)
    if last is not None:
        wait = min_interval - (now - last)
        if wait > 0:
            time.sleep(wait)
    _LAST_HIT[host] = time.monotonic()


def fetch(
    url: str,
    *,
    jurisdiction: str,
    portal_name: str,
    source_type: SourceType,
    dest_dir: Path | None = None,
    title: str | None = None,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_UA,
    retries: int = 2,
    respect_robots: bool = False,
    min_interval: float = 0.0,
    browser_fallback: bool = False,
    client: httpx.Client | None = None,
) -> FetchResult:
    """Fetch a single URL live and return its bytes + a RawDocument record.

    Bytes are persisted content-addressed under `dest_dir` when given. A non-2xx
    response is returned (status captured) rather than raised; only transport
    errors after `retries` propagate.

    When ``browser_fallback`` is set and the server answers 403/429 (anti-bot,
    e.g. Singapore SSO), the URL is re-fetched with a headless browser and the
    rendered DOM replaces the body — provided Playwright is installed. If it is
    not, the original 403 is returned unchanged (graceful degradation).
    """
    if respect_robots and not robots_allows(url, user_agent):
        raise PermissionError(f"robots.txt disallows fetching {url}")

    host = urlparse(url).netloc
    _rate_limit(host, min_interval)

    owns_client = client is None
    client = client or httpx.Client(
        follow_redirects=True, timeout=timeout, headers={"User-Agent": user_agent}
    )
    try:
        last_exc: Exception | None = None
        response = None
        for attempt in range(retries + 1):
            try:
                response = client.get(url)
                if response.status_code < 500:
                    break
            except httpx.TransportError as exc:  # network blip — retry
                last_exc = exc
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
        if response is None:
            raise last_exc or httpx.TransportError(f"failed to fetch {url}")
    finally:
        if owns_client:
            client.close()

    body = response.content
    status = response.status_code
    final_url = str(response.url)
    content_type = response.headers.get(
        "content-type", "application/octet-stream"
    ).split(";")[0].strip()

    if browser_fallback and status in (403, 429):
        rendered = _browser_render(final_url, timeout=timeout)
        if rendered is not None:
            body = rendered.html.encode("utf-8")
            status = rendered.status or status
            final_url = rendered.final_url
            content_type = rendered.content_type

    sha = sha256_bytes(body)

    bytes_path = ""
    if dest_dir is not None:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        ext = "pdf" if is_pdf(content_type, body) else "bin"
        p = dest_dir / f"{sha.split(':', 1)[1]}.{ext}"
        if not p.exists():
            p.write_bytes(body)
        bytes_path = str(p)

    document = RawDocument(
        document_id=_document_id(jurisdiction, sha),
        source_url=final_url,
        retrieval_timestamp=datetime.now(timezone.utc),
        http_status=status,
        sha256=sha,
        content_type=content_type,
        bytes_path=bytes_path,
        portal_name=portal_name,
        jurisdiction=jurisdiction,
        source_type=source_type,
        title=title,
    )
    return FetchResult(document=document, body=body)


def fetch_url(
    url: str,
    *,
    jurisdiction: str,
    portal_name: str,
    source_type: SourceType,
    dest_dir: Path,
    title: str | None = None,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_UA,
) -> RawDocument:
    """Back-compat wrapper: fetch and persist, returning only the RawDocument."""
    return fetch(
        url,
        jurisdiction=jurisdiction,
        portal_name=portal_name,
        source_type=source_type,
        dest_dir=dest_dir,
        title=title,
        timeout=timeout,
        user_agent=user_agent,
    ).document


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
    """Register a local file as if it had been crawled (manual-upload fallback)."""
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


def crawl(profile: SourceProfile, *, dest_dir: Path) -> Iterable[FetchResult]:
    """Walk every `http` portal in the profile and yield FetchResults.

    Slice 1 fetches each portal's landing URL live; instrument-URL discovery
    (search + sitemap) is the next slice.
    """
    dest_dir = Path(dest_dir)
    for portal in profile.portals:
        if portal.fetch_method is not FetchMethod.http:
            continue
        yield fetch(
            str(portal.url),
            jurisdiction=profile.iso_code,
            portal_name=portal.name,
            source_type=portal.source_type,
            dest_dir=dest_dir,
            title=portal.name,
            min_interval=1.0,
        )


__all__ = [
    "FetchResult",
    "fetch",
    "fetch_url",
    "ingest_local_file",
    "crawl",
    "is_pdf",
    "robots_allows",
    "DEFAULT_UA",
]
