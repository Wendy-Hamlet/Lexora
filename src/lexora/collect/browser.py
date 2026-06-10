"""Headless-browser fetch fallback (Playwright, optional).

Some authoritative portals cannot be read by a plain HTTP client:

  * Singapore SSO (sso.agc.gov.sg) returns HTTP 403 to non-browser clients.
  * Australia's Federal Register of Legislation is a JavaScript SPA whose
    instrument links exist only after client-side rendering.

This module renders such pages with a headless Chromium via Playwright and
returns the final DOM as HTML, so the rest of the pipeline (discovery's link
harvester, the HTML extractor) is unchanged. Playwright is an *optional*
dependency: :func:`is_available` reports whether it can be used, and callers
degrade gracefully (the orchestrator records the original 403 instead) when it
is not installed.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass
class RenderedResult:
    status: int
    final_url: str
    html: str
    content_type: str = "text/html"


def is_available() -> bool:
    """True if Playwright AND a Chromium build are installed and launchable."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        return True
    except Exception:
        return False


class BrowserSession:
    """A reusable headless-Chromium session: launch once, render many URLs.

    Relaunching Chromium per query (the one-shot :func:`render`) is slow and, on
    rate-limited anti-bot portals like SG SSO, makes a burst of fresh launches
    return the homepage / an empty shell instead of search results. Rendering many
    URLs through one persistent context (shared cookies, sequential navigations)
    is both faster and far more stable. Use as a context manager::

        with BrowserSession() as s:
            html = s.render(url).html
    """

    def __init__(self, *, user_agent: str = DEFAULT_UA, timeout: float = 30.0):
        self._ua = user_agent
        self._timeout = timeout
        self._pw = None
        self._browser = None
        self._ctx = None

    def __enter__(self) -> BrowserSession:
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # pragma: no cover - only without playwright
            raise RuntimeError(
                "Playwright is not installed; run `pip install playwright` and "
                "`playwright install chromium`."
            ) from exc
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._ctx = self._browser.new_context(user_agent=self._ua)
        return self

    def render(
        self,
        url: str,
        *,
        wait_until: str = "networkidle",
        settle_ms: int = 1500,
        wait_selector: str | None = None,
        timeout: float | None = None,
    ) -> RenderedResult:
        timeout_ms = int((timeout or self._timeout) * 1000)
        page = self._ctx.new_page()
        try:
            response = page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            if wait_selector:
                # results may be XHR-driven; wait for them, but don't fail hard
                with contextlib.suppress(Exception):
                    page.wait_for_selector(wait_selector, timeout=timeout_ms)
            page.wait_for_timeout(settle_ms)
            html = page.content()
            status = response.status if response is not None else 0
            final_url = page.url
        finally:
            page.close()
        return RenderedResult(status=status, final_url=final_url, html=html)

    def __exit__(self, *exc) -> None:
        try:
            if self._browser is not None:
                self._browser.close()
        finally:
            if self._pw is not None:
                self._pw.stop()


def render(
    url: str,
    *,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_UA,
    wait_until: str = "networkidle",
    settle_ms: int = 1500,
    wait_selector: str | None = None,
) -> RenderedResult:
    """Render ``url`` in headless Chromium and return the final DOM as HTML.

    One-shot convenience wrapper around :class:`BrowserSession` (launches and
    closes a browser per call). For many URLs, hold a :class:`BrowserSession`
    open instead. Raises ``RuntimeError`` if Playwright is not installed — callers
    that want graceful degradation should gate on :func:`is_available` first.
    """
    with BrowserSession(user_agent=user_agent, timeout=timeout) as session:
        return session.render(
            url, wait_until=wait_until, settle_ms=settle_ms, wait_selector=wait_selector
        )


__all__ = ["RenderedResult", "BrowserSession", "is_available", "render", "DEFAULT_UA"]
