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


def render(
    url: str,
    *,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_UA,
    wait_until: str = "networkidle",
    settle_ms: int = 1500,
) -> RenderedResult:
    """Render ``url`` in headless Chromium and return the final DOM as HTML.

    Raises ``RuntimeError`` if Playwright is not installed — callers that want
    graceful degradation should gate on :func:`is_available` first.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - exercised only without playwright
        raise RuntimeError(
            "Playwright is not installed; run `pip install playwright` and "
            "`playwright install chromium`."
        ) from exc

    timeout_ms = int(timeout * 1000)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(user_agent=user_agent)
            page = context.new_page()
            response = page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            page.wait_for_timeout(settle_ms)  # let late XHR-driven content settle
            html = page.content()
            status = response.status if response is not None else 0
            final_url = page.url
        finally:
            browser.close()
    return RenderedResult(status=status, final_url=final_url, html=html)


__all__ = ["RenderedResult", "is_available", "render", "DEFAULT_UA"]
