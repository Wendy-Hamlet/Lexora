"""HTML → canonical text via DOM parsing.

Each extracted block carries its DOM anchor (CSS selector) so a citation can
point back to the exact element in the original page.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HtmlBlock:
    dom_anchor: str
    text: str
    char_start: int
    char_end: int


def extract_html(html_bytes: bytes, base_url: str) -> list[HtmlBlock]:  # pragma: no cover
    """Return DOM-anchored text blocks from raw HTML.

    TODO: parse with BeautifulSoup + lxml, walk structural tags
    (article, section, p, h1..h6), build CSS-selector anchors, compute global
    character offsets across the page.
    """
    raise NotImplementedError("Implement HTML extractor.")
