"""HTML → canonical text via DOM parsing.

Each extracted block carries a DOM anchor (an id/CSS hint) so a citation can
point back to the element in the original page, and global character offsets so
the structure parser and the verbatim validator can work on the same string as
they do for PDFs.

The global text is the block texts joined by ``BLOCK_SEPARATOR`` (the same
separator the PDF extractor uses between pages), so downstream code is format
agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

BLOCK_SEPARATOR = "\n\n"

# Tags whose text we never want in the canonical legal text.
_DROP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "form", "svg"]
# Block-level tags that become one HtmlBlock each.
_BLOCK_TAGS = ["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "td"]


@dataclass
class HtmlBlock:
    dom_anchor: str
    text: str
    char_start: int
    char_end: int


def _anchor_for(tag: Tag, index: int) -> str:
    """Build a stable-ish anchor: prefer an element id, else id of nearest
    ancestor, else a tag#ordinal fallback."""
    if tag.get("id"):
        return f"#{tag['id']}"
    for parent in tag.parents:
        if isinstance(parent, Tag) and parent.get("id"):
            return f"#{parent['id']} {tag.name}:nth({index})"
    return f"{tag.name}:nth({index})"


def _clean(text: str) -> str:
    # collapse runs of intra-line whitespace but keep the block as one line
    return " ".join(text.split())


def extract_html(html_bytes: bytes, base_url: str = "") -> list[HtmlBlock]:
    """Return DOM-anchored text blocks from raw HTML, with global char offsets.

    Blocks are emitted in document order. Empty/whitespace blocks are skipped.
    """
    html = html_bytes.decode("utf-8", errors="replace") if isinstance(html_bytes, bytes) else html_bytes
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_DROP_TAGS):
        tag.decompose()

    body = soup.body or soup
    blocks: list[HtmlBlock] = []
    cursor = 0
    seen: set[int] = set()

    candidates = body.find_all(_BLOCK_TAGS)
    if not candidates:
        # no recognisable block tags — fall back to the whole body as one block
        text = _clean(body.get_text(" "))
        if text:
            blocks.append(HtmlBlock(dom_anchor="body", text=text, char_start=0, char_end=len(text)))
        return blocks

    for index, tag in enumerate(candidates):
        if id(tag) in seen:
            continue
        # Skip a block that merely wraps other block tags (avoid double counting);
        # only emit leaf-ish blocks.
        if tag.find(_BLOCK_TAGS):
            continue
        seen.add(id(tag))
        text = _clean(tag.get_text(" "))
        if not text:
            continue
        char_start = cursor
        char_end = cursor + len(text)
        blocks.append(
            HtmlBlock(
                dom_anchor=_anchor_for(tag, index),
                text=text,
                char_start=char_start,
                char_end=char_end,
            )
        )
        cursor = char_end + len(BLOCK_SEPARATOR)
    return blocks


def assemble_global_text(blocks: list[HtmlBlock]) -> str:
    """Reconstruct the global text whose offsets match each block."""
    return BLOCK_SEPARATOR.join(b.text for b in blocks)


__all__ = ["HtmlBlock", "BLOCK_SEPARATOR", "extract_html", "assemble_global_text"]
