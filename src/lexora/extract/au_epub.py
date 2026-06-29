"""Combine a multi-document AU EPUB into one HTML body.

The Federal Register serves a compilation's full prose as an EPUB whose content
is SPLIT across ``OEBPS/document_1/document_1.html``, ``document_2/…`` … A large
Act keeps its operative Schedule in the later parts — e.g. the Criminal Code Act
1995's computer offences (Part 10.7) live in ``document_2``/``document_3``, NOT in
``document_1``. Fetching only ``document_1`` therefore silently drops the very
provisions a sectoral statute is cited for.

``combine_au_epub`` detects a ``document_1.html`` URL, fetches the sibling parts
in order until one is missing, and splices every part's ``<body>`` into a single
HTML document. The structure parser then sees the whole Act (one ``<body>`` with
all block tags parses correctly, where concatenating full ``<html>`` documents
would leave a lenient parser reading only the first).
"""
from __future__ import annotations

import re
import time

import httpx
from bs4 import BeautifulSoup

# .../epub/OEBPS/document_1/document_1.html  -> capture the stem around the "1"s.
_DOC1 = re.compile(r"(?P<a>.*/epub/OEBPS/document_)1(?P<b>/document_)1(?P<c>\.html)$")
_MAX_PARTS = 50  # safety bound; real Acts have a handful


def _inner_body(html_bytes: bytes) -> str:
    soup = BeautifulSoup(html_bytes, "lxml")
    body = soup.body or soup
    return "".join(str(x) for x in body.contents)


def combine_au_epub(
    url: str, first_body: bytes, client: httpx.Client, *, inter_delay: float = 0.3
) -> bytes | None:
    """If ``url`` is an AU EPUB ``document_1.html``, fetch ``document_2..N`` and
    return one ``<html><body>`` splicing every part. Returns ``None`` when the URL
    is not a document_1 EPUB part or the Act is single-document (nothing to add),
    so the caller keeps the original body. Sibling fetches reuse ``client`` (so a
    serial/throttle-safe caller stays serial) with a small spacing delay."""
    m = _DOC1.match(url)
    if not m:
        return None
    parts = [_inner_body(first_body)]
    n = 2
    while n <= _MAX_PARTS:
        sib = f"{m.group('a')}{n}{m.group('b')}{n}{m.group('c')}"
        try:
            r = client.get(sib)
        except Exception:
            break
        if r.status_code != 200 or len(r.content) < 200:
            break
        parts.append(_inner_body(r.content))
        n += 1
        if inter_delay:
            time.sleep(inter_delay)
    if len(parts) == 1:
        return None  # single-document EPUB — nothing to combine
    return ("<html><body>" + "\n".join(parts) + "</body></html>").encode("utf-8")


__all__ = ["combine_au_epub"]
