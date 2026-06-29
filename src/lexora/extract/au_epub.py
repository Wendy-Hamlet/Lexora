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

Section-heading normalisation. AU numbers its operative sections in *spaced*
style ("5  Object of this Act"), but the HTML extractor collapses runs of
whitespace to a single space, so the structure parser's spaced opener (which
requires 2+ spaces, to avoid matching every "1 apple") never fires and every
section's number is lost — clauses then mis-label as the front commencement
table's row numbers. The EPUB tags each real section number explicitly with
``<span class="CharSectno">N</span>``; we append a dot to it ("N.") so the
parser's *dotted* opener detects the genuine heading. This runs for every part
(so single-document EPUBs are normalised too — hence a single-doc EPUB returns
its normalised body, not ``None``).
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
    # Make every real section number ("<span class=CharSectno>5</span>") parse as a
    # dotted heading ("5.") — the only robust section signal once whitespace is
    # collapsed (see module docstring). TOC/cross-reference numbers are plain
    # <span> without this class, so genuine headings alone are rewritten.
    for span in soup.find_all("span", class_="CharSectno"):
        num = span.get_text().strip()
        # Skip numbers that already carry a dot — a trailing-dot heading ("5.") is
        # done, and the Criminal Code's decimal style ("477.1") must NOT gain a
        # second dot (it would make the dotted opener mis-read it as section "1").
        if num and "." not in num:
            span.string = f"{num}."
    body = soup.body or soup
    return "".join(str(x) for x in body.contents)


def combine_au_epub(
    url: str, first_body: bytes, client: httpx.Client, *, inter_delay: float = 0.3
) -> bytes | None:
    """If ``url`` is an AU EPUB ``document_1.html``, fetch ``document_2..N`` and
    return one ``<html><body>`` splicing every part, with section headings
    normalised (see module docstring). Returns ``None`` only when the URL is not a
    document_1 EPUB part, so the caller keeps the original body; a *single*-document
    EPUB still returns its normalised body (the heading fix must apply there too).
    Sibling fetches reuse ``client`` (so a serial/throttle-safe caller stays serial)
    with a small spacing delay."""
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
    return ("<html><body>" + "\n".join(parts) + "</body></html>").encode("utf-8")


__all__ = ["combine_au_epub"]
