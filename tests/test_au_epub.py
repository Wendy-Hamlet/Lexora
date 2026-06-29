"""Multi-document AU EPUB combination (offline)."""
from __future__ import annotations

from lexora.extract.au_epub import combine_au_epub
from lexora.extract.html_extractor import assemble_global_text, extract_html

_BASE = ("https://www.legislation.gov.au/C2004A04868/2026-03-14/2026-03-14/"
         "text/original/epub/OEBPS/document_{n}/document_{n}.html")


class _FakeClient:
    """Serves document_2/3 then 404 (end of parts); records the URLs requested."""

    def __init__(self, parts: dict[int, str]):
        self.parts = parts
        self.requested: list[str] = []

    def get(self, url):
        self.requested.append(url)
        for n, html in self.parts.items():
            if f"document_{n}/document_{n}.html" in url:
                return _Resp(200, html.encode("utf-8"))
        return _Resp(404, b"not found")


class _Resp:
    def __init__(self, status_code: int, content: bytes):
        self.status_code = status_code
        self.content = content


def test_combine_splices_all_parts():
    pad = "<p>" + "x " * 120 + "</p>"  # keep each part above the 200-byte stub floor
    doc1 = ("<html><body><p>1 Short title</p>" + pad + "</body></html>").encode("utf-8")
    parts = {
        2: "<html><body><p>477.1 Unauthorised access to data</p>" + pad + "</body></html>",
        3: "<html><body><p>478.1 Unauthorised impairment</p>" + pad + "</body></html>",
    }
    client = _FakeClient(parts)
    combined = combine_au_epub(_BASE.format(n=1), doc1, client, inter_delay=0)
    assert combined is not None
    text = assemble_global_text(extract_html(combined))
    # all three parts present in one body
    assert "Short title" in text
    assert "Unauthorised access to data" in text
    assert "Unauthorised impairment" in text
    # probed document_2, document_3, then document_4 (the 404 that stops it)
    assert any("document_4" in u for u in client.requested)


def test_single_document_epub_returns_normalised_body():
    # A single-document EPUB still returns its (normalised) body, not None — the
    # section-heading fix must apply even when there is nothing to splice.
    client = _FakeClient({})
    out = combine_au_epub(_BASE.format(n=1), b"<html><body><p>x</p></body></html>",
                          client, inter_delay=0)
    assert out is not None
    assert b"x" in out


def test_charsectno_becomes_dotted_heading():
    # "<span class=CharSectno>5</span>  Object" -> "5." so the dotted opener fires.
    doc1 = (b'<html><body><p class="ActHead5">'
            b'<span class="CharSectno">5</span><span>&#xa0; </span>'
            b'<span>Object of this Act</span></p></body></html>')
    client = _FakeClient({})
    out = combine_au_epub(_BASE.format(n=1), doc1, client, inter_delay=0)
    text = assemble_global_text(extract_html(out))
    assert "5. Object of this Act" in text


def test_decimal_section_number_is_kept():
    # Criminal Code style: a single CharSectno "4.1" must stay "4.1" (not gain a
    # second dot) so the AU decimal opener can detect it.
    doc1 = (b'<html><body><p class="ActHead5">'
            b'<span class="CharSectno">4.1</span><span>&#xa0; </span>'
            b'<span>Physical elements</span></p></body></html>')
    out = combine_au_epub(_BASE.format(n=1), doc1, _FakeClient({}), inter_delay=0)
    text = assemble_global_text(extract_html(out))
    assert "4.1 Physical elements" in text
    assert "4.1." not in text


def test_hyphen_section_number_fragments_are_merged():
    # ITAA 1997 style: the number "1-1" is split across three CharSectno spans
    # (1, U+2011, 1) — merge them and fold the non-breaking hyphen to "-".
    doc1 = ('<html><body><p class="ActHead5">'
            '<span class="CharSectno">1</span>'
            '<span class="CharSectno">‑</span>'
            '<span class="CharSectno">1</span>'
            '<span>&#xa0; </span><span>Short title</span></p></body></html>'
            ).encode()
    out = combine_au_epub(_BASE.format(n=1), doc1, _FakeClient({}), inter_delay=0)
    text = assemble_global_text(extract_html(out))
    assert "1-1 Short title" in text


def test_non_epub_url_is_ignored():
    client = _FakeClient({2: "<html><body><p>y</p></body></html>"})
    assert combine_au_epub("https://e.gov/some.pdf", b"%PDF", client) is None
    assert client.requested == []  # never probed
