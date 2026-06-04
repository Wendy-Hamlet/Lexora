"""Tests for the HTML extractor."""
from __future__ import annotations

from lexora.extract.html_extractor import (
    BLOCK_SEPARATOR,
    assemble_global_text,
    extract_html,
)

HTML = b"""
<html><head><title>x</title><style>p{color:red}</style></head>
<body>
  <nav>menu we don't want</nav>
  <h1 id="title">Personal Data Protection Act</h1>
  <p id="s26">26. An organisation must not transfer any personal data overseas.</p>
  <p>13. An organisation must not collect personal data without consent.</p>
  <script>tracker()</script>
</body></html>
"""


def test_drops_script_style_nav():
    blocks = extract_html(HTML)
    joined = " ".join(b.text for b in blocks)
    assert "tracker" not in joined
    assert "color:red" not in joined
    assert "menu we don't want" not in joined


def test_blocks_have_text_and_anchors():
    blocks = extract_html(HTML)
    texts = [b.text for b in blocks]
    assert any("must not transfer" in t for t in texts)
    assert any("without consent" in t for t in texts)
    # the element with an id keeps an id-based anchor
    s26 = next(b for b in blocks if "must not transfer" in b.text)
    assert s26.dom_anchor.startswith("#s26")


def test_offsets_reconstruct_global_text():
    blocks = extract_html(HTML)
    global_text = assemble_global_text(blocks)
    for b in blocks:
        assert global_text[b.char_start:b.char_end] == b.text
    # separator length is consistent
    if len(blocks) >= 2:
        gap = blocks[1].char_start - blocks[0].char_end
        assert gap == len(BLOCK_SEPARATOR)


def test_whitespace_is_collapsed():
    blocks = extract_html(b"<html><body><p>a   b\n\tc</p></body></html>")
    assert blocks[0].text == "a b c"
