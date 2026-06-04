"""Format-general structure parser tests.

Covers the two common-law numbering styles (dotted SG/MY, spaced AU), the
year-false-positive rejection that previously mis-parsed AU Acts, and style
auto-selection.
"""
from __future__ import annotations

from lexora.extract.html_extractor import HtmlBlock
from lexora.structure.legal_parser import _detect_boundaries, parse_structure_html


def _secs(text: str) -> list[str]:
    return [sec for _, sec, _ in _detect_boundaries(text)]


def test_detect_dotted_style_sg_my():
    text = "1. Short title\n\n26.—(1) An organisation must not transfer personal data."
    boundaries = _detect_boundaries(text)
    secs = [sec for _, sec, _ in boundaries]
    assert "1" in secs and "26" in secs
    # the inline subsection of 26.—(1) is captured
    assert ("26", "1") in [(sec, sub) for _, sec, sub in boundaries]


def test_detect_spaced_style_au():
    text = "1  Short title\n\n2A  Objects of this Act\n\n13  Interference with privacy"
    assert _secs(text) == ["1", "2A", "13"]


def test_year_with_dot_is_not_a_section():
    text = "2010.\n\nsome recital text\n\n5. Real provision body here"
    secs = _secs(text)
    assert "2010" not in secs
    assert "5" in secs


def test_spaced_style_wins_over_stray_year_dot():
    # AU-like doc: spaced sections dominate; a stray "2012." year-dot is dropped.
    text = "1  Short title\n\n2  Commencement\n\n3  Application\n\n2012.\n\n4  Crown"
    assert _secs(text) == ["1", "2", "3", "4"]


def test_parse_structure_html_spaced_emits_verbatim_clauses():
    text = "13  Interference with privacy\n\n14  Australian Privacy Principles apply"
    block = HtmlBlock(dom_anchor="#s", text=text, char_start=0, char_end=len(text))
    clauses = parse_structure_html("doc", [block])
    paths = [c.structural_path for c in clauses]
    assert "Section 13" in paths and "Section 14" in paths
    # spans are verbatim slices of the source text
    for c in clauses:
        assert text[c.span.char_start:c.span.char_end] == c.span.text
