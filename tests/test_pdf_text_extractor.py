"""Unit tests for the PDF text-extractor cleaners.

These operate on per-page text (``list[str]``) before global char offsets are
assigned, so they are pure and verbatim-safe: anything they remove never enters a
CanonicalSpan. Each guards against absorbing a running header/footer, a
navigational page header, or document-final trailing matter into a provision.
"""
from __future__ import annotations

from lexora.extract.pdf_text_extractor import (
    _is_nav_header,
    _strip_nav_headers,
    _strip_running_lines,
    _strip_trailing_matter,
)

# --- cross-page running header/footer ----------------------------------------

def test_running_footer_repeated_across_pages_is_removed():
    bodies = ["consent must be obtained.", "data must be protected.", "access may be refused.",
              "breach must be notified.", "records must be accurate.", "transfers are restricted."]
    pages = [f"Informal Consolidation\n{b}\nAct 2012\n{i}" for i, b in enumerate(bodies, 1)]
    out = _strip_running_lines(pages)
    assert all("Informal Consolidation" not in p for p in out)
    assert all(b in p for b, p in zip(bodies, out, strict=True))


def test_operative_line_is_not_mistaken_for_a_running_line():
    # A distinctive sentence that appears on only one page must survive.
    pages = ["header line repeats\nunique operative clause text here." for _ in range(5)]
    pages.append("header line repeats\na different unique provision body.")
    out = _strip_running_lines(pages)
    assert "a different unique provision body." in out[-1]


# --- navigational running header (legislation.gov.au) ------------------------

def test_nav_header_signatures_detected():
    assert _is_nav_header("Section 26WR")
    assert _is_nav_header("Clause 8")
    assert _is_nav_header("Notification of eligible data breaches  Part IIIC")
    assert _is_nav_header("Schedule 1  Australian Privacy Principles")
    assert _is_nav_header("Division 1  Introduction")


def test_real_divisional_headings_are_not_nav_headers():
    # SG's heading is ALL-CAPS; AU's carries an em dash. The parser needs both, so
    # neither may be treated as a navigational header.
    assert not _is_nav_header("PART 4")
    assert not _is_nav_header("Part 3—Dealing with personal information")
    assert not _is_nav_header("Schedule 1—Australian Privacy Principles")
    # a prose cross-reference (single spaces) is not a header cell
    assert not _is_nav_header("a data user under Division 2 of Part II")


def test_strip_nav_headers_keeps_operative_text():
    page = ("Notification of eligible data breaches  Part IIIC\n"
            "Section 26WR\n"
            "(4) The statement must set out the contact details.")
    out = _strip_nav_headers([page])[0]
    assert "Part IIIC" not in out
    assert "Section 26WR" not in out
    assert "(4) The statement must set out the contact details." in out


# --- document-final trailing matter ------------------------------------------

def test_trailing_imprint_and_amendment_table_removed():
    pages = [
        "5. An operative provision ending properly.",
        "LAWS OF MALAYSIA\nAct 709\nLIST OF AMENDMENTS\nAmending law\n- NIL -",
        "Act 709\nLIST OF SECTIONS AMENDED\nSection\n- NIL -\n"
        "DICETAK OLEH\nPERCETAKAN NASIONAL MALAYSIA BERHAD\n"
        "BAGI PIHAK DAN DENGAN PERINTAH KERAJAAN MALAYSIA",
    ]
    out = _strip_trailing_matter(pages)
    assert out[0] == "5. An operative provision ending properly."
    assert out[1] == ""  # pure end-matter page blanked
    assert out[2] == ""
    for marker in ("LIST OF AMENDMENTS", "LIST OF SECTIONS AMENDED", "DICETAK"):
        assert all(marker not in p for p in out)


def test_trailing_matter_keeps_operative_prose_before_the_marker():
    # If the closing imprint shares a page with operative text, only the imprint tail
    # is cut; the sentence above it (which ends in a full stop) is kept.
    pages = ["earlier page.", "The final operative sentence ends here.\nPRINTED BY the Government Printer"]
    out = _strip_trailing_matter(pages)
    assert out[-1] == "The final operative sentence ends here."


def test_no_trailing_matter_is_a_no_op():
    pages = ["one.", "two.", "three."]
    assert _strip_trailing_matter(list(pages)) == pages
