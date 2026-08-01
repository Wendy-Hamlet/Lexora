"""Printing a document's own title must not be able to break the run's log."""
from __future__ import annotations

import io
import sys

from lexora.console import console_safe


class _GbkStdout(io.StringIO):
    encoding = "gbk"


class _Utf8Stdout(io.StringIO):
    encoding = "utf-8"


def test_a_private_use_character_does_not_reach_a_gbk_console(monkeypatch):
    """Verbatim from the 2026-08-02 Malaysian run: a portal published a PDF named
    'WJW24\\uf0220906 Act 710.pdf'. U+F022 is PRIVATE USE -- it means whatever the font
    that produced it says it means -- and it raised UnicodeEncodeError inside
    logging.emit, dumping a traceback into the log and losing the line. The document had
    mapped fine; only the report of it broke."""
    monkeypatch.setattr(sys, "stdout", _GbkStdout())
    out = console_safe("WJW240906 Act 710.pdf")
    out.encode("gbk")                       # the whole point: this must not raise
    assert out.startswith("WJW24")
    assert out.endswith("Act 710.pdf")
    assert "" not in out


def test_a_console_that_can_take_it_keeps_the_real_characters(monkeypatch):
    """Lossy is for display and only where display forces it. A UTF-8 terminal shows the
    title the portal actually published."""
    monkeypatch.setattr(sys, "stdout", _Utf8Stdout())
    assert console_safe("AKTA PERLINDUNGAN DATA PERIBADI — 2024") == \
        "AKTA PERLINDUNGAN DATA PERIBADI — 2024"


def test_plain_text_is_returned_unchanged(monkeypatch):
    monkeypatch.setattr(sys, "stdout", _GbkStdout())
    assert console_safe("PERSONAL DATA PROTECTION ACT 2010") == \
        "PERSONAL DATA PROTECTION ACT 2010"
    assert console_safe("") == ""


def test_a_stream_with_no_encoding_attribute_falls_back_to_ascii(monkeypatch):
    """pytest's own capture and a redirected file both do this."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    out = console_safe("café  statute")
    out.encode("ascii")
    assert "statute" in out
