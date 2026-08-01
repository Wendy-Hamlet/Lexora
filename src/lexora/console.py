"""Make a string safe to print on the console this project actually runs on.

Windows consoles here encode with GBK, and a document title is not ours to choose: it
comes out of whatever PDF a portal published. Three separate times this has cost us
something on that difference.

* The live cost meter carried a Yen sign and its thread died on the FIRST emit, so a
  two-hour paid run reported nothing while the unit test -- capturing in UTF-8 -- stayed
  green.
* A Malaysian document called ``WJW24\\uf0220906 Act 710.pdf`` (U+F022 is a PRIVATE-USE
  character, meaningful only to the font that produced it) raised UnicodeEncodeError
  inside ``logging.emit`` on 2026-08-02. That one was survivable -- ``logging`` catches
  handler errors -- but it dumped a full traceback into a log a judge might be reading,
  and the line it was trying to print was lost.

The rule this encodes: **text we CHOOSE is ASCII; text we RECEIVE is made printable.**
Never the other way round -- silently transliterating a law's real name into the data
would be far worse than an ugly log line.
"""
from __future__ import annotations

import sys


def console_safe(text: str, *, placeholder: str = "?") -> str:
    """``text`` with anything the console cannot encode replaced.

    Uses the console's own encoding when it has one (so a UTF-8 terminal keeps the real
    characters) and falls back to ASCII. Only for DISPLAY: never store or export the
    result, because it is lossy by construction.
    """
    if not text:
        return text
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return text.encode(encoding, "replace").decode(encoding, "replace").replace(
            "�", placeholder)
    return text
