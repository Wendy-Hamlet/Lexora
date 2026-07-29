"""Law-name normalisation for the submission outputs.

The crawled document title is whatever the portal put on the page. Singapore's
SSO renders the statute inside a full application shell, so the visible text we
capture runs the real title straight into the page furniture:

    Cybersecurity Act 2018 Current version as at 14 Jul 2026 Part 3 PROVIDER-OWNED
    CRITICAL INFORMATION INFRASTRUCTURE ... Actions Download PDF (434.4 KB) Add to
    My Collections Amend

"Law Name" is the second column a reviewer reads, so the title has to be the
statute and nothing else. The real name always comes first and the shell always
opens with SSO's "Current version as at <date>" banner, which gives an exact cut
point rather than a length heuristic.

This normalises the *presentation* only: no offsets, quotes, indicator verdicts
or rationales depend on it. Genuinely long titles (Australia's amendment Acts,
Malaysia's sectoral Codes of Practice, PDPC advisories) are left untouched — the
cut is anchored to the boilerplate, not to a character budget.
"""
from __future__ import annotations

import re

# SSO's "current version" banner — the first token of the page shell.
_SSO_SHELL = re.compile(r"\s+Current version as at\b.*$", re.IGNORECASE | re.DOTALL)


def normalize_law_name(title: str | None) -> str:
    """Return the statute title with portal page-furniture stripped."""
    if not title:
        return ""
    name = _SSO_SHELL.sub("", str(title))
    return re.sub(r"\s+", " ", name).strip()


# Malaysia's AGC portal serves each Act as a file whose NAME is whatever the uploader
# called it, and that name is what discovery captures as the title. Eight of the eleven
# Malaysian laws tagged NEW in the round-1 submission carried one in the Law Name column:
#
#     Act 706 ori.pdf
#     Act 712_online 1 Sept. 2024-final (5.9.2024).pdf
#     DRAF KEDUA AKTA 701 (final)(KU) (1).pdf
#     20150604_A1487_BI_Act A1487.pdf
#
# Law Name is the column that says WHICH instrument the row is about, on the rows that
# carry the NEW claim. A filename there reads like a directory listing.
# Every rule here DELETES a bounded token. None of them is anchored ".*$": a greedy
# "drop everything after this word" rule turned "Mei 2019 Reprint Online Act 678.pdf" into
# "Mei 2019", throwing away the only part that identified the Act. A ragged name is a
# cosmetic problem; a name with the instrument removed from it is a wrong answer.
_FILE_EXT = re.compile(r"\.(?:pdf|docx?|html?|txt)\s*$", re.I)
_UPLOAD_NOISE = re.compile(
    r"""(?ix)
      \s*\((?:\d+|final|ku|bi|dalam\s+talian[^)]*|reprint[^)]*|[\d.\s/]+)\)  # (1) (final) (KU)
    | \s*_(?:unlocked|online|final|ori|bi|ku)\b
    | \s*\b(?:ori|final|unlocked)\s*$
    | \s*\bas\s+at\s+\d{1,2}\s+\w+\s+\d{4}\s*$                               # "as at 1 Jan 2024"
    | \s*\b\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\s*$                             # "23.11.2021"
    | ^\s*\d{1,2}\.\s+                                                       # "09. Act A1441"
    | ^\s*\d{8}_                                                             # "20150604_..."
    """,
)

# A name that already ENDS like a law title is a law title, whatever noise it contains.
# Without this, Singapore's "Online Safety (Relief and Accountability) Act 2025" was
# flagged as a file name because it happens to contain the word "Online" — and a title
# wrongly flagged is a title this module is licensed to overwrite from the document text.
_ENDS_LIKE_A_LAW = re.compile(
    r"(?i)\b(?:act|regulations?|rules|order|code|ordinance|enactment|decree|bill|"
    r"guidelines?|advisory|notice|standard)\b[\s,]*(?:\(?(?:19|20)\d{2}\)?)?\s*$"
)

# The Act's own masthead: an "Act <n>" line, then up to four lines whose join ends in
# "ACT <year>". The anchor is load-bearing and deliberately strict. Without it, every
# Malaysian reprint's copyright page — "UNDER THE AUTHORITY OF THE REVISION OF LAWS ACT
# 1968" — yields the confident, wrong title "Revision of Laws Act 1968". Recovering 40% of
# titles safely beats recovering 90% with forgeries in them.
_ACT_NUMBER_LINE = re.compile(r"^Act\s+[A-Z]?\d+[A-Z]?$", re.I)
_TITLE_TAIL = re.compile(r"\b(?:ACT|ENACTMENT|ORDINANCE)\s+(?:19|20)\d{2}$", re.I)


def looks_like_a_filename(title: str | None) -> bool:
    """True when the portal handed us a file name rather than a law name.

    A file extension always convicts. Otherwise upload noise only convicts a name that
    does not already end like a law title (see ``_ENDS_LIKE_A_LAW``)."""
    if not title:
        return False
    t = str(title).strip()
    if _FILE_EXT.search(t):
        return True
    return bool(_UPLOAD_NOISE.search(t)) and not _ENDS_LIKE_A_LAW.search(t)


def statute_title_from_text(document_text: str, max_lines: int = 24) -> str:
    """The Act's own title as printed on its masthead, or ``""`` if not clearly there.

    Never guesses: see ``_ACT_NUMBER_LINE`` for why the anchor cannot be relaxed.
    """
    lines = [ln.strip() for ln in (document_text or "").split("\n") if ln.strip()]
    for i, line in enumerate(lines[:max_lines]):
        if not _ACT_NUMBER_LINE.fullmatch(line):
            continue
        parts: list[str] = []
        for nxt in lines[i + 1 : i + 5]:
            if nxt != nxt.upper() or not any(c.isalpha() for c in nxt):
                break  # title lines are set in caps; stop at the first that is not
            parts.append(nxt)
            joined = re.sub(r"\s+", " ", " ".join(parts))
            if _TITLE_TAIL.search(joined):
                return joined.title()
    return ""


def resolve_law_name(title: str | None, document_text: str = "") -> str:
    """The best available name for the instrument.

    A portal title that already reads like a law name is kept untouched — this only ever
    engages when the portal gave us a file name. Then the Act's own masthead wins if it
    states one; otherwise the file name is *cleaned*, never replaced, so the result cannot
    name an instrument the document does not.
    """
    name = normalize_law_name(title)
    if not looks_like_a_filename(name):
        return name
    from_text = statute_title_from_text(document_text)
    if from_text:
        return from_text
    cleaned = _FILE_EXT.sub("", name)
    while True:
        stripped = _UPLOAD_NOISE.sub("", cleaned).strip(" -_,.")
        if stripped == cleaned.strip(" -_,."):
            break
        cleaned = stripped
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_,.")
    return cleaned or name
