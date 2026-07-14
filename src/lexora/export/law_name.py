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
