"""Instrument lifecycle detection — the official *enforced-only* discipline.

The RDTII inventory must carry only **enforced** measures: pending drafts and
repealed/spent instruments are out of scope (internal guide p.8). This module
decides one instrument's :class:`~lexora.models.source.InstrumentStatus` from the
signals available, in strict reliability order:

1. **Portal structured channel** — authoritative. The AU register's ``isInForce``
   flag / ``status`` string is a machine-readable verdict; nothing beats it.
2. **Title-level markers** — strong. Portals stamp lifecycle onto the title
   itself (SG SSO renders ``Personal Data Protection Act 2012 (Repealed)``).
3. **Body-text markers** — *deliberately conservative*. Free-text repeal
   detection is brittle in exactly the way the 6.1/6.4 keyword work exposed: an
   in-force Act that *repeals another* law is full of "is repealed", so a naive
   body scan would wrongly retire it. We therefore only honour a whole-instrument
   header signal ("This Act is repealed", an "(Repealed)" stamp) and explicit
   draft markers ("Exposure Draft", "[Draft for consultation]"), never a bare
   "repealed" anywhere in the text.

Design invariants (mirror :mod:`lexora.classify.boundaries`):
- **Positive-signal only.** Absence of an in-force confirmation yields
  ``unknown``, NOT ``repealed`` — :func:`is_enforced` treats ``unknown`` as
  enforced, so a valid law is never dropped on uncertainty. We only ever act on a
  clear retire/draft signal.
- **No finding, only judging.** This classifies an already-discovered
  instrument; it never searches for or ranks instruments.
"""
from __future__ import annotations

import re

from lexora.models.source import InstrumentStatus

# --- portal `status` strings (AU register and similar) -----------------------
# Matched case-insensitively as substrings of the portal's own status label.
_PORTAL_REPEALED = ("repeal", "ceased", "spent", "expired", "revoked", "superseded")
_PORTAL_DRAFT = ("not yet in force", "notinforce", "not in force", "before parliament")

# --- title-level markers (strong) --------------------------------------------
# SG SSO and others stamp lifecycle in parentheses on the rendered title.
_TITLE_REPEALED = re.compile(r"\(\s*(?:repealed|spent|ceased|revoked)\b", re.I)
_TITLE_DRAFT = re.compile(
    r"\b(?:exposure draft|draft (?:bill|for consultation))\b"
    r"|\bbill\s+\d{1,4}\s+of\s+\d{4}\b"  # "Bill 12 of 2025" — a bill, not an Act
    r"|\(\s*draft\b",
    re.I,
)

# --- body-text markers (conservative; whole-instrument header only) ----------
# Only a header that retires THIS instrument as a whole. Anchored to a leading
# "This Act/Regulation/Ordinance" so an Act that repeals *another* law (whose body
# says "the X Act is repealed") is not itself mistaken for retired.
_TEXT_REPEALED = re.compile(
    r"\bthis\s+(?:act|regulation|ordinance|decree|law)\s+(?:is|has been|was)\s+repealed\b",
    re.I,
)
_TEXT_DRAFT = re.compile(
    r"\bexposure draft\b|\[\s*draft[^\]]*\]|\bfor public consultation\b",
    re.I,
)
# How much leading text to scan for a header repeal/draft note (the stamp lives at
# the top of a reprint/compilation, not buried in operative provisions).
_TEXT_HEAD_CHARS = 1500


def detect_status(
    *,
    in_force: bool | None = None,
    portal_status: str = "",
    title: str = "",
    text: str = "",
) -> InstrumentStatus:
    """Classify an instrument's lifecycle status from the available signals.

    Reliability order: portal status string > ``in_force`` flag > title marker >
    head-of-text marker. Returns ``unknown`` when no signal fires (the caller's
    enforced-only filter then keeps the instrument).
    """
    label = (portal_status or "").lower()
    if label:
        if any(m in label for m in _PORTAL_REPEALED):
            return InstrumentStatus.repealed
        if any(m in label for m in _PORTAL_DRAFT):
            return InstrumentStatus.draft
        if "in force" in label or "inforce" in label:
            return InstrumentStatus.in_force

    if in_force is True:
        return InstrumentStatus.in_force
    # in_force is False alone is ambiguous (future commencement vs retired) — fall
    # through to the text/title signals rather than guessing.

    if title:
        if _TITLE_REPEALED.search(title):
            return InstrumentStatus.repealed
        if _TITLE_DRAFT.search(title):
            return InstrumentStatus.draft

    if text:
        head = text[:_TEXT_HEAD_CHARS]
        if _TEXT_REPEALED.search(head):
            return InstrumentStatus.repealed
        if _TEXT_DRAFT.search(head):
            return InstrumentStatus.draft

    return InstrumentStatus.unknown


def is_enforced(status: InstrumentStatus) -> bool:
    """Whether an instrument passes the enforced-only filter.

    ``unknown`` counts as enforced: we drop only on a positive repealed/draft
    signal, never on the absence of an in-force confirmation."""
    return status not in (InstrumentStatus.repealed, InstrumentStatus.draft)


def merge_status(primary: InstrumentStatus, secondary: InstrumentStatus) -> InstrumentStatus:
    """Combine a portal-channel verdict (``primary``) with a text/title verdict.

    A definite ``primary`` wins (it is the authoritative channel). When the portal
    said ``unknown`` (no connector), defer to the ``secondary`` text/title signal.
    A ``primary`` ``in_force`` is NOT overridden by a text marker — the register is
    more current than a stale "(Repealed)" stamp baked into an archived PDF."""
    if primary is not InstrumentStatus.unknown:
        return primary
    return secondary


__all__ = ["detect_status", "is_enforced", "merge_status"]
