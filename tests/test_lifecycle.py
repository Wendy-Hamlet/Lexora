"""Instrument lifecycle detection — the enforced-only discipline (classify/lifecycle.py).

Pins the reliability order (portal channel > in_force flag > title marker > head
text), the positive-signal-only invariant (unknown stays enforced), and that a
law which *repeals another* law is not itself mistaken for retired.
"""
from __future__ import annotations

from lexora.classify.lifecycle import detect_status, is_enforced, merge_status
from lexora.models.source import InstrumentStatus


# --- portal channel (authoritative) ------------------------------------------
def test_portal_status_repealed():
    assert detect_status(portal_status="Repealed") is InstrumentStatus.repealed
    assert detect_status(portal_status="Ceased") is InstrumentStatus.repealed
    assert detect_status(portal_status="Spent") is InstrumentStatus.repealed


def test_in_force_flag():
    assert detect_status(in_force=True) is InstrumentStatus.in_force
    # in_force False alone is ambiguous (future commencement vs retired) -> unknown.
    assert detect_status(in_force=False) is InstrumentStatus.unknown


def test_portal_status_beats_flag():
    # The explicit status label wins over the boolean flag.
    assert detect_status(in_force=True, portal_status="Repealed") is InstrumentStatus.repealed


# --- title markers (strong) --------------------------------------------------
def test_title_repealed_stamp():
    title = "Personal Data Protection Act 2012 (Repealed)"
    assert detect_status(title=title) is InstrumentStatus.repealed


def test_title_draft_bill():
    assert detect_status(title="Data Protection Bill 12 of 2025") is InstrumentStatus.draft
    assert detect_status(title="Privacy Reform (Exposure Draft)") is InstrumentStatus.draft


def test_title_in_force_act_is_unknown_not_forced():
    # A plain in-force Act title carries no stamp -> unknown (kept by the filter).
    assert detect_status(title="Personal Data Protection Act 2012") is InstrumentStatus.unknown


# --- body text (conservative) ------------------------------------------------
def test_text_self_repeal_header():
    text = "This Act is repealed with effect from 1 January 2026.\n\nFormerly in force..."
    assert detect_status(text=text) is InstrumentStatus.repealed


def test_text_repealing_another_law_is_not_self_retired():
    # An IN-FORCE Act whose body repeals a DIFFERENT law must not be flagged.
    text = (
        "An Act to provide for the protection of personal data.\n"
        "65. The Spam Control Act 2007 is repealed.\n"
        "66. The Junk Communications Act is repealed and replaced by this Act."
    )
    assert detect_status(text=text) is InstrumentStatus.unknown


def test_body_repeal_must_be_in_head():
    # A self-repeal note buried far past the header window is not honoured
    # (the conservative scan only trusts a top-of-document stamp).
    text = "x" * 4000 + " This Act is repealed."
    assert detect_status(text=text) is InstrumentStatus.unknown


# --- is_enforced / merge -----------------------------------------------------
def test_is_enforced_policy():
    assert is_enforced(InstrumentStatus.in_force) is True
    assert is_enforced(InstrumentStatus.unknown) is True  # never drop on uncertainty
    assert is_enforced(InstrumentStatus.repealed) is False
    assert is_enforced(InstrumentStatus.draft) is False


def test_merge_prefers_definite_portal_verdict():
    assert merge_status(InstrumentStatus.in_force, InstrumentStatus.repealed) is InstrumentStatus.in_force
    # No portal verdict -> defer to the text/title signal.
    assert merge_status(InstrumentStatus.unknown, InstrumentStatus.repealed) is InstrumentStatus.repealed
    assert merge_status(InstrumentStatus.unknown, InstrumentStatus.unknown) is InstrumentStatus.unknown
