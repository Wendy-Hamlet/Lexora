"""Telling a portal's refusal apart from its answer (pure logic, no network).

Discovery used to read "this page lists no instruments" as "this economy has no such
law". Measured on the 2026-07-27 Singapore run, that reading was wrong every single
time: of 67 query renders, 16 were a 923-byte CloudFront "403 ERROR / Request blocked"
page and 20 were a 39-byte empty document — and not one was a genuine empty result set.
Three of the refused queries named known instruments (Computer Misuse Act, Criminal
Procedure Code, Banking Act 1970).

The pages below are the real captured bodies, trimmed.
"""
from __future__ import annotations

import pytest

from lexora.collect.discovery import (
    PageOutcome,
    acquisition_log,
    classify_page,
    page_is_usable,
    reset_acquisition_log,
)

# The exact CloudFront body SSO served, trimmed from 923 bytes.
BLOCKED = (
    '<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">'
    "<html><head>"
    "<title>ERROR: The request could not be satisfied</title>"
    "</head><body><h1>403 ERROR</h1>"
    "<h2>The request could not be satisfied.</h2><hr>Request blocked."
    "</body></html>"
)
# Exactly what Playwright returns when a navigation completes onto nothing (39 bytes).
UNRENDERED = "<html><head></head><body></body></html>"

RESULTS = (
    '<html><body><div class="results">'
    '<a href="/Act/PDPA2012">Personal Data Protection Act 2012</a>'
    "</div></body></html>"
)
EMPTY = (
    '<html><body><div class="results">'
    "<p>Your search did not match any legislation.</p>"
    "</div></body></html>"
)


@pytest.fixture(autouse=True)
def _fresh_log():
    reset_acquisition_log()


@pytest.mark.parametrize(
    ("page", "has_results", "expected"),
    [
        (BLOCKED, False, PageOutcome.blocked),
        (UNRENDERED, False, PageOutcome.unrendered),
        ("", False, PageOutcome.unrendered),
        ("   \n ", False, PageOutcome.unrendered),
        (RESULTS, True, PageOutcome.results),
        (EMPTY, False, PageOutcome.empty),
    ],
)
def test_classification(page, has_results, expected):
    assert classify_page(page, has_results=has_results) is expected


def test_bytes_and_str_agree():
    assert classify_page(BLOCKED.encode(), has_results=False) is PageOutcome.blocked


@pytest.mark.parametrize("body", [
    "<html><head><title>403 Forbidden</title></head><body><h1>403 ERROR</h1></body></html>",
    "<html><body>Access Denied</body></html>",
    "<html><head><title>Attention Required! | Cloudflare</title></head><body>x</body></html>",
    "<html><body>Checking your browser before accessing the site.</body></html>",
])
def test_other_wall_vendors_are_refusals_too(body):
    """Round 2's portals will meet the same walls behind different vendors."""
    assert classify_page(body, has_results=False) is PageOutcome.blocked


def test_a_block_is_not_mistaken_for_an_answer_just_because_it_is_long():
    """Padding must not buy a refusal the benefit of the doubt."""
    padded = BLOCKED + "<!-- " + "x" * 50_000 + " -->"
    assert classify_page(padded, has_results=False) is PageOutcome.blocked


class TestRetryPredicate:
    """Retry a refusal; accept an answer. The old predicate could not tell them apart,
    so it spent three backoff renders (~20 s) on each empty answer AND each block."""

    def test_refusals_are_retried(self):
        assert page_is_usable(BLOCKED) is False
        assert page_is_usable(UNRENDERED) is False

    def test_answers_are_accepted_immediately(self):
        assert page_is_usable(RESULTS) is True
        assert page_is_usable(EMPTY) is True


class TestAcquisitionLog:
    def test_it_counts_what_the_portals_did(self):
        log = acquisition_log()
        log.record(PageOutcome.results, "Personal Data Protection Act 2012")
        log.record(PageOutcome.empty, "Nonexistent Act 1900")
        log.record(PageOutcome.blocked, "Computer Misuse Act")
        log.record(PageOutcome.unrendered, "Criminal Procedure Code")
        assert log.asked == 4
        assert log.refused == 2
        assert log.blocked_queries == ["Computer Misuse Act", "Criminal Procedure Code"]

    def test_an_empty_answer_is_not_a_refusal(self):
        """The whole point: 'no such law' and 'the portal would not say' differ."""
        log = acquisition_log()
        log.record(PageOutcome.empty, "Nonexistent Act 1900")
        assert log.refused == 0
        assert log.blocked_queries == []

    def test_summary_reports_each_outcome(self):
        log = acquisition_log()
        for outcome in (PageOutcome.results, PageOutcome.blocked, PageOutcome.blocked):
            log.record(outcome, "q")
        s = log.summary()
        assert "3 portal quer(ies)" in s and "2 blocked" in s

    def test_reset_starts_a_fresh_tally(self):
        acquisition_log().record(PageOutcome.blocked, "q")
        assert reset_acquisition_log().asked == 0
