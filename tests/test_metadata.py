"""Generic document-metadata extractor (Law Number / Last Amended) + its source-check.

Offline: a fake LLM client stands in for the endpoint so these pin the contract
(inert default, year/number source-verification = fabrication guard, graceful
fallback) with no network.
"""
from __future__ import annotations

from lexora.cite.metadata import MetadataExtractor, make_metadata_extractor

_TEXT = (
    "LAWS OF MALAYSIA\nACT 709\nPERSONAL DATA PROTECTION ACT 2010\n"
    "Date of Royal Assent: 2 June 2010\n"
    "... [body] ...\nEndnote: this reprint incorporates amendments in force as at 2024.\n"
)


class _FakeClient:
    def __init__(self, payload: dict) -> None:
        self._p = payload
        self.calls = 0

    def chat(self, system, user, json_schema=None):  # noqa: ANN001
        self.calls += 1
        return self._p


class _BoomClient:
    def chat(self, system, user, json_schema=None):  # noqa: ANN001
        raise RuntimeError("backend down")


def test_inert_extractor_returns_blanks():
    assert make_metadata_extractor(use_llm=False)._client is None
    assert MetadataExtractor(client=None).extract(_TEXT, "Malaysia", "PDPA 2010") == ("", "", "")


def test_extracts_verified_number_and_year():
    gen = MetadataExtractor(_FakeClient({"law_number": "Act 709", "last_amended": "2024"}))
    assert gen.extract(_TEXT, "Malaysia", "Personal Data Protection Act 2010") == ("2024", "Act 709", "")
    assert gen.extracted == 1 and gen.rejected == 0


def test_year_pulled_from_prose_when_present_in_text():
    gen = MetadataExtractor(_FakeClient({"law_number": "Act 709", "last_amended": "as at 2024"}))
    assert gen.extract(_TEXT, "Malaysia", "PDPA")[0] == "2024"


def test_fabricated_number_is_rejected():
    # "Act 999" is not printed in the document -> dropped, not shipped.
    gen = MetadataExtractor(_FakeClient({"law_number": "Act 999", "last_amended": "2024"}))
    last_amended, law_number, _ = gen.extract(_TEXT, "Malaysia", "PDPA")
    assert law_number == "" and last_amended == "2024"
    assert gen.rejected == 1


def test_fabricated_year_is_rejected():
    gen = MetadataExtractor(_FakeClient({"law_number": "Act 709", "last_amended": "2099"}))
    last_amended, law_number, _ = gen.extract(_TEXT, "Malaysia", "PDPA")
    assert last_amended == "" and law_number == "Act 709"
    assert gen.rejected == 1


def test_multi_token_number_verifies_each_digit_run():
    text = "Privacy Act 1988 ... No. 119 of 1988 ... compiled 14 October 2024."
    gen = MetadataExtractor(_FakeClient({"law_number": "No. 119 of 1988", "last_amended": "2024"}))
    assert gen.extract(text, "Australia", "Privacy Act 1988") == ("2024", "No. 119 of 1988", "")


def test_sentinel_field_falls_through_to_blank():
    # The model could not determine last_amended from the text and said so via the
    # sentinel; it must come back blank so the caller falls back to the anchor —
    # NOT be treated as a value and NOT counted as a source-check rejection.
    from lexora.cite.metadata import SENTINEL_NONE

    gen = MetadataExtractor(
        _FakeClient({"law_number": "Act 709", "last_amended": SENTINEL_NONE})
    )
    last_amended, law_number, _ = gen.extract(_TEXT, "Malaysia", "PDPA")
    assert last_amended == "" and law_number == "Act 709"
    assert gen.rejected == 0  # a sentinel is "absent", not a fabrication


def test_own_knowledge_goes_to_review_note_never_to_answer():
    # The model's background-knowledge value must surface ONLY in the review note,
    # tagged, and must never become the law_number/last_amended answer.
    gen = MetadataExtractor(_FakeClient({
        "law_number": "<<NONE>>",
        "last_amended": "<<NONE>>",
        "notes": "law_number(by knowledge)=Act 709",
    }))
    last_amended, law_number, review_note = gen.extract(_TEXT, "Malaysia", "PDPA")
    assert last_amended == "" and law_number == ""          # sentinel -> blank answer
    assert "own knowledge" in review_note.lower()
    assert "Act 709" in review_note                          # preserved for analysts
    assert "NOT used as answer" in review_note


def test_block_mode_locates_history_block_not_blind_tail():
    from lexora.cite.metadata import _document_window
    # Long doc: masthead, a big body, then a located history section near the end.
    head = "PERSONAL DATA PROTECTION ACT 2012\n2020 REVISED EDITION\n"
    body = "section text ... " * 1000  # pushes the history block past a blind tail
    history = "LEGISLATIVE HISTORY\n1. Act 26 of 2012 - the enacting Act.\n"
    text = head + body + history + ("endnote padding " * 50)
    win = _document_window(text, "block")
    assert "Act 26 of 2012" in win          # the clipped-by-fixed entry is now sent
    assert win.startswith("PERSONAL DATA")   # masthead still included
    assert len(win) < len(text)             # but it is NOT the whole document


def test_block_mode_falls_back_to_full_when_no_history_marker():
    from lexora.cite.metadata import _document_window
    text = "MALAYSIA ACT 709\n" + ("a clean reprint of statutory provisions only. " * 500)
    assert len(text) > 6000  # long enough to trigger windowing
    # Hybrid: no recognizable history heading -> send the WHOLE document.
    assert _document_window(text, "block") == text


def test_backend_error_returns_blanks_and_counts():
    gen = MetadataExtractor(_BoomClient())
    assert gen.extract(_TEXT, "Malaysia", "PDPA") == ("", "", "")
    assert gen.error_count == 1 and gen.last_error_type == "RuntimeError"


def test_resolve_doc_metadata_precedence():
    """Structured portal value wins; the LLM extractor fills only the gap. (The
    curated anchor tier was scrapped 2026-06-21 — see git history.)"""
    from datetime import datetime, timezone

    from lexora.models.source import LegalSystem, RawDocument, SourceProfile, SourceType
    from lexora.pipeline import _resolve_doc_metadata

    profile = SourceProfile(
        jurisdiction="Malaysia", iso_code="MY", primary_language="en", legal_system=LegalSystem.common,
    )
    doc = RawDocument(
        document_id="d", source_url="https://e.gov/x", retrieval_timestamp=datetime.now(timezone.utc),
        http_status=200, sha256="sha256:x", content_type="application/pdf", bytes_path="/tmp/x",
        portal_name="P", jurisdiction="MY", source_type=SourceType.primary,
        title="Personal Data Protection Act 2010", law_number="Act 709", last_amended="",  # connector gave number only
    )
    extractor = MetadataExtractor(_FakeClient({"law_number": "Act 999", "last_amended": "2024"}))
    last_amended, law_number, _note = _resolve_doc_metadata(doc, _TEXT, profile, extractor)
    assert law_number == "Act 709"   # structured portal value, NOT the LLM's "Act 999"
    assert last_amended == "2024"    # LLM extractor filled the gap the portal left blank


def test_resolve_doc_metadata_sentinel_yields_blank_no_anchor():
    """With the anchor scrapped, a sentinelled field stays blank rather than being
    backfilled — the model declined and there is no curated fallback to inject a
    (possibly stale) value."""
    from datetime import datetime, timezone

    from lexora.models.source import LegalSystem, RawDocument, SourceProfile, SourceType
    from lexora.pipeline import _resolve_doc_metadata

    profile = SourceProfile(
        jurisdiction="Singapore", iso_code="SG", primary_language="en", legal_system=LegalSystem.common,
    )
    doc = RawDocument(
        document_id="d", source_url="https://e.gov/x", retrieval_timestamp=datetime.now(timezone.utc),
        http_status=200, sha256="sha256:x", content_type="application/pdf", bytes_path="/tmp/x",
        portal_name="P", jurisdiction="SG", source_type=SourceType.primary,
        title="Personal Data Protection Act 2012", law_number="", last_amended="",
    )
    text = "Personal Data Protection Act 2012\n... body ..."
    extractor = MetadataExtractor(_FakeClient({"law_number": "<<NONE>>", "last_amended": "<<NONE>>"}))
    last_amended, law_number, _note = _resolve_doc_metadata(doc, text, profile, extractor)
    assert last_amended == "" and law_number == ""


# --- the principal Act vs one of its amendments --------------------------------
#
# Singapore's SSO prints an ORDERED legislative history whose first entry is the
# principal Act. The masthead does NOT carry the number: it wraps the title over
# two lines ("Personal Data Protection" / "Act 2012"), and the page-furniture
# cleaner used to drop only the long half — leaving "Act 2012", a fragment that
# reads exactly like a law number and which the model duly copied.

_SSO_PDPA = (
    "THE STATUTES OF THE REPUBLIC OF SINGAPORE\n"
    "PERSONAL DATA PROTECTION\nACT 2012\n2020 REVISED EDITION\n"
    "... [body] ...\n"
    "LEGISLATIVE HISTORY\n"
    "This Legislative History is a service provided by the Law Revision Commission\n"
    "on a best-efforts basis. It is not part of the Act.\n"
    "1. Act 26 of 2012 - Personal Data Protection Act 2012\n"
    "Bill : 24/2012\n"
    "2. Act 40 of 2020 - Personal Data Protection (Amendment) Act 2020\n"
)

# This one was RENAMED, so its history opens with the OLD title.
_SSO_CDCSA = (
    "THE STATUTES OF THE REPUBLIC OF SINGAPORE\n"
    "CHILD DEVELOPMENT CO-SAVINGS\nACT 2001\n2020 REVISED EDITION\n"
    "... [body] ...\n"
    "LEGISLATIVE HISTORY\n"
    "(Formerly known as the Children Development Co-Savings Act (2002 Ed.))\n"
    "This Legislative History is a service provided by the Law Revision Commission\n"
    "on a best-efforts basis. It is not part of the Act.\n"
    "1. Act 13 of 2001 - Children Development Co-Savings Act 2001\n"
    "2. Act 46 of 2024 - Child Development Co-Savings (Amendment) Act 2024\n"
)


def test_masthead_fragment_is_overruled_by_the_history_block():
    gen = MetadataExtractor(_FakeClient({"law_number": "Act 2012", "last_amended": "2020"}))
    _, law_number, _ = gen.extract(_SSO_PDPA, "SG", "Personal Data Protection Act 2012")
    assert law_number == "Act 26 of 2012"
    assert gen.overridden == 1


def test_an_amendments_number_is_not_taken_for_the_acts():
    """The renamed Act: the model matched our title against entry 2 and reported a
    2024 number for a 2001 Act. The year check catches it; the history block then
    supplies the right one."""
    gen = MetadataExtractor(_FakeClient({"law_number": "Act 46 of 2024", "last_amended": "2024"}))
    last, law_number, _ = gen.extract(_SSO_CDCSA, "SG", "Child Development Co-Savings Act 2001")
    assert law_number == "Act 13 of 2001"
    assert last == "2024"          # last_amended IS the amendment year - untouched
    assert gen.rejected == 1


def test_year_check_is_silent_when_a_number_carries_no_year():
    # Malaysia's "Act 709" encodes no year - it must not be rejected.
    gen = MetadataExtractor(_FakeClient({"law_number": "Act 709", "last_amended": "2024"}))
    _, law_number, _ = gen.extract(_TEXT, "MY", "Personal Data Protection Act 2010")
    assert law_number == "Act 709"
    assert gen.rejected == 0


def test_no_history_block_leaves_the_model_answer_standing():
    # Australia / Malaysia have no SSO-style ordered history -> nothing to override.
    au = ("Privacy Act 1988\nNo. 119 of 1988\nCompilation No. 30\n"
          "... [body] ...\nEndnotes\nAmendment history\n")
    gen = MetadataExtractor(_FakeClient({"law_number": "No. 119 of 1988", "last_amended": "1988"}))
    _, law_number, _ = gen.extract(au, "AU", "Privacy Act 1988")
    assert law_number == "No. 119 of 1988"
    assert gen.overridden == 0
