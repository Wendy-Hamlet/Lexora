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


class _TwoChannelClient:
    """Answers the extraction and recall prompts differently, and records which
    system prompt each call carried."""

    def __init__(self, extraction: dict, recall: dict) -> None:
        self._extraction, self._recall = extraction, recall
        self.users: list[str] = []

    def chat(self, system, user, json_schema=None):  # noqa: ANN001
        self.users.append(user)
        recalling = "NOT given its text" in system
        return dict(self._recall if recalling else self._extraction)


def test_recall_is_asked_in_its_own_call_with_no_document_text():
    """The channel's whole value is being independent of the document.

    Its predecessor rode along in the extraction call and, measured on four Malaysian
    laws, restated what it had just read instead of recalling anything."""
    client = _TwoChannelClient(
        {"law_number": "Act 709", "last_amended": "2024"},
        {"law_number": "Act 709", "last_amended": "2024"},
    )
    gen = MetadataExtractor(client, recall=True)
    gen.extract(_TEXT, "Malaysia", "Personal Data Protection Act 2010")

    assert len(client.users) == 2
    extraction_user, recall_user = client.users
    assert "Document text" in extraction_user
    assert "Personal Data Protection Act 2010" in recall_user
    assert "LAWS OF MALAYSIA" not in recall_user  # no document text reaches it
    assert "Endnote" not in recall_user


def test_only_a_disagreement_is_written_down():
    """Agreement is not evidence: on the Food Act 1983 both channels said 2006 and the
    real answer was neither. A conflict is the only thing that tells an analyst where
    to look, so it is the only thing that becomes a note."""
    agreeing = _TwoChannelClient(
        {"law_number": "Act 709", "last_amended": "2024"},
        {"law_number": "Act 709", "last_amended": "2024"},
    )
    gen = MetadataExtractor(agreeing, recall=True)
    assert gen.extract(_TEXT, "Malaysia", "PDPA")[2] == ""
    assert gen.recall_conflicts == 0

    disagreeing = _TwoChannelClient(
        {"law_number": "Act 709", "last_amended": "2024"},
        {"law_number": "Act 709", "last_amended": "2019"},
    )
    gen = MetadataExtractor(disagreeing, recall=True)
    note = gen.extract(_TEXT, "Malaysia", "PDPA")[2]
    assert "last_amended: document says 2024, recall says 2019" in note
    assert "NOT used as answer" in note
    assert "law_number" not in note  # that field agreed
    assert gen.recall_conflicts == 1


def test_recall_never_changes_an_answer():
    client = _TwoChannelClient(
        {"law_number": "<<NONE>>", "last_amended": "<<NONE>>"},
        {"law_number": "Act 709", "last_amended": "2019"},
    )
    gen = MetadataExtractor(client, recall=True)
    last_amended, law_number, note = gen.extract(_TEXT, "Malaysia", "PDPA")
    assert (last_amended, law_number) == ("", "")  # falls through to the next tier
    assert note == ""  # nothing to disagree WITH -- a blank is not a contradiction


def test_recall_declines_rather_than_guesses():
    """An unrecognised law must come back blank. A plausible-looking guess here would
    be read as independent confirmation, which is the opposite of this channel's job."""
    client = _TwoChannelClient(
        {"law_number": "Act 709", "last_amended": "2024"},
        {"law_number": "<<NONE>>", "last_amended": "I am not certain about this law"},
    )
    gen = MetadataExtractor(client, recall=True)
    assert gen.recall("Malaysia", "Some Obscure Act 1961") == ("", "")
    assert gen.recalled == 0


def test_recall_is_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("LEXORA_METADATA_RECALL", raising=False)
    client = _TwoChannelClient({"law_number": "Act 709", "last_amended": "2024"},
                               {"law_number": "Act 1", "last_amended": "1999"})
    gen = MetadataExtractor(client)  # no explicit recall= -> env decides
    gen.extract(_TEXT, "Malaysia", "PDPA")
    assert len(client.users) == 1  # extraction only; no second call was paid for


def test_recall_is_asked_once_per_law_not_once_per_document():
    client = _TwoChannelClient({"law_number": "Act 709", "last_amended": "2024"},
                               {"law_number": "Act 709", "last_amended": "2019"})
    gen = MetadataExtractor(client, recall=True)
    for _ in range(3):
        gen.extract(_TEXT, "Malaysia", "Personal Data Protection Act 2010")
    assert sum(1 for u in client.users if "Document text" not in u) == 1


def test_an_act_number_conflict_is_about_the_number_not_the_punctuation():
    from lexora.cite.metadata import recall_conflict_note

    assert recall_conflict_note(("", "Act 709"), ("", "Act No. 709")) == ""
    assert "law_number" in recall_conflict_note(("", "Act 709"), ("", "Act 710"))


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
