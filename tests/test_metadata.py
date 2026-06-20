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
    assert MetadataExtractor(client=None).extract(_TEXT, "Malaysia", "PDPA 2010") == ("", "")


def test_extracts_verified_number_and_year():
    gen = MetadataExtractor(_FakeClient({"law_number": "Act 709", "last_amended": "2024"}))
    assert gen.extract(_TEXT, "Malaysia", "Personal Data Protection Act 2010") == ("2024", "Act 709")
    assert gen.extracted == 1 and gen.rejected == 0


def test_year_pulled_from_prose_when_present_in_text():
    gen = MetadataExtractor(_FakeClient({"law_number": "Act 709", "last_amended": "as at 2024"}))
    assert gen.extract(_TEXT, "Malaysia", "PDPA")[0] == "2024"


def test_fabricated_number_is_rejected():
    # "Act 999" is not printed in the document -> dropped, not shipped.
    gen = MetadataExtractor(_FakeClient({"law_number": "Act 999", "last_amended": "2024"}))
    last_amended, law_number = gen.extract(_TEXT, "Malaysia", "PDPA")
    assert law_number == "" and last_amended == "2024"
    assert gen.rejected == 1


def test_fabricated_year_is_rejected():
    gen = MetadataExtractor(_FakeClient({"law_number": "Act 709", "last_amended": "2099"}))
    last_amended, law_number = gen.extract(_TEXT, "Malaysia", "PDPA")
    assert last_amended == "" and law_number == "Act 709"
    assert gen.rejected == 1


def test_multi_token_number_verifies_each_digit_run():
    text = "Privacy Act 1988 ... No. 119 of 1988 ... compiled 14 October 2024."
    gen = MetadataExtractor(_FakeClient({"law_number": "No. 119 of 1988", "last_amended": "2024"}))
    assert gen.extract(text, "Australia", "Privacy Act 1988") == ("2024", "No. 119 of 1988")


def test_backend_error_returns_blanks_and_counts():
    gen = MetadataExtractor(_BoomClient())
    assert gen.extract(_TEXT, "Malaysia", "PDPA") == ("", "")
    assert gen.error_count == 1 and gen.last_error_type == "RuntimeError"


def test_resolve_doc_metadata_precedence():
    """connector value wins; LLM extractor fills the gap and beats the anchor."""
    from datetime import datetime, timezone

    from lexora.models.source import InstrumentMeta, LegalSystem, RawDocument, SourceProfile, SourceType
    from lexora.pipeline import _resolve_doc_metadata

    profile = SourceProfile(
        jurisdiction="Malaysia", iso_code="MY", primary_language="en", legal_system=LegalSystem.common,
        instrument_metadata={"Personal Data Protection Act 2010":
                             InstrumentMeta(last_amended="1999", law_number="Act 000")},
    )
    doc = RawDocument(
        document_id="d", source_url="https://e.gov/x", retrieval_timestamp=datetime.now(timezone.utc),
        http_status=200, sha256="sha256:x", content_type="application/pdf", bytes_path="/tmp/x",
        portal_name="P", jurisdiction="MY", source_type=SourceType.primary,
        title="Personal Data Protection Act 2010", law_number="Act 709", last_amended="",  # connector gave number only
    )
    extractor = MetadataExtractor(_FakeClient({"law_number": "Act 709", "last_amended": "2024"}))
    last_amended, law_number = _resolve_doc_metadata(doc, _TEXT, profile, extractor)
    assert law_number == "Act 709"   # connector (not the anchor's "Act 000")
    assert last_amended == "2024"    # LLM extractor filled the gap (not the anchor's "1999")
