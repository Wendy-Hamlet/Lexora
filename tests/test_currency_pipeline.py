"""End-to-end amendment-currency adjudication through the pipeline helper.

Drives `_apply_currency_flags` with a real principal + its amending Act in the same
working set, and asserts the per-provision verdicts the legal team specified:
untouched -> CURRENT, amended -> AMENDED (+ amending text), deleted+commenced ->
REPEALED (then dropped under enforced_only), term-rename -> annotated.
"""
from __future__ import annotations

from datetime import datetime, timezone

from lexora.models.citation import Citation, DiscoveryTag, ReviewStatus
from lexora.models.source import (
    LegalSystem,
    RawDocument,
    SourceProfile,
    SourceType,
)
from lexora.pipeline import DemoArtifacts, _apply_currency_flags

PRINCIPAL_TEXT = """
LAWS OF MALAYSIA
ACT 709
PERSONAL DATA PROTECTION ACT 2010
Date of Royal Assent : 2 June 2010
An Act to regulate the processing of personal data in commercial transactions.
"""

AMENDMENT_TEXT = """
LAWS OF MALAYSIA
Act A1727
PERSONAL DATA PROTECTION (AMENDMENT) ACT 2024
This Act comes into operation on 1 January 2025.
An Act to amend the Personal Data Protection Act 2010.
2. The Personal Data Protection Act 2010 [Act 709], referred to as the "principal
Act", is amended as set out in this Act.
4. The principal Act is amended by substituting for the words "data user" wherever
appearing the words "data controller".
5. Section 6 of the principal Act is amended by substituting for subsection (1) the
following subsection.
6. Section 129 of the principal Act is deleted.
"""


def _doc(sha: str, title: str, text: str, number: str, last_amended: str) -> DemoArtifacts:
    doc = RawDocument(
        document_id=sha, source_url="https://lom.agc.gov.my/x", http_status=200,
        retrieval_timestamp=datetime.now(timezone.utc), sha256=sha,
        content_type="application/pdf", bytes_path=f"{sha}.pdf", portal_name="AGC",
        jurisdiction="Malaysia", source_type=SourceType.primary, title=title,
        law_number=number, last_amended=last_amended,
    )
    return DemoArtifacts(document=doc, clauses=[], citations=[], document_text=text)


def _cit(sha: str, section: str, quote: str) -> Citation:
    return Citation(
        economy="Malaysia", title="Personal Data Protection Act 2010", law_number="Act 709",
        last_amended="2010", indicator_id="P7-I1", article_path=f"S. {section}",
        discovery_tag=DiscoveryTag.known, page_or_dom_anchor="p.1", quote=quote,
        source_url="https://lom.agc.gov.my/x", confidence=0.9, clause_id=f"c{section}",
        retrieval_timestamp=datetime.now(timezone.utc), document_hash=sha,
        jurisdiction="Malaysia", legal_form="statute", char_start=0, char_end=5,
        review_status=ReviewStatus.verified,
    )


def _run():
    principal = _doc("H709", "Personal Data Protection Act 2010", PRINCIPAL_TEXT,
                     "Act 709", "2010")
    amendment = _doc("HA1727", "Personal Data Protection (Amendment) Act 2024",
                     AMENDMENT_TEXT, "Act A1727", "2024")
    c_untouched = _cit("H709", "40", "A data user may collect personal data ...")
    c_amended = _cit("H709", "6", "A data user shall not process personal data ...")
    c_repealed = _cit("H709", "129", "transfer of personal data outside Malaysia ...")
    cits = [c_untouched, c_amended, c_repealed]
    profile = SourceProfile(
        jurisdiction="Malaysia", iso_code="MY", primary_language="en",
        legal_system=LegalSystem.common,
    )
    _apply_currency_flags([principal, amendment], cits, profile)
    return c_untouched, c_amended, c_repealed


def test_untouched_provision_is_current_with_rename_note():
    untouched, _, _ = _run()
    # Section 40 was not amended -> the coarse doc-level flag is cleared to CURRENT,
    # but the quote uses "data user", so the term-rename is still annotated.
    assert untouched.currency_status == "CURRENT"
    assert untouched.review_status is ReviewStatus.verified  # not raised to review
    assert 'renamed to "data controller"' in untouched.notes


def test_amended_provision_flagged_and_carries_amendment_text():
    _, amended, _ = _run()
    assert amended.currency_status == "AMENDED"
    assert amended.review_status is ReviewStatus.amendment_review
    assert amended.amendment_text  # amending Act's own verbatim, kept beside original
    assert "Act A1727 (2024)" in amended.amended_by


def test_repealed_provision_flagged_repealed():
    _, _, repealed = _run()
    assert repealed.currency_status == "REPEALED"
    assert repealed.review_status is ReviewStatus.amendment_review
    assert "repealed by Act A1727 (2024)" in repealed.notes


# A title-only amendment (no "[Act 709]" bracket) — the real gazetted A1727 names its
# principal by title, while a PDPA citation is keyed by the Act number; the
# number/title candidate-key bridge must still connect them.
TITLE_ONLY_AMENDMENT = """
LAWS OF MALAYSIA
Act A1727
PERSONAL DATA PROTECTION (AMENDMENT) ACT 2024
This Act comes into operation on 1 January 2025.
An Act to amend the Personal Data Protection Act 2010.
6. Section 129 of the principal Act is deleted.
"""


class _RecordingExtractor:
    """Stub LLM amendment extractor: returns a canned instruction set and records the
    texts it was asked to extract (to assert the LLM-first path actually ran). Carries
    a non-None ``_client`` so the pipeline treats it as a live LLM and can pool it."""

    def __init__(self, instructions):
        self._client = object()
        self._instructions = instructions
        self.seen: list[str] = []

    def extract(self, text: str):
        self.seen.append(text)
        return list(self._instructions)


def test_llm_first_extractor_is_preferred_over_regex():
    from lexora.cite.amendments import AmendmentInstruction, InstructionKind, Operation

    principal = _doc("H709", "Personal Data Protection Act 2010", PRINCIPAL_TEXT,
                     "Act 709", "2010")
    # An amending Act whose REGEX parse would NOT see section 40 as deleted; the LLM
    # extractor reports the deletion, so adjudication must follow the LLM, not regex.
    amendment = _doc("HA1727", "Personal Data Protection (Amendment) Act 2024",
                     AMENDMENT_TEXT, "Act A1727", "2024")
    c40 = _cit("H709", "40", "A data user may collect personal data ...")
    extractor = _RecordingExtractor([
        AmendmentInstruction(kind=InstructionKind.section_op, op=Operation.delete,
                             target_section="40", old_term="", new_term="", raw=""),
    ])
    _apply_currency_flags([principal, amendment], [c40], SourceProfile(
        jurisdiction="Malaysia", iso_code="MY", primary_language="en",
        legal_system=LegalSystem.common), extractor=extractor)
    assert extractor.seen == [AMENDMENT_TEXT]          # LLM path ran on the amending Act
    assert c40.currency_status == "REPEALED"           # followed the LLM's deletion
    assert "repealed by Act A1727 (2024)" in c40.notes


def test_parallel_extraction_over_multiple_amending_acts():
    # Two amending Acts in the working set, workers>1 -> the LLM extractor is called
    # concurrently for each; both verdicts must land.
    from lexora.cite.amendments import AmendmentInstruction, InstructionKind, Operation

    principal = _doc("H709", "Personal Data Protection Act 2010", PRINCIPAL_TEXT,
                     "Act 709", "2010")
    amd_a = _doc("HA1", "Personal Data Protection (Amendment) Act 2024",
                 AMENDMENT_TEXT, "Act A1727", "2024")
    amd_b = _doc("HA2", "Personal Data Protection (Amendment) Act 2024",
                 TITLE_ONLY_AMENDMENT, "Act A1800", "2024")
    c6 = _cit("H709", "6", "A data user shall not process personal data ...")
    extractor = _RecordingExtractor([
        AmendmentInstruction(kind=InstructionKind.section_op, op=Operation.amend,
                             target_section="6", old_term="", new_term="", raw="amended"),
    ])
    _apply_currency_flags([principal, amd_a, amd_b], [c6], SourceProfile(
        jurisdiction="Malaysia", iso_code="MY", primary_language="en",
        legal_system=LegalSystem.common), extractor=extractor, workers=4)
    assert len(extractor.seen) == 2                    # both amending Acts extracted
    assert c6.currency_status == "AMENDED"


def test_inert_extractor_falls_back_to_regex():
    # An inert extractor (_client None, extract -> []) must not suppress the regex floor.
    class _Inert:
        _client = None

        def extract(self, text):
            return []

    principal = _doc("H709", "Personal Data Protection Act 2010", PRINCIPAL_TEXT,
                     "Act 709", "2010")
    amendment = _doc("HA1727", "Personal Data Protection (Amendment) Act 2024",
                     AMENDMENT_TEXT, "Act A1727", "2024")
    c = _cit("H709", "129", "transfer of personal data outside Malaysia ...")
    _apply_currency_flags([principal, amendment], [c], SourceProfile(
        jurisdiction="Malaysia", iso_code="MY", primary_language="en",
        legal_system=LegalSystem.common), extractor=_Inert())
    assert c.currency_status == "REPEALED"             # regex parser still adjudicated


def test_title_only_amendment_matches_number_keyed_citation():
    principal = _doc("H709", "Personal Data Protection Act 2010", PRINCIPAL_TEXT,
                     "Act 709", "2010")
    amendment = _doc("HA1727", "Personal Data Protection (Amendment) Act 2024",
                     TITLE_ONLY_AMENDMENT, "Act A1727", "2024")
    c = _cit("H709", "129", "transfer of personal data outside Malaysia ...")  # Act 709
    from lexora.cite.amendments import detect_amends_target
    # Sanity: this amendment names the principal by TITLE only (number empty).
    assert detect_amends_target(TITLE_ONLY_AMENDMENT)[0] == ""
    _apply_currency_flags([principal, amendment], [c], SourceProfile(
        jurisdiction="Malaysia", iso_code="MY", primary_language="en",
        legal_system=LegalSystem.common))
    assert c.currency_status == "REPEALED"  # bridged despite number-vs-title keying
    assert c.source_version == "ORIGINAL"
