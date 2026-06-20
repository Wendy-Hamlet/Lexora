"""Enforced-only filter wiring (pipeline._citations_from_clauses).

A positively repealed/draft instrument contributes NO citations to the inventory
(official scope, internal guide p.8); an in-force or status-unknown instrument is
kept. The drop is opt-out via ``enforced_only=False`` for audit/inspection.
"""
from __future__ import annotations

from datetime import datetime, timezone

from lexora.models.clause import CanonicalSpan, Clause
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import (
    InstrumentStatus,
    LegalSystem,
    RawDocument,
    SourceProfile,
    SourceType,
)
from lexora.pipeline import _citations_from_clauses

# A genuine minimum-retention duty -> maps to 7.3 (P7-I3) when the instrument is live.
_MIN = ("A service provider must keep the information for the retention period of "
        "2 years, and must retain it for at least that period.")


def _profile() -> SourceProfile:
    return SourceProfile(
        jurisdiction="Singapore", iso_code="SG", primary_language="en",
        legal_system=LegalSystem.common,
        keywords_by_indicator={"7.3": {"en": ["retention", "retain", "keep"]}},
    )


def _doc(*, status: InstrumentStatus = InstrumentStatus.unknown, title: str = "Test Act") -> RawDocument:
    return RawDocument(
        document_id="d", source_url="https://e.gov/x", retrieval_timestamp=datetime.now(timezone.utc),
        http_status=200, sha256="sha256:x", content_type="application/pdf", bytes_path="/tmp/x",
        portal_name="P", jurisdiction="SG", source_type=SourceType.primary, title=title, status=status)


def _clause() -> Clause:
    return Clause(clause_id="S. 187", document_id="d", structural_path="S. 187",
                  span=CanonicalSpan(span_id="s", document_id="d",
                                     char_start=0, char_end=len(_MIN), text=_MIN))


def _ind() -> list[RDTIIIndicator]:
    return [RDTIIIndicator(rdtii_id="7.3", submission_id="P7-I3", pillar=7,
                           name="Minimum period of data retention requirements",
                           description="Whether the law imposes a minimum period for retaining data.")]


def _run(doc: RawDocument, *, enforced_only: bool = True, document_text: str = ""):
    return _citations_from_clauses([_clause()], doc, _profile(), _ind(), "statute",
                                   top_k=1, min_score=0.0, enforced_only=enforced_only,
                                   document_text=document_text)


def test_in_force_instrument_is_kept():
    assert [c.indicator_id for c in _run(_doc(status=InstrumentStatus.in_force))] == ["P7-I3"]


def test_unknown_status_is_kept():
    # Never drop on uncertainty.
    assert [c.indicator_id for c in _run(_doc(status=InstrumentStatus.unknown))] == ["P7-I3"]


def test_repealed_portal_status_drops_all_citations():
    assert _run(_doc(status=InstrumentStatus.repealed)) == []


def test_repealed_title_stamp_drops_all_citations():
    # Even with UNKNOWN portal status, an SG-SSO-style "(Repealed)" title retires it.
    assert _run(_doc(title="Old Data Act 1998 (Repealed)")) == []


def test_enforced_only_off_keeps_repealed_for_audit():
    cites = _run(_doc(status=InstrumentStatus.repealed), enforced_only=False)
    assert [c.indicator_id for c in cites] == ["P7-I3"]


def test_self_repeal_header_in_text_drops():
    text = "This Act is repealed with effect from 2026.\n" + _MIN
    assert _run(_doc(), document_text=text) == []
