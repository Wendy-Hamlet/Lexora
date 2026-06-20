"""P6/P7 boundary discipline — the tightening scope filter (classify/boundaries.py).

Pins the 7.3 minimum-retention vs retention-limitation discriminator both as a unit
predicate and end-to-end (a limitation clause is dropped from 7.3; a minimum-retention
duty is kept), and that the filter never touches indicators with no rule.
"""
from __future__ import annotations

from datetime import datetime, timezone

from lexora.classify.boundaries import admits_clause
from lexora.models.clause import CanonicalSpan, Clause
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import LegalSystem, RawDocument, SourceProfile, SourceType
from lexora.pipeline import _citations_from_clauses

# SG PDPA s.25 — a retention LIMITATION (the opposite of a minimum-retention duty).
_LIMIT = ("An organisation must cease to retain its documents containing personal data "
          "as soon as it is reasonable to assume that the purpose for which it was "
          "collected is no longer being served by retention.")
# A genuine minimum-retention duty (AU TIA-style mandatory data retention).
_MIN = ("A service provider must keep, or cause to be kept, the information for the "
        "retention period of 2 years, and must retain it for at least that period.")
_NEUTRAL = "An organisation shall appoint a data protection officer responsible for compliance."


def test_unit_7_3_excludes_limitation_admits_minimum():
    assert admits_clause("7.3", _LIMIT) is False
    assert admits_clause("7.3", _MIN) is True
    assert admits_clause("7.3", _NEUTRAL) is True  # neutral -> not excluded


def test_unit_no_rule_indicator_always_admits():
    assert admits_clause("7.4", _LIMIT) is True
    assert admits_clause("7.1", _LIMIT) is True


# --- 6.1 ban / localisation  vs  6.4 conditional flow ------------------------
# SG PDPA s.26 — prohibition BUT permitted subject to comparable protection -> 6.4.
_SG_S26 = ("An organisation must not transfer personal data to a country or territory "
           "outside Singapore except in circumstances prescribed under this Act to ensure "
           "a standard of protection comparable to the protection under this Act.")
# MY PDPA s.129 — prohibition unless the destination is specified by the Minister -> 6.4.
_MY_S129 = ("A data user shall not transfer any personal data to a place outside Malaysia "
            "unless to a place specified by the Minister.")
# AU My Health Records — outright data-localisation, no transfer condition -> 6.1.
_AU_LOCALISE = ("A registered repository operator must not hold or process My Health Record "
                "information, or take any record of it, outside Australia.")


def test_6_1_excludes_conditional_transfer():
    assert admits_clause("6.1", _SG_S26) is False   # conditional -> 6.4, not 6.1
    assert admits_clause("6.1", _MY_S129) is False
    assert admits_clause("6.1", _AU_LOCALISE) is True  # pure localisation stays 6.1


def test_6_4_excludes_pure_ban_admits_conditional():
    assert admits_clause("6.4", _AU_LOCALISE) is False  # pure localisation -> 6.1, not 6.4
    assert admits_clause("6.4", _SG_S26) is True        # conditional flow stays 6.4
    assert admits_clause("6.4", _MY_S129) is True


def test_6_1_6_4_routing_has_no_gap():
    # every transfer clause lands in exactly one of 6.1 / 6.4 (never dropped from both)
    for text in (_SG_S26, _MY_S129, _AU_LOCALISE):
        assert admits_clause("6.1", text) or admits_clause("6.4", text)


def _profile() -> SourceProfile:
    return SourceProfile(
        jurisdiction="Singapore", iso_code="SG", primary_language="en",
        legal_system=LegalSystem.common,
        keywords_by_indicator={"7.3": {"en": ["retention", "retain", "keep records"]}},
    )


def _doc() -> RawDocument:
    return RawDocument(
        document_id="d", source_url="https://e.gov/x", retrieval_timestamp=datetime.now(timezone.utc),
        http_status=200, sha256="sha256:x", content_type="application/pdf", bytes_path="/tmp/x",
        portal_name="P", jurisdiction="SG", source_type=SourceType.primary, title="Test Act")


def _clause(cid: str, text: str) -> Clause:
    return Clause(clause_id=cid, document_id="d", structural_path=cid,
                  span=CanonicalSpan(span_id=cid + ".s", document_id="d",
                                     char_start=0, char_end=len(text), text=text))


def _ind_73() -> list[RDTIIIndicator]:
    return [RDTIIIndicator(rdtii_id="7.3", submission_id="P7-I3", pillar=7,
                           name="Minimum period of data retention requirements",
                           description="Whether the law imposes a minimum period for retaining data.")]


def test_e2e_limitation_clause_not_emitted_for_7_3():
    clauses = [_clause("S. 25", _LIMIT)]
    cites = _citations_from_clauses(clauses, _doc(), _profile(), _ind_73(),
                                    "statute", top_k=1, min_score=0.0)
    assert cites == []  # s.25 limitation is excluded from 7.3


def test_e2e_minimum_retention_clause_is_emitted_for_7_3():
    clauses = [_clause("S. 187", _MIN)]
    cites = _citations_from_clauses(clauses, _doc(), _profile(), _ind_73(),
                                    "statute", top_k=1, min_score=0.0)
    assert [c.indicator_id for c in cites] == ["P7-I3"]
