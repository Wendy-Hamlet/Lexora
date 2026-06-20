"""Multi-section output + the rel_floor precision gate (WS-5).

Production maps an indicator to several relevant sections of one instrument
(top_k>1), but a secondary section is emitted only when it is genuinely
competitive with the best (rel_floor). Pins: top_k=1 stays single-section;
top_k>1 with rel_floor=0 emits the extra section; a high rel_floor drops the
weaker section while always keeping the strongest.
"""
from __future__ import annotations

from datetime import datetime, timezone

from lexora.models.clause import CanonicalSpan, Clause
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import LegalSystem, RawDocument, SourceProfile, SourceType
from lexora.pipeline import _citations_from_clauses

# Two sections both relevant to a comprehensive-framework indicator, at clearly
# different match strengths; a third, unrelated, is below any floor.
_STRONG = ("An organisation must protect personal data in its possession by making "
           "reasonable security arrangements to prevent unauthorised access.")
_WEAK = "An organisation may make arrangements for the protection of records."
_OFFTOPIC = "The Minister may by notification prescribe fees and forms under this Act."


def _profile() -> SourceProfile:
    return SourceProfile(
        jurisdiction="Singapore", iso_code="SG", primary_language="en",
        legal_system=LegalSystem.common,
        keywords_by_indicator={"7.1": {"en": [
            "protect personal data reasonable security arrangements unauthorised access"]}},
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


def _ind() -> list[RDTIIIndicator]:
    return [RDTIIIndicator(rdtii_id="7.1", submission_id="P7-I1", pillar=7,
                           name="Comprehensive data protection framework",
                           description="protect personal data security arrangements")]


def _clauses() -> list[Clause]:
    return [_clause("S. 24", _STRONG), _clause("S. 11", _WEAK), _clause("S. 99", _OFFTOPIC)]


def _run(*, top_k: int, rel_floor: float):
    cites = _citations_from_clauses(_clauses(), _doc(), _profile(), _ind(), "statute",
                                    top_k=top_k, min_score=0.0, rel_floor=rel_floor)
    return [c.clause_id for c in cites]


def test_top_k_1_is_single_section():
    assert _run(top_k=1, rel_floor=0.6) == ["S. 24"]  # only the strongest section


def test_multi_section_emits_extra_when_floor_open():
    # top_k=2, no relative gate -> both relevant sections surface.
    ids = _run(top_k=2, rel_floor=0.0)
    assert "S. 24" in ids and "S. 11" in ids and len(ids) == 2


def test_rel_floor_drops_weaker_section_keeps_strongest():
    # A strict relative floor keeps only the top section (the weaker one falls below
    # rel_floor * top_score), but never drops the strongest.
    ids = _run(top_k=3, rel_floor=0.97)
    assert ids == ["S. 24"]


def test_offtopic_section_dropped_by_production_floor():
    # Under the production relative gate the zero-overlap fees/forms section falls
    # far below rel_floor * top_score and is never emitted, even at top_k=3.
    assert "S. 99" not in _run(top_k=3, rel_floor=0.6)
