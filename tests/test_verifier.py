"""Constrained LLM verifier (P-3) — offline, no server.

A deterministic ``FakeClient`` stands in for the OpenAI-compatible endpoint so
these tests pin the *contract* without a model: the verifier is tightening-only
(it picks among the given clauses or abstains, never invents one), it never lets
the model write quote text (span id comes from the canonical clause), and the
pipeline wiring drops abstained mappings and routes "uncertain" to human review.
"""
from __future__ import annotations

from datetime import datetime, timezone

from lexora.classify.verifier import Verifier, make_verifier
from lexora.models.citation import ClaimLabel, ReviewStatus
from lexora.models.clause import CanonicalSpan, Clause
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import (
    LegalSystem,
    PortalSpec,
    RawDocument,
    SourceProfile,
    SourceType,
)


class FakeClient:
    """Returns a canned response dict and records the prompt it was given."""

    def __init__(self, response: dict | Exception):
        self.response = response
        self.last_user = None

    def chat(self, system: str, user: str, json_schema=None) -> dict:
        self.last_user = user
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _clause(cid: str, text: str) -> Clause:
    return Clause(
        clause_id=cid, document_id="doc", structural_path=f"S. {cid}",
        span=CanonicalSpan(span_id=cid + ".s", document_id="doc", char_start=0,
                           char_end=len(text), text=text),
    )


def _ind() -> RDTIIIndicator:
    return RDTIIIndicator(
        rdtii_id="7.3", submission_id="P7-I3", pillar=7, name="data retention",
        description="minimum period records must be retained",
    )


def _candidates() -> list[Clause]:
    return [
        _clause("c1", "records must be retained for at least seven years"),
        _clause("c2", "the Minister may by order prescribe forms"),
    ]


def test_verify_selects_named_clause_and_fills_span_from_canonical():
    client = FakeClient({"clause_id": "c1", "label": "match", "confidence": 0.9,
                         "rationale": "explicit retention period"})
    claim = Verifier(client).verify(_ind(), _candidates())
    assert claim is not None
    assert claim.clause_id == "c1"
    assert claim.indicator_id == "P7-I3"
    assert claim.label is ClaimLabel.match
    # span id is taken from the canonical clause, never from the model output
    assert claim.quote_span_id == "c1.s"
    # the prompt enumerates candidate ids but no quote text is requested back
    assert 'clause_id="c1"' in client.last_user


def test_verify_abstains_on_no_match():
    client = FakeClient({"clause_id": None, "label": "no_match", "confidence": 0.1})
    assert Verifier(client).verify(_ind(), _candidates()) is None


def test_verify_abstains_on_hallucinated_id():
    # The model names a clause that is not in the candidate set -> drop (tightening).
    client = FakeClient({"clause_id": "c99", "label": "match", "confidence": 1.0})
    assert Verifier(client).verify(_ind(), _candidates()) is None


def test_verify_uncertain_is_kept_as_uncertain():
    client = FakeClient({"clause_id": "c1", "label": "uncertain", "confidence": 0.5})
    claim = Verifier(client).verify(_ind(), _candidates())
    assert claim is not None and claim.label is ClaimLabel.uncertain


def test_verify_backend_failure_abstains_instead_of_raising():
    client = FakeClient(RuntimeError("endpoint down"))
    assert Verifier(client).verify(_ind(), _candidates()) is None


def test_verify_empty_candidates_returns_none():
    client = FakeClient({"clause_id": "c1", "label": "match"})
    assert Verifier(client).verify(_ind(), []) is None


def test_make_verifier_off_by_default():
    assert make_verifier() is None
    assert make_verifier(use_llm=False) is None


# --- pipeline wiring --------------------------------------------------------

def _doc() -> RawDocument:
    return RawDocument(
        document_id="doc", source_url="https://e.gov/x", http_status=200,
        retrieval_timestamp=datetime.now(timezone.utc), sha256="sha256:abc",
        content_type="application/pdf", bytes_path="/tmp/x", portal_name="P",
        jurisdiction="SG", source_type=SourceType.primary, title="Some Act 1968",
    )


def _profile() -> SourceProfile:
    return SourceProfile(
        jurisdiction="Singapore", iso_code="SG", primary_language="en",
        legal_system=LegalSystem.common,
        portals=[PortalSpec(name="P", url="https://e.gov/", source_type=SourceType.primary)],
    )


def test_pipeline_verifier_drops_abstained_indicator():
    from lexora.pipeline import _citations_from_clauses

    verifier = Verifier(FakeClient({"clause_id": None, "label": "no_match"}))
    cites = _citations_from_clauses(
        _candidates(), _doc(), _profile(), [_ind()], "statute", top_k=2, min_score=0.0,
        verifier=verifier,
    )
    assert cites == []  # tightened out entirely


def test_pipeline_verifier_match_publishes_verified():
    from lexora.pipeline import _citations_from_clauses

    verifier = Verifier(FakeClient({"clause_id": "c1", "label": "match", "confidence": 0.9}))
    cites = _citations_from_clauses(
        _candidates(), _doc(), _profile(), [_ind()], "statute", top_k=2, min_score=0.0,
        verifier=verifier,
    )
    assert len(cites) == 1
    assert cites[0].clause_id == "c1"
    assert cites[0].review_status is ReviewStatus.verified


def test_pipeline_verifier_uncertain_routes_to_conflict_review():
    from lexora.pipeline import _citations_from_clauses

    verifier = Verifier(FakeClient({"clause_id": "c1", "label": "uncertain"}))
    cites = _citations_from_clauses(
        _candidates(), _doc(), _profile(), [_ind()], "statute", top_k=2, min_score=0.0,
        verifier=verifier,
    )
    assert len(cites) == 1
    assert cites[0].review_status is ReviewStatus.conflict_review
    assert "uncertain" in cites[0].notes.lower()


def test_pipeline_without_verifier_is_unchanged():
    from lexora.pipeline import _citations_from_clauses

    cites = _citations_from_clauses(
        _candidates(), _doc(), _profile(), [_ind()], "statute", top_k=2, min_score=0.0,
    )
    # both keyword-passing clauses materialize when no verifier narrows them
    assert {c.clause_id for c in cites} == {"c1", "c2"}
    assert all(c.review_status is ReviewStatus.verified for c in cites)
