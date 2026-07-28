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
    verifier = Verifier(client)
    assert verifier.verify(_ind(), _candidates()) is None
    assert verifier.error_count == 1
    assert verifier.last_error_type == "RuntimeError"


# --- per-clause 9-in-1 judge ------------------------------------------------

def _inds() -> list[RDTIIIndicator]:
    return [
        RDTIIIndicator(rdtii_id="6.4", submission_id="P6-I4", pillar=6,
                       name="cross-border", description="conditional transfer abroad"),
        RDTIIIndicator(rdtii_id="7.3", submission_id="P7-I3", pillar=7,
                       name="retention", description="minimum retention period"),
        RDTIIIndicator(rdtii_id="7.5", submission_id="P7-I5", pillar=7,
                       name="govt access", description="state access to data"),
    ]


def test_judge_clause_returns_supported_indicator_subset():
    # The clause supports two indicators; one returned id is hallucinated and dropped.
    client = FakeClient({"indicators": ["P7-I3", "P7-I5", "P9-I9"], "rationale": "x"})
    got = Verifier(client, mode="per_clause").judge_clause(
        _clause("c1", "records must be retained and may be disclosed to an agency"), _inds()
    )
    assert got == {"P7-I3", "P7-I5"}  # hallucinated P9-I9 filtered out
    # the prompt carries the one clause and enumerates the indicator ids
    assert "P6-I4" in client.last_user and "CLAUSE" in client.last_user


def test_judge_clause_empty_means_supports_nothing():
    client = FakeClient({"indicators": [], "rationale": "off-topic"})
    assert Verifier(client, mode="per_clause").judge_clause(_clause("c1", "forms"), _inds()) == set()


def test_judge_clause_backend_error_returns_none():
    v = Verifier(FakeClient(RuntimeError("down")), mode="per_clause")
    assert v.judge_clause(_clause("c1", "x"), _inds()) is None
    assert v.error_count == 1 and v.last_error_type == "RuntimeError"


def test_judge_clause_unparseable_returns_none():
    assert Verifier(FakeClient({"nope": 1}), mode="per_clause").judge_clause(
        _clause("c1", "x"), _inds()
    ) is None


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


# --- per-cell verifier (universal precision lane) ---------------------------

def test_judge_each_keeps_returned_subset_only():
    # model keeps c1, and names a hallucinated id which is dropped (tightening).
    client = FakeClient({"keep": ["c1", "c99"], "rationale": "c1 on-point"})
    kept = Verifier(client, mode="per_cell").judge_each(_ind(), _candidates())
    assert kept == {"c1"}  # subset of candidates only; hallucinated c99 dropped


def test_judge_each_empty_keep_is_a_real_drop_all():
    client = FakeClient({"keep": [], "rationale": "none on-point"})
    assert Verifier(client, mode="per_cell").judge_each(_ind(), _candidates()) == set()


def test_judge_each_backend_error_returns_none_for_keep_all():
    # None is the sentinel the pipeline reads as "keep all" (never worse than baseline).
    client = FakeClient(RuntimeError("endpoint down"))
    v = Verifier(client, mode="per_cell")
    assert v.judge_each(_ind(), _candidates()) is None
    assert v.error_count == 1


def test_make_verifier_per_cell_mode():
    # (offline) just the mode plumbing — no client constructed when use_llm=False.
    assert make_verifier(use_llm=False, mode="per_cell") is None


def test_pipeline_per_cell_drops_off_topic_keeps_on_topic():
    from lexora.pipeline import _citations_from_clauses

    # c1 (retention) is on-topic for 7.3; c2 (forms) is off-topic -> dropped.
    verifier = Verifier(FakeClient({"keep": ["c1"]}), mode="per_cell")
    cites = _citations_from_clauses(
        _candidates(), _doc(), _profile(), [_ind()], "statute", top_k=2, min_score=0.0,
        verifier=verifier,
    )
    assert {c.clause_id for c in cites} == {"c1"}  # off-topic c2 tightened out


def test_pipeline_per_cell_error_keeps_all_not_drops():
    from lexora.pipeline import _citations_from_clauses

    # backend error -> judge_each None -> keep ALL passing (baseline), never delete.
    verifier = Verifier(FakeClient(RuntimeError("down")), mode="per_cell")
    cites = _citations_from_clauses(
        _candidates(), _doc(), _profile(), [_ind()], "statute", top_k=2, min_score=0.0,
        verifier=verifier,
    )
    assert {c.clause_id for c in cites} == {"c1", "c2"}  # both kept on error


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


# --- per-clause 9-in-1 verifier (pipeline wiring) ---------------------------

class _ClauseAwareClient:
    """A per_clause fake: reads the clause text from the prompt and returns the
    indicators that clause supports, so different clauses get different verdicts."""

    def chat(self, system: str, user: str, json_schema=None) -> dict:
        # "seven years" appears only in c1's clause text (not the indicator block).
        if "seven years" in user:                    # c1 — retention provision
            return {"indicators": ["P7-I3"], "rationale": "retention period"}
        return {"indicators": [], "rationale": "off-topic"}  # c2 — forms


def test_pipeline_per_clause_assigns_only_supported_clause():
    from lexora.pipeline import _citations_from_clauses

    # c1 (retention) -> P7-I3; c2 (forms) -> nothing. One 9-in-1 call per clause.
    verifier = Verifier(_ClauseAwareClient(), mode="per_clause")
    cites = _citations_from_clauses(
        _candidates(), _doc(), _profile(), [_ind()], "statute", top_k=2, min_score=0.0,
        verifier=verifier,
    )
    assert {(c.indicator_id, c.clause_id) for c in cites} == {("P7-I3", "c1")}


class FlakyClient:
    """Raises on the first ``n_errors`` calls, then answers. Models an endpoint that is
    intermittently bad rather than dead."""

    def __init__(self, n_errors: int, response: dict):
        self.remaining = n_errors
        self.response = response

    def chat(self, system: str, user: str, json_schema=None) -> dict:
        if self.remaining > 0:
            self.remaining -= 1
            raise RuntimeError("down")
        return self.response


def test_pipeline_per_clause_incidental_error_still_drops_that_clause(monkeypatch):
    from lexora.pipeline import _citations_from_clauses

    # One failure out of two is below the degrade threshold, so the old contract holds:
    # the erroring clause contributes nothing and no mapping is fabricated for it.
    monkeypatch.setenv("LEXORA_JUDGE_DEGRADE_AT", "0.5")
    verifier = Verifier(FlakyClient(1, {"indicators": ["P7-I3"]}), mode="per_clause")
    cites = _citations_from_clauses(
        _candidates(), _doc(), _profile(), [_ind()], "statute", top_k=2, min_score=0.0,
        verifier=verifier,
    )
    assert len(cites) == 1  # the one clause that was actually judged
    assert "DEGRADED" not in cites[0].notes


def test_pipeline_per_clause_systematic_failure_degrades_to_ranking_lane(monkeypatch):
    from lexora.pipeline import _citations_from_clauses

    # Every judgement fails. Dropping every clause would publish an EMPTY document that
    # looks exactly like "this law has nothing" -- so the document falls back to the
    # key-free BM25 lane instead, and every row says so.
    monkeypatch.setenv("LEXORA_JUDGE_DEGRADE_AT", "0.5")
    # The degraded lane's precision floor is measured separately (scripts/bench_fallback.py);
    # switch it off here so this test covers the degrade path itself and does not silently
    # become a test of whether two synthetic clauses clear a BM25 threshold.
    monkeypatch.setenv("LEXORA_DEGRADED_MIN_SCORE", "0")
    verifier = Verifier(FakeClient(RuntimeError("down")), mode="per_clause")
    cites = _citations_from_clauses(
        _candidates(), _doc(), _profile(), [_ind()], "statute", top_k=2, min_score=0.0,
        verifier=verifier,
    )
    assert cites, "a dead judge must degrade, not silently publish nothing"
    assert all("DEGRADED" in c.notes for c in cites)
    assert all("LLM judge unavailable" in c.notes for c in cites)


def test_pipeline_per_clause_degrade_gate_can_be_disabled(monkeypatch):
    from lexora.pipeline import _citations_from_clauses

    # No ratio can exceed 1.1, so the gate never fires and the pre-gate behaviour returns.
    monkeypatch.setenv("LEXORA_JUDGE_DEGRADE_AT", "1.1")
    verifier = Verifier(FakeClient(RuntimeError("down")), mode="per_clause")
    cites = _citations_from_clauses(
        _candidates(), _doc(), _profile(), [_ind()], "statute", top_k=2, min_score=0.0,
        verifier=verifier,
    )
    assert cites == []
