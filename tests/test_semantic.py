"""Dense-channel logic (P2) — offline, no model download.

The fastembed backend is exercised live elsewhere; here a deterministic
bag-of-words ``FakeEmbedder`` (cosine == normalized token overlap) pins the
*orchestration*: the AU concept->title crosswalk, the SG/MY candidate re-rank and
the clause-level BM25+dense fusion, none of which should depend on a real model to
be correct.
"""
from __future__ import annotations

import re

import numpy as np

from lexora.models.clause import CanonicalSpan, Clause
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import LegalSystem, PortalSpec, SourceProfile, SourceType
from lexora.semantic.embedder import cosine_topk, reciprocal_rank_fusion

_W = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Hashed bag-of-words embedder; cosine ~= token overlap.

    Tokens hash into a fixed-width vector so every ``encode`` call returns the
    same dimension (a real model has a fixed dim) regardless of call order."""

    DIM = 512

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for r, text in enumerate(texts):
            for tok in _W.findall(text.lower()):
                out[r, hash(tok) % self.DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


# --- pure helpers -----------------------------------------------------------

def test_cosine_topk_orders_and_truncates():
    docs = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    q = np.array([1.0, 0.0], dtype=np.float32)
    top = cosine_topk(q, docs, k=2)
    assert [i for i, _ in top] == [0, 2]  # exact match first, 45° next
    assert cosine_topk(q, np.zeros((0, 0), dtype=np.float32), k=3) == []


def test_reciprocal_rank_fusion_rewards_agreement():
    bm25 = ["a", "b", "c"]
    dense = ["b", "a", "d"]
    fused = reciprocal_rank_fusion([bm25, dense], k=60)
    # 'b' is rank2+rank1, 'a' is rank1+rank2 -> tie above c/d which each appear
    # once at rank3 (so c and d themselves tie).
    assert fused["a"] == fused["b"] > fused["c"]
    assert fused["c"] == fused["d"]


# --- AU semantic crosswalk --------------------------------------------------

def _ind(sub: str, text: str) -> RDTIIIndicator:
    return RDTIIIndicator(
        rdtii_id=sub.split("-")[0][1] + "." + sub.split("I")[-1],
        submission_id=sub, pillar=int(sub[1]), name=text, description=text,
        discovery_queries=[text],
    )


def _au_portal() -> PortalSpec:
    return PortalSpec(
        name="FRL", url="https://www.legislation.gov.au/", source_type=SourceType.primary,
        full_text=False,
    )


def test_au_crosswalk_attributes_titles_to_right_indicator():
    from lexora.collect.strategies import au_semantic_crosswalk

    catalogue = [
        {"id": "C1", "name": "Telecommunications Interception Access Act 1979", "isPrincipal": True},
        {"id": "C2", "name": "My Health Records Act 2012", "isPrincipal": True},
        {"id": "C3", "name": "Dairy Produce Marketing Act", "isPrincipal": True},
    ]
    inds = [
        _ind("P7-I3", "telecommunications interception access retention"),
        _ind("P6-I1", "health records storage"),
    ]
    hits = au_semantic_crosswalk(
        _au_portal(), inds, embedder=FakeEmbedder(), top_k=1, min_sim=0.2,
        catalogue=catalogue, known_instruments=[],
    )
    by_url = {h.url.split("/")[-2]: h for h in hits}
    assert by_url["C1"].indicator_hits == ["P7-I3"]
    assert by_url["C2"].indicator_hits == ["P6-I1"]
    assert "C3" not in by_url  # irrelevant title stays below min_sim
    assert all(h.discovery_tag == "NEW" for h in hits)  # nothing in known list


def test_au_crosswalk_tags_known_when_title_matches_known_list():
    from lexora.collect.strategies import au_semantic_crosswalk

    catalogue = [{"id": "C1", "name": "Privacy Act 1988", "isPrincipal": True}]
    inds = [_ind("P7-I1", "privacy act personal data protection")]
    hits = au_semantic_crosswalk(
        _au_portal(), inds, embedder=FakeEmbedder(), top_k=1, min_sim=0.1,
        catalogue=catalogue, known_instruments=["Privacy Act 1988"],
    )
    assert hits and hits[0].discovery_tag == "KNOWN"
    assert hits[0].matched_instrument == "Privacy Act 1988"


# --- SG/MY candidate re-rank ------------------------------------------------

def test_semantic_rerank_promotes_concept_relevant_candidate():
    from lexora.collect.discovery import DiscoveryResult, _semantic_rerank

    agg = {
        "k1": DiscoveryResult(
            url="https://e/Act/A", title="Widget Marketing Act", source_type=SourceType.primary,
            score=0.9, via="api", is_pdf_link=False),
        "k2": DiscoveryResult(
            url="https://e/Act/B", title="Income Tax record retention Act", source_type=SourceType.primary,
            score=0.9, via="api", is_pdf_link=False),
    }
    by_key = {"k1": {"P7-I3"}, "k2": {"P7-I3"}}
    inds = [_ind("P7-I3", "minimum data retention record keeping")]
    _semantic_rerank(agg, by_key, inds, FakeEmbedder())
    assert agg["k2"].score > agg["k1"].score  # concept-relevant title wins the tie


# --- clause-level fusion ----------------------------------------------------

def _clause(cid: str, text: str) -> Clause:
    return Clause(
        clause_id=cid, document_id="doc", structural_path=cid,
        span=CanonicalSpan(span_id=cid + ".s", document_id="doc", char_start=0,
                           char_end=len(text), text=text),
    )


def _profile() -> SourceProfile:
    return SourceProfile(
        jurisdiction="Singapore", iso_code="SG", primary_language="en",
        legal_system=LegalSystem.common,
        portals=[PortalSpec(name="P", url="https://e/", source_type=SourceType.primary)],
    )


def test_fusion_surfaces_dense_only_clause():
    from lexora.classify.retrieval import build_index, retrieve_candidates

    clauses = [
        _clause("c1", "the quick brown fox jumps"),
        _clause("c2", "records must be retained for seven years by every company"),
        _clause("c3", "miscellaneous unrelated provisions about fishing"),
    ]
    index = build_index(clauses)
    ind = RDTIIIndicator(
        rdtii_id="7.3", submission_id="P7-I3", pillar=7,
        name="retention", description="retained records seven years company",
        keywords=["retained", "records"],
    )
    hits = retrieve_candidates(ind, _profile(), index, top_k=2, embedder=FakeEmbedder())
    assert hits[0].clause_id == "c2"
    assert hits[0].source == "fused"
