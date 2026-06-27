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

def test_embedding_model_resolves_both_env_var_names(monkeypatch):
    # G-1b wiring: config sets LEXORA_EMBEDDING_MODEL but the embedder historically
    # only read LEXORA_EMBED_MODEL, so the promised multilingual model was dead.
    from lexora.semantic.embedder import _FALLBACK_MODEL, resolve_model_name

    monkeypatch.delenv("LEXORA_EMBED_MODEL", raising=False)
    monkeypatch.delenv("LEXORA_EMBEDDING_MODEL", raising=False)
    assert resolve_model_name() == _FALLBACK_MODEL  # light English default

    monkeypatch.setenv("LEXORA_EMBEDDING_MODEL", "BAAI/bge-m3")
    assert resolve_model_name() == "BAAI/bge-m3"  # config knob now honoured

    monkeypatch.setenv("LEXORA_EMBED_MODEL", "legacy/override")
    assert resolve_model_name() == "legacy/override"  # explicit legacy var wins


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


def test_boilerplate_section_is_dropped_from_candidates():
    # An "Interpretation" front-matter clause that is vocabulary-dense for the
    # indicator must not be returned: it is non-operative and otherwise floats to
    # the top of the dense channel for every indicator (P-4 finding).
    from lexora.classify.retrieval import _is_boilerplate, build_index, retrieve_candidates

    boilerplate = _clause(
        "c1", '2. Interpretation In this Act, "personal data", "transfer", '
        '"organisation" and "consent" have the meanings given below')
    operative = _clause(
        "c2", "26. An organisation must not transfer personal data to a country "
        "outside Singapore except with comparable protection")
    assert _is_boilerplate(boilerplate) and not _is_boilerplate(operative)

    index = build_index([boilerplate, operative])
    ind = RDTIIIndicator(
        rdtii_id="6.4", submission_id="P6-I4", pillar=6, name="transfer",
        description="transfer of personal data to a country outside with consent",
        keywords=["transfer", "personal", "data", "consent", "organisation"],
    )
    hits = retrieve_candidates(ind, _profile(), index, top_k=2, embedder=FakeEmbedder())
    ids = [h.clause_id for h in hits]
    assert "c1" not in ids and "c2" in ids
    # with the filter off, the boilerplate clause is eligible again
    kept = retrieve_candidates(ind, _profile(), index, top_k=2, embedder=FakeEmbedder(),
                               drop_boilerplate=False)
    assert "c1" in [h.clause_id for h in kept]


class DictEmbedder:
    """Embedder with hand-assigned 2-D vectors per text, so the dense ranking is
    fully controlled and decoupled from BM25 (a bag-of-words fake can't diverge
    from BM25 on purpose; this can)."""

    def __init__(self, vecs: dict[str, list[float]]):
        self._vecs = vecs

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out = np.zeros((len(texts), 2), dtype=np.float32)
        for r, t in enumerate(texts):
            out[r] = self._vecs.get(t.strip(), [0.0, 0.0])
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def test_bm25_anchored_rank1_protects_precision_against_dense_demotion():
    # BM25 ranks cA (the on-point clause) first; the dense channel disagrees and
    # ranks a vocabulary decoy (cC) first. Plain RRF lets the decoy take rank 1;
    # the bm25-anchored default pins BM25's top hit back at rank 1 (the P-4 fix).
    from lexora.classify.retrieval import build_index, retrieve_candidates

    cA = _clause("26", "transfer personal data outside the country to a place")
    cB = _clause("99", "general miscellaneous matters about forms")
    cC = _clause("8", "personal data may be collected after consent is given")
    index = build_index([cA, cB, cC])
    ind = RDTIIIndicator(
        rdtii_id="6.4", submission_id="P6-I4", pillar=6, name="transfer",
        description="transfer of personal data outside the country", keywords=["transfer"],
    )
    concept = "transfer of personal data outside the country transfer"
    emb = DictEmbedder({
        concept: [1.0, 0.0],
        cA.span.text: [0.0, 1.0],   # dense thinks cA is unrelated
        cC.span.text: [1.0, 0.0],   # dense ranks the decoy top
        cB.span.text: [0.7, 0.7],
    })

    bm25 = retrieve_candidates(ind, _profile(), index, top_k=1, use_semantic=False)
    assert bm25[0].clause_id == cA.clause_id  # BM25's precise pick

    plain = retrieve_candidates(ind, _profile(), index, top_k=1, embedder=emb,
                                anchor_bm25_top1=False)
    anchored = retrieve_candidates(ind, _profile(), index, top_k=1, embedder=emb,
                                   anchor_bm25_top1=True)
    assert plain[0].clause_id == cC.clause_id      # dense demotes the right clause
    assert anchored[0].clause_id == cA.clause_id   # anchoring restores it


# --- cross-encoder rerank orchestration -------------------------------------

class FakeReranker:
    """Deterministic cross-encoder stub: scores each doc by how many of the given
    ``prefer`` phrases it contains, so a test fully controls the rerank order
    independently of BM25 (the real model is exercised live elsewhere)."""

    def __init__(self, prefer: list[str]):
        self._prefer = [p.lower() for p in prefer]

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        scored = [
            (i, float(sum(p in d.lower() for p in self._prefer)))
            for i, d in enumerate(documents)
        ]
        return sorted(scored, key=lambda pair: pair[1], reverse=True)


def test_reranker_reorders_pool_keeps_bm25_score_and_tag():
    # BM25 surfaces both clauses on shared vocabulary; the cross-encoder prefers the
    # genuinely on-point one. The rerank result must put it first, tag the hit
    # "reranked", and keep the RAW BM25 score (so the confidence gate is unchanged).
    from lexora.classify.retrieval import build_index, retrieve_candidates

    decoy = _clause(
        "c1", "transfer of functions and transfer of staff between agencies on transfer")
    onpoint = _clause(
        "c2", "an organisation must not transfer personal data outside the country "
        "unless comparable protection applies to the transfer")
    index = build_index([decoy, onpoint])
    ind = RDTIIIndicator(
        rdtii_id="6.4", submission_id="P6-I4", pillar=6, name="transfer",
        description="cross-border transfer of personal data subject to conditions",
        keywords=["transfer"],
    )

    # Baseline BM25 order (no rerank) — decoy is vocabulary-dense on "transfer".
    base = retrieve_candidates(ind, _profile(), index, top_k=2, use_semantic=False)
    base_score = {h.clause_id: h.score for h in base}

    rr = FakeReranker(prefer=["personal data", "comparable protection"])
    hits = retrieve_candidates(ind, _profile(), index, top_k=2, use_semantic=False,
                               reranker=rr)
    assert [h.clause_id for h in hits] == ["c2", "c1"]   # cross-encoder reorders
    assert all(h.source == "reranked" for h in hits)
    # raw BM25 score is preserved per clause (rerank changes order, not the scale)
    assert hits[0].score == base_score["c2"]
    assert hits[1].score == base_score["c1"]


def test_reranker_takes_precedence_over_dense():
    # When both a dense embedder and a reranker are supplied, the cross-encoder owns
    # the final order (the dense fusion path is skipped) — the documented precedence.
    from lexora.classify.retrieval import build_index, retrieve_candidates

    c1 = _clause("c1", "alpha provision about widgets and transfer")
    c2 = _clause("c2", "beta provision about transfer of personal data abroad")
    index = build_index([c1, c2])
    ind = RDTIIIndicator(
        rdtii_id="6.4", submission_id="P6-I4", pillar=6, name="transfer",
        description="transfer of personal data abroad", keywords=["transfer"],
    )
    rr = FakeReranker(prefer=["personal data"])
    hits = retrieve_candidates(ind, _profile(), index, top_k=2, embedder=FakeEmbedder(),
                               reranker=rr)
    assert hits[0].clause_id == "c2"
    assert hits[0].source == "reranked"  # not "fused"
