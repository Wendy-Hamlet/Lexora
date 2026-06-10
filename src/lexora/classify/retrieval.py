"""Retrieval: BM25, optionally fused with a dense channel.

The BM25 channel alone drives the verbatim contract end-to-end. When the optional
dense backend (:mod:`lexora.semantic.embedder`) is installed, a sentence-embedding
channel is fused in by reciprocal-rank fusion: dense retrieval surfaces clauses
whose wording differs from the indicator's keywords (e.g. an "retain ... for seven
years" provision under a "minimum data retention" indicator) and reorders the pool
so the semantically-right section ranks first. Reported scores stay on the BM25
scale, so the orchestrator's confidence gate (`tanh(bm25/5)`) is unchanged whether
or not the dense channel is active.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from lexora.models.clause import Clause
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import SourceProfile

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


@dataclass
class RetrievalHit:
    clause_id: str
    score: float
    source: str  # "bm25" | "dense" | "fused"


class BM25Index:
    """A keyword index over a list of clauses, scored with Okapi BM25.

    Holds an optional, lazily-built dense embedding of the same clauses so the
    dense channel embeds each document's clauses once and reuses them across all
    indicators mapped against that document."""

    def __init__(self, clauses: list[Clause]):
        self.clauses = clauses
        self._tokenized = [_tokenize(c.span.text) for c in clauses]
        if not self._tokenized:
            self._bm25 = None
        else:
            self._bm25 = BM25Okapi(self._tokenized)
        self._clause_vecs = None  # lazy dense matrix, (n_clauses, dim)

    def query(self, terms: list[str], top_k: int = 20) -> list[RetrievalHit]:
        """Return the top-k clauses by BM25 score (unfiltered).

        BM25 raw scores can be negative when the corpus is small and IDF is
        unusual (e.g. only 2 documents); callers apply their own `min_score`
        cutoff rather than this index pre-filtering.
        """
        if self._bm25 is None or not terms:
            return []
        scores = self._bm25.get_scores(terms)
        ranked = sorted(
            ((idx, float(score)) for idx, score in enumerate(scores)),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]
        return [
            RetrievalHit(clause_id=self.clauses[idx].clause_id, score=score, source="bm25")
            for idx, score in ranked
        ]

    def bm25_scores(self, terms: list[str]):
        """Raw BM25 score for every clause (array aligned to ``self.clauses``)."""
        if self._bm25 is None or not terms:
            return None
        return self._bm25.get_scores(terms)

    def dense_ranking(self, query_text: str, embedder, top_k: int) -> list[int]:
        """Clause indices ranked by cosine similarity to ``query_text``.

        Embeds the clauses once (cached on the index) and the query each call."""
        from lexora.semantic.embedder import cosine_topk

        if not self.clauses or not query_text:
            return []
        if self._clause_vecs is None:
            self._clause_vecs = embedder.encode([c.span.text for c in self.clauses])
        qvec = embedder.encode([query_text])
        if qvec.size == 0 or self._clause_vecs.size == 0:
            return []
        return [idx for idx, _ in cosine_topk(qvec[0], self._clause_vecs, top_k)]


def build_index(clauses: list[Clause]) -> BM25Index:
    return BM25Index(clauses)


def _expand_query(
    indicator: RDTIIIndicator,
    profile: SourceProfile,
    language: str | None = None,
) -> list[str]:
    """Build the BM25 query token bag.

    Sources, in priority order:
      1. indicator.description (the canonical RDTII text)
      2. indicator.keywords (English seed terms)
      3. profile.keywords_by_indicator[indicator.id][lang] for the active language
         (jurisdiction-specific synonyms, may be non-English)
    """
    tokens = _tokenize(indicator.description)
    for kw in indicator.keywords:
        tokens.extend(_tokenize(kw))
    lang = language or profile.primary_language
    by_lang = profile.keywords_by_indicator.get(indicator.id, {})
    for kw in by_lang.get(lang, []):
        tokens.extend(_tokenize(kw))
    return tokens


def _concept_text(
    indicator: RDTIIIndicator,
    profile: SourceProfile,
    language: str | None = None,
) -> str:
    """Natural-language concept text for the dense channel (vs the BM25 token bag).

    Embeddings work on phrasing, not token bags, so this joins the indicator's
    description, keywords and the jurisdiction's synonym pack into one string."""
    parts = [indicator.description, *indicator.keywords]
    lang = language or profile.primary_language
    parts.extend(profile.keywords_by_indicator.get(indicator.id, {}).get(lang, []))
    return " ".join(p for p in parts if p)


def _maybe_embedder(use_semantic: bool):
    if not use_semantic:
        return None
    from lexora.semantic import embedder as emb_mod

    if not emb_mod.is_available():
        return None
    try:
        return emb_mod.get_embedder()
    except Exception:
        return None


def retrieve_candidates(
    indicator: RDTIIIndicator,
    profile: SourceProfile,
    index: BM25Index,
    *,
    top_k: int = 5,
    language: str | None = None,
    use_semantic: bool = True,
    embedder=None,
    pool_k: int = 20,
) -> list[RetrievalHit]:
    """Return clause candidates for a single indicator.

    BM25-only unless the dense backend is available: then the BM25 and dense
    rankings of the clause pool are fused by reciprocal-rank fusion and the top
    ``top_k`` are returned. Each hit keeps its raw BM25 score, so the caller's
    normalized confidence gate behaves identically with or without the dense
    channel — fusion changes *which* clauses rank first, not the score scale.
    """
    terms = _expand_query(indicator, profile, language)
    scores = index.bm25_scores(terms)
    if scores is None:
        return []
    score_by_id = {c.clause_id: float(scores[i]) for i, c in enumerate(index.clauses)}
    bm25_order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    emb = embedder if embedder is not None else _maybe_embedder(use_semantic)
    if emb is None:
        return [
            RetrievalHit(index.clauses[i].clause_id, float(scores[i]), "bm25")
            for i in bm25_order[:top_k]
        ]

    dense_idx = index.dense_ranking(_concept_text(indicator, profile, language), emb, pool_k)
    if not dense_idx:
        return [
            RetrievalHit(index.clauses[i].clause_id, float(scores[i]), "bm25")
            for i in bm25_order[:top_k]
        ]

    from lexora.semantic.embedder import reciprocal_rank_fusion

    bm25_ids = [index.clauses[i].clause_id for i in bm25_order[:pool_k]]
    dense_ids = [index.clauses[i].clause_id for i in dense_idx]
    fused = reciprocal_rank_fusion([bm25_ids, dense_ids])
    ordered = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:top_k]
    return [RetrievalHit(cid, score_by_id[cid], "fused") for cid in ordered]


__all__ = ["BM25Index", "RetrievalHit", "build_index", "retrieve_candidates"]
