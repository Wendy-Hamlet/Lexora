"""Retrieval: BM25 over clauses (Slice 0).

Dense BGE-M3 embeddings and reciprocal-rank fusion are layered in a later
slice; the BM25 channel alone is enough to drive the verbatim contract
end-to-end and to demonstrate the orchestrator handing only `clause_id`s to
the cite stage.
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
    """A keyword index over a list of clauses, scored with Okapi BM25."""

    def __init__(self, clauses: list[Clause]):
        self.clauses = clauses
        self._tokenized = [_tokenize(c.span.text) for c in clauses]
        if not self._tokenized:
            self._bm25 = None
        else:
            self._bm25 = BM25Okapi(self._tokenized)

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


def retrieve_candidates(
    indicator: RDTIIIndicator,
    profile: SourceProfile,
    index: BM25Index,
    *,
    top_k: int = 5,
    language: str | None = None,
) -> list[RetrievalHit]:
    """Return BM25-ranked clause candidates for a single indicator."""
    return index.query(_expand_query(indicator, profile, language), top_k=top_k)


__all__ = ["BM25Index", "RetrievalHit", "build_index", "retrieve_candidates"]
