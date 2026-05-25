"""Hybrid retrieval: BM25 + multilingual embeddings.

For each RDTII indicator, return top-k candidate clauses across the corpus of
clauses for a given jurisdiction. Keyword expansion is fed from the indicator
config and the jurisdiction profile's `keywords_by_indicator` block.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalHit:
    clause_id: str
    score: float
    source: str   # "bm25" | "dense" | "fused"


def retrieve_candidates(
    indicator_id: str,
    jurisdiction: str,
    top_k: int = 20,
) -> list[RetrievalHit]:  # pragma: no cover
    """Return ranked clause candidates for the given indicator.

    TODO:
      1. BM25 over canonical text (rank_bm25 or OpenSearch/Tantivy).
      2. Dense retrieval with BGE-M3 (multilingual).
      3. Reciprocal-rank fusion to combine.
      4. Optional cross-encoder reranker on top-k.
    """
    raise NotImplementedError("Implement hybrid retrieval.")
