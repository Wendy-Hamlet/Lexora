"""Stage 4 — classification.

Hybrid retrieval (BM25 + multilingual embeddings) returns candidate clause IDs
for each RDTII indicator. A constrained LLM verifier maps retrieved clauses to
indicators or abstains. The verifier is structurally prevented from emitting
free text, citing material outside the candidate set, or deciding legal
precedence — see docs/anti_hallucination.md.
"""
