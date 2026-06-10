"""Semantic (dense-embedding) layer — optional.

Keyword matching alone (BM25 in classify, token-overlap in discovery) misses
instruments whose statutory vocabulary differs from the RDTII concept phrasing:
an Income Tax Act that says "keep records for seven years" never lexically hits
a "minimum data retention" query, and Australia's name-only portal cannot find a
statute that is not already in the known list. This package adds a dense channel
— sentence embeddings + cosine similarity — used to (a) build a semantic
crosswalk from indicator concepts to a portal's title catalogue (AU), (b)
re-rank harvested candidates so the right sectoral law survives the budget cut
(SG/MY), and (c) fuse with BM25 for clause->indicator mapping.

The embedding backend is an *optional* dependency (fastembed). Every caller gates
on :func:`embedder.is_available` and degrades to the keyword-only behaviour when
it is absent, so the offline test suite and a bare install keep working.
"""
