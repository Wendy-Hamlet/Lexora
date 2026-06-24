"""RDTII indicator model — loaded from configs/rdtii_indicators.yaml."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RDTIIIndicator(BaseModel):
    """One regulatory indicator in the UN ESCAP RDTII 2.1 framework.

    Two codes are carried deliberately:
      * ``rdtii_id``      — the canonical framework number (e.g. "6.4"). This is
                            what the gold Round 1 Database uses and what we key
                            jurisdiction keyword packs by.
      * ``submission_id`` — the official output code (e.g. "P6-I4") required in
                            the "Indicator ID" column of OUTPUT_TEMPLATE.
    """

    rdtii_id: str  # e.g. "6.4"
    submission_id: str  # e.g. "P6-I4"
    pillar: int
    name: str
    description: str
    # The official RDTII 2.1 Guide's FULL "long" definition of this indicator
    # (verbatim-faithful), carrying the boundary rules the one-line `description`
    # omits — e.g. 6.1-ban vs 6.4-conditional, 6.2-storage vs 6.3-infrastructure,
    # 7.3-duration vs 6.2-location. Fed to the LLM verifier/rationale so it applies
    # the authoritative discriminators, not just shared vocabulary. Empty when the
    # YAML has no long_definition for the indicator.
    long_definition: str = ""
    scoring_criteria: str = ""
    possible_scores: list[float] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    # Concept phrases used to DISCOVER instruments for this indicator (full-text
    # portal search), as opposed to `keywords` which score clauses within a
    # document. A few natural-language phrases that surface the operative rule
    # across the flagship law AND sectoral statutes (e.g. 7.3 "keep records for
    # at least" reaches the Income Tax / Employment Acts, not just the PDPA).
    discovery_queries: list[str] = Field(default_factory=list)

    @property
    def id(self) -> str:
        """Canonical id, used for internal lookups (retrieval, keyword packs)."""
        return self.rdtii_id

    def query_phrases(self, limit: int | None = None) -> list[str]:
        """Phrases to drive per-indicator discovery search.

        Prefers the curated `discovery_queries`; falls back to the indicator
        name plus its keywords so an indicator with no curated phrases still
        discovers something. `limit` caps the count (browser-rendered portals
        such as SG SSO pay one page fetch per phrase)."""
        phrases = list(self.discovery_queries) or [self.name, *self.keywords]
        return phrases[:limit] if limit else phrases
