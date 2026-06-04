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
    scoring_criteria: str = ""
    possible_scores: list[float] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    @property
    def id(self) -> str:
        """Canonical id, used for internal lookups (retrieval, keyword packs)."""
        return self.rdtii_id
