"""RDTII indicator model — loaded from configs/rdtii_indicators.yaml."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RDTIIIndicator(BaseModel):
    """One indicator in the UN ESCAP RDTII 2.1 framework."""

    id: str  # e.g. "6.1", "7.3"
    pillar: int
    name: str
    description: str
    keywords: list[str] = Field(default_factory=list)
