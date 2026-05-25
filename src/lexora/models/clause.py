"""Canonical span + clause models.

A `CanonicalSpan` is a character-level location inside a document's canonical text.
A `Clause` is a structural unit (article/section/paragraph) that wraps one or more
spans.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CanonicalSpan(BaseModel):
    """A pointer into a stored canonical text. Citations are built from these."""

    span_id: str
    document_id: str
    page_number: int | None = None
    dom_anchor: str | None = None
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Clause(BaseModel):
    """A legal-structure unit (article/section/paragraph)."""

    clause_id: str
    document_id: str
    structural_path: str  # e.g. "Chapter III > Article 12 > Paragraph 2"
    article_number: str | None = None
    section_number: str | None = None
    paragraph_number: str | None = None
    span: CanonicalSpan
    is_citable: bool = True  # False if span.ocr_confidence below threshold
