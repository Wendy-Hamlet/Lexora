"""Pydantic data contracts shared across pipeline stages.

These models are the canonical interface between modules. Changing a field
here is a breaking change — coordinate before merging.
"""
from lexora.models.citation import Citation, EvidenceClaim, ReviewStatus
from lexora.models.clause import CanonicalSpan, Clause
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import LegalSystem, PortalSpec, RawDocument, SourceProfile, SourceType

__all__ = [
    "Citation",
    "EvidenceClaim",
    "ReviewStatus",
    "CanonicalSpan",
    "Clause",
    "RDTIIIndicator",
    "LegalSystem",
    "PortalSpec",
    "RawDocument",
    "SourceProfile",
    "SourceType",
]
