"""Stage 5 — citation building + the verbatim contract validator.

This is the core anti-hallucination boundary. Every EvidenceClaim must pass the
validator before becoming a Citation. See docs/anti_hallucination.md.
"""
from lexora.cite.validator import OCR_CITABLE_THRESHOLD, normalize, validate_claim

__all__ = ["normalize", "validate_claim", "OCR_CITABLE_THRESHOLD"]
