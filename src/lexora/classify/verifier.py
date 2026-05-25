"""Constrained LLM verifier.

The verifier takes (indicator, candidate clauses) and returns an EvidenceClaim
referring to clause IDs only. It is structurally prevented from emitting quote
text — quote text is filled by the orchestrator from canonical storage.
"""
from __future__ import annotations

from lexora.models.citation import EvidenceClaim
from lexora.models.clause import Clause
from lexora.models.indicator import RDTIIIndicator


def verify(
    indicator: RDTIIIndicator,
    candidates: list[Clause],
) -> EvidenceClaim | None:  # pragma: no cover
    """Ask the LLM to select 0 or 1 clauses supporting the indicator.

    Implementation notes:
      - Use JSON-mode / function-calling so the model output is the EvidenceClaim
        Pydantic schema by construction.
      - The prompt must enumerate the candidate clause IDs, the indicator
        definition, and the four allowed labels (match | no_match | uncertain).
      - The prompt must instruct: "Return clause IDs only. Do not write quote
        text. Abstain if no clause clearly supports the indicator."
      - Return None when the model abstains.
    """
    raise NotImplementedError("Implement constrained LLM verifier.")
