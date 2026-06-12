"""Constrained LLM verifier — a TIGHTENING gate on top of the verbatim contract.

The verifier takes (indicator, candidate clauses already past BM25 + verbatim)
and decides whether one of them *actually supports* the indicator. It exists to
cut the wrong-indicator / weak-mapping failure mode: BM25/dense surfacing a clause
means it shares vocabulary, not that the provision is on-point.

Two invariants make it safe:

* **Tightening only.** It chooses among the clauses the retriever already passed,
  or abstains. It can never introduce a clause, relax a gate, or widen coverage —
  the worst it does is drop a citation or flag it for review.
* **No quote text.** It returns clause IDs and a label; ``quote_span_id`` is
  filled from the canonical :class:`Clause` it selected, never from model output.
  The model is structurally unable to write the quote that ships.

The backend is optional (:mod:`lexora.classify.llm_client`); :func:`make_verifier`
returns ``None`` when the LLM is disabled or unavailable, and the pipeline then
behaves exactly as before (BM25 + verbatim only). Default is OFF, so the offline
test suite and a bare install never need a server.
"""
from __future__ import annotations

from lexora.models.citation import ClaimLabel, EvidenceClaim
from lexora.models.clause import Clause
from lexora.models.indicator import RDTIIIndicator

_CLAUSE_TEXT_CAP = 1200  # keep the prompt bounded; statutes have huge clauses
_VALID_LABELS = {label.value for label in ClaimLabel}

_SYSTEM = (
    "You are a legal-mapping auditor for the UN ESCAP RDTII framework. You are "
    "given one regulatory indicator and a short list of candidate statutory "
    "clauses already retrieved for it. Decide whether ONE clause clearly and "
    "substantively supports the indicator — not merely shares vocabulary.\n"
    "Rules:\n"
    "- Choose at most one clause. Abstain (clause_id=null, label=\"no_match\") if "
    "none is clearly on-point. When unsure, prefer abstaining or \"uncertain\".\n"
    "- You may ONLY pick from the given clause_id values. Never invent an id.\n"
    "- Do NOT write or quote any clause text. Return identifiers only.\n"
    "Respond with a single JSON object: "
    '{"clause_id": <string|null>, "label": "match"|"uncertain"|"no_match", '
    '"confidence": <0.0-1.0>, "rationale": <short string>}.'
)


class Verifier:
    """Wraps an LLM client to judge (indicator, candidate clauses)."""

    def __init__(self, client) -> None:
        self._client = client
        self.error_count = 0
        self.last_error_type: str | None = None

    def verify(
        self,
        indicator: RDTIIIndicator,
        candidates: list[Clause],
    ) -> EvidenceClaim | None:
        """Select 0 or 1 candidate clause that supports ``indicator``.

        Returns an :class:`EvidenceClaim` (clause + label, with the span id taken
        from the chosen canonical clause) for a ``match``/``uncertain`` verdict,
        or ``None`` to abstain (``no_match``, an unparseable reply, or a model
        that named a clause not in the candidate set — all treated as "drop").
        """
        if not candidates:
            return None
        by_id = {c.clause_id: c for c in candidates}
        try:
            data = self._client.chat(_SYSTEM, self._user_prompt(indicator, candidates),
                                     json_schema=_RESPONSE_SCHEMA)
        except Exception as exc:
            # A backend failure must not fabricate or block — treat as abstain so
            # the run can continue, while callers can still report the failure.
            self.error_count += 1
            self.last_error_type = type(exc).__name__
            return None

        clause_id = data.get("clause_id")
        label = str(data.get("label", "")).lower()
        if not clause_id or clause_id not in by_id:
            return None  # abstain / hallucinated id -> drop
        if label not in _VALID_LABELS or label == ClaimLabel.no_match.value:
            return None
        chosen = by_id[clause_id]
        confidence = data.get("confidence", 0.5)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.5
        return EvidenceClaim(
            indicator_id=indicator.submission_id,
            clause_id=chosen.clause_id,
            quote_span_id=chosen.span.span_id,  # from canonical clause, never the model
            label=ClaimLabel(label),
            confidence=confidence,
        )

    @staticmethod
    def _user_prompt(indicator: RDTIIIndicator, candidates: list[Clause]) -> str:
        lines = [
            f"Indicator {indicator.submission_id} — {indicator.name}",
            f"Definition: {indicator.description}",
        ]
        if indicator.scoring_criteria:
            lines.append(f"Scoring criteria: {indicator.scoring_criteria}")
        lines.append("\nCandidate clauses:")
        for c in candidates:
            text = c.span.text.strip().replace("\n", " ")
            if len(text) > _CLAUSE_TEXT_CAP:
                text = text[:_CLAUSE_TEXT_CAP] + " …"
            lines.append(f'- clause_id="{c.clause_id}" ({c.structural_path}): {text}')
        return "\n".join(lines)


# Documented for the server's structured-output mode; also spelled out in _SYSTEM.
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "clause_id": {"type": ["string", "null"]},
        "label": {"type": "string", "enum": ["match", "uncertain", "no_match"]},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["label"],
}


def make_verifier(use_llm: bool = False) -> Verifier | None:
    """Construct a :class:`Verifier`, or ``None`` when the LLM gate is off.

    ``use_llm`` is the explicit opt-in (the ``--verify`` flag / caller choice).
    Even when requested, returns ``None`` if the ``openai`` SDK is not installed,
    so callers can wire it unconditionally and the run degrades gracefully.
    """
    if not use_llm:
        return None
    from lexora.classify import llm_client

    if not llm_client.is_available():
        return None
    try:
        return Verifier(llm_client.LlmClient())
    except Exception:
        return None


def verify(
    indicator: RDTIIIndicator,
    candidates: list[Clause],
) -> EvidenceClaim | None:
    """Module-level convenience: build a default verifier and run one judgement.

    Returns ``None`` (abstain) when the LLM backend is unavailable, so this is
    safe to call unconditionally."""
    v = make_verifier(use_llm=True)
    return v.verify(indicator, candidates) if v is not None else None


__all__ = ["Verifier", "make_verifier", "verify"]
