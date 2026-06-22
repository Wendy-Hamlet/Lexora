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
    "- Base your judgement ONLY on the provided clause text and indicator "
    "definition, not on your own background knowledge. If your reasoning relies on "
    "anything outside the provided text, say so explicitly in 'rationale' so an "
    "analyst can review it — it must not change which clause you would pick from the "
    "text alone.\n"
    "Respond with a single JSON object: "
    '{"clause_id": <string|null>, "label": "match"|"uncertain"|"no_match", '
    '"confidence": <0.0-1.0>, "rationale": <short string>}.'
)


_PER_CELL_SYSTEM = (
    "You are a legal-mapping auditor for the UN ESCAP RDTII framework. You are "
    "given ONE regulatory indicator and several candidate statutory clauses already "
    "retrieved for it by a keyword/semantic search. Your job is to remove only the "
    "clauses that are CLEARLY off-topic, while keeping every clause that plausibly "
    "relates to this indicator's subject matter. A retriever passed these clauses "
    "because they share vocabulary; you exist to cut the cases where shared words are "
    "the ONLY connection (a broad statute flooding every indicator), not to demand "
    "that each clause be the single most central provision.\n"
    "Decision rule — DEFAULT TO KEEP:\n"
    "- KEEP a clause if its provision is on, about, or materially supports THIS "
    "indicator's topic — even if it is only partially on-point, one of several "
    "relevant clauses, or a supporting/qualifying provision rather than the core "
    "rule. Borderline relevance is a KEEP.\n"
    "- DROP a clause ONLY when its subject matter is unmistakably different from the "
    "indicator and it merely happens to share some vocabulary — e.g. a criminal "
    "offence and penalty, a bare definition with no operative rule, a pure "
    "cross-reference to an unrelated Act, or an administrative/procedural power with "
    "no bearing on the indicator's measure.\n"
    "- When you are genuinely unsure whether a clause is off-topic, KEEP it.\n"
    "- The keep list normally contains one or more clause_ids; return an empty list "
    "only when EVERY candidate is clearly off-topic.\n"
    "- You may ONLY return clause_id values from the given list. Never invent an id.\n"
    "- Do NOT write or quote any clause text. Return identifiers only.\n"
    "- Base your judgement ONLY on the provided clause text and indicator definition, "
    "not on outside knowledge. If your reasoning relies on anything outside the text, "
    "say so in 'rationale'.\n"
    'Respond with a single JSON object: {"keep": [<clause_id>, ...], '
    '"rationale": <short string>}.'
)

_PER_CELL_SCHEMA = {
    "type": "object",
    "properties": {
        "keep": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["keep"],
}


class Verifier:
    """Wraps an LLM client to judge (indicator, candidate clauses).

    ``mode`` selects the verdict shape:

    * ``"pick_one"`` (default, legacy ``--verify``): choose ≤1 supporting clause
      per indicator or abstain — :meth:`verify`.
    * ``"per_cell"`` (``--verify-cells``): judge EVERY (clause × indicator) cell
      keep/drop and return the kept subset — :meth:`judge_each`. This is the
      universal precision lane that kills the "broad statute floods all 9
      indicators on shared vocabulary" failure mode.
    """

    def __init__(self, client, *, mode: str = "pick_one") -> None:
        self._client = client
        self.mode = mode
        self.error_count = 0
        self.last_error_type: str | None = None

    def judge_each(
        self,
        indicator: RDTIIIndicator,
        candidates: list[Clause],
    ) -> set[str] | None:
        """Per-cell keep/drop: return the set of clause_ids that substantively
        support ``indicator``.

        Tightening only — the result is always a SUBSET of the candidate ids
        (hallucinated ids are dropped). An empty set is a real "drop all" verdict.
        Returns ``None`` ONLY on a backend error / unparseable reply, which the
        caller treats as "keep all" (fall back to the un-verified baseline) so an
        endpoint outage can never silently delete citations — separating
        error-drop from a true no-match abstention.
        """
        if not candidates:
            return set()
        by_id = {c.clause_id: c for c in candidates}
        try:
            data = self._client.chat(
                _PER_CELL_SYSTEM, self._user_prompt(indicator, candidates),
                json_schema=_PER_CELL_SCHEMA,
            )
        except Exception as exc:
            self.error_count += 1
            self.last_error_type = type(exc).__name__
            return None  # error -> caller keeps all (never worse than baseline)
        keep = data.get("keep") if isinstance(data, dict) else None
        if not isinstance(keep, list):
            return None  # unparseable -> treat as error, keep all
        return {str(cid) for cid in keep if str(cid) in by_id}

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


def make_verifier(use_llm: bool = False, *, mode: str = "pick_one") -> Verifier | None:
    """Construct a :class:`Verifier`, or ``None`` when the LLM gate is off.

    ``use_llm`` is the explicit opt-in (the ``--verify`` / ``--verify-cells`` flag).
    ``mode`` is ``"pick_one"`` (legacy) or ``"per_cell"`` (universal precision lane).
    Even when requested, returns ``None`` if the ``openai`` SDK is not installed,
    so callers can wire it unconditionally and the run degrades gracefully.
    """
    if not use_llm:
        return None
    from lexora.classify import llm_client

    if not llm_client.is_available():
        return None
    try:
        return Verifier(llm_client.LlmClient(), mode=mode)
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
