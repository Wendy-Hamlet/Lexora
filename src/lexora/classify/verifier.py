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
from lexora.models.clause import CanonicalSpan, Clause
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


_PER_CLAUSE_SYSTEM = (
    "You are a legal-mapping auditor for the UN ESCAP RDTII framework. You are given the full "
    "list of indicators, then ONE statutory clause. Decide which indicators (if any) THIS "
    "single clause substantively supports as direct primary-source evidence — not merely shares "
    "vocabulary with. A clause may support zero, one, or several indicators.\n"
    "Decision rule:\n"
    "- Mark an indicator only when THIS clause's operative provision falls WITHIN that "
    "indicator's official definition and its boundaries with look-alike indicators. Judge the "
    "clause on its own text, not the rest of the Act.\n"
    "- An incidental mention, a bare definition with no operative rule, or a pure cross-reference "
    "to another Act is NOT support — leave such indicators out.\n"
    "- Distinguish look-alikes by the definitions: e.g. a conditional cross-border transfer rule "
    "is 6.4 not 6.1; storage location is 6.2 not retention duration 7.3.\n"
    "POLARITY: for indicators phrased as 'Lack of <framework>' (P7-I1 comprehensive "
    "data-protection framework; P7-I2 dedicated cybersecurity framework), a clause that PROVIDES "
    "or constitutes part of that framework IS supporting evidence.\n"
    # The two rules below resolve a UNIT MISMATCH, measured on the MY PDPA against the legal
    # group's provision-level gold (scripts/bench_judge.py). A framework indicator's official
    # definition asks an ECONOMY-level yes/no ("does a comprehensive horizontal framework
    # exist"), but our output unit is one clause. Judging each clause against an economy-level
    # test, the model admitted only the umbrella provisions and dropped the constituent
    # obligations: P7-I1 scored 2/9 gold sections. Spelling out that the whole is evidenced by
    # its parts took it to 9/9. The second rule is the necessary brake -- without it the model
    # reads "part of the framework" as the entire statute and marks 50 of 146 sections,
    # machinery included (precision 20%); bounding it to substantive duties and rights keeps
    # 9/9 while halving the output (30-33 citations, precision 33-37%) and, as a side effect,
    # makes the verdict markedly more stable across identical runs (88% -> 97% of clauses).
    "FRAMEWORK INDICATORS ARE EVIDENCED PROVISION BY PROVISION. Such an indicator's official "
    "definition describes the framework as a WHOLE, but you are mapping ONE clause at a time, "
    "and the whole is evidenced by each of its parts. So mark P7-I1 on EVERY clause that enacts "
    "a core data-protection obligation or data-subject right - the data-protection principles, "
    "notice and consent, security, retention, access, correction, withdrawal of consent, "
    "sensitive/special categories, the regulator's enforcement powers, and cross-border transfer "
    "each count INDIVIDUALLY and on their own. Do NOT reserve the mark for the umbrella or "
    "principles clause, and do NOT drop a clause merely because it covers only one part of the "
    "framework.\n"
    "BUT A FRAMEWORK IS NOT ITS WHOLE STATUTE. Mark P7-I1 only where the clause states a "
    "SUBSTANTIVE duty owed by data controllers/processors or a RIGHT held by data subjects. Do "
    "NOT mark the statute's machinery: registration, licensing or certification procedure, "
    "appeals and tribunal constitution, the regulator's appointment/staffing/funding, inspection "
    "and search procedure, offences and penalties, service of notices, bare definitions, "
    "commencement, savings and transitional provisions, and schedules of forms. If a clause "
    "would read as 'how the regime is administered' rather than 'what a data controller must do "
    "or what a data subject may demand', leave P7-I1 out.\n"
    "- Base your judgement ONLY on the provided clause text and the indicator definitions, not on "
    "outside knowledge. You may ONLY return submission_id values from the given list.\n"
    'Respond with a single JSON object: {"indicators": [<submission_id>, ...], '
    '"rationale": <short string>}.'
)

_PER_CLAUSE_SCHEMA = {
    "type": "object",
    "properties": {
        "indicators": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["indicators"],
}

# Stand-in clause used ONLY to render the prompt template for fingerprinting the verdict
# cache (see Verifier._prompt_fingerprint). Never sent to a model. Its content is
# irrelevant, but it must never change: it is part of the cache key, so editing it would
# invalidate every stored verdict for no reason.
_SENTINEL_CLAUSE = Clause(
    clause_id="__fingerprint__",
    document_id="__fingerprint__",
    structural_path="__fingerprint__",
    span=CanonicalSpan(
        span_id="__fingerprint__", document_id="__fingerprint__",
        char_start=0, char_end=0, text="",
    ),
)


class Verifier:
    """Wraps an LLM client to judge (indicator, candidate clauses).

    ``mode`` selects the verdict shape:

    * ``"pick_one"`` (default, legacy ``--verify``): choose ≤1 supporting clause
      per indicator or abstain — :meth:`verify`.
    * ``"per_cell"`` (``--verify-cells``): judge EVERY (clause × indicator) cell
      keep/drop and return the kept subset — :meth:`judge_each`. This is the
      universal precision lane that kills the "broad statute floods all 9
      indicators on shared vocabulary" failure mode.
    * ``"per_clause"`` (9-in-1 per clause): judge ONE clause against ALL indicators
      in a single call — :meth:`judge_clause`. This is the relevance decision
      itself (not a tightening pass on a prior attribution): a focused single-clause
      question the model answers reliably, where a full-text 9-in-1 skim over a
      whole Act is noisy and buries sectoral provisions.
    """

    def __init__(self, client, *, mode: str = "pick_one", cache=None) -> None:
        self._client = client
        self.mode = mode
        self.error_count = 0
        self.last_error_type: str | None = None
        # Verdict cache (per_clause only; see judge_cache). None = disabled.
        self._cache = cache
        # Prompt fingerprint PER indicator catalogue, not one for the verifier's lifetime.
        # The fingerprint IS the question, and callers legitimately ask different ones from
        # the same verifier: the regime-2 brute judge narrows `indicators` per document, so
        # one run can put a three-indicator catalogue and the full nine-indicator catalogue
        # to the same model. Pinning the first one seen meant a verdict taken against three
        # indicators was later served, unchanged, as the answer to the nine-indicator
        # question -- silently dropping the six that were never in the prompt. That is
        # exactly the "cached answer outliving the question that produced it" this module's
        # docstring promises cannot happen.
        self._fingerprints: dict[tuple[str, ...], str] = {}
        # Clauses this verifier actually put to the model, vs served from cache. Counted
        # here rather than on the cache so the numbers survive the cache being switched
        # off -- a cost report that says "0 clauses judged" because the cache is absent
        # is not reporting a cold run, it is failing to report at all.
        self.judged = 0
        self.from_cache = 0

    def judge_clause(
        self,
        clause: Clause,
        indicators: list[RDTIIIndicator],
    ) -> set[str] | None:
        """9-in-1 per clause: return the submission_ids of the indicators THIS one
        clause substantively supports (possibly empty, possibly several).

        The result is always a SUBSET of the given indicators' ids (hallucinated
        ids dropped). An empty set is a real "supports nothing". Returns ``None``
        only on a backend error / unparseable reply, which the caller treats as a
        skip (the clause contributes no citation) so an outage cannot fabricate
        mappings.

        The verdict is a pure function of (model, indicator catalogue, clause text), so it
        is cached across runs when one is configured — see :mod:`lexora.classify.judge_cache`.
        A failure is never cached: caching ``None`` would let one outage erase a citation
        permanently."""
        if not indicators:
            return set()
        valid = {i.submission_id for i in indicators}

        key = None
        if self._cache is not None:
            signature = tuple(i.submission_id for i in indicators)
            fingerprint = self._fingerprints.get(signature)
            if fingerprint is None:
                fingerprint = self._fingerprints[signature] = self._prompt_fingerprint(
                    indicators
                )
            key = self._cache.key(self._model_id(), fingerprint, clause)
            cached = self._cache.get(key)
            if cached is not None:
                self.from_cache += 1
                # Belt and braces: the fingerprint now pins this exact catalogue, so a hit
                # can only carry ids we asked about. Keep the intersection anyway -- it
                # costs nothing and a stale row from an older schema cannot leak through.
                return cached & valid

        try:
            data = self._client.chat(
                _PER_CLAUSE_SYSTEM, self._clause_prompt(clause, indicators),
                json_schema=_PER_CLAUSE_SCHEMA,
            )
        except Exception as exc:
            self.error_count += 1
            self.last_error_type = type(exc).__name__
            return None
        got = data.get("indicators") if isinstance(data, dict) else None
        if not isinstance(got, list):
            return None
        verdict = {str(x) for x in got if str(x) in valid}
        self.judged += 1
        if self._cache is not None and key is not None:
            self._cache.put(key, self._model_id(), verdict)
        return verdict

    def _model_id(self) -> str:
        return str(getattr(self._client, "model", "") or "")

    def _prompt_fingerprint(self, indicators: list[RDTIIIndicator]) -> str:
        """Fingerprint the question, template and all.

        Render the REAL prompt with a sentinel clause, so the hash covers not only the
        indicator definitions but the template that presents them -- field order, headers,
        wording. Changing how we ask can change the answer, so it must change the key.
        """
        from lexora.classify.judge_cache import prompt_fingerprint

        probe = self._clause_prompt(_SENTINEL_CLAUSE, indicators)
        return prompt_fingerprint(_PER_CLAUSE_SYSTEM, probe)

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
    def _clause_prompt(clause: Clause, indicators: list[RDTIIIndicator]) -> str:
        """The full indicator catalogue, THEN one clause (9-in-1 per clause).

        The order is load-bearing, and not for the model's sake. Prompt caching keys on a
        common PREFIX: the provider bills a repeated prefix at a fraction of the input rate
        (GLM: Y2 vs Y8 per 1M). The catalogue is 3,051 tokens and identical on every call;
        the clause is 68-247. With the clause first, the very first token differs each time,
        so nothing after it can ever match -- the catalogue was re-billed at full price on
        all 22,445 calls of the 2026-07-14 run, 68.5M tokens of the 73.2M total (94%), and
        the bill showed exactly the ~11% cache rate you would predict from the system prompt
        alone. Catalogue first makes the whole 3,051-token prefix cacheable.

        Putting the clause LAST also happens to be the better prompt (rules before the thing
        being judged, and the judged text sits in the most recent position), but that is a
        bonus -- the reason is the invoice.
        """
        text = clause.span.text.strip().replace("\n", " ")
        if len(text) > _CLAUSE_TEXT_CAP:
            text = text[:_CLAUSE_TEXT_CAP] + " …"
        lines = ["INDICATORS:"]
        for i in indicators:
            b = f"- {i.submission_id} ({i.name}): {i.description}"
            if i.long_definition:
                b += f"\n  Definition (scope + boundaries): {i.long_definition}"
            lines.append(b)
        lines += ["", f"CLAUSE ({clause.structural_path}):", text]
        return "\n".join(lines)

    @staticmethod
    def _user_prompt(indicator: RDTIIIndicator, candidates: list[Clause]) -> str:
        lines = [
            f"Indicator {indicator.submission_id} — {indicator.name}",
            f"Definition: {indicator.description}",
        ]
        if indicator.long_definition:
            # The authoritative RDTII Guide long definition carries the boundary
            # rules that separate look-alike indicators (e.g. a conditional-flow
            # clause is 6.4 not 6.1; storage is 6.2 not 6.3; retention=duration not
            # location). Drop a clause that the long definition assigns to a
            # DIFFERENT indicator.
            lines.append(
                "Official RDTII Guide definition (authoritative — apply its scope and "
                f"its boundaries with neighbouring indicators):\n{indicator.long_definition}"
            )
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


def make_verifier(
    use_llm: bool = False, *, mode: str = "pick_one", model: str | None = None
) -> Verifier | None:
    """Construct a :class:`Verifier`, or ``None`` when the LLM gate is off.

    ``use_llm`` is the explicit opt-in (the ``--verify`` / ``--verify-cells`` flag).
    ``mode`` is ``"pick_one"`` (legacy), ``"per_cell"`` (precision lane), or
    ``"per_clause"`` (9-in-1 per clause — the relevance decision itself). ``model``
    overrides the configured LLM model; ``per_clause`` additionally honours
    ``LEXORA_BRUTE_MODEL`` when it is set, so the relevance decision can be pinned to a
    different backend than the rest of the run. Unset means "follow ``LEXORA_LLM_MODEL``"
    — see the call site below for why anything else breaks the documented config swap.
    Even when requested, returns ``None`` if the ``openai`` SDK is not installed,
    so callers can wire it unconditionally and the run degrades gracefully.
    """
    if not use_llm:
        return None
    from lexora.classify import llm_client

    if not llm_client.is_available():
        return None
    if model is None and mode == "per_clause":
        from lexora.config import env_value

        # LEXORA_BRUTE_MODEL is an OVERRIDE, not a second source of truth. It used to
        # default to a hardcoded vendor model name, which quietly broke the one promise the
        # No-Vendor-Lock-in rubric actually tests: someone who points LEXORA_LLM_BASE_URL at
        # a local Ollama and sets LEXORA_LLM_MODEL=llama3 -- exactly what our README tells
        # them to do -- still had this lane ask that server for `deepseek-v4-flash`. The
        # server 404s, every judgement fails, and since this lane produces essentially all
        # of our output, "swap the model by changing .env" was false for the only lane that
        # matters. Empty (unset) now means "whatever LEXORA_LLM_MODEL says".
        model = env_value("LEXORA_BRUTE_MODEL", "") or None

    # The verdict cache serves the per_clause lane only: that is the one asking a question
    # with no state behind it (model + catalogue + clause text -> indicator set), and the
    # one that fires tens of thousands of times per run. A cache failure must never take
    # the run down with it, so fall back to an uncached judge.
    cache = None
    if mode == "per_clause":
        from lexora.classify import judge_cache

        if judge_cache.cache_enabled():
            try:
                cache = judge_cache.JudgeCache()
            except Exception:  # unwritable path, locked db, ...
                cache = None
    try:
        return Verifier(llm_client.LlmClient(model=model), mode=mode, cache=cache)
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
