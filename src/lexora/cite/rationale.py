"""Mapping Rationale generation — deterministic template with an optional LLM layer.

The submission CSV "Mapping Rationale" column explains WHY a provision maps to its
indicator; the Policy judge reads it (official cap: 300 chars). Two generators:

* :func:`template_rationale` — deterministic, no network. Cites the provision,
  names the indicator, and lists the curated concept phrases that literally appear
  in the clause text. Always available; the safe floor. Built to stay within the
  300-char limit BY CONSTRUCTION (whole phrases appended only while they fit), so
  it is never a broken mid-sentence truncation.
* :class:`RationaleGenerator` with an LLM client — paraphrases the mechanism into a
  fluent sentence. Strictly guarded: the model explains the mapping and must NOT
  reproduce the provision text. An n-gram copy check rejects a leaking output; any
  error / empty / over-length / copy falls back to the template, so a citation
  always gets a rationale.

Red line: neither path ever writes the Verbatim Snippet — that is copied from the
canonical span by :func:`lexora.cite.citation_builder.build_citation`. The LLM only
authors the *explanation*. The generator is INERT by default (no client → template
only), mirroring the verifier, so the offline test suite needs no server.
"""
from __future__ import annotations

import re
from collections import Counter

from lexora.models.clause import Clause
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import SourceProfile

RATIONALE_MAX_CHARS = 300
_COPY_NGRAM = 6  # a 6+ word run copied from the provision counts as reproducing it


def template_rationale(
    indicator: RDTIIIndicator,
    profile: SourceProfile,
    clause: Clause,
    article_path: str,
) -> str:
    """Deterministic Mapping Rationale (no LLM). See module docstring."""
    loc = article_path or "This provision"
    base = f"{loc} maps to {indicator.submission_id} ({indicator.name})."
    if len(base) >= RATIONALE_MAX_CHARS:
        return base[:RATIONALE_MAX_CHARS]

    text_l = clause.span.text.lower()
    by_lang = profile.keywords_by_indicator.get(indicator.rdtii_id, {})
    seen: set[str] = set()
    matched: list[str] = []
    for lang in (profile.primary_language, *profile.additional_languages, *by_lang):
        for phrase in by_lang.get(lang, []):
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            if key and key in text_l:
                matched.append(phrase)
    if not matched:
        return base

    prefix = f"{base} Indicator concept(s) present: "
    chosen: list[str] = []
    for phrase in matched:
        trial = [*chosen, f"'{phrase}'"]
        if len(prefix + "; ".join(trial) + ".") > RATIONALE_MAX_CHARS:
            break
        chosen = trial
    if not chosen:
        return base
    return prefix + "; ".join(chosen) + "."


def _words(text: str) -> list[str]:
    return text.lower().split()


def copies_provision(rationale: str, provision_text: str, n: int = _COPY_NGRAM) -> bool:
    """True if any run of ``n`` consecutive provision words appears verbatim in the
    rationale — the guard that keeps the LLM explaining, not reproducing, the text."""
    words = _words(provision_text)
    if len(words) < n:
        # Very short provision: treat the whole thing appearing verbatim as a copy.
        return bool(words) and " ".join(words) in rationale.lower()
    r = rationale.lower()
    return any(" ".join(words[i:i + n]) in r for i in range(len(words) - n + 1))


_SYSTEM = (
    "You are a legal-tech assistant for the UN ESCAP RDTII project. Given a legal "
    "provision and one RDTII regulatory indicator, write ONE concise English "
    "sentence (max 280 characters) explaining WHY the provision maps to that "
    "indicator. Requirements: (1) reference the section/article number; (2) name "
    "the regulatory mechanism in your own words; (3) make the indicator linkage "
    "explicit. Hard rules: do NOT quote or copy the provision's wording verbatim — "
    "paraphrase the mechanism only. Base the rationale ONLY on the provided "
    "provision text and indicator definition; do NOT introduce facts from your own "
    "background knowledge into the rationale.\n"
    "This tool performs the MAPPING step only. Never assess, predict, suggest or "
    "mention a score for the indicator, in either field — scoring is a separate human "
    "step and is not being asked of you.\n"
    "Separate channel for your own knowledge: if you have relevant background "
    "knowledge worth flagging (e.g. the provision was later amended, or context not "
    "in the text), put it in 'notes' for analyst review only — it must NOT appear in "
    "the rationale and is NEVER used as the answer.\n"
    'Output a JSON object: {"rationale": "...", "notes": "..."}.'
)

# The prompt above asks; this enforces. Lexora's stated scope is ESCAP's Step 1 —
# find the provision and cite it — and it deliberately emits no Raw Score. The
# round-1 submission nevertheless shipped 184 rows whose Notes column speculated
# about one, 45 of them asserting a value outright ("The indicator score for
# Singapore would be 0"), plus one that managed "whether the framework is scored 0
# or 0". A policy judge reads that column, so an unasked-for score there is not a
# harmless aside: it is the tool contradicting its own scope in the deliverable.
_SENTENCE = re.compile(r"[^.!?]+[.!?]?")
_SCORE_TALK = re.compile(r"\bscor(?:e|es|ed|ing)\b", re.IGNORECASE)


def strip_score_talk(note: str) -> str:
    """Drop whole sentences that talk about scoring; "" if nothing else is left.

    Sentence-granular rather than word-granular so a surviving note still reads as
    prose. Every mention goes, not only assertions: the note is review-only, so
    losing a remark about scoring costs nothing, while keeping one invites the
    reader to treat it as this tool's output.
    """
    kept = [s for s in _SENTENCE.findall(note or "") if not _SCORE_TALK.search(s)]
    return " ".join(s.strip() for s in kept if s.strip()).strip()

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"rationale": {"type": "string"}, "notes": {"type": "string"}},
    "required": ["rationale"],
}
_REVIEW_PREFIX = "LLM rationale (own knowledge, NOT used as answer):"
_CLAUSE_TEXT_CAP = 1200


class RationaleGenerator:
    """Produces a Mapping Rationale, preferring the LLM and falling back to the
    deterministic template on any failure, over-length, or verbatim-copy output.

    ``client`` is ``None`` for the inert (template-only) generator. Counters let a
    run report how often the LLM was used vs fell back.

    WHY THE REASONS ARE COUNTED SEPARATELY. 29.1% of the Round-1 submission's rationales
    were template despite the LLM layer being on, and for two months nobody could say why,
    because a single ``fallbacks`` counter was incremented from five different places for
    five different reasons. "Read the counter" cannot answer the question the counter was
    added for. Four of those five reasons are OUR guards rejecting the model's answer, not
    the model failing to give one -- a distinction that decides whether the fix is a better
    prompt or a less trigger-happy guard.

    ``fallback_reasons`` is keyed by the "+"-joined set of every reason that fired, so the
    counts sum exactly to ``fallbacks`` AND overlap stays visible: a rationale that both
    copies the provision and runs long will not be fixed by relaxing the copy check alone.
    """

    def __init__(self, client=None, cache=None) -> None:
        self._client = client
        self._cache = cache
        self.llm_used = 0
        self.fallbacks = 0
        self.fallback_reasons: Counter[str] = Counter()
        self.score_talk_stripped = 0
        self.error_count = 0
        self.last_error_type: str | None = None

    def _record_fallback(self, *reasons: str) -> None:
        self.fallbacks += 1
        self.fallback_reasons["+".join(sorted(reasons))] += 1

    def _model_id(self) -> str:
        return str(getattr(self._client, "model", "") or "")

    def _cached(self, user: str) -> dict | None:
        """The model's RAW previous answer to this exact question, if we have it.

        Deliberately NOT the accepted rationale: the guards below re-run on every read, so
        tuning them is measurable over the whole corpus instead of being invisible on cached
        rows. See :mod:`lexora.cite.rationale_cache`."""
        if self._cache is None:
            return None
        return self._cache.get(self._cache.key(self._model_id(), _SYSTEM, user))

    def _store(self, user: str, data: dict) -> None:
        if self._cache is None or not isinstance(data, dict):
            return
        from lexora.cite.rationale_cache import system_fingerprint

        self._cache.put(
            self._cache.key(self._model_id(), _SYSTEM, user),
            self._model_id(),
            data,
            system_fingerprint(_SYSTEM),
        )

    def fallback_summary(self) -> str:
        """Reasons, commonest first — e.g. ``copied_provision 41, empty 7``."""
        return ", ".join(
            f"{reason} {n}" for reason, n in self.fallback_reasons.most_common()
        )

    def generate(
        self,
        indicator: RDTIIIndicator,
        profile: SourceProfile,
        clause: Clause,
        article_path: str,
    ) -> tuple[str, str]:
        """Return ``(rationale, review_note)``. ``rationale`` is the answer (LLM or
        template fallback); ``review_note`` carries the model's own-knowledge
        channel for analyst review (empty on the template path) and is never the
        answer."""
        template = template_rationale(indicator, profile, clause, article_path)
        if self._client is None:
            return template, ""
        user = self._user_prompt(indicator, profile, clause, article_path)
        data = self._cached(user)
        if data is None:
            try:
                data = self._client.chat(_SYSTEM, user, json_schema=_RESPONSE_SCHEMA)
            except Exception as exc:  # noqa: BLE001 — never let the LLM break a citation
                self.error_count += 1
                self.last_error_type = type(exc).__name__
                self._record_fallback("backend_error")
                return template, ""
            self._store(user, data)

        rationale = (data.get("rationale") or "").strip()
        raw_note = (data.get("notes") or "").strip()
        review_note = strip_score_talk(raw_note)
        if review_note != raw_note:
            self.score_talk_stripped += 1
        review_note = f"{_REVIEW_PREFIX} {review_note}" if review_note else ""
        # Every reason is evaluated, not short-circuited: which guard fired is the whole
        # point (see the class docstring), and an `or` chain can only ever name the first.
        reasons: list[str] = []
        if not rationale:
            reasons.append("empty")
        # A rationale that scores is out of scope, not merely wordy: fall back to the
        # deterministic template rather than ship it.
        if _SCORE_TALK.search(rationale):
            reasons.append("score_talk")
        if len(rationale) > RATIONALE_MAX_CHARS:
            reasons.append("too_long")
        if copies_provision(rationale, clause.span.text):
            reasons.append("copied_provision")
        if reasons:
            self._record_fallback(*reasons)
            return template, review_note
        self.llm_used += 1
        return rationale, review_note

    @staticmethod
    def _user_prompt(
        indicator: RDTIIIndicator,
        profile: SourceProfile,
        clause: Clause,
        article_path: str,
    ) -> str:
        text = clause.span.text.strip().replace("\n", " ")
        if len(text) > _CLAUSE_TEXT_CAP:
            text = text[:_CLAUSE_TEXT_CAP] + " …"
        lines = [
            f"Indicator: {indicator.submission_id} — {indicator.name}",
            f"Indicator description: {indicator.description}",
        ]
        if indicator.long_definition:
            lines.append(
                "Official RDTII Guide definition (authoritative — ground the rationale "
                f"in its scope and boundaries):\n{indicator.long_definition}"
            )
        if indicator.scoring_criteria:
            lines.append(f"Scoring criteria: {indicator.scoring_criteria}")
        lines += [
            f"Economy: {profile.jurisdiction}",
            f"Provision: {article_path}",
            f'Provision text: """{text}"""',
            "Write the rationale JSON.",
        ]
        return "\n".join(lines)


def _open_cache():
    """Best-effort cache. A store we cannot open must cost money, never correctness."""
    from lexora.cite import rationale_cache

    if not rationale_cache.cache_enabled():
        return None
    try:
        return rationale_cache.RationaleCache()
    except Exception:  # noqa: BLE001 — unwritable path, locked file, read-only volume
        return None


def make_rationale_generator(use_llm: bool = False) -> RationaleGenerator:
    """Construct a :class:`RationaleGenerator`. Always returns a usable generator:
    template-only when ``use_llm`` is false or the LLM backend is unavailable, so
    callers can wire it unconditionally and a citation always gets a rationale."""
    if not use_llm:
        return RationaleGenerator(client=None)
    from lexora.classify import llm_client

    if not llm_client.is_available():
        return RationaleGenerator(client=None)
    try:
        return RationaleGenerator(client=llm_client.LlmClient(), cache=_open_cache())
    except Exception:  # noqa: BLE001
        return RationaleGenerator(client=None)


__all__ = [
    "RATIONALE_MAX_CHARS",
    "RationaleGenerator",
    "copies_provision",
    "strip_score_talk",
    "make_rationale_generator",
    "template_rationale",
]
