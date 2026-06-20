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
    "explicit. Hard rule: do NOT quote or copy the provision's wording verbatim — "
    "paraphrase the mechanism only. "
    'Output a JSON object: {"rationale": "..."}.'
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"rationale": {"type": "string"}},
    "required": ["rationale"],
}
_CLAUSE_TEXT_CAP = 1200


class RationaleGenerator:
    """Produces a Mapping Rationale, preferring the LLM and falling back to the
    deterministic template on any failure, over-length, or verbatim-copy output.

    ``client`` is ``None`` for the inert (template-only) generator. Counters let a
    run report how often the LLM was used vs fell back."""

    def __init__(self, client=None) -> None:
        self._client = client
        self.llm_used = 0
        self.fallbacks = 0
        self.error_count = 0
        self.last_error_type: str | None = None

    def generate(
        self,
        indicator: RDTIIIndicator,
        profile: SourceProfile,
        clause: Clause,
        article_path: str,
    ) -> str:
        template = template_rationale(indicator, profile, clause, article_path)
        if self._client is None:
            return template
        try:
            data = self._client.chat(
                _SYSTEM,
                self._user_prompt(indicator, profile, clause, article_path),
                json_schema=_RESPONSE_SCHEMA,
            )
        except Exception as exc:  # noqa: BLE001 — never let the LLM break a citation
            self.error_count += 1
            self.last_error_type = type(exc).__name__
            self.fallbacks += 1
            return template

        rationale = (data.get("rationale") or "").strip()
        if (
            not rationale
            or len(rationale) > RATIONALE_MAX_CHARS
            or copies_provision(rationale, clause.span.text)
        ):
            self.fallbacks += 1
            return template
        self.llm_used += 1
        return rationale

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
        if indicator.scoring_criteria:
            lines.append(f"Scoring criteria: {indicator.scoring_criteria}")
        lines += [
            f"Economy: {profile.jurisdiction}",
            f"Provision: {article_path}",
            f'Provision text: """{text}"""',
            "Write the rationale JSON.",
        ]
        return "\n".join(lines)


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
        return RationaleGenerator(client=llm_client.LlmClient())
    except Exception:  # noqa: BLE001
        return RationaleGenerator(client=None)


__all__ = [
    "RATIONALE_MAX_CHARS",
    "RationaleGenerator",
    "copies_provision",
    "make_rationale_generator",
    "template_rationale",
]
