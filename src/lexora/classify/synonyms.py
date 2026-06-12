"""LLM cross-lingual synonym generation (G-3a) — query-side expansion only.

The retrieval cliff on a new jurisdiction is two-sided. G-1a/G-1c fixed the
*corpus* side (a Chinese/Thai clause now tokenizes and parses); this fixes the
*query* side: BM25's query is built from the English RDTII indicator text, which
shares no tokens with a Chinese statute, so coverage stays zero until the query
carries local-language legal terms. Section 2.9 produced those terms by hand for
AU; doing it by hand for every indicator × language does not scale.

This module asks the LLM to produce, for one indicator and a target language, the
operative legal phrases that jurisdiction's statutes would actually use — fed into
``profile.keywords_by_indicator[indicator.id][lang]`` exactly where the hand-tuned
synonyms already live. Turning "fit-to-answer" hand tuning into a reproducible,
language-general step is the whole point of G-3a.

**Red line (same as the verifier).** The LLM only emits *query* terms. It never
writes, quotes, or attributes clause text; the verbatim citation always comes
from canonical storage. Generated terms are search hints, not evidence, so a
hallucinated term can at worst cost a little precision — never a fabricated quote.
Output is meant to pass through human review before being committed to a config
(see scripts/gen_synonyms.py), mirroring the roadmap's "LLM drafts, human gates".
"""
from __future__ import annotations

from lexora.models.indicator import RDTIIIndicator

_MAX_TERMS = 12  # keep query expansion bounded; long bags dilute BM25
_MAX_TERM_LEN = 80  # a synonym is a phrase, not a paragraph

_SYSTEM = (
    "You expand a single regulatory indicator from the UN ESCAP RDTII framework "
    "into local-language legal SEARCH TERMS for keyword retrieval over that "
    "jurisdiction's statutes. \n"
    "Rules:\n"
    "- Output ONLY operative legal phrases / synonyms as they would appear in the "
    "target language's legislation (e.g. the statutory term of art, not a gloss).\n"
    "- Target language is given; write the terms in THAT language's script.\n"
    "- Do NOT quote, invent, or attribute any specific statute, section, or case. "
    "These are search hints, not citations.\n"
    "- No explanations, numbering, or commentary.\n"
    'Respond with a single JSON object: {"terms": [<string>, ...]} with at most '
    f"{_MAX_TERMS} concise terms."
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"terms": {"type": "array", "items": {"type": "string"}}},
    "required": ["terms"],
}


def _clean_terms(raw) -> list[str]:
    """Normalize the model's term list: strings only, trimmed, deduped, bounded."""
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        term = " ".join(item.split()).strip("\"'·-—–•* \t")
        if not term or len(term) > _MAX_TERM_LEN:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= _MAX_TERMS:
            break
    return out


class SynonymGenerator:
    """Wraps an LLM client to draft query-side synonyms for one indicator."""

    def __init__(self, client) -> None:
        self._client = client
        self.error_count = 0
        self.last_error_type: str | None = None

    def generate(self, indicator: RDTIIIndicator, language: str) -> list[str]:
        """Return local-language search terms for ``indicator`` (``[]`` on failure).

        A backend error degrades to no expansion — the base English query still
        runs — so this is safe to call unconditionally.
        """
        try:
            data = self._client.chat(
                _SYSTEM, self._user_prompt(indicator, language), json_schema=_RESPONSE_SCHEMA
            )
        except Exception as exc:
            self.error_count += 1
            self.last_error_type = type(exc).__name__
            return []
        return _clean_terms(data.get("terms"))

    @staticmethod
    def _user_prompt(indicator: RDTIIIndicator, language: str) -> str:
        lines = [
            f"Target language: {language}",
            f"Indicator {indicator.submission_id} — {indicator.name}",
            f"Definition: {indicator.description}",
        ]
        if indicator.keywords:
            lines.append("English seed terms: " + ", ".join(indicator.keywords))
        return "\n".join(lines)


def make_synonym_generator(use_llm: bool = False) -> SynonymGenerator | None:
    """Construct a :class:`SynonymGenerator`, or ``None`` when the LLM is off.

    Mirrors :func:`lexora.classify.verifier.make_verifier`: explicit opt-in, and
    ``None`` when the ``openai`` SDK is absent so callers can wire it
    unconditionally and degrade to the base (English) query.
    """
    if not use_llm:
        return None
    from lexora.classify import llm_client

    if not llm_client.is_available():
        return None
    try:
        return SynonymGenerator(llm_client.LlmClient())
    except Exception:
        return None


__all__ = ["SynonymGenerator", "make_synonym_generator"]
