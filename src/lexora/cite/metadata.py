"""Document-metadata extraction — the generic LLM fallback for the submission's
``Law Number / Ref`` and ``Last Amended`` columns.

Structured portal channels (e.g. the AU register OData ``number``/``year`` fields)
are the primary, most reliable source and are captured at discovery time. This
module is the GENERALIZER for everything else: any law on a portal without a
structured connector — including NEW laws and finals-round economies — by reading
the official front/back matter (masthead, endnotes, compilation/amendment notes)
of the already-fetched document.

An LLM reads the text and returns the law number + most-recent-amendment year. The
output is then VERIFIED against the source: every numeric token the model returns
must actually appear in the document text, so the model can surface a printed value
but never fabricate one. A field that fails verification is dropped (left blank).
The extractor is INERT by default (no client → returns blanks), mirroring the
verifier / rationale generator, so the offline test suite needs no server.
"""
from __future__ import annotations

import re

_HEAD_CHARS = 3500  # masthead / front matter (title, act number, assent dates)
_TAIL_CHARS = 2500  # endnotes / amendment history / compilation notes

_YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
_DIGIT_RUN = re.compile(r"\d+")

_SYSTEM = (
    "You extract two metadata fields from the official front/back matter of a law. "
    "(1) law_number — the official act/law number exactly as printed (e.g. 'Act 709', "
    "'No. 119 of 1988', 'No. 9 of 2018'). (2) last_amended — the YEAR (YYYY) of the most "
    "recent amendment or compilation; if the law was never amended, the year it was made. "
    "Use ONLY values explicitly printed in the provided text — do NOT infer or guess. If a "
    "field is not present, return an empty string for it. "
    'Output a JSON object: {"law_number": "...", "last_amended": "..."}.'
)
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"law_number": {"type": "string"}, "last_amended": {"type": "string"}},
    "required": ["law_number", "last_amended"],
}


def _document_window(text: str) -> str:
    """Head + tail of the document — where legal metadata lives — keeping the prompt
    bounded for long statutes."""
    if len(text) <= _HEAD_CHARS + _TAIL_CHARS:
        return text
    return f"{text[:_HEAD_CHARS]}\n\n[...]\n\n{text[-_TAIL_CHARS:]}"


def _verify_in_text(value: str, text_lower: str) -> bool:
    """True if every numeric token of ``value`` appears verbatim in the source text.
    A fabrication guard that tolerates formatting differences ('No. 119 of 1988' vs
    'No. 119, 1988') while rejecting numbers/years not printed in the document."""
    runs = _DIGIT_RUN.findall(value)
    return bool(runs) and all(run in text_lower for run in runs)


class MetadataExtractor:
    """Reads (last_amended, law_number) from a document's text via the LLM, then
    verifies each against the source. ``client`` is ``None`` for the inert
    extractor (returns blanks). Counters let a run report usage."""

    def __init__(self, client=None) -> None:
        self._client = client
        self.extracted = 0
        self.rejected = 0
        self.error_count = 0
        self.last_error_type: str | None = None

    def extract(self, document_text: str, jurisdiction: str, title: str) -> tuple[str, str]:
        """Return ``(last_amended, law_number)``; either may be ``""`` when absent,
        unverifiable, or the backend is unavailable."""
        if self._client is None or not document_text:
            return "", ""
        window = _document_window(document_text)
        user = (
            f"Jurisdiction: {jurisdiction}\n"
            f"Law title: {title}\n\n"
            f"Document text (front and back matter):\n\"\"\"{window}\"\"\"\n"
            "Return the metadata JSON."
        )
        try:
            data = self._client.chat(_SYSTEM, user, json_schema=_RESPONSE_SCHEMA)
        except Exception as exc:  # noqa: BLE001 — never let extraction break a citation
            self.error_count += 1
            self.last_error_type = type(exc).__name__
            return "", ""

        text_lower = document_text.lower()
        law_number = (data.get("law_number") or "").strip()
        last_amended = (data.get("last_amended") or "").strip()

        if law_number and not _verify_in_text(law_number, text_lower):
            self.rejected += 1
            law_number = ""
        # last_amended must be a 4-digit year that is actually printed in the text.
        ym = _YEAR_RE.search(last_amended)
        if not ym or ym.group(0) not in text_lower:
            if last_amended:
                self.rejected += 1
            last_amended = ""
        else:
            last_amended = ym.group(0)

        if law_number or last_amended:
            self.extracted += 1
        return last_amended, law_number


def make_metadata_extractor(use_llm: bool = False) -> MetadataExtractor:
    """Construct a :class:`MetadataExtractor`. Returns the inert (blank-returning)
    extractor when ``use_llm`` is false or the LLM backend is unavailable, so
    callers can wire it unconditionally."""
    if not use_llm:
        return MetadataExtractor(client=None)
    from lexora.classify import llm_client

    if not llm_client.is_available():
        return MetadataExtractor(client=None)
    try:
        return MetadataExtractor(client=llm_client.LlmClient())
    except Exception:  # noqa: BLE001
        return MetadataExtractor(client=None)


__all__ = ["MetadataExtractor", "make_metadata_extractor"]
