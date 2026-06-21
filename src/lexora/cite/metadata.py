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
_BLOCK_MAX = 12000  # cap on the located legislative-history block

# Cross-jurisdiction structural markers for the legislative-history / endnote
# section, where the official law number and amendment years live. A blind
# fixed-size tail can clip the first entry (the enacting "Act N of YYYY"); locating
# the block by heading and sending it whole is jurisdiction-agnostic — every marker
# is a section LABEL, not a country-specific citation pattern. Lower-cased; matched
# only in the document's latter half so body prose ("an amendment to ...") is ignored.
_HISTORY_MARKERS = (
    "legislative history",
    "it is not part of the act",
    "amendment history",
    "endnotes",
    "endnote",
    "list of amendments",
    "amendments incorporated",
    "table of amendments",
    "this act has been amended",
    "notes to the",          # AU compilations: "Notes to the Privacy Act 1988"
    "amendment record",
)


def _locate_history_block(text: str) -> int | None:
    """Return the start index of the legislative-history / endnote block, or None.

    Searches only the latter half of the document (the section is always near the
    end) for the EARLIEST marker, then backs up a little to include its heading."""
    hay = text.lower()
    floor = len(text) // 2
    best: int | None = None
    for marker in _HISTORY_MARKERS:
        i = hay.find(marker, floor)
        if i != -1 and (best is None or i < best):
            best = i
    return None if best is None else max(0, best - 80)

_YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
_DIGIT_RUN = re.compile(r"\d+")

# Sentinel the model returns when a field cannot be determined FROM THE PROVIDED
# TEXT. Distinct, ASCII, and digit-free so it never passes the numeric source-check
# and is trivially detected for fallback to the curated anchor / portal channel.
SENTINEL_NONE = "<<NONE>>"

_SYSTEM = (
    "You extract two metadata fields from the official front/back matter of a law, "
    "using ONLY what is explicitly printed in the text provided to you.\n"
    "(1) law_number — the official act/law number exactly as printed (e.g. 'Act 709', "
    "'No. 119 of 1988', 'No. 9 of 2018').\n"
    "(2) last_amended — the YEAR (YYYY) stated for the most recent AMENDMENT; if the law "
    "was never amended, the year it was MADE/ENACTED. A printing, compilation, reprint, "
    "'current version as at', download or retrieval date is NOT an amendment — never "
    "report such a date as last_amended.\n"
    "Rules: Use ONLY values explicitly printed in the provided text. Do NOT infer, guess, "
    f"or use your own background knowledge as the answer. If a field cannot be determined "
    f"from the provided text, return the exact string \"{SENTINEL_NONE}\" for that field "
    "(do not return an empty string, and do not substitute a printing/compilation date).\n"
    "Separate channel for your own knowledge: if you happen to know a value from your own "
    "training (NOT printed in the provided text), put it in 'notes' for analyst review, e.g. "
    "\"law_number(by knowledge)=Act X; last_amended(by knowledge)=YYYY\". The 'notes' field "
    "is for review only and is NEVER used as the answer.\n"
    'Output a JSON object: {"law_number": "...", "last_amended": "...", "notes": "..."}.'
)
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "law_number": {"type": "string"},
        "last_amended": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["law_number", "last_amended"],
}


def _document_window(text: str, mode: str = "block") -> str:
    """The slice of a statute sent to the extractor — where legal metadata lives —
    keeping the prompt bounded for long documents.

    Modes:
    * ``"full"`` — the whole document (Plan A: maximal recall, highest token cost).
    * ``"block"`` — HYBRID (recommended): masthead head + the LOCATED
      legislative-history block when a section heading is found (precise + cheap);
      otherwise escalates to the FULL document, so a jurisdiction with no
      recognizable endnote heading (e.g. a clean reprint) still gets every field
      rather than silently losing it to a blind tail. Only marker-less documents
      pay the full-text token cost.
    * ``"fixed"`` — blind head + tail (the original; can clip the first endnote).
    """
    if len(text) <= _HEAD_CHARS + _TAIL_CHARS:
        return text
    if mode == "full":
        return text
    if mode == "block":
        start = _locate_history_block(text)
        if start is not None and start >= _HEAD_CHARS:
            block = text[start:start + _BLOCK_MAX]
            return f"{text[:_HEAD_CHARS]}\n\n[...]\n\n{block}"
        return text  # no recognizable history block -> send the whole document
    return f"{text[:_HEAD_CHARS]}\n\n[...]\n\n{text[-_TAIL_CHARS:]}"


_REVIEW_PREFIX = "LLM metadata (own knowledge, NOT used as answer):"


def _format_review_note(note: str) -> str:
    """Tag the model's own-knowledge channel so it is unmistakably review-only and
    never confusable with a sourced answer. Empty in, empty out."""
    note = (note or "").strip()
    if not note or note == SENTINEL_NONE:
        return ""
    return f"{_REVIEW_PREFIX} {note}"


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

    def extract(
        self, document_text: str, jurisdiction: str, title: str,
        window_mode: str = "block",
    ) -> tuple[str, str, str]:
        """Return ``(last_amended, law_number, review_note)``.

        ``last_amended`` / ``law_number`` may be ``""`` when absent, sentinelled
        (the model could not determine the field from the text), unverifiable, or
        the backend is unavailable — the caller then falls back to the next tier
        (portal channel / curated anchor). ``review_note`` carries the model's
        own-knowledge channel for analyst review; it is NEVER used as an answer."""
        if self._client is None or not document_text:
            return "", "", ""
        window = _document_window(document_text, window_mode)
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
            return "", "", ""

        text_lower = document_text.lower()
        law_number = (data.get("law_number") or "").strip()
        last_amended = (data.get("last_amended") or "").strip()
        review_note = (data.get("notes") or "").strip()

        # Sentinel ("cannot determine from the provided text") -> blank -> fall back.
        if law_number == SENTINEL_NONE:
            law_number = ""
        if last_amended == SENTINEL_NONE:
            last_amended = ""

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
        return last_amended, law_number, _format_review_note(review_note)


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
