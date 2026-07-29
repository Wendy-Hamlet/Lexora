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
    'Output a JSON object: {"law_number": "...", "last_amended": "..."}.'
)
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "law_number": {"type": "string"},
        "last_amended": {"type": "string"},
    },
    "required": ["law_number", "last_amended"],
}

# --- the recall channel -------------------------------------------------------------
#
# What this replaced, and why. The extraction prompt above used to carry a second
# instruction: "if you happen to know a value from your own training, put it in 'notes'".
# Measured on four Malaysian laws (2026-07-29), that channel does not recall — it
# RESTATES. One answer quoted the supplied text back ("The text explicitly states…"),
# one silently declined, one answered about the TITLE we had handed it, and on the Food
# Act 1983 both channels agreed on 2006 and both were wrong. Zero rows of the round-1
# submission ever carried a `(by knowledge)=` note.
#
# Three things were wrong with it, and this is the fix for each:
#   1. it shared a call with extraction, so the text it had just read was the cheapest
#      thing to say -> recall is now its OWN call and is given NO document text at all;
#   2. its answer was conditioned on a title we supplied -> the name passed in comes from
#      `resolve_law_name`, i.e. the law's own masthead, not the portal's file name;
#   3. nothing rewarded abstention, so it always produced something -> the prompt now
#      says plainly that "I don't know" is the useful answer.
#
# The output is never an answer and never a submission field. It earns its cost only by
# CONTRADICTING the text-based extraction, so only disagreements are reported: the Food
# Act case is the standing proof that agreement between the two is not evidence.
_RECALL_SYSTEM = (
    "You are asked what you already know about a named law. You are NOT given its text, "
    "and you must not ask for it.\n"
    "Answer ONLY from your own training knowledge of this specific law.\n"
    "(1) law_number — the official act/law number as it is conventionally cited "
    "(e.g. 'Act 709', 'No. 119 of 1988').\n"
    "(2) last_amended — the YEAR (YYYY) of the most recent amendment you know this law "
    "to have received. The year inside the law's NAME is its year of enactment and is "
    "NOT evidence about amendments — do not derive one field from the other.\n"
    f"If you do not recognise this particular law, or you are not confident of a field, "
    f"return the exact string \"{SENTINEL_NONE}\" for that field.\n"
    "Saying you do not know IS the useful answer here. This channel exists only to "
    "contradict a separate reading of the document, so a plausible-looking guess is "
    "worse than a blank: it would be mistaken for independent confirmation.\n"
    'Output a JSON object: {"law_number": "...", "last_amended": "..."}.'
)
_RECALL_SCHEMA = {
    "type": "object",
    "properties": {
        "law_number": {"type": "string"},
        "last_amended": {"type": "string"},
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


_REVIEW_PREFIX = "LLM recall disagrees with the document (review only, NOT used as answer):"


def recall_conflict_note(
    extracted: tuple[str, str], recalled: tuple[str, str]
) -> str:
    """A review note naming the fields where recall and the document disagree, or "".

    Each argument is ``(last_amended, law_number)``. A field is compared only when BOTH
    channels produced a value: a blank on either side is an absence of evidence, and an
    agreement is not evidence either — on the Food Act 1983 the two channels agreed on
    2006 and the real answer was neither. Only a disagreement tells an analyst where to
    look, so only a disagreement is written down.
    """
    ex_amended, ex_number = ((v or "").strip() for v in extracted)
    rc_amended, rc_number = ((v or "").strip() for v in recalled)
    parts = []
    if ex_number and rc_number and _number_key(ex_number) != _number_key(rc_number):
        parts.append(f"law_number: document says {ex_number!r}, recall says {rc_number!r}")
    if ex_amended and rc_amended and ex_amended != rc_amended:
        parts.append(
            f"last_amended: document says {ex_amended}, recall says {rc_amended}"
        )
    return f"{_REVIEW_PREFIX} " + "; ".join(parts) if parts else ""


def _number_key(value: str) -> str:
    """An act number's comparable core: digits only, so 'Act 709' == 'ACT 709' ==
    'Act No. 709' and a conflict means the NUMBERS differ, not the punctuation."""
    return "-".join(_DIGIT_RUN.findall(value))


def _verify_in_text(value: str, text_lower: str) -> bool:
    """True if every numeric token of ``value`` appears verbatim in the source text.
    A fabrication guard that tolerates formatting differences ('No. 119 of 1988' vs
    'No. 119, 1988') while rejecting numbers/years not printed in the document."""
    runs = _DIGIT_RUN.findall(value)
    return bool(runs) and all(run in text_lower for run in runs)


# A law-number citation carrying its own year: "Act 26 of 2012", "No. 119 of 1988".
_NUMBERED = re.compile(r"\b(?:Act|No\.?)\s*(\d+)\s+of\s+(\d{4})\b", re.I)
# A title that ends in its year of enactment: "Personal Data Protection Act 2012".
_TITLE_YEAR = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\s*$")


def year_agrees_with_title(law_number: str, title: str) -> bool:
    """False only when both carry a year AND the years disagree.

    The failure this catches: an amendment's number copied in place of the
    principal Act's. Singapore's Child Development Co-Savings Act 2001 was RENAMED,
    so its history block opens with the old name ("Children Development…"); the
    model, matching on the title we gave it, skipped that entry and reported the
    most recent amendment instead — "Act 46 of 2024" for a 2001 Act. A principal
    Act's number always carries the year in its own title, so the years must agree.
    Silent (True) when either side has no year: Malaysia's "Act 709" is a valid
    number that simply does not encode one."""
    n = _NUMBERED.search(law_number or "")
    t = _TITLE_YEAR.search((title or "").strip())
    if not n or not t:
        return True
    return n.group(2) == t.group(1)


# SSO prints an ordered legislative history whose FIRST entry is always the
# principal Act ("1. Act 26 of 2012 — Personal Data Protection Act 2012"), with the
# amendments numbered after it. That ordering is the document's own authority on
# which number is the Act's — far stronger than asking a model to pick. The gap
# absorbs the Commission's disclaimer and any "(Formerly known as …)" line.
_PRINCIPAL_IN_HISTORY = re.compile(
    r"LEGISLATIVE\s+HISTORY\b.{0,500}?(?:^|\n)\s*1\.\s*(Act\s+\d+\s+of\s+\d{4})\b",
    re.IGNORECASE | re.DOTALL,
)


def principal_act_number(text: str) -> str:
    """The principal Act's number taken from the ordered legislative history, or "".

    Structural, not generative — it reads the document's own numbering. Absent that
    structure (Australia, Malaysia) it returns "" and the LLM answer stands."""
    m = _PRINCIPAL_IN_HISTORY.search(text or "")
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


class MetadataExtractor:
    """Reads (last_amended, law_number) from a document's text via the LLM, then
    verifies each against the source. ``client`` is ``None`` for the inert
    extractor (returns blanks). Counters let a run report usage."""

    def __init__(self, client=None, *, recall: bool | None = None) -> None:
        self._client = client
        self._recall_enabled = recall_enabled() if recall is None else recall
        # One law appears in many documents (consolidations, amendment Acts, PDF
        # variants); its recall answer does not depend on which one we are holding.
        self._recalled: dict[tuple[str, str], tuple[str, str]] = {}
        self.extracted = 0
        self.rejected = 0
        self.overridden = 0  # structural history beat the model's law_number
        self.recalled = 0
        self.recall_conflicts = 0
        self.error_count = 0
        self.last_error_type: str | None = None

    def recall(self, jurisdiction: str, law_name: str) -> tuple[str, str]:
        """``(last_amended, law_number)`` from the model's training knowledge alone.

        No document text is passed — see the note above :data:`_RECALL_SYSTEM`. Blank
        for either field the model declines, which is the answer this channel is there
        to make cheap. Memoised per (jurisdiction, law name)."""
        if self._client is None or not self._recall_enabled or not law_name:
            return "", ""
        key = (jurisdiction, law_name)
        if key in self._recalled:
            return self._recalled[key]

        user = (
            f"Jurisdiction: {jurisdiction}\n"
            f"Law: {law_name}\n"
            "What do you know about this law? Return the JSON."
        )
        try:
            data = self._client.chat(_RECALL_SYSTEM, user, json_schema=_RECALL_SCHEMA)
        except Exception as exc:  # noqa: BLE001 — a review channel never breaks a run
            self.error_count += 1
            self.last_error_type = type(exc).__name__
            return "", ""

        number = (data.get("law_number") or "").strip()
        amended = (data.get("last_amended") or "").strip()
        if number == SENTINEL_NONE:
            number = ""
        if amended == SENTINEL_NONE:
            amended = ""
        # A year is the only shape this field may take; anything else is the model
        # narrating rather than recalling.
        ym = _YEAR_RE.fullmatch(amended)
        amended = ym.group(0) if ym else ""
        if number or amended:
            self.recalled += 1
        self._recalled[key] = (amended, number)
        return amended, number

    def extract(
        self, document_text: str, jurisdiction: str, title: str,
        window_mode: str = "block",
    ) -> tuple[str, str, str]:
        """Return ``(last_amended, law_number, review_note)``.

        ``last_amended`` / ``law_number`` may be ``""`` when absent, sentinelled
        (the model could not determine the field from the text), unverifiable, or
        the backend is unavailable — the caller then falls back to the next tier
        (portal channel / curated anchor). ``review_note`` is non-empty only when the
        separate recall channel CONTRADICTS what this call read out of the document;
        it is never an answer and never a submission field."""
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

        # Sentinel ("cannot determine from the provided text") -> blank -> fall back.
        if law_number == SENTINEL_NONE:
            law_number = ""
        if last_amended == SENTINEL_NONE:
            last_amended = ""

        if law_number and not _verify_in_text(law_number, text_lower):
            self.rejected += 1
            law_number = ""
        # An amendment's number reported as the Act's own: the years disagree.
        if law_number and not year_agrees_with_title(law_number, title):
            self.rejected += 1
            law_number = ""
        # The document's ordered history outranks the model's choice, and also
        # recovers a number the checks above just dropped.
        principal = principal_act_number(document_text)
        if principal and year_agrees_with_title(principal, title):
            if principal != law_number:
                self.overridden += 1
            law_number = principal
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

        # The independent second opinion, asked in its own call with none of the text
        # above in front of it. It cannot change either answer — only report that the
        # two readings disagree, which is where an analyst should look.
        note = recall_conflict_note(
            (last_amended, law_number), self.recall(jurisdiction, title)
        )
        if note:
            self.recall_conflicts += 1
        return last_amended, law_number, note


def recall_enabled() -> bool:
    """Whether to ask the recall channel (``LEXORA_METADATA_RECALL``, default OFF).

    Opt-in because it is one extra call per distinct law and produces no submission
    field — it only annotates rows where the document and the model's own knowledge
    disagree. Read through :func:`lexora.config.env_value` so it can be set in ``.env``
    alongside the endpoint, not only as a real environment variable."""
    from lexora.config import env_value

    return env_value("LEXORA_METADATA_RECALL", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def make_metadata_extractor(
    use_llm: bool = False, *, recall: bool | None = None
) -> MetadataExtractor:
    """Construct a :class:`MetadataExtractor`. Returns the inert (blank-returning)
    extractor when ``use_llm`` is false or the LLM backend is unavailable, so
    callers can wire it unconditionally. ``recall`` overrides
    :func:`recall_enabled` for the second-opinion channel."""
    if not use_llm:
        return MetadataExtractor(client=None, recall=recall)
    from lexora.classify import llm_client

    if not llm_client.is_available():
        return MetadataExtractor(client=None, recall=recall)
    try:
        return MetadataExtractor(client=llm_client.LlmClient(), recall=recall)
    except Exception:  # noqa: BLE001
        return MetadataExtractor(client=None, recall=recall)


__all__ = [
    "MetadataExtractor",
    "make_metadata_extractor",
    "recall_conflict_note",
    "recall_enabled",
]
