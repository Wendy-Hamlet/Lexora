"""LLM-assisted amendment-instruction extraction — the generalizer for the brittle
deterministic parser in :mod:`lexora.cite.amendments`.

The regex parser (`parse_amendment_instructions`) is tuned to the Westminster /
Commonwealth drafting register and breaks on phrasings it has not seen (coordinated
quoted terms, line-wrapped "principal Act", a non-English lexicon, US "striking …
inserting …"). This module reads the same amending-Act text with an LLM and returns
the SAME `AmendmentInstruction` objects, so a caller can swap or blend the two.

It mirrors the optional-dependency contract of :mod:`lexora.cite.metadata`:
INERT by default (no client -> returns ``[]``), so the offline suite never needs a
server. The model's output is SOURCE-VERIFIED — a target section, a renamed term or
an evidence snippet it returns must actually appear in the document text — so the
model can surface a printed instruction but never fabricate one. The shared
`Operation` / `InstructionKind` taxonomy is the same one the deterministic path and
any future non-English lexicon emit, so downstream adjudication is identical.

Red line (same as the regex path): this only DESCRIBES what an amending Act does; it
never writes a consolidated provision. ``raw`` carries the model's verbatim evidence
snippet, copied from the source, not synthesized prose.
"""
from __future__ import annotations

import re

from lexora.cite.amendments import (
    AmendmentInstruction,
    InstructionKind,
    Operation,
    section_of,
)

# Keyed by lower-cased value so the model may return either case (InstructionKind
# values are upper-case, Operation values lower-case).
_KIND = {k.value.lower(): k for k in InstructionKind}
_OP = {o.value.lower(): o for o in Operation}

_SYSTEM = (
    "You extract the textual amendment instructions an amending Act performs on its "
    "principal Act, using ONLY what is explicitly printed in the text provided.\n"
    "Return a JSON object {\"instructions\": [ ... ]}. Each instruction is an object:\n"
    "  operation: one of amend | substitute | delete | insert | rename\n"
    "  scope: one of\n"
    "     global_rename  - an act-wide term substitution applied 'wherever appearing'\n"
    "     definition_or_principle - a change to an interpretation/definition section or a stated principle\n"
    "     section_op     - a change confined to one numbered section\n"
    "  target_section: the principal-Act section number affected (e.g. '6', '129', '7A'), or '' for a global_rename\n"
    "  old_term: for a rename, the quoted term being replaced (e.g. 'data user'); else ''\n"
    "  new_term: for a rename, the replacement term (e.g. 'data controller'); else ''\n"
    "  evidence: a SHORT verbatim snippet, copied exactly from the provided text, that states this instruction\n"
    "Rules: use ONLY values printed in the provided text; do NOT infer or use background "
    "knowledge. Every target_section number, old_term, new_term and evidence snippet MUST "
    "appear verbatim in the text. If the document is not an amending Act, return an empty list."
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "instructions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "scope": {"type": "string"},
                    "target_section": {"type": "string"},
                    "old_term": {"type": "string"},
                    "new_term": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["operation", "scope"],
            },
        }
    },
    "required": ["instructions"],
}

_MAX_CHARS = 24000  # an amending Act is short; cap defends against an outlier


def _printed_as_a_section(text: str, number: str) -> bool:
    """True if ``number`` is printed the way a section number is printed.

    The source check used to be a bare substring search, which for a one-digit section
    is no check at all: "6" is inside "16", "2016" and every page number — measured on
    twelve real amending Acts, EVERY single-digit section passed. This asks for the
    reference form instead (``section 6``, ``s. 6``, ``paragraph 6``, or a line that
    opens ``6.``), which cuts the false-pass surface from 39% to 25% over 1..200. It is
    an improvement, not a solution — low numbers still pass legitimately, because an
    amending Act's own clauses are numbered 1, 2, 3. The real guard for the one
    operation that can destroy a row is :func:`_delete_is_substantiated`.
    """
    return re.search(
        rf"(?:\bsections?\s+|\bss?\.\s*|^\s*|\bparagraph\s+){re.escape(number)}\b",
        text, re.IGNORECASE | re.MULTILINE,
    ) is not None


class AmendmentInstructionExtractor:
    """Reads amendment instructions from an amending Act via the LLM, then verifies
    each against the source. ``client`` is ``None`` for the inert extractor (returns
    ``[]``). Counters let a run report usage / rejections."""

    def __init__(self, client=None) -> None:
        self._client = client
        self.extracted = 0
        self.rejected = 0
        self.error_count = 0

    def extract(self, document_text: str) -> list[AmendmentInstruction]:
        """Return the amending Act's instructions, or ``[]`` when the backend is
        unavailable, the document is not an amendment, or nothing verifies."""
        if self._client is None or not document_text:
            return []
        text = document_text[:_MAX_CHARS]
        user = f'Amending Act text:\n"""{text}"""\nReturn the instructions JSON.'
        try:
            data = self._client.chat(_SYSTEM, user, json_schema=_RESPONSE_SCHEMA)
        except Exception:  # noqa: BLE001 — never let extraction break a run
            self.error_count += 1
            return []
        low = document_text.lower()
        out: list[AmendmentInstruction] = []
        for item in (data or {}).get("instructions", []) or []:
            instr = self._coerce(item, low)
            if instr is not None:
                out.append(instr)
        if out:
            self.extracted += 1
        return out

    def _coerce(self, item: dict, text_lower: str) -> AmendmentInstruction | None:
        """Map one model item to a verified :class:`AmendmentInstruction`, or ``None``
        when it is malformed or fails the source check."""
        op = _OP.get(str(item.get("operation", "")).strip().lower())
        kind = _KIND.get(str(item.get("scope", "")).strip().lower())
        if op is None or kind is None:
            self.rejected += 1
            return None
        section = section_of(str(item.get("target_section", "")) or "") or \
            str(item.get("target_section", "")).strip()
        old_term = str(item.get("old_term", "")).strip()
        new_term = str(item.get("new_term", "")).strip()
        evidence = str(item.get("evidence", "")).strip()

        # Source-verify: any concrete value the model returns must be printed in the
        # document. A section/term/evidence not in the text is a fabrication -> drop.
        if section and not _printed_as_a_section(text_lower, section):
            self.rejected += 1
            return None
        if old_term and old_term.lower() not in text_lower:
            old_term = ""
        if new_term and new_term.lower() not in text_lower:
            new_term = ""
        if evidence and evidence.lower() not in text_lower:
            evidence = ""
        if kind is InstructionKind.global_rename and not old_term:
            self.rejected += 1  # a rename with no verifiable term is unusable
            return None
        if kind is not InstructionKind.global_rename and not section:
            self.rejected += 1  # a section op with no verifiable section is unusable
            return None
        # A `delete` is the one operation that DESTROYS output: it becomes
        # CurrencyStatus.REPEALED, and `enforced_only` then drops that citation from the
        # submission. Until now an instruction whose evidence snippet failed the source
        # check kept the instruction and merely blanked the evidence, so a repeal could
        # be carried entirely by a section number that appears somewhere in the text.
        # Require a repeal to be quoted from the document, and to quote THIS section.
        # Failing this guard keeps the citation, which is the recoverable direction —
        # a dropped row leaves no trace anywhere in the output.
        if op is Operation.delete and not self._delete_is_substantiated(evidence, section):
            self.rejected += 1
            return None
        return AmendmentInstruction(
            kind=kind, op=op, target_section=section,
            old_term=old_term, new_term=new_term, raw=evidence,
        )

    @staticmethod
    def _delete_is_substantiated(evidence: str, section: str) -> bool:
        """A repeal must be carried by a verified quotation that names its own section.

        ``evidence`` has already been blanked if it did not appear verbatim in the
        document, so a non-empty value here is text the Act really prints. Requiring
        the section number inside it stops a snippet about section 26 from repealing
        section 6."""
        if not evidence:
            return False
        return not section or _printed_as_a_section(evidence, section)


def make_amendment_extractor(use_llm: bool = False) -> AmendmentInstructionExtractor:
    """Construct an extractor. Returns the inert (``[]``-returning) extractor when
    ``use_llm`` is false or the LLM backend is unavailable, so callers can wire it
    unconditionally — mirroring :func:`lexora.cite.metadata.make_metadata_extractor`."""
    if not use_llm:
        return AmendmentInstructionExtractor(client=None)
    from lexora.classify import llm_client

    if not llm_client.is_available():
        return AmendmentInstructionExtractor(client=None)
    try:
        return AmendmentInstructionExtractor(client=llm_client.LlmClient())
    except Exception:  # noqa: BLE001
        return AmendmentInstructionExtractor(client=None)


def llm_enabled() -> bool:
    """Whether the env opts into LLM amendment extraction (``LEXORA_AMENDMENT_LLM=1``).

    Read through :func:`lexora.config.env_value` so it is settable in ``.env`` beside the
    endpoint it needs, not only as a real environment variable — the same half-visible
    knob the embedding model had."""
    from lexora.config import env_value

    return env_value("LEXORA_AMENDMENT_LLM", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


__all__ = [
    "AmendmentInstructionExtractor",
    "make_amendment_extractor",
    "llm_enabled",
]
