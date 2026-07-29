"""Amendment detection & currency assessment — flag citations whose SOURCE text
may be stale because a later instrument amended the law.

The ``Last Amended`` column (see :mod:`lexora.cite.metadata`) reports the most
recent amendment year *a single document admits in its own front/back matter*. A
clean "as made" original (e.g. the Malaysia PDPA 2010 PDF) carries no record of a
SEPARATE later amending Act, so the field silently reports the enactment year and
a citation drawn from it can quote provisions a later Act has since repealed.

This module closes that gap WITHOUT attempting full point-in-time consolidation
(the hard part). It detects, per law, the set of amending instruments and compares
their years against what the source document incorporates, turning a silent stale
quote into an explicit ``AMENDMENT_REVIEW`` flag plus a note for the analyst.

Four signals, in the priority the integrator chose (B/A/D primary, C backstop):

* **B — corpus cross-reference (deterministic, generalizes to NEW laws).** An
  amending Act is highly formulaic ("An Act to amend the <Principal Act>",
  "principal Act"). When the amending instrument is in the fetched corpus we read
  the edge straight off its long title — no registry, works for unknown laws.
* **A — in-document consolidation point.** A compiled/reprint document states the
  point it incorporates to ("Incorporating all amendments up to ...", "as at
  <date>", "Revised up to <year>"). This is the cutoff a citation from that
  document is current to.
* **D — portal-supplied amendment year.** The portal channel may report a
  most-recent-amendment year on the principal (``RawDocument.last_amended``) that
  post-dates what the document text incorporates.
* **C — curated registry (backstop only).** ``SourceProfile.amended_by`` maps a
  known principal to its amendments. Only useful for known laws (useless for NEW),
  so it is the LAST tier — it fills laws the corpus/portal signals missed.

A law may be amended MANY times; every amendment is a distinct, valid event and is
kept in the chain. Effectiveness is "later overrides earlier", so the chain is held
in chronological order and currency is judged against the WHOLE chain: any
amendment newer than the source's incorporation point is reported as missing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class CurrencyStatus(str, Enum):
    """Whether a citation's source text is current w.r.t. known amendments.

    The first three are document-level (we know an amendment exists but not which
    sections); the last two are provision-level verdicts available once the amending
    Act's instructions are parsed (Tier-2)."""

    current = "CURRENT"  # source incorporates every known amendment (or this section untouched)
    stale_risk = "STALE_RISK"  # a later amendment exists that the source pre-dates
    unknown = "UNKNOWN"  # no amendment signal either way (treated as ok)
    amended = "AMENDED"  # this exact section was amended/substituted by a later Act
    repealed = "REPEALED"  # this exact section was deleted/repealed (and commenced)


_YEAR = r"(1[6-9]\d{2}|20\d{2})"
_YEAR_RE = re.compile(r"\b" + _YEAR + r"\b")

# --- identity of the instrument in front of us (from its masthead) ---
# "Act 709", "Act A1727" (amending Acts in Malaysia take a letter-prefixed number),
# "Act 854". First hit in the masthead is the instrument's own number.
_ACT_NUMBER_RE = re.compile(r"\bAct\s+([A-Z]?\d+[A-Z]?)\b", re.IGNORECASE)
# A masthead title line ending in a 4-digit year, e.g. "PERSONAL DATA PROTECTION
# ACT 2010". Allows the common "(AMENDMENT)" infix.
_ACT_TITLE_RE = re.compile(r"([A-Z][A-Za-z()]+(?:\s+[A-Za-z()]+){0,8}\s+ACT,?\s+\d{4})")

# --- Signal B: this document is an amending Act targeting a principal ---
# Long-title form: "An Act to amend the Personal Data Protection Act 2010".
_TO_AMEND_RE = re.compile(r"to amend the\s+(.{3,80}?Act,?\s+\d{4})", re.IGNORECASE)
# Broader amendment SIGNAL (classification only, not target identification). AU's
# theme-named omnibus Acts carry a GENERIC long title — "An Act to amend legislation
# relating to telecommunications, and for related purposes" / "to amend the law
# relating to ..." — which names no single principal, so `_TO_AMEND_RE` misses it and
# the Act is misread as an as-made ORIGINAL. The "An Act to amend ..." long-title
# formula is itself a reliable amendment marker (a principal's long title reads "An
# Act relating to / to provide for / about ..."), so it distinguishes amendment from
# original even when the target is unnamed. Used ONLY by `classify_version`;
# `detect_amends_target` keeps the stricter forms (it must return a real principal).
_AMEND_LONGTITLE_RE = re.compile(r"\bAn Act\b[^.\n]{0,40}?\bto amend\b", re.IGNORECASE)
# Definition form: "The Personal Data Protection Act 2010 [Act 709] ... principal
# Act" — captures both the principal's title and its bracketed Act number.
_PRINCIPAL_RE = re.compile(
    r"(.{3,80}?Act,?\s+\d{4})\s*\[Act\s+([A-Z]?\d+[A-Z]?)\][^.]{0,80}?principal\s+Act",
    re.IGNORECASE,
)

# --- Signal A: the consolidation point a compiled document incorporates to ---
_INCORP_RES = (
    re.compile(r"incorporating\s+(?:all\s+)?amendments?\b[^.\n]*?" + _YEAR, re.IGNORECASE),
    re.compile(r"\bas at\b[^.\n]*?" + _YEAR, re.IGNORECASE),
    re.compile(r"\brevised\b(?:\s+up\s+to)?[^.\n]*?" + _YEAR, re.IGNORECASE),
    re.compile(r"\bamendments?\s+incorporated\b[^.\n]*?" + _YEAR, re.IGNORECASE),
    # AU Federal Register compilation masthead, two phrasings seen: "Includes
    # amendments up to: Act No. 79, 2021" and "Includes amendments: Act No. 75,
    # 2025". The "Act No. NN," before the year contains a period, so this one stays
    # on its line ([^\n]) rather than stopping at the first dot; the amending Act's
    # own 2-3 digit number can't satisfy the 4-digit _YEAR, so the match lands on
    # the trailing compilation year.
    re.compile(r"includes\s+amendments?\b[^\n]*?" + _YEAR, re.IGNORECASE),
)

_HEAD = 4000  # masthead window
_TAIL = 4000  # endnote / compilation-note window


def _ends(text: str) -> str:
    """Masthead + endnote slices where legislative-history language lives, so the
    deterministic regexes never scan megabytes of body prose."""
    if len(text) <= _HEAD + _TAIL:
        return text
    return text[:_HEAD] + "\n" + text[-_TAIL:]


def normalize_key(*, number: str = "", title: str = "") -> str:
    """A stable principal-law key. Prefers the Act NUMBER (jurisdiction-native,
    collision-free); falls back to a normalized title. Empty when neither is
    usable, so callers can skip un-keyable rows."""
    num = (number or "").strip().lower()
    m = _ACT_NUMBER_RE.search(num) if num else None
    if m:
        return "act:" + m.group(1).lower()
    if num and re.fullmatch(r"[a-z]?\d+[a-z]?", num):
        return "act:" + num
    t = (title or "").lower()
    t = re.sub(r"\[act\s+[a-z]?\d+[a-z]?\]", " ", t)  # drop bracketed number
    t = re.sub(r"\(amendment\)", " ", t)  # an amendment shares its principal's key
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    return ("title:" + t) if t else ""


def candidate_keys(*, number: str = "", title: str = "") -> list[str]:
    """All keys a principal law may be referenced by — its Act-NUMBER key AND its
    TITLE key. An amending Act frequently cites the principal by title only while the
    principal document carries its number, so matching the two requires trying both."""
    keys: list[str] = []
    if number:
        nk = normalize_key(number=number)
        if nk.startswith("act:"):
            keys.append(nk)
    if title:
        tk = normalize_key(title=title)
        if tk.startswith("title:") and tk not in keys:
            keys.append(tk)
    return keys


@dataclass(frozen=True)
class AmendmentEvent:
    """One amending instrument acting on a principal law. ``year`` orders the chain
    (later overrides earlier); ``detected_by`` records which signal found it."""

    year: int
    amending_id: str = ""  # e.g. "Act A1727" (blank when only a year is known)
    amending_title: str = ""
    detected_by: str = ""  # "corpus" | "portal" | "registry"

    def label(self) -> str:
        base = self.amending_id or self.amending_title or "amendment"
        return f"{base} ({self.year})"


@dataclass
class Identity:
    """What an instrument says it is, parsed from its masthead."""

    number: str = ""
    title: str = ""
    year: int | None = None

    @property
    def key(self) -> str:
        return normalize_key(number=self.number, title=self.title)


def parse_identity(text: str) -> Identity:
    """Read (number, title, year) from a document's masthead. Best-effort: any
    field may be blank when the masthead does not print it."""
    head = text[:_HEAD]
    num = _ACT_NUMBER_RE.search(head)
    title_m = _ACT_TITLE_RE.search(head)
    title = title_m.group(1).strip() if title_m else ""
    year = None
    if title:
        y = _YEAR_RE.search(title)
        year = int(y.group(0)) if y else None
    return Identity(number=("Act " + num.group(1)) if num else "", title=title, year=year)


def detect_amends_target(text: str) -> tuple[str, str] | None:
    """Signal B. If this document is an amending Act, return the principal it amends
    as ``(target_number, target_title)`` (either may be ``""``); else ``None``.

    Reads the formulaic amending-Act language in the masthead. ``target_number`` is
    populated when the principal is cited as "<Title> [Act NNN]"."""
    head = _ends(text)
    pm = _PRINCIPAL_RE.search(head)
    if pm:
        return ("Act " + pm.group(2), pm.group(1).strip())
    tm = _TO_AMEND_RE.search(head)
    if tm:
        return ("", tm.group(1).strip())
    return None


def detect_incorporated_to(text: str) -> int | None:
    """Signal A. The latest year the document says it incorporates amendments to,
    or ``None`` when it makes no such claim (an "as made" original)."""
    head = _ends(text)
    years: list[int] = []
    for rx in _INCORP_RES:
        years.extend(int(m.group(1)) for m in rx.finditer(head))
    return max(years) if years else None


class AmendmentIndex:
    """Principal-law -> chronological chain of :class:`AmendmentEvent`.

    Build from the fetched corpus (Signal B) and the portal channel (D), then layer
    the curated registry (C) as a backstop for laws the live signals missed. Lookup
    returns the full chain so a multi-amendment law is judged against every event."""

    def __init__(self) -> None:
        self._chains: dict[str, list[AmendmentEvent]] = {}

    def _add(self, key: str, event: AmendmentEvent) -> None:
        if not key:
            return
        chain = self._chains.setdefault(key, [])
        # De-dup by (amending_id, year): the same amendment may surface from several
        # signals. Prefer the more specific record (one carrying an amending_id).
        for i, e in enumerate(chain):
            if e.year == event.year and (
                e.amending_id == event.amending_id
                or not e.amending_id
                or not event.amending_id
            ):
                if not e.amending_id and event.amending_id:
                    chain[i] = event
                return
        chain.append(event)

    def add_from_corpus(self, documents: list[tuple[str, str]]) -> None:
        """Signal B. ``documents`` is ``[(title_hint, text), ...]`` for every fetched
        instrument. Any that is an amending Act contributes an event to its target's
        chain, keyed by the target's number (preferred) or title."""
        for title_hint, text in documents:
            target = detect_amends_target(text)
            if target is None:
                continue
            t_num, t_title = target
            ident = parse_identity(text)
            y = ident.year
            if y is None:
                continue
            event = AmendmentEvent(
                year=y,
                amending_id=ident.number or (title_hint or "").strip(),
                amending_title=ident.title or t_title,
                detected_by="corpus",
            )
            for key in candidate_keys(number=t_num, title=t_title):
                self._add(key, event)

    def add_portal(self, *, principal_key: str, year: int, amending_id: str = "") -> None:
        """Signal D. A portal-reported amendment year on the principal."""
        if principal_key and year:
            self._add(principal_key, AmendmentEvent(year=year, amending_id=amending_id,
                                                    detected_by="portal"))

    def add_registry(self, amended_by: dict[str, list[dict]]) -> None:
        """Signal C (backstop). ``amended_by`` maps a principal's native number (or
        title) to ``[{"by": "Act A1727", "year": 2024}, ...]``. Added last so live
        signals win; only fills laws not already covered for that year."""
        for principal, events in (amended_by or {}).items():
            key = normalize_key(number=str(principal), title=str(principal))
            for ev in events or []:
                year = ev.get("year")
                if year is None:
                    continue
                self._add(key, AmendmentEvent(year=int(year), amending_id=str(ev.get("by", "")),
                                              detected_by="registry"))

    def events_for(self, key: str) -> list[AmendmentEvent]:
        """The chain for a principal, chronological (oldest first; last = newest)."""
        return sorted(self._chains.get(key, []), key=lambda e: e.year)

    def events_for_any(self, keys: list[str]) -> list[AmendmentEvent]:
        """The merged, de-duplicated chain across several keys a principal may be
        referenced by (Act-number + title) — chronological."""
        seen: set[tuple[int, str]] = set()
        out: list[AmendmentEvent] = []
        for k in keys:
            for e in self._chains.get(k, []):
                sig = (e.year, e.amending_id)
                if sig in seen:
                    continue
                seen.add(sig)
                out.append(e)
        return sorted(out, key=lambda e: e.year)


@dataclass
class CurrencyAssessment:
    status: CurrencyStatus
    incorporated_to: int | None
    missing: list[AmendmentEvent] = field(default_factory=list)

    def amended_by_label(self) -> str:
        """Human list of the amendments newer than the source's incorporation point."""
        return "; ".join(e.label() for e in self.missing)


def assess_currency(
    *, principal_key: str = "", keys: list[str] | None = None,
    incorporated_to: int | None, index: AmendmentIndex,
    self_consolidated: bool = False,
) -> CurrencyAssessment:
    """Compare what a source document incorporates against the known amendment chain.

    Pass ``keys`` (the principal's Act-number + title candidate keys) to match an
    amendment that referenced the principal by either form, or ``principal_key`` for a
    single key. ``incorporated_to`` is the cutoff the citation's source text is current
    to — Signal A's consolidation point, else the year in the document's ``Last
    Amended`` (an "as made" original incorporates nothing past its enactment year). Any
    amendment newer than the cutoff is reported as missing (later overrides earlier, so
    every newer event matters, not just the last)."""
    events = index.events_for_any(keys) if keys else index.events_for(principal_key)
    if not events:
        # No amendment chain known. A source that is itself a CONSOLIDATION states its
        # own currency point in its masthead ("incorporates amendments to YYYY"), so
        # with no later amendment surfaced it is CURRENT to that point. An "as made"
        # ORIGINAL makes no such claim — even if a portal supplied an enactment-year
        # cutoff — so its status is genuinely UNKNOWN until an amendment is found.
        status = CurrencyStatus.current if self_consolidated else CurrencyStatus.unknown
        return CurrencyAssessment(status, incorporated_to, [])
    cutoff = incorporated_to if incorporated_to is not None else -1
    missing = [e for e in events if e.year > cutoff]
    status = CurrencyStatus.stale_risk if missing else CurrencyStatus.current
    return CurrencyAssessment(status, incorporated_to, missing)


def currency_note(assessment: CurrencyAssessment) -> str:
    """Analyst-facing note for a STALE_RISK citation; empty otherwise."""
    if assessment.status is not CurrencyStatus.stale_risk:
        return ""
    inc = assessment.incorporated_to
    inc_txt = f"amendments up to {inc}" if inc is not None else "the original as-made text"
    return (
        f"Source reflects {inc_txt}; later amending instrument(s) exist: "
        f"{assessment.amended_by_label()}. Verify this provision against the "
        f"consolidated in-force version."
    )


# ---------------------------------------------------------------------------
# Version classification + amendment-instruction parsing (Tier-2 groundwork).
#
# Mirrors the legal team's manual workflow: (2.1) does the new version CONTAIN the
# old (a consolidated reprint) or is it a delta amendment? and (2.1.2.1) which
# instructions touch DEFINITIONS / PRINCIPLES (broad cascade) vs a specific section?
# This stage only PARSES; per-citation adjudication (Tier-2 step 2) consumes it.
#
# Red line preserved: the parser reads the amending Act's own instruction text and
# records the old/new TERMS and target sections verbatim — it never synthesizes a
# consolidated provision. Downstream only attaches the amending Act's own wording.
# ---------------------------------------------------------------------------


class VersionKind(str, Enum):
    """How a fetched document relates to the law's text (legal step 2.1)."""

    original = "ORIGINAL"  # principal "as made"; no later amendments folded in
    amendment_delta = "AMENDMENT_DELTA"  # an amending Act: instructions, not full text
    consolidated = "CONSOLIDATED"  # a reprint that incorporates amendments (self-contained)


class Operation(str, Enum):
    """The textual operation an instruction performs on the principal."""

    amend = "amend"
    substitute = "substitute"
    delete = "delete"
    insert = "insert"
    rename = "rename"


class InstructionKind(str, Enum):
    """Cascade class of an instruction — drives flag severity (legal step 2.1.2.1).

    ``global_rename`` and ``definition_or_principle`` have broad, act-wide effect
    (a renamed term or a changed definition ripples through every provision that
    uses it); ``section_op`` is a localized change to one numbered section."""

    global_rename = "GLOBAL_RENAME"
    definition_or_principle = "DEFINITION_OR_PRINCIPLE"
    section_op = "SECTION_OP"


@dataclass(frozen=True)
class AmendmentInstruction:
    """One parsed amendment instruction. ``raw`` keeps the amending Act's own words
    (never synthesized); ``old_term``/``new_term`` carry a rename's verbatim terms."""

    kind: InstructionKind
    op: Operation
    target_section: str = ""
    old_term: str = ""
    new_term: str = ""
    raw: str = ""


_Q = r"[\"“”'‘’]"  # straight + curly quotes
_NQ = r"[^\"“”]"  # any non-double-quote (term body)

# Whole-act term substitution: the "wherever appearing" marker is what makes it
# GLOBAL rather than a one-section edit. Real gazette clauses list several quoted
# forms ("data user" AND "data users"), so extra coordinated quoted terms are
# tolerated before "wherever"; group 1/2 keep the first old/new term for the note.
_RENAME_RE = re.compile(
    r"substitut\w*\s+for\s+the\s+(?:word|expression)s?\s+"
    + _Q + r"(" + _NQ + r"+)" + _Q
    + r"(?:\s*(?:,|and|or)\s*" + _Q + _NQ + r"+" + _Q + r")*"
    + r"\s+wherever\s+(?:appearing|they\s+appear|it\s+appears)\b"
    + r"(?:" + _NQ + r"*?the\s+(?:word|expression)s?\s+" + _Q + r"(" + _NQ + r"+)" + _Q + r")?",
    re.IGNORECASE | re.DOTALL,
)
# "Section 6 of the principal Act is amended / deleted / repealed / substituted"
_SECTION_OP_RE = re.compile(
    r"[Ss]ection\s+(\d+[A-Za-z]?)\s+of\s+the\s+principal\s+Act\s+is\s+"
    r"(amended|deleted|repealed|substituted)",
    re.IGNORECASE,
)
# A brand-new section: "inserting after section 7 the following".
_INSERT_RE = re.compile(
    r"inserting\s+(?:after|before)\s+section\s+(\d+[A-Za-z]?)", re.IGNORECASE
)
# A section op touching the interpretation/definitions section or a principle has
# act-wide reach -> the higher-severity DEFINITION_OR_PRINCIPLE class.
_DEFN_HINT_RE = re.compile(r"\b(interpretation|definition|principle)s?\b", re.IGNORECASE)
_VERB_OP = {"deleted": Operation.delete, "repealed": Operation.delete,
            "substituted": Operation.substitute}


def _squeeze(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def classify_version(text: str) -> VersionKind:
    """Legal step 2.1: is this the original, a delta amendment, or a consolidation?

    An amending Act announces itself in its masthead long title ("An Act to amend
    the ... Act 20xx" / "the principal Act"); a consolidation carries an
    incorporation note; anything else is treated as the original as-made text."""
    head = text[:_HEAD]
    if (
        _TO_AMEND_RE.search(head)
        or _PRINCIPAL_RE.search(head)
        or _AMEND_LONGTITLE_RE.search(head)
    ):
        return VersionKind.amendment_delta
    if detect_incorporated_to(text) is not None:
        return VersionKind.consolidated
    return VersionKind.original


def parse_amendment_instructions(text: str) -> list[AmendmentInstruction]:
    """Parse an amending Act's instructions into classified operations.

    Deterministic, tuned to the Westminster / Commonwealth drafting register
    ("the principal Act is amended by deleting/substituting/inserting ...",
    "wherever appearing"). NOT a universal grammar: Australian schedule-item tables
    and US "striking/inserting" differ, and non-English jurisdictions need a
    per-language lexicon or an LLM extractor sharing this same Operation taxonomy."""
    out: list[AmendmentInstruction] = []
    for m in _RENAME_RE.finditer(text):
        out.append(AmendmentInstruction(
            kind=InstructionKind.global_rename, op=Operation.rename,
            old_term=m.group(1).strip(), new_term=(m.group(2) or "").strip(),
            raw=_squeeze(m.group(0)),
        ))
    for m in _SECTION_OP_RE.finditer(text):
        verb = m.group(2).lower()
        op = _VERB_OP.get(verb, Operation.amend)
        window = text[m.start(): m.start() + 240]
        kind = (InstructionKind.definition_or_principle
                if _DEFN_HINT_RE.search(window) else InstructionKind.section_op)
        out.append(AmendmentInstruction(
            kind=kind, op=op, target_section=m.group(1), raw=_squeeze(m.group(0)),
        ))
    for m in _INSERT_RE.finditer(text):
        out.append(AmendmentInstruction(
            kind=InstructionKind.section_op, op=Operation.insert,
            target_section=m.group(1), raw=_squeeze(m.group(0)),
        ))
    return out


def affected_sections(instructions: list[AmendmentInstruction]) -> set[str]:
    """The set of principal-Act section numbers a list of instructions touches."""
    return {i.target_section for i in instructions if i.target_section}


def has_global_rename(instructions: list[AmendmentInstruction]) -> bool:
    """True if any instruction is an act-wide term substitution (broad cascade)."""
    return any(i.kind is InstructionKind.global_rename for i in instructions)


_TITLE_CORE_RE = re.compile(r"\s*\bAct\b\s*\d", re.IGNORECASE)


def amendment_search_queries(title: str, *, word_tokenised: bool = False) -> list[str]:
    """Portal queries that would surface amendments to a principal law, DERIVED from
    its title — general (works for any law), not a hardcoded amendment name.

    The driver tags every fetched document (``classify_version``) and, for each
    ORIGINAL, runs these queries to look for ITS amendments, rather than chasing a
    fixed list of known amending Acts. From "Personal Data Protection Act 2010" the
    core "Personal Data Protection" yields "...(Amendment) Act" / "...Amendment".

    ``word_tokenised`` says the portal's search matches a bag of words rather than the
    literal string (Singapore SSO's ``PhraseType=AllWords``). There the two variants
    are the SAME question: punctuation is not a token, and the only word that differs,
    "Act", appears in every statute title — so "X (Amendment) Act" and "X Amendment"
    match the same set, and the shorter one is the less constrained of the two.
    Measured on the 2026-07-27 Singapore run: of 21 pairs actually issued, 20 returned
    an identical instrument set (the 21st differed only because one of the two was
    refused by the CDN). Half those renders bought nothing.
    """
    t = (title or "").strip()
    if not t:
        return []
    core = _TITLE_CORE_RE.split(t, maxsplit=1)[0].strip()
    core = re.sub(r"\(amendment\)", "", core, flags=re.IGNORECASE).strip()
    core = re.sub(r"\s+", " ", core)
    if len(core) < 3:
        return []
    if word_tokenised:
        return [f"{core} Amendment"]
    return [f"{core} (Amendment) Act", f"{core} Amendment"]


_SECTION_OF_RE = re.compile(
    r"\b(?:S\.?|Sec\.?|Section|Reg\.?|Regulation|Art\.?|Article)\s*(\d+[A-Za-z]?)",
    re.IGNORECASE,
)
# "comes into operation on a date to be appointed" (Malaysia/Commonwealth) means the
# provision is NOT automatically in force — the safe signal that a repeal has not
# commenced. An explicit date or "on the date of publication" means it has.
_NOT_COMMENCED_RE = re.compile(
    r"comes?\s+into\s+(?:operation|force)\s+on\s+a\s+date\s+to\s+be\s+appointed",
    re.IGNORECASE,
)


# A schedule-qualified path ("Schedule 1 > Section 90(4)"). The parser namespaces such
# clauses (``::sch1-s90``) precisely because a consolidated Act restarts numbering inside
# each Schedule, so the schedule's paragraph 90 and the main body's section 90 are
# different provisions.
_SCHEDULE_PATH_RE = re.compile(r"^\s*Schedule\b", re.IGNORECASE)


def section_of(article_path: str) -> str:
    """The base section number in a citation's article path ("S. 26(1)(a)" -> "26"),
    so a provision can be matched against an amendment's affected sections.

    Empty when no section token is present, AND empty for a schedule-qualified path.
    ``adjudicate_provision`` matches an instruction's target on this token by string
    equality, and the instruction parser only recognises main-body targets ("Section 6 of
    the principal Act is deleted"). Returning the bare number for "Schedule 1 > Section
    90(4)" made an instruction repealing the MAIN BODY's section 90 also repeal the
    schedule's paragraph 90 — and a REPEALED verdict is DELETED from the submission by
    ``enforced_only``, silently. Such a citation now falls back to the document-level
    verdict (STALE_RISK at worst), which flags rather than deletes.

    Restoring provision-level adjudication inside a Schedule needs the instruction parser
    to recognise schedule-qualified targets first; until it does, there is nothing safe to
    match against."""
    if _SCHEDULE_PATH_RE.match(article_path or ""):
        return ""
    m = _SECTION_OF_RE.search(article_path or "")
    return m.group(1) if m else ""


def is_commenced(text: str) -> bool:
    """Whether an amending Act appears to be in force. Conservative: returns False
    only on a clear "date to be appointed" signal (repeal not yet commenced -> a
    deleted provision is flagged but NOT excluded); True otherwise."""
    return _NOT_COMMENCED_RE.search(text[:_HEAD] or "") is None


@dataclass
class ProvisionVerdict:
    """Per-provision adjudication: the currency status for ONE cited section, the
    amending Act's own verbatim for an amended/repealed section (never synthesized),
    and an analyst note. ``status is None`` means "leave the document-level verdict"."""

    status: CurrencyStatus | None
    amendment_text: str = ""
    note: str = ""


def adjudicate_provision(
    *,
    section: str,
    quote: str,
    instructions: list[AmendmentInstruction],
    amend_label: str,
    commenced: bool = True,
) -> ProvisionVerdict:
    """Decide a single cited provision against an amendment's parsed instructions.

    Mirrors the three cases the legal team set: untouched -> keep the original
    (downgraded to CURRENT, since this section was not among those amended); amended
    -> AMENDED, carrying the amending Act's own text alongside the original; deleted
    -> REPEALED (only when commenced — otherwise flagged STALE_RISK pending
    commencement). An act-wide term rename whose old term appears in the quoted text
    adds a low-severity note regardless of the section verdict (broad cascade)."""
    # No section token to adjudicate on (a Preamble, an APP item, a Schedule paragraph --
    # see `section_of`). "Not among the amended sections" would be a conclusion we have not
    # reached: it clears the document-level flag to CURRENT and tells the analyst the Act
    # "did not amend section " with the number missing. Abstain instead, and leave the
    # coarse verdict standing.
    if not section:
        return ProvisionVerdict(None)
    ops = [i for i in instructions if i.target_section and i.target_section == section]
    notes: list[str] = []

    rename_note = ""
    for r in (i for i in instructions if i.kind is InstructionKind.global_rename):
        if r.old_term and r.old_term.lower() in (quote or "").lower():
            rename_note = (
                f'term "{r.old_term}" renamed to "{r.new_term}" by {amend_label}'
            )
            break

    status: CurrencyStatus | None = None
    amendment_text = ""
    if any(i.op is Operation.delete for i in ops):
        amendment_text = "; ".join(i.raw for i in ops if i.op is Operation.delete)
        if commenced:
            status = CurrencyStatus.repealed
            notes.append(f"Section {section} repealed by {amend_label}")
        else:
            status = CurrencyStatus.stale_risk
            notes.append(
                f"Section {section} repealed by {amend_label} — commencement "
                f"unconfirmed; not excluded"
            )
    elif any(i.op in (Operation.amend, Operation.substitute, Operation.insert) for i in ops):
        status = CurrencyStatus.amended
        amendment_text = "; ".join(i.raw for i in ops)
        defprinc = any(i.kind is InstructionKind.definition_or_principle for i in ops)
        notes.append(
            f"Section {section} {'definition/principle ' if defprinc else ''}"
            f"amended by {amend_label}"
        )
    else:
        # The law was amended, but not THIS section -> the cited text is unaffected
        # in substance; clear the coarse document-level flag to CURRENT. A term rename
        # touching the quote is a terminology caveat only (the note carries it), not a
        # substantive change, so the status stays CURRENT and the row is not escalated.
        status = CurrencyStatus.current
        if not rename_note:
            notes.append(f"{amend_label} did not amend section {section}")

    if rename_note:
        notes.append(rename_note)
    return ProvisionVerdict(status, amendment_text, " | ".join(n for n in notes if n))


__all__ = [
    "CurrencyStatus",
    "AmendmentEvent",
    "Identity",
    "AmendmentIndex",
    "CurrencyAssessment",
    "normalize_key",
    "candidate_keys",
    "parse_identity",
    "detect_amends_target",
    "detect_incorporated_to",
    "assess_currency",
    "currency_note",
    # Version classification + instruction parsing (Tier-2 groundwork)
    "VersionKind",
    "Operation",
    "InstructionKind",
    "AmendmentInstruction",
    "classify_version",
    "parse_amendment_instructions",
    "affected_sections",
    "has_global_rename",
    "amendment_search_queries",
    "ProvisionVerdict",
    "section_of",
    "is_commenced",
    "adjudicate_provision",
]
