"""Amendment detection & currency assessment (lexora.cite.amendments).

Deterministic / offline — no LLM, no network. Exercises the four signals the
integrator prioritised (B corpus, A in-doc, D portal, C registry backstop), the
multi-amendment chain (every amendment kept, chronological, later-overrides-earlier),
and the stale-vs-current verdict.
"""
from __future__ import annotations

from lexora.cite.amendments import (
    AmendmentIndex,
    CurrencyStatus,
    InstructionKind,
    Operation,
    VersionKind,
    adjudicate_provision,
    affected_sections,
    amendment_search_queries,
    assess_currency,
    classify_version,
    currency_note,
    detect_amends_target,
    detect_incorporated_to,
    has_global_rename,
    is_commenced,
    normalize_key,
    parse_amendment_instructions,
    parse_identity,
    section_of,
)

# A masthead in the Malaysian gazette style: an amending Act that targets a principal.
AMENDING_A1727 = """
LAWS OF MALAYSIA
Act A1727
PERSONAL DATA PROTECTION (AMENDMENT) ACT 2024
Date of Royal Assent ... 18 October 2024
An Act to amend the Personal Data Protection Act 2010.
ENACTED by the Parliament of Malaysia as follows:
1. This Act may be cited as the Personal Data Protection (Amendment) Act 2024.
2. The Personal Data Protection Act 2010 [Act 709], which is referred to as the
"principal Act", is amended as set out in this Act.
3. Section 2 of the principal Act is amended by deleting the words ...
"""

PRINCIPAL_709_AS_MADE = """
LAWS OF MALAYSIA
ACT 709
PERSONAL DATA PROTECTION ACT 2010
Date of Royal Assent : 2 June 2010
An Act to regulate the processing of personal data in commercial transactions.
ENACTED by the Parliament of Malaysia as follows:
PART I - PRELIMINARY
"""

# A compiled/reprint document that states its incorporation point (Signal A).
CONSOLIDATED_REPRINT = """
PERSONAL DATA PROTECTION ACT 2010
Incorporating all amendments up to and including Act A1727 of 2024
Reprint as at 1 January 2025
"""


def test_parse_identity_reads_number_title_year():
    ident = parse_identity(PRINCIPAL_709_AS_MADE)
    assert ident.number == "Act 709"
    assert "PERSONAL DATA PROTECTION ACT 2010" in ident.title
    assert ident.year == 2010
    assert ident.key == "act:709"


def test_normalize_key_amendment_shares_principal_when_keyed_by_number():
    # The amendment is keyed to the principal via its bracketed [Act 709] number.
    assert normalize_key(number="Act 709") == "act:709"
    assert normalize_key(number="709") == "act:709"
    # Title-only fallback strips the "(Amendment)" infix so it collapses onto the
    # principal's title key.
    assert normalize_key(title="Personal Data Protection (Amendment) Act 2024") == \
        normalize_key(title="Personal Data Protection Act 2024")


def test_detect_amends_target_principal_form():
    target = detect_amends_target(AMENDING_A1727)
    assert target is not None
    t_num, t_title = target
    assert t_num == "Act 709"
    assert "Personal Data Protection Act 2010" in t_title


def test_detect_amends_target_none_for_principal():
    assert detect_amends_target(PRINCIPAL_709_AS_MADE) is None


def test_detect_incorporated_to():
    assert detect_incorporated_to(CONSOLIDATED_REPRINT) == 2025
    assert detect_incorporated_to(PRINCIPAL_709_AS_MADE) is None


def test_detect_incorporated_to_au_compilation_phrasings():
    # AU Federal Register compilation mastheads, both observed wordings; the amending
    # Act's own 2-digit number must not be read as the year.
    assert detect_incorporated_to(
        "Privacy Act 1988\nCompilation No. 104\nIncludes amendments up to: Act No. 79, 2021"
    ) == 2021
    assert detect_incorporated_to(
        "Privacy Act 1988\nCompilation No. 104\nIncludes amendments: Act No. 75, 2025"
    ) == 2025


def test_signal_b_corpus_builds_chain():
    idx = AmendmentIndex()
    idx.add_from_corpus([
        ("Personal Data Protection Act 2010", PRINCIPAL_709_AS_MADE),
        ("Personal Data Protection (Amendment) Act 2024", AMENDING_A1727),
    ])
    events = idx.events_for("act:709")
    assert len(events) == 1
    assert events[0].year == 2024
    assert events[0].detected_by == "corpus"


def test_as_made_original_flagged_stale():
    idx = AmendmentIndex()
    idx.add_from_corpus([("amend", AMENDING_A1727)])
    # The 2010 original incorporates nothing past 2010 -> the 2024 amendment is missing.
    a = assess_currency(principal_key="act:709", incorporated_to=2010, index=idx)
    assert a.status is CurrencyStatus.stale_risk
    assert [e.year for e in a.missing] == [2024]
    note = currency_note(a)
    assert "2024" in note and "consolidated in-force version" in note


def test_consolidated_reprint_is_current():
    idx = AmendmentIndex()
    idx.add_from_corpus([("amend", AMENDING_A1727)])
    a = assess_currency(principal_key="act:709", incorporated_to=2025, index=idx)
    assert a.status is CurrencyStatus.current
    assert a.missing == []
    assert currency_note(a) == ""


def test_no_amendment_signal_is_unknown_not_stale():
    idx = AmendmentIndex()
    a = assess_currency(principal_key="act:709", incorporated_to=2010, index=idx)
    assert a.status is CurrencyStatus.unknown
    assert a.missing == []


def test_self_consolidated_with_no_chain_is_current_not_unknown():
    # A source that is itself a consolidation states its own currency point, so with
    # no amendment chain known it is CURRENT to that point — not UNKNOWN (which stays
    # reserved for an as-made original, the default flag).
    idx = AmendmentIndex()
    a = assess_currency(
        principal_key="act:709", incorporated_to=2026, index=idx, self_consolidated=True
    )
    assert a.status is CurrencyStatus.current
    assert a.missing == []
    # Same inputs but as an as-made original -> UNKNOWN.
    b = assess_currency(principal_key="act:709", incorporated_to=2026, index=idx)
    assert b.status is CurrencyStatus.unknown


def test_multiple_amendments_later_overrides_and_partial_incorporation():
    """A law amended three times; a source incorporating to 2018 is missing the two
    later amendments (every newer event matters, not just the last)."""
    idx = AmendmentIndex()
    idx.add_registry({"709": [
        {"by": "Act A100", "year": 2016},
        {"by": "Act A200", "year": 2020},
        {"by": "Act A300", "year": 2024},
    ]})
    chain = idx.events_for("act:709")
    assert [e.year for e in chain] == [2016, 2020, 2024]  # chronological
    a = assess_currency(principal_key="act:709", incorporated_to=2018, index=idx)
    assert a.status is CurrencyStatus.stale_risk
    assert [e.year for e in a.missing] == [2020, 2024]


def test_registry_backstop_and_corpus_dedup():
    """Registry (C) fills what corpus (B) missed, and the two are de-duped on the
    same year — corpus's richer record (with amending_id) is kept."""
    idx = AmendmentIndex()
    idx.add_from_corpus([("Act A1727", AMENDING_A1727)])  # year 2024, id "Act A1727"
    idx.add_registry({"709": [{"by": "Act A1727", "year": 2024}]})
    events = idx.events_for("act:709")
    assert len(events) == 1  # de-duped, not doubled
    assert events[0].detected_by == "corpus"


# --- Version classification + instruction parsing (Tier-2 groundwork) ---

# A delta amending Act exercising every instruction class: a global term rename, a
# definition-section edit, a substantive section substitution, a section deletion,
# and a brand-new inserted section.
AMEND_BODY = """
LAWS OF MALAYSIA
Act A1727
PERSONAL DATA PROTECTION (AMENDMENT) ACT 2024
An Act to amend the Personal Data Protection Act 2010.
ENACTED by the Parliament of Malaysia as follows:
2. The Personal Data Protection Act 2010 [Act 709], referred to as the "principal
Act", is amended as set out in this Act.
3. Section 4 of the principal Act is amended by inserting the following definition
of "data controller".
4. The principal Act is amended by substituting for the words "data user" wherever
appearing the words "data controller".
5. Section 6 of the principal Act is amended by substituting for subsection (1) the
following subsection.
6. Section 129 of the principal Act is deleted.
7. The principal Act is amended by inserting after section 7 the following new
section.
"""


def test_classify_version_three_kinds():
    assert classify_version(AMEND_BODY) is VersionKind.amendment_delta
    assert classify_version(PRINCIPAL_709_AS_MADE) is VersionKind.original
    assert classify_version(CONSOLIDATED_REPRINT) is VersionKind.consolidated


def test_classify_omnibus_amendment_with_generic_long_title():
    # AU theme-named omnibus Acts carry a generic long title that names no single
    # principal ("An Act to amend legislation relating to ...") — the strict
    # "to amend the X Act YYYY" form misses it, so it WAS misread as an as-made
    # original. The broad "An Act to amend ..." long-title marker classifies it.
    omnibus = (
        "Telecommunications Legislation Amendment (Information Disclosure, National "
        "Interest and Other Measures) Act 2023\nNo. 17, 2023\n"
        "An Act to amend legislation relating to telecommunications, and for related "
        "purposes\nContents\n1 Short title\n2 Commencement\n"
    )
    assert classify_version(omnibus) is VersionKind.amendment_delta
    # A genuine as-made principal whose long title does NOT say "to amend" stays original.
    principal = (
        "Telecommunications Act 1997\nNo. 47, 1997\n"
        "An Act relating to telecommunications, and for related purposes\nContents\n"
    )
    assert classify_version(principal) is VersionKind.original


def test_parse_global_rename():
    instrs = parse_amendment_instructions(AMEND_BODY)
    renames = [i for i in instrs if i.kind is InstructionKind.global_rename]
    assert len(renames) == 1
    r = renames[0]
    assert r.op is Operation.rename
    assert r.old_term == "data user"
    assert r.new_term == "data controller"
    assert has_global_rename(instrs)


def test_parse_definition_vs_section_op_classification():
    instrs = parse_amendment_instructions(AMEND_BODY)
    by_sec = {i.target_section: i for i in instrs if i.target_section}
    # Section 4 edits a DEFINITION -> high-cascade class.
    assert by_sec["4"].kind is InstructionKind.definition_or_principle
    # Section 6 is an ordinary substantive edit.
    assert by_sec["6"].kind is InstructionKind.section_op
    assert by_sec["6"].op is Operation.amend


def test_parse_delete_and_insert_ops():
    instrs = parse_amendment_instructions(AMEND_BODY)
    by_sec = {i.target_section: i for i in instrs if i.target_section}
    assert by_sec["129"].op is Operation.delete
    assert by_sec["7"].op is Operation.insert
    assert affected_sections(instrs) == {"4", "6", "129", "7"}


def test_parser_ignores_non_amendment_text():
    assert parse_amendment_instructions(PRINCIPAL_709_AS_MADE) == []


# --- Per-provision adjudication (Tier-2 step 2) ---

def test_section_of_parses_article_path():
    assert section_of("S. 26(1)(a)") == "26"
    assert section_of("Section 6") == "6"
    assert section_of("Art. 14") == "14"
    assert section_of("Preamble") == ""


def test_an_act_title_on_a_revised_line_is_not_a_consolidation_point():
    """`incorporated_to` is the year a compiled document says it folds amendments up to,
    and `assess_currency` judges every citation against it.

    `\\brevised\\b[^.\\n]*?<year>` matched Singapore's standing boilerplate "REVISED EDITION
    OF THE LAWS ACT 1983" -- the name of the enabling statute -- on 92 documents, so a 2018
    Act reported that it incorporated amendments only up to 1983. It also matched the SSO
    label "Revised Edition — Cybersecurity Act 2018", where the year is the Act's own.
    Every "revised" match in the corpus was one of those two shapes: 92 of 303 Singapore
    documents carried a fabricated consolidation point, and after this change 302 carry
    none, which is the honest answer.
    """
    from lexora.cite.amendments import detect_incorporated_to

    # Act titles on a "revised" line are not revision dates.
    assert detect_incorporated_to("REVISED EDITION OF THE LAWS ACT 1983") is None
    assert detect_incorporated_to("Revised Edition — Cybersecurity Act 2018") is None
    assert detect_incorporated_to("Revised Edition — Official Secrets Act 1935") is None

    # Genuine consolidation claims still read, including Malaysia's reprint marker --
    # 52 Malaysian documents keep a real value spanning 2006-2025.
    assert detect_incorporated_to("Revised—2024") == 2024
    assert detect_incorporated_to("Revised Edition 2020") == 2020
    assert detect_incorporated_to("revised up to 31 December 2019") == 2019
    assert detect_incorporated_to("Incorporating all amendments up to 1 January 2006") == 2006


def test_a_title_year_is_not_an_act_number():
    """`_ACT_NUMBER_RE` is written for Malaysia, where the number FOLLOWS the word
    ("Act 709"). Singapore and Australia put the YEAR there -- "Personal Data Protection
    Act 2012" -- so the first match in the masthead was stored as the instrument's number.

    Measured over the corpus before the fix: 295 of 303 Singapore documents (97%), 34 of 50
    Australian and 46 of 102 Malaysian ones carried a year as their number. `Identity.key`
    feeds `candidate_keys`, which is how an amending Act is matched to the principal it
    amends, so every instrument of a given year collided on one key -- "Act 1967" was
    shared by eleven documents across Malaysia and Singapore.
    """
    from lexora.cite.amendments import normalize_key, parse_identity

    sg = "PERSONAL DATA PROTECTION ACT 2012\nAn Act to govern the collection of data.\n"
    assert parse_identity(sg).number == ""          # honest: SG numbers are "Act N of YYYY"
    assert parse_identity(sg).year == 2012          # the year is still read, as a year

    # Malaysia prints both; the real number must win over the title's year.
    my = "LAWS OF MALAYSIA\nAct 709\nPERSONAL DATA PROTECTION ACT 2010\n"
    assert parse_identity(my).number == "Act 709"
    amending = "LAWS OF MALAYSIA\nAct A1727\nPERSONAL DATA PROTECTION (AMENDMENT) ACT 2024\n"
    assert parse_identity(amending).number == "Act A1727"

    # And the key builder does not depend on its caller having been careful.
    assert normalize_key(number="Act 709") == "act:709"
    assert normalize_key(number="Act 26 of 2012") == "act:26"
    assert normalize_key(number="Act 2012", title="Personal Data Protection Act 2012") \
        != "act:2012"
    assert normalize_key(number="2012", title="Personal Data Protection Act 2012") \
        != "act:2012"


def test_repealed_and_substituted_is_a_substitution_not_a_removal():
    """"Repealed AND the following substituted" is how the Westminster register REPLACES a
    section -- the provision goes on existing with new words.

    Reading the leading verb alone made it a deletion, a deletion becomes
    CurrencyStatus.repealed, and `enforced_only` DROPS repealed rows from the submission.
    Measured on a Singapore replay: PDPA s.24, the Protection Obligation and plainly in
    force, was deleted from the CSV on exactly this sentence -- and because it survived in
    the per-document artifacts, the JSON sidecar carried 552 rows against the CSV's 551.
    """
    from lexora.cite.amendments import (
        CurrencyStatus,
        Operation,
        adjudicate_provision,
        parse_amendment_instructions,
    )

    def op_for(sentence: str) -> Operation:
        return parse_amendment_instructions(sentence)[0].op

    # A real repeal is still a repeal.
    assert op_for("Section 24 of the principal Act is repealed.") is Operation.delete
    # Every common replacement formula is a substitution.
    for s in (
        "Section 24 of the principal Act is repealed and the following section substituted:",
        "Section 24 of the principal Act is repealed and substituted by the following:",
        "Section 24 of the principal Act is repealed and there is substituted the following:",
        "Section 24 of the principal Act is deleted and replaced by the following section:",
    ):
        assert op_for(s) is Operation.substitute, s

    # A substitution belonging to the NEXT instruction must not rescue this one.
    two = ("Section 24 of the principal Act is repealed. Section 25 of the principal Act "
           "is repealed and the following substituted:")
    ops = [i.op for i in parse_amendment_instructions(two)]
    assert ops == [Operation.delete, Operation.substitute]

    # End to end: a replaced section is flagged, not deleted.
    instrs = parse_amendment_instructions(
        "Section 24 of the principal Act is repealed and the following section substituted:"
    )
    verdict = adjudicate_provision(section="24", quote="x", instructions=instrs,
                                   amend_label="Act 26 of 2020")
    assert verdict.status is CurrencyStatus.amended


def test_a_schedule_paragraph_is_not_the_main_body_section_of_the_same_number():
    """A consolidated Act restarts numbering inside each Schedule — that is why the parser
    namespaces schedule clauses. `adjudicate_provision` matches an instruction's target by
    string equality on this token, and the instruction parser only recognises main-body
    targets ("Section 6 of the principal Act is deleted"), so returning the bare number
    here let an instruction repealing the MAIN BODY's section 90 also repeal
    "Schedule 1 > Section 90(4)" — which `enforced_only` then DELETES from the submission,
    silently. Both paths are real rows in the round-1 submission.
    """
    from lexora.cite.amendments import (
        AmendmentInstruction,
        CurrencyStatus,
        InstructionKind,
        Operation,
        adjudicate_provision,
    )

    assert section_of("Section 90(4)") == "90"
    assert section_of("Schedule 1 > Section 90(4)") == ""
    assert section_of("Schedule 2 > Section 27J(1A)") == ""

    instr = AmendmentInstruction(
        kind=InstructionKind.section_op, op=Operation.delete, target_section="90",
        raw="Section 90 of the principal Act is deleted.",
    )
    kw = dict(quote="x", instructions=[instr], amend_label="Act A123")
    # The main body's section 90 is genuinely repealed...
    assert adjudicate_provision(section=section_of("Section 90(4)"), **kw).status \
        is CurrencyStatus.repealed
    # ...and the schedule paragraph of the same number is left to the document-level
    # verdict, which flags rather than deletes.
    assert adjudicate_provision(
        section=section_of("Schedule 1 > Section 90(4)"), **kw
    ).status is None


def test_is_commenced():
    assert is_commenced("This Act comes into operation on 1 January 2025.") is True
    assert is_commenced(
        "This Act comes into operation on a date to be appointed by the Minister."
    ) is False


def _instrs():
    return parse_amendment_instructions(AMEND_BODY)


def test_adjudicate_untouched_section_downgraded_to_current():
    v = adjudicate_provision(
        section="40", quote="An organisation may ...", instructions=_instrs(),
        amend_label="Act A1727 (2024)",
    )
    assert v.status is CurrencyStatus.current
    assert "did not amend section 40" in v.note


def test_adjudicate_amended_section_carries_amendment_text():
    v = adjudicate_provision(
        section="6", quote="A data user shall ...", instructions=_instrs(),
        amend_label="Act A1727 (2024)",
    )
    assert v.status is CurrencyStatus.amended
    assert v.amendment_text  # the amending Act's own instruction, verbatim
    assert "Section 6" in v.note


def test_adjudicate_deleted_section_repealed_when_commenced_else_stale():
    commenced = adjudicate_provision(
        section="129", quote="...", instructions=_instrs(),
        amend_label="Act A1727 (2024)", commenced=True,
    )
    assert commenced.status is CurrencyStatus.repealed
    pending = adjudicate_provision(
        section="129", quote="...", instructions=_instrs(),
        amend_label="Act A1727 (2024)", commenced=False,
    )
    assert pending.status is CurrencyStatus.stale_risk
    assert "commencement unconfirmed" in pending.note


def test_adjudicate_global_rename_annotates_matching_quote():
    # An untouched section whose quote uses the renamed term gets the rename note.
    v = adjudicate_provision(
        section="40", quote="A data user shall not transfer personal data ...",
        instructions=_instrs(), amend_label="Act A1727 (2024)",
    )
    assert 'renamed to "data controller"' in v.note


# --- General amendment-search query derivation (discovery) ---

def test_amendment_search_queries_general_from_title():
    qs = amendment_search_queries("Personal Data Protection Act 2010")
    assert "Personal Data Protection (Amendment) Act" in qs
    assert "Personal Data Protection Amendment" in qs
    # Works for any law, not a hardcoded amendment name.
    assert amendment_search_queries("Cyber Security Act 2024")[0] == \
        "Cyber Security (Amendment) Act"


def test_amendment_search_queries_strips_existing_amendment_and_blanks():
    # An amendment's own title collapses onto its principal's core.
    assert amendment_search_queries("Personal Data Protection (Amendment) Act 2024")[0] == \
        "Personal Data Protection (Amendment) Act"
    assert amendment_search_queries("") == []
    assert amendment_search_queries("Act 2010") == []


def test_amendment_search_queries_collapse_on_a_word_tokenised_portal():
    """On a bag-of-words portal the two variants ask the same question.

    Singapore SSO searches with `PhraseType=AllWords`: punctuation is not a token, and
    the only word that differs between the variants -- "Act" -- appears in every
    statute title, so "X (Amendment) Act" and "X Amendment" match the same set.
    Measured on the 2026-07-27 run: 20 of the 21 pairs actually issued returned an
    identical instrument set (the 21st differed only because the CDN refused one of
    the two), so half of those renders bought nothing.
    """
    both = amendment_search_queries("Personal Data Protection Act 2010")
    one = amendment_search_queries("Personal Data Protection Act 2010",
                                   word_tokenised=True)
    assert len(both) == 2
    # The surviving query is the LESS constrained of the pair (it drops a word).
    assert one == ["Personal Data Protection Amendment"]
    assert set(one).issubset(both)


def test_word_tokenised_collapse_keeps_the_blank_and_strip_behaviour():
    assert amendment_search_queries("", word_tokenised=True) == []
    assert amendment_search_queries("Act 2010", word_tokenised=True) == []
    assert amendment_search_queries("Cyber Security (Amendment) Act 2024",
                                    word_tokenised=True) == ["Cyber Security Amendment"]
