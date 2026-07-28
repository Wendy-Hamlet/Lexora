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
