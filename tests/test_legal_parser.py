"""Format-general structure parser tests.

Covers the two common-law numbering styles (dotted SG/MY, spaced AU), the
year-false-positive rejection that previously mis-parsed AU Acts, and style
auto-selection.
"""
from __future__ import annotations

from lexora.extract.html_extractor import HtmlBlock
from lexora.structure.legal_parser import (
    _cn_to_int,
    _detect_boundaries,
    _looks_unstructured,
    _normalize_numeral,
    parse_structure_html,
)


def _secs(text: str) -> list[str]:
    return [num for _, num, _, _ in _detect_boundaries(text)]


def test_detect_dotted_style_sg_my():
    text = "1. Short title\n\n26.—(1) An organisation must not transfer personal data."
    boundaries = _detect_boundaries(text)
    secs = [num for _, num, _, _ in boundaries]
    assert "1" in secs and "26" in secs
    # the inline subsection of 26.—(1) is captured
    assert ("26", "1") in [(num, sub) for _, num, sub, _ in boundaries]


def test_detect_spaced_style_au():
    text = "1  Short title\n\n2A  Objects of this Act\n\n13  Interference with privacy"
    assert _secs(text) == ["1", "2A", "13"]


def test_year_with_dot_is_not_a_section():
    text = "2010.\n\nsome recital text\n\n5. Real provision body here"
    secs = _secs(text)
    assert "2010" not in secs
    assert "5" in secs


def test_spaced_style_wins_over_stray_year_dot():
    # AU-like doc: spaced sections dominate; a stray "2012." year-dot is dropped.
    text = "1  Short title\n\n2  Commencement\n\n3  Application\n\n2012.\n\n4  Crown"
    assert _secs(text) == ["1", "2", "3", "4"]


def test_prose_document_yields_no_forged_section_numbers():
    """A document with no numbering must yield nothing, not three invented sections.

    The style vote is winner-takes-all with no floor, so on prose whichever opener finds
    two or three accidental matches wins outright. Measured on the OAIC "Summary of version
    changes to APP guidelines": 75,669 characters, no sections, and three WRAPPED PROSE
    LINES beginning "9.2 (see Chapter 9) …" became clauses whose structural_path read
    "Section 9.2". Three such rows reached outputs/submission_glm52_20260712.csv — a
    citation to a provision that does not exist, the same class of fault as the forged
    "Act 2012" law number.
    """
    filler = ("This guidance explains how the Australian Privacy Principles apply to "
              "entities and does not create obligations of its own. " * 300)
    text = (
        f"{filler}\n"
        "9.2 (see Chapter 9) and APP 10.2 (see Chapter 10) are discussed below.\n"
        f"{filler}\n"
        "3.1 but nevertheless retains it under APP 4, because the entity considers\n"
        f"{filler}\n"
        "12.5 (see Chapter 12), APPs 13.1 and 13.2 (see Chapter 13) also apply.\n"
        f"{filler}"
    )
    assert len(text) > 60_000, "the guard is a ratio; the fixture must be prose-sized"
    # The openers still MATCH -- the guard is about what the match is worth, not whether
    # the regex fires -- so the check belongs at document level.
    assert len(_detect_boundaries(text)) == 3
    assert _looks_unstructured(text)

    block = HtmlBlock(dom_anchor="#s", text=text, char_start=0, char_end=len(text))
    assert parse_structure_html("doc", [block]) == []


def test_a_real_statute_is_not_rejected_by_the_prose_guard():
    # The guard is a ratio, so an ordinary Act -- even a long one with long provisions --
    # must stay well clear of it. Measured over the 452-document corpus, the median is
    # 1,338 characters per boundary and the highest genuine document 7,526.
    body = "An organisation must not transfer personal data outside Singapore. " * 60
    text = "\n\n".join(f"{n}.—(1) {body}" for n in range(1, 12))
    assert len(text) // 11 > 3_000  # ~4k chars per section, above the corpus p90
    assert not _looks_unstructured(text)
    assert len(_detect_boundaries(text)) == 11


def test_a_long_schedule_with_two_real_paragraphs_survives():
    """The guard is a DOCUMENT-level ratio, deliberately.

    Applying the same ratio per part deleted two genuine numbered paragraphs of the
    Healthcare Services Act's Schedule 1 (measured: 76 -> 74 clauses): a long schedule
    carrying few, long provisions scores exactly like prose. The document as a whole is
    densely numbered, so it must be judged as a whole.
    """
    body = "A licensable healthcare service means a service of a kind described. " * 220
    text = (
        "\n\n".join(f"{n}.—(1) Short operative provision {n}." for n in range(1, 40))
        + "\n\nSchedule 1—LICENSABLE HEALTHCARE SERVICES\n\n"
        + f"1. For the purposes of the definition, {body}\n\n2. In this Schedule, {body}"
    )
    clauses = parse_structure_html(
        "doc", [HtmlBlock(dom_anchor="#s", text=text, char_start=0, char_end=len(text))]
    )
    paths = {c.structural_path for c in clauses}
    assert "Schedule 1 > Section 1" in paths and "Schedule 1 > Section 2" in paths


def test_parse_structure_html_spaced_emits_verbatim_clauses():
    text = "13  Interference with privacy\n\n14  Australian Privacy Principles apply"
    block = HtmlBlock(dom_anchor="#s", text=text, char_start=0, char_end=len(text))
    clauses = parse_structure_html("doc", [block])
    paths = [c.structural_path for c in clauses]
    assert "Section 13" in paths and "Section 14" in paths
    # spans are verbatim slices of the source text
    for c in clauses:
        assert text[c.span.char_start:c.span.char_end] == c.span.text


# --- Schedule-aware parsing (AU Privacy Act 1988 shape) ----------------------

# A consolidated-Act fragment: a TOC pointer to Schedule 1, a spaced main body
# whose sections re-appear by number inside the schedules, an APP-bearing
# Schedule 1, and a plain Schedule 2.
_AU_DOC = (
    "Contents\n\n"
    "Schedule 1 — Australian Privacy Principles .......... 55\n\n"
    "1  Short title\n\n"
    "6  Interpretation\n\n"
    "8  Commonwealth records\n\n"
    "Schedule 1—Australian Privacy Principles\n\n"
    "Australian Privacy Principle 6—use or disclosure of personal information\n\n"
    "6.1  An APP entity that holds personal information may use it.\n\n"
    "Australian Privacy Principle 8—cross-border disclosure of personal information\n\n"
    "8.1  Before an APP entity discloses personal information to an overseas recipient,"
    " the entity must take reasonable steps.\n\n"
    "8.2  Subclause 8.1 does not apply in the listed circumstances.\n\n"
    "Schedule 2—Other matters\n\n"
    "1  Definitions for this Schedule\n\n"
    "2  Application of this Schedule\n"
)


def _parsed_au():
    block = HtmlBlock(dom_anchor="#d", text=_AU_DOC, char_start=0, char_end=len(_AU_DOC))
    clauses = parse_structure_html("doc", [block])
    return clauses, {c.clause_id: c for c in clauses}


def test_schedule_clause_ids_do_not_collide():
    clauses, by_id = _parsed_au()
    ids = [c.clause_id for c in clauses]
    assert len(ids) == len(set(ids))  # the silent-clobber bug is gone
    # main body s1 and Schedule-2 s1 are now distinct addressable clauses
    assert "doc::s1" in by_id
    assert "doc::sch2-s1" in by_id


def test_main_body_ids_stay_unprefixed_for_back_compat():
    _, by_id = _parsed_au()
    # the TOC "Schedule 1 ..." pointer line must NOT open a part: the body that
    # follows it keeps its plain section ids.
    for cid in ("doc::s1", "doc::s6", "doc::s8"):
        assert cid in by_id
    assert by_id["doc::s8"].structural_path == "Section 8"


def test_app8_is_addressable_inside_schedule_1():
    _, by_id = _parsed_au()
    # APP 8 (AU indicator 6.4 = cross-border disclosure) is now a citable clause,
    # distinct from main-body section 8.
    assert "doc::sch1-app8" in by_id
    app8 = by_id["doc::sch1-app8"]
    assert app8.structural_path == "Schedule 1 > Australian Privacy Principle 8"
    assert app8.section_number == "8"
    assert "cross-border disclosure" in app8.span.text
    # its sub-items collapse to namespaced sub-clauses
    assert by_id["doc::sch1-app8-1"].structural_path == (
        "Schedule 1 > Australian Privacy Principle 8.1"
    )
    assert by_id["doc::sch1-app8-1"].paragraph_number == "1"


def test_all_schedule_spans_are_verbatim():
    clauses, _ = _parsed_au()
    for c in clauses:
        assert _AU_DOC[c.span.char_start:c.span.char_end] == c.span.text


def test_no_schedule_means_unchanged_flat_namespace():
    text = "1  Short title\n\n13  Interference with privacy"
    block = HtmlBlock(dom_anchor="#s", text=text, char_start=0, char_end=len(text))
    by_id = {c.clause_id: c for c in parse_structure_html("doc", [block])}
    assert set(by_id) == {"doc::s1", "doc::s13"}  # no parts -> pre-Schedule behaviour


def test_multiletter_section_suffix_detected():
    # AU inserts amending sections as 6A / 6AA / 6AB; a single-letter cap folded
    # 6AA's body into section 6 (so its "(1)" reappeared as a second s6-1).
    text = "6  Interpretation\n\n6AA  Responsible person\n\n6AB  Permitted purpose"
    assert _secs(text) == ["6", "6AA", "6AB"]


def test_two_space_running_header_does_not_open_a_schedule():
    # A consolidated Act repeats a page header "Schedule 1  Australian Privacy
    # Principles" (two spaces, no dash) on every page; only the dashed divisional
    # heading is a real part boundary, so the body section keeps its plain id.
    text = (
        "Schedule 1  Australian Privacy Principles\n\n"
        "1  Short title\n\n"
        "Schedule 1  Australian Privacy Principles\n\n"
        "6  Interpretation"
    )
    block = HtmlBlock(dom_anchor="#s", text=text, char_start=0, char_end=len(text))
    by_id = {c.clause_id: c for c in parse_structure_html("doc", [block])}
    assert set(by_id) == {"doc::s1", "doc::s6"}  # no sch1- prefix appeared


def test_repeated_marker_gets_unique_suffix():
    # A section whose body restarts a "(1)" list (nested numbering) would mint two
    # s5-1 ids; the uniqueness net keeps both, suffixing the later one.
    text = "5  Definitions\n\n(1) first sense of the term\n\n(1) second sense of the term"
    block = HtmlBlock(dom_anchor="#s", text=text, char_start=0, char_end=len(text))
    ids = [c.clause_id for c in parse_structure_html("doc", [block])]
    assert ids.count("doc::s5-1") == 1
    assert "doc::s5-1~2" in ids
    assert len(ids) == len(set(ids))  # invariant: ids are unique per document


# --- Part/Division lead-in block attaches to the following section -----------

def test_part_division_block_does_not_bleed_onto_previous_clause():
    # A consolidated Act inserts a whole "PART 4 / TITLE / Division 1 — …" block
    # plus an edition tag and the next section's marginal heading between two
    # sections. The block must attach to the FOLLOWING section, not bleed onto the
    # previous clause's verbatim tail.
    text = (
        "12.  The first section ends with operative text here.\n\n"
        "[40/2020]\n"
        "PART 4\n"
        "COLLECTION, USE AND DISCLOSURE OF\n"
        "PERSONAL DATA\n"
        "Division 1 — Consent\n"
        "Consent required\n"
        "13.  An organisation must obtain consent before collecting data."
    )
    block = HtmlBlock(dom_anchor="#s", text=text, char_start=0, char_end=len(text))
    by_id = {c.clause_id: c for c in parse_structure_html("doc", [block])}
    s12, s13 = by_id["doc::s12"], by_id["doc::s13"]
    # the previous clause's tail is clean — no Part/Division/edition lead-in
    assert "PART 4" not in s12.span.text
    assert "Division 1" not in s12.span.text
    assert "[40/2020]" not in s12.span.text
    assert s12.span.text.rstrip().endswith("operative text here.")
    # the lead-in block now heads the section it introduces
    assert s13.span.text.lstrip().startswith("[40/2020]")
    assert "PART 4" in s13.span.text and "Consent required" in s13.span.text
    # verbatim invariant holds for both
    for c in (s12, s13):
        assert text[c.span.char_start:c.span.char_end] == c.span.text


def test_wrapped_prose_part_reference_is_not_treated_as_a_heading():
    # "…under Division 2 of\nPart II;" line-starts with "Part II" but ends with a
    # semicolon — it is operative prose, not a divisional heading, so it must stay
    # inside its own clause and never be pulled onto the next section.
    text = (
        "93.  A decision to refuse registration under Division 2 of\n"
        "Part II; and any related matter.\n\n"
        "94.  The next section."
    )
    block = HtmlBlock(dom_anchor="#s", text=text, char_start=0, char_end=len(text))
    by_id = {c.clause_id: c for c in parse_structure_html("doc", [block])}
    assert "Part II;" in by_id["doc::s93"].span.text
    assert "Part II" not in by_id["doc::s94"].span.text


# --- APP real heading (clause-number prefix) + interior heading pull ----------

def test_app_real_heading_with_clause_number_wins_over_overview():
    # The front "Overview" lists each principle WITHOUT a number; the real heading
    # carries a leading clause number ("8  Australian Privacy Principle 8—…"). The
    # numbered heading must be the one that opens the principle, otherwise the next
    # principle's heading bleeds onto the previous one's last item.
    text = (
        "Schedule 1—Australian Privacy Principles\n\n"
        "Overview\n\n"
        "Australian Privacy Principle 8—cross-border disclosure of personal information\n\n"
        "Australian Privacy Principle 9—government related identifiers\n\n"
        "Part 4  Integrity of personal information\n\n"
        "8  Australian Privacy Principle 8—cross-border disclosure of personal information\n\n"
        "8.1  Before disclosing, the entity must take reasonable steps.\n\n"
        "9  Australian Privacy Principle 9—government related identifiers\n\n"
        "9.1  An organisation must not adopt a government related identifier.\n"
    )
    block = HtmlBlock(dom_anchor="#d", text=text, char_start=0, char_end=len(text))
    by_id = {c.clause_id: c for c in parse_structure_html("doc", [block])}
    assert "doc::sch1-app8-1" in by_id
    app8_1 = by_id["doc::sch1-app8-1"]
    # the last item of APP 8 stops at its own sentence, not the APP 9 heading
    assert app8_1.span.text.rstrip().endswith("reasonable steps.")
    assert "Principle 9" not in app8_1.span.text
    # verbatim invariant
    for c in by_id.values():
        assert text[c.span.char_start:c.span.char_end] == c.span.text


def test_subsection_marginal_heading_attaches_to_following_subsection():
    text = (
        "80TB  Monitoring powers\n\n"
        "(2) The information is correct. It includes powers of entry.\n"
        "Matters subject to monitoring\n"
        "(3) The following matters are subject to monitoring."
    )
    block = HtmlBlock(dom_anchor="#s", text=text, char_start=0, char_end=len(text))
    by_id = {c.clause_id: c for c in parse_structure_html("doc", [block])}
    assert by_id["doc::s80TB-2"].span.text.rstrip().endswith("powers of entry.")
    assert "Matters subject to monitoring" in by_id["doc::s80TB-3"].span.text


# --- G-1c multi-script civil-law articles ------------------------------------

def test_chinese_numeral_parsing():
    assert _cn_to_int("一") == 1
    assert _cn_to_int("十") == 10
    assert _cn_to_int("十三") == 13
    assert _cn_to_int("二十四") == 24
    assert _cn_to_int("二十六") == 26
    assert _cn_to_int("一百零五") == 105


def test_normalize_numeral_across_scripts():
    assert _normalize_numeral("13") == "13"       # ascii
    assert _normalize_numeral("１３") == "13"       # fullwidth
    assert _normalize_numeral("๑๓") == "13"       # thai digits
    assert _normalize_numeral("十三") == "13"       # chinese numerals
    assert _normalize_numeral("条") == ""          # not a numeral


def test_chinese_articles_are_addressable():
    # 第N条 articles become citable clauses with ::aN ids and "Article N" paths;
    # native Chinese numerals are folded to ASCII so the id is stable.
    text = "第一条 本法的目的。\n\n第十三条 跨境提供个人信息的规定。\n\n第二十四条 安全措施。"
    block = HtmlBlock(dom_anchor="#d", text=text, char_start=0, char_end=len(text))
    clauses = parse_structure_html("doc", [block])
    by_id = {c.clause_id: c for c in clauses}
    assert "doc::a1" in by_id
    assert "doc::a13" in by_id
    assert "doc::a24" in by_id
    art13 = by_id["doc::a13"]
    assert art13.structural_path == "Article 13"
    assert art13.article_number == "13"
    # verbatim contract: the stored span slices back to the source text.
    assert text[art13.span.char_start : art13.span.char_end] == art13.span.text


def test_thai_articles_are_addressable():
    text = "มาตรา ๑ บทนำ\n\nมาตรา ๑๓ การส่งข้อมูลข้ามพรมแดน\n\nมาตรา ๒๔ มาตรการรักษาความปลอดภัย"
    block = HtmlBlock(dom_anchor="#d", text=text, char_start=0, char_end=len(text))
    by_id = {c.clause_id: c for c in parse_structure_html("doc", [block])}
    assert {"doc::a1", "doc::a13", "doc::a24"} <= set(by_id)


def test_english_act_is_not_misparsed_as_articles():
    # An English Act that merely mentions "article" in prose must still parse as
    # sections — the winner-takes-all picks the dominant style, not stray matches.
    text = (
        "1  Short title\n\n"
        "2  This Act gives effect to Article 17 of the Convention.\n\n"
        "13  Cross-border disclosure"
    )
    by_id = {
        c.clause_id: c
        for c in parse_structure_html(
            "doc", [HtmlBlock(dom_anchor="#s", text=text, char_start=0, char_end=len(text))]
        )
    }
    assert set(by_id) == {"doc::s1", "doc::s2", "doc::s13"}
    assert not any("::a" in cid for cid in by_id)
