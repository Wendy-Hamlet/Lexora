"""Law-name normalisation — the submission's second column."""
from __future__ import annotations

import pytest

from lexora.export.law_name import (
    looks_like_a_filename,
    normalize_law_name,
    resolve_law_name,
    statute_title_from_text,
)

# Verbatim from the 2026-07-14 submission: Malaysia's AGC portal serves each Act as a file
# whose NAME is whatever the uploader called it, and that name reached the Law Name column
# on eight of the eleven Malaysian rows carrying the NEW claim.
MY_FILENAMES = [
    ("Act 706 ori.pdf", "Act 706"),
    ("Act 758 Final.pdf", "Act 758"),
    ("Act A1727.pdf", "Act A1727"),
    ("ACT 791 as at 1 January 2024 (Final).pdf", "ACT 791"),
    ("Act 590(Reprint 1 Jan 2006)", "Act 590"),
    ("Act 670 (Dalam talian 2025)", "Act 670"),
    ("DRAF KEDUA AKTA 701 (final)(KU) (1).pdf", "DRAF KEDUA AKTA 701"),
    ("09. Act A1441", "Act A1441"),
    ("INCOME TAX ACT 1967 (ACT 53) 23.11.2021.pdf", "INCOME TAX ACT 1967 (ACT 53)"),
]

MY_MASTHEAD = (
    "LAWS OF MALAYSIA\nONLINE VERSION OF UPDATED\nTEXT OF REPRINT\n"
    "Act 758\nFINANCIAL SERVICES ACT 2013\nAs at 6 October 2023\n"
)


@pytest.mark.parametrize(("raw", "expected"), MY_FILENAMES)
def test_a_portal_filename_is_cleaned_not_shipped(raw, expected):
    assert looks_like_a_filename(raw)
    assert resolve_law_name(raw) == expected


def test_the_act_states_its_own_name_and_that_wins():
    # Better than any cleanup of the file name: the document says what it is.
    assert statute_title_from_text(MY_MASTHEAD) == "Financial Services Act 2013"
    assert resolve_law_name("Act 758 Final.pdf", MY_MASTHEAD) == "Financial Services Act 2013"


def test_the_masthead_anchor_is_not_relaxed_to_the_copyright_page():
    """Every Malaysian reprint carries "UNDER THE AUTHORITY OF THE REVISION OF LAWS ACT
    1968". Dropping the "Act <n>" anchor to raise the hit rate would turn that boilerplate
    into a confident, wrong law name on every such document."""
    copyright_page = (
        "REPRINT\nAs at 1 October 2018\nPUBLISHED BY\n"
        "THE COMMISSIONER OF LAW REVISION, MALAYSIA\n"
        "UNDER THE AUTHORITY OF THE REVISION OF LAWS ACT 1968\n2018\n"
    )
    assert statute_title_from_text(copyright_page) == ""
    assert resolve_law_name("Act 643 Reprint 2006_unlocked", copyright_page) \
        == "Act 643 Reprint 2006"


def test_a_real_law_name_is_never_touched():
    # "Online Safety ... Act 2025" contains a word the upload-noise rules look for, so the
    # filename test must be settled by how the name ENDS, not by what it contains -- a
    # title wrongly flagged is a title this module is licensed to overwrite from the text.
    good = [
        "Online Safety (Relief and Accountability) Act 2025",
        "Personal Data Protection Act 2012",
        "Telecommunications and Other Legislation Amendment (Assistance and Access) Act 2018",
        "Personal Data Protection Code of Practice For the Utilities Sector (Electricity)",
    ]
    for name in good:
        assert not looks_like_a_filename(name), name
        assert resolve_law_name(name, MY_MASTHEAD) == name


def test_cleanup_never_removes_the_instrument_it_names():
    # A greedy "drop everything after this word" rule turned this into "Mei 2019".
    out = resolve_law_name("Mei 2019 Reprint Online Act 678.pdf")
    assert "Act 678" in out
    # Parenthesised parts of a real title survive the noise rules.
    assert resolve_law_name("A1779 - ATOMIC ENERGY LICENSING (AMENDMENT) ACT 2025.pdf") \
        == "A1779 - ATOMIC ENERGY LICENSING (AMENDMENT) ACT 2025"

# Verbatim from the 2026-07-14 run: SSO renders the statute inside a page shell,
# and the crawled title swallowed the whole thing.
SSO_CYBER = (
    "Cybersecurity Act 2018 Current version as at 14 Jul 2026 Part 3 "
    "PROVIDER-OWNED CRITICAL INFORMATION INFRASTRUCTURE PROVIDER-OWNED CRITICAL "
    "INFORMATION INFRASTRUCTURE Act 19 of 2024 Actions Download PDF (434.4 KB) "
    "Add to My Collections Amend"
)
SSO_PDPA = (
    "Personal Data Protection Act 2012 Current version as at 14 Jul 2026 "
    "Actions Download PDF (1.2 MB) Add to My Collections"
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (SSO_CYBER, "Cybersecurity Act 2018"),
        (SSO_PDPA, "Personal Data Protection Act 2012"),
        ("Income Tax Act 1947 Current version as at 14 Jul 2026", "Income Tax Act 1947"),
    ],
)
def test_strips_the_sso_page_shell(raw, expected):
    assert normalize_law_name(raw) == expected


@pytest.mark.parametrize(
    "title",
    [
        # Australia's amendment Acts really are this long — the cut is anchored to
        # the boilerplate, never to a length budget.
        "Telecommunications and Other Legislation Amendment (Assistance and Access) Act 2018",
        "Personally Controlled Electronic Health Records (Consequential Amendments) Act 2012",
        "Personal Data Protection Code of Practice For Banking Sector And Financial Institutions",
        "Privacy Act 1988",
    ],
)
def test_leaves_genuine_titles_alone(title):
    assert normalize_law_name(title) == title


def test_collapses_whitespace_from_html_extraction():
    assert normalize_law_name("Banking\n  Act\t1970") == "Banking Act 1970"


def test_empty_and_none():
    assert normalize_law_name(None) == ""
    assert normalize_law_name("") == ""
    assert normalize_law_name("   ") == ""


def test_no_shell_banner_is_a_no_op():
    assert normalize_law_name("Data Sharing Act 2025") == "Data Sharing Act 2025"


def _sg_citation(title: str, quote: str | None = None):
    from datetime import datetime, timezone

    from lexora.models.citation import Citation, DiscoveryTag, ReviewStatus

    quote = quote or "An organisation must not transfer any personal data to a country..."
    return Citation(
        economy="Singapore",
        title=title,
        law_number="Act 26 of 2012",
        last_amended="2020",
        indicator_id="P7-I1",
        article_path="Section 26",
        discovery_tag=DiscoveryTag.known,
        page_or_dom_anchor="p.40",
        quote=quote,
        mapping_rationale="Transfer limitation obligation.",
        source_url="https://sso.agc.gov.sg/Act/PDPA2012",
        confidence=0.9,
        notes="",
        clause_id="c1",
        retrieval_timestamp=datetime.now(timezone.utc),
        jurisdiction="SG",
        legal_form="statute",
        document_hash="sha256:deadbeef",
        char_start=0,
        char_end=len(quote),
        review_status=ReviewStatus.verified,
    )


def test_the_csv_exporter_emits_the_clean_name(tmp_path):
    """The exporter, not just the helper, must emit the clean name."""
    import csv as _csv

    from lexora.export.csv_exporter import to_csv

    out = tmp_path / "sub.csv"
    to_csv([_sg_citation(SSO_PDPA)], out)
    row = next(iter(_csv.DictReader(out.open(encoding="utf-8-sig"))))
    assert row["Law Name"] == "Personal Data Protection Act 2012"


def test_an_oversized_cell_is_reported_not_truncated(tmp_path, caplog):
    """A spreadsheet cell holds 32,767 characters and Excel truncates a longer one on
    paste without saying so. The official template is an xlsx, and 2 of the 669 round-1
    rows exceed it (48,078 and 42,165 characters, OAIC guidance chapters that parsed into
    one enormous "section") -- on the Verbatim Snippet, the column a reviewer checks by
    hand.

    The snippet is the clause's exact span and ships beside its char offsets, so cutting
    it here would break the correspondence that makes it checkable. Warn instead.
    """
    import csv as _csv
    import logging

    from lexora.export.csv_exporter import to_csv

    huge = "A provision that goes on. " * 1400  # ~36k characters
    out = tmp_path / "sub.csv"
    with caplog.at_level(logging.WARNING):
        to_csv([_sg_citation("Personal Data Protection Act 2012", quote=huge)], out)

    row = next(iter(_csv.DictReader(out.open(encoding="utf-8-sig"))))
    assert row["Verbatim Snippet"] == huge, "the data must reach the file intact"
    assert "spreadsheet limit" in caplog.text
    assert "Verbatim Snippet" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        to_csv([_sg_citation("Personal Data Protection Act 2012")], out)
    assert "spreadsheet limit" not in caplog.text


def test_the_audit_csv_keeps_the_raw_title(tmp_path):
    """Provenance is the audit trail's job: it records what the portal served."""
    import csv as _csv

    from lexora.export.csv_exporter import to_audit_csv

    out = tmp_path / "audit.csv"
    to_audit_csv([_sg_citation(SSO_PDPA)], out)
    row = next(iter(_csv.DictReader(out.open(encoding="utf-8-sig"))))
    assert row["title"] == SSO_PDPA
