"""Law-name normalisation — the submission's second column."""
from __future__ import annotations

import pytest

from lexora.export.law_name import normalize_law_name

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


def _sg_citation(title: str):
    from datetime import datetime, timezone

    from lexora.models.citation import Citation, DiscoveryTag, ReviewStatus

    quote = "An organisation must not transfer any personal data to a country..."
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


def test_the_audit_csv_keeps_the_raw_title(tmp_path):
    """Provenance is the audit trail's job: it records what the portal served."""
    import csv as _csv

    from lexora.export.csv_exporter import to_audit_csv

    out = tmp_path / "audit.csv"
    to_audit_csv([_sg_citation(SSO_PDPA)], out)
    row = next(iter(_csv.DictReader(out.open(encoding="utf-8-sig"))))
    assert row["title"] == SSO_PDPA
