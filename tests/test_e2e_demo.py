"""End-to-end smoke test for the Slice 0 pipeline.

Synthesizes a tiny PDPA-shaped PDF at runtime, runs the full pipeline against
it, and asserts the verbatim contract held: every emitted quote is a
byte-for-byte slice of the canonical span text, and review_status is VERIFIED.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # PyMuPDF
import pytest

from lexora.collect.profile_loader import load_profile
from lexora.export.csv_exporter import SUBMISSION_COLUMNS, to_csv
from lexora.export.jsonld_exporter import to_jsonld
from lexora.indicators import load_indicators
from lexora.models.citation import ReviewStatus
from lexora.pipeline import run_demo_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent


SECTION_26_BODY = (
    "An organisation must not transfer any personal data to a country or "
    "territory outside Singapore except in accordance with requirements "
    "prescribed under this Act to ensure that organisations provide a standard "
    "of protection to personal data so transferred that is comparable to the "
    "protection under this Act."
)
SECTION_13_BODY = (
    "An organisation must not, on or after the appointed day, collect, use or "
    "disclose personal data about an individual unless the individual gives, "
    "or is deemed to have given, his or her consent under this Act to the "
    "collection, use or disclosure, as the case may be."
)


def _make_pdf(path: Path) -> None:
    """Build a tiny one-page PDF whose text layer survives full extraction.

    `insert_textbox` wraps inside the rect, unlike `insert_text` which would
    silently clip at the page edge and truncate each Section body.
    """
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(72, 72, 540, 770)
    text = (
        "Personal Data Protection Act 2012 (Excerpt for testing)\n\n"
        f"13.  {SECTION_13_BODY}\n\n"
        f"26.  {SECTION_26_BODY}\n"
    )
    page.insert_textbox(rect, text, fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()


@pytest.fixture
def synthetic_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "pdpa_excerpt.pdf"
    _make_pdf(pdf_path)
    return pdf_path


def test_demo_pipeline_end_to_end(tmp_path: Path, synthetic_pdf: Path) -> None:
    profile = load_profile(REPO_ROOT / "configs" / "jurisdictions" / "sg.yaml")
    indicators = load_indicators(REPO_ROOT / "configs" / "rdtii_indicators.yaml")

    artifacts = run_demo_pipeline(
        pdf_path=synthetic_pdf,
        profile=profile,
        indicators=indicators,
        source_url="https://sso.agc.gov.sg/Act/PDPA2012",
        portal_name="Singapore Statutes Online",
        title="PDPA 2012 (test excerpt)",
        dest_dir=tmp_path / "raw",
        top_k=1,
        # BM25 scores are unbounded and can be negative when the corpus is
        # tiny (only 2 clauses here); take the top-1 regardless of sign.
        min_score=-10.0,
    )

    assert len(artifacts.pages) == 1
    assert len(artifacts.clauses) >= 2, "expected at least Section 13 and Section 26"
    assert len(artifacts.citations) >= 1

    sections = {c.section_number for c in artifacts.clauses}
    assert {"13", "26"}.issubset(sections)

    for citation in artifacts.citations:
        clause = next(c for c in artifacts.clauses if c.clause_id == citation.clause_id)
        assert citation.quote == clause.span.text, "quote must be a verbatim copy"
        assert citation.review_status is ReviewStatus.verified
        assert citation.document_hash.startswith("sha256:")
        assert citation.char_end > citation.char_start
        # indicator_id must be the official submission code, e.g. "P6-I4"
        assert re.fullmatch(r"P[67]-I\d", citation.indicator_id), citation.indicator_id
        assert citation.economy == "Singapore"

    # The cross-border provision (Section 26) must surface for some P6 indicator.
    # NB: the verbatim snippet keeps the PDF's line wrap ("outside\nSingapore"),
    # so match a phrase that does not straddle the wrap.
    cross_border = [c for c in artifacts.citations if "transfer any personal data" in c.quote]
    assert cross_border, "the Section 26 cross-border provision should be cited"
    assert any(c.indicator_id.startswith("P6") for c in cross_border)

    # JSON-LD export round-trip
    out_path = tmp_path / "out.jsonld"
    n = to_jsonld(artifacts.citations, out_path)
    assert n == len(artifacts.citations)
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == n
    first = json.loads(lines[0])
    assert first["@context"]["indicator_id"] == "rdtii:indicator"
    assert first["quote"]  # non-empty

    # Submission CSV must match the official template header exactly.
    csv_path = tmp_path / "out.csv"
    m = to_csv(artifacts.citations, csv_path)
    assert m == len(artifacts.citations)
    header = csv_path.read_text(encoding="utf-8-sig").splitlines()[0]
    assert header.split(",") == [label for label, _ in SUBMISSION_COLUMNS]


def test_submission_columns_match_official_template():
    """Guards the exact OUTPUT_TEMPLATE_31MAY.xlsx column names and order."""
    expected = [
        "Economy", "Law Name", "Law Number / Ref", "Last Amended",
        "Indicator ID", "Article / Section", "Discovery Tag",
        "Location Reference", "Verbatim Snippet", "Mapping Rationale",
        "Source URL", "Confidence", "Notes",
    ]
    assert [label for label, _ in SUBMISSION_COLUMNS] == expected
