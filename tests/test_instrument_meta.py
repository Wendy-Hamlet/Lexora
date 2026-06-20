"""Instrument-metadata backfill: Last Amended / Law Number resolution.

The fetched document carries neither the amendment year nor the official law
number, so the pipeline backfills them from curated, source-verified per-instrument
metadata in the jurisdiction YAMLs. These pin the resolver's matching contract and
that the real configs parse + resolve.
"""
from __future__ import annotations

from pathlib import Path

from lexora.collect.profile_loader import load_profile
from lexora.models.source import (
    InstrumentMeta,
    LegalSystem,
    SourceProfile,
)
from lexora.pipeline import _resolve_instrument_meta

_JURIS = Path(__file__).resolve().parent.parent / "configs" / "jurisdictions"


def _profile() -> SourceProfile:
    return SourceProfile(
        jurisdiction="Malaysia", iso_code="MY", primary_language="en",
        legal_system=LegalSystem.common,
        known_instrument_ids={"709": "Personal Data Protection Act 2010"},
        instrument_metadata={
            "Personal Data Protection Act 2010": InstrumentMeta(last_amended="2024", law_number="Act 709"),
            "Computer Crimes Act 1997": InstrumentMeta(last_amended="2006", law_number="Act 563"),
        },
    )


def test_exact_title_resolves():
    assert _resolve_instrument_meta("Personal Data Protection Act 2010", _profile()) == ("2024", "Act 709")


def test_fuzzy_title_with_extra_tokens_resolves():
    # discovery titles often carry the Act number / casing noise
    assert _resolve_instrument_meta("Personal Data Protection Act (Act 709) 2010", _profile()) == ("2024", "Act 709")


def test_unknown_title_returns_blank():
    assert _resolve_instrument_meta("Some Unrelated Regulation 2020", _profile()) == ("", "")


def test_blank_title_or_no_metadata_returns_blank():
    assert _resolve_instrument_meta(None, _profile()) == ("", "")
    bare = SourceProfile(jurisdiction="X", iso_code="XX", primary_language="en",
                         legal_system=LegalSystem.common)
    assert _resolve_instrument_meta("Personal Data Protection Act 2010", bare) == ("", "")


def test_filename_title_falls_back_to_act_number():
    # MY Fess returns filenames; the canonical name does not fuzzy-match, but the
    # known Act number embedded in the title does.
    assert _resolve_instrument_meta("akta709_BI_2010_reprint.pdf", _profile()) == ("2024", "Act 709")


def test_document_metadata_preferred_over_curated_anchor():
    """The structured value captured from the portal wins; the curated anchor only
    fills a field the connector could not supply."""
    from datetime import datetime, timezone

    from lexora.indicators import load_indicators
    from lexora.models.clause import CanonicalSpan, Clause
    from lexora.models.source import RawDocument, SourceType
    from lexora.pipeline import _citations_from_clauses

    profile = load_profile(_JURIS / "au.yaml")  # anchor: Privacy Act -> 2024 / No. 119 of 1988
    inds = [i for i in load_indicators(_JURIS.parent / "rdtii_indicators.yaml") if i.rdtii_id == "6.4"]
    text = ("Before an APP entity discloses personal information about an individual to an "
            "overseas recipient, the entity must take such steps as are reasonable.")
    clause = Clause(clause_id="d::app8", document_id="d", structural_path="APP 8",
                    span=CanonicalSpan(span_id="d::app8.s", document_id="d",
                                       char_start=0, char_end=len(text), text=text))
    doc = RawDocument(
        document_id="d", source_url="https://www.legislation.gov.au/C2004A03712/latest",
        retrieval_timestamp=datetime.now(timezone.utc), http_status=200, sha256="sha256:x",
        content_type="application/pdf", bytes_path="/tmp/x", portal_name="AU Register",
        jurisdiction="AU", source_type=SourceType.primary, title="Privacy Act 1988",
        law_number="No. 119 of 1988", last_amended="",  # connector gave number, not date
    )
    cites = _citations_from_clauses([clause], doc, profile, inds, "statute", top_k=1, min_score=0.0)
    assert cites, "expected a citation"
    c = cites[0]
    assert c.law_number == "No. 119 of 1988"   # from the document (connector)
    assert c.last_amended == "2024"            # gap filled from the curated anchor


def test_real_configs_parse_and_resolve():
    expected = {
        "sg": ("Personal Data Protection Act 2012", ("2021", "")),
        "au": ("Privacy Act 1988", ("2024", "No. 119 of 1988")),
        "my": ("Personal Data Protection Act 2010", ("2024", "Act 709")),
    }
    for iso, (title, want) in expected.items():
        profile = load_profile(_JURIS / f"{iso}.yaml")
        assert profile.instrument_metadata, f"{iso}.yaml has no instrument_metadata"
        assert _resolve_instrument_meta(title, profile) == want
