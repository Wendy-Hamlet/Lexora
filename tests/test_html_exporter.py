"""Unit tests for the reviewer console (pure rendering, no network, no browser).

The page is a view of the CSV, so what these guard is that it stays one: the same rows,
the verbatim text unaltered, and nothing that would stop it opening on a laptop with no
network in the middle of a demo.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from lexora.export.html_exporter import to_html
from lexora.models.citation import Citation, DiscoveryTag


def _cite(**over) -> Citation:
    base = dict(
        economy="Malaysia",
        jurisdiction="MY",
        title="Personal Data Protection Act 2010",
        law_number="Act 709",
        indicator_id="P6-I4",
        article_path="S. 129(1)",
        discovery_tag=DiscoveryTag.known,
        page_or_dom_anchor="14",
        quote="A data user shall not transfer any personal data to a place outside Malaysia.",
        mapping_rationale="Default prohibition on outbound transfer maps to P6-I4.",
        source_url="https://lom.agc.gov.my/act-709.pdf",
        confidence=0.91,
        clause_id="c-1",
        retrieval_timestamp=datetime(2026, 7, 14, 23, 31, tzinfo=timezone.utc),
        legal_form="statute",
        char_start=100,
        char_end=176,
        document_hash="sha256:" + "a" * 64,
    )
    base.update(over)
    return Citation(**base)


def test_every_row_becomes_a_card(tmp_path):
    out = tmp_path / "v.html"
    n = to_html([_cite(), _cite(indicator_id="P7-I1")], out)
    assert n == 2
    assert out.read_text(encoding="utf-8").count('class="card"') == 2


def test_the_page_opens_with_no_network(tmp_path):
    """A demo laptop may have no route out; an external asset would render a blank page."""
    out = tmp_path / "v.html"
    to_html([_cite()], out)
    page = out.read_text(encoding="utf-8")
    assert not re.search(r'(src|href)\s*=\s*"https?://(?!lom\.agc)', page)
    assert "<style>" in page and "<script>" in page


def test_the_quote_survives_verbatim(tmp_path):
    """The audit trail is the point: the rendered quote must be the stored one."""
    quote = 'The Minister may <exempt> a class of "data users" & impose conditions.'
    out = tmp_path / "v.html"
    to_html([_cite(quote=quote)], out)
    page = out.read_text(encoding="utf-8")
    assert "&lt;exempt&gt;" in page and "&amp;" in page  # escaped, not dropped
    assert "<exempt>" not in page                        # and not injected as markup


def test_facets_come_from_the_data(tmp_path):
    out = tmp_path / "v.html"
    to_html([_cite(indicator_id="P6-I4"),
             _cite(indicator_id="P7-I1", economy="Singapore", jurisdiction="SG",
                   discovery_tag=DiscoveryTag.new)], out)
    page = out.read_text(encoding="utf-8")
    for value in ("P6-I4", "P7-I1", "NEW", "KNOWN", "Singapore", "Malaysia"):
        assert f'data-value="{value}"' in page or f'>{value}<' in page


def test_counts_are_reported_and_add_up(tmp_path):
    out = tmp_path / "v.html"
    to_html([_cite(discovery_tag=DiscoveryTag.new),
             _cite(discovery_tag=DiscoveryTag.new),
             _cite(discovery_tag=DiscoveryTag.known)], out)
    page = out.read_text(encoding="utf-8")
    assert "<b>3</b><span>provisions</span>" in page
    assert "<b>2</b><span>NEW</span>" in page
    assert "<b>1</b><span>KNOWN</span>" in page


def test_offsets_and_hash_are_shown(tmp_path):
    """What makes the quote checkable rather than merely quoted."""
    out = tmp_path / "v.html"
    to_html([_cite()], out)
    page = out.read_text(encoding="utf-8")
    assert "chars 100" in page and "176" in page
    assert "sha256:" in page


def test_an_empty_run_still_renders(tmp_path):
    out = tmp_path / "v.html"
    assert to_html([], out) == 0
    assert "No provision matches" in out.read_text(encoding="utf-8")


@pytest.mark.parametrize("missing", ["mapping_rationale", "notes", "last_amended",
                                     "page_or_dom_anchor", "law_number"])
def test_optional_fields_may_be_absent(tmp_path, missing):
    """Half the template columns are optional; a blank one must not break the page.

    ``source_url`` is deliberately not in this list: the model types it ``HttpUrl`` with
    no default, so a citation without a source cannot be constructed in the first place.
    """
    out = tmp_path / "v.html"
    to_html([_cite(**{missing: ""})], out)
    assert out.read_text(encoding="utf-8").count('class="card"') == 1
