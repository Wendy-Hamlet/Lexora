"""Malaysia's statute-book feeds — the discovery route that does not go through search.

Fixtures are trimmed verbatim from the live feeds on 2026-08-01, including the two
traps that the first version of this module walked into.
"""
from __future__ import annotations

import json

import httpx
import pytest

from lexora.collect import my_inventory
from lexora.collect.my_inventory import fetch_inventory, reset_cache

PRINCIPAL = {"recordsTotal": 2, "records": [
    {"ACTNO_LEGISLATION": "709",
     "LEGISLATIONTITLEBI": '<a href="act-detail.php?act=709&lang=BI">PERSONAL DATA '
                           'PROTECTION ACT 2010</a>',
     "COMMENCEMENTREMARKBI": "15-11-2013 [P.U.(B) 466/2013]",
     "DOC2DOWNLOADBI": '<a href="../../../ilims/upload/portal/akta/Act 709 ori.pdf">pdf</a>'},
    {"ACTNO_LEGISLATION": "884",
     "LEGISLATIONTITLEBI": "JOHOR BAHRU-SINGAPORE RAPID TRANSIT SYSTEM LINK ACT 2026",
     "COMMENCEMENTREMARKBI": "NOT YET IN FORCE", "DOC2DOWNLOADBI": ""},
]}

# Act A1727 is a gold instrument and it carries BOTH traps: the field named BI holds the
# MALAY title, and the direct download anchor is empty while the file is reachable only
# through the `generatepdf` blob.
AMENDMENT = {"recordsTotal": 1, "records": [
    {"ACTNO_LEGISLATION": "A1727",
     "TajukBI": "AKTA PERLINDUNGAN DATA PERIBADI (PINDAAN) 2024",
     "LEGISLATIONTITLEBI": "PERSONAL DATA PROTECTION (AMENDMENT) ACT 2024",
     "COMMENCEMENTREMARKBI": "24/12/2024 [P.U. (B) 522/2024]",
     "DOC2DOWNLOADBI": "",
     "DOC2DOWNLOADBIgeneratepdf": json.dumps(
         {"icon": "pdf-en-printed.png",
          "path": "/upload/portal/akta/outputaktap/2430673_BI/",
          "docName": "Act A1727.pdf"})},
]}

REPEALED = {"recordsTotal": 1, "records": [
    {"ILA_ACT_NO": "762", "TITLEBI": "GOODS AND SERVICES TAX ACT 2014",
     "REPEALEDBY": "805", "REPEALTITLEBI": "GOODS AND SERVICES (REPEAL) ACT 2018"},
]}

BY_FEED = {
    "json-principal-2024.php": PRINCIPAL,
    "json-updated-2024.php": {"recordsTotal": 0, "records": []},
    "json-amendment-2024.php": AMENDMENT,
    "json-repealed-2024.php": REPEALED,
}


def _client(missing: set[str] | None = None) -> httpx.Client:
    missing = missing or set()

    def handler(request: httpx.Request) -> httpx.Response:
        feed = request.url.path.lstrip("/")
        if feed in missing:
            return httpx.Response(500, text="Internal Server Error")
        return httpx.Response(200, json=BY_FEED[feed])

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


@pytest.fixture(autouse=True)
def _fresh():
    reset_cache()
    yield
    reset_cache()


def test_the_inventory_names_and_locates_every_act():
    with _client() as client:
        inv = fetch_inventory(client)
    assert set(inv) == {"709", "884", "A1727", "762"}
    pdpa = inv["709"]
    assert pdpa.title_en == "PERSONAL DATA PROTECTION ACT 2010"
    # page-relative href resolved against the portal, not left as "../../../ilims/..."
    assert pdpa.pdf_url == "https://lom.agc.gov.my/ilims/upload/portal/akta/Act%20709%20ori.pdf"
    assert pdpa.in_force is True


def test_a_field_named_bi_is_not_evidence_that_it_holds_english():
    """The amendment feed's TajukBI is the MALAY title. The first version of this module
    read that field and then kept "the longest title", which is how the Personal Data
    Protection (Amendment) Act 2024 came out named in Malay -- its Malay title is one
    character longer than its English one."""
    with _client() as client:
        inv = fetch_inventory(client)
    assert inv["A1727"].title_en == "PERSONAL DATA PROTECTION (AMENDMENT) ACT 2024"


def test_a_gold_act_whose_download_anchor_is_empty_is_still_reachable():
    with _client() as client:
        inv = fetch_inventory(client)
    assert inv["A1727"].pdf_url == (
        "https://lom.agc.gov.my/ilims/upload/portal/akta/outputaktap/2430673_BI/Act A1727.pdf")


def test_the_portal_states_the_repeal_chain_itself():
    with _client() as client:
        inv = fetch_inventory(client)
    gst = inv["762"]
    assert gst.repealed is True
    assert (gst.repealed_by, gst.repealed_by_title) == (
        "805", "GOODS AND SERVICES (REPEAL) ACT 2018")
    assert inv["709"].repealed is False


def test_not_yet_in_force_is_carried_through():
    with _client() as client:
        inv = fetch_inventory(client)
    assert inv["884"].in_force is False
    assert inv["A1727"].in_force is True


def test_one_dead_feed_does_not_cost_the_whole_inventory():
    """Three quarters of a statute book beats an exception: `reprint` answered 500 during
    the survey, and a route that collapses when one listing is down is not a fallback."""
    with _client(missing={"json-amendment-2024.php"}) as client:
        inv = fetch_inventory(client)
    assert "A1727" not in inv
    assert inv["709"].title_en == "PERSONAL DATA PROTECTION ACT 2010"
    assert inv["762"].repealed is True


def test_the_inventory_is_read_once_per_process():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=BY_FEED[request.url.path.lstrip("/")])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetch_inventory(client)
        fetch_inventory(client)
    assert calls["n"] == len(my_inventory._FEEDS) + 1  # feeds + repealed, fetched once
