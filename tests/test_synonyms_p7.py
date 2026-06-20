"""Section-targeted P7 synonyms for SG / MY (WS-5).

Pins the synonym packs that won SG P7-I1/P7-I4 and MY P7-I1 hit@1 on the flagship
PDPAs: the comprehensive-framework / DPO indicators must rank their operative
provision above a vocabulary-near decoy. Synthetic clauses (no cached PDF needed)
mirror the real section wording; the indicator's keyword pack comes from the LIVE
profile YAML, so a regression in the synonyms breaks this test.
"""
from __future__ import annotations

from pathlib import Path

from lexora.classify.retrieval import build_index, retrieve_candidates
from lexora.collect.profile_loader import load_profile
from lexora.indicators import load_indicators
from lexora.models.clause import CanonicalSpan, Clause

REPO = Path(__file__).resolve().parent.parent
JURIS = REPO / "configs" / "jurisdictions"
INDICATORS = REPO / "configs" / "rdtii_indicators.yaml"


def _clause(section: str, text: str) -> Clause:
    cid = f"doc::s{section}"
    return Clause(clause_id=cid, document_id="doc", structural_path=f"Section {section}",
                  section_number=section,
                  span=CanonicalSpan(span_id=cid + ".s", document_id="doc",
                                     char_start=0, char_end=len(text), text=text))


def _ind(indicators, submission_id):
    return next(i for i in indicators if i.submission_id == submission_id)


def _top_section(profile, indicator, clauses) -> str:
    index = build_index(clauses)
    hits = retrieve_candidates(indicator, profile, index, top_k=1, use_semantic=False)
    cid = hits[0].clause_id
    return next(c for c in clauses if c.clause_id == cid).section_number


# Section wording mirrors the SG PDPA operative provisions (paraphrased, not copied
# verbatim — this is a retrieval test, not a citation).
_SG_S13 = ("An organisation must not collect, use or disclose personal data about an "
           "individual unless the individual gives consent under this Act.")
_SG_S24 = ("An organisation must protect personal data in its possession by making "
           "reasonable security arrangements to prevent unauthorised access.")
_SG_S11 = ("An organisation must designate one or more individuals to be responsible for "
           "ensuring that the organisation complies with this Act.")
_DECOY = "Miscellaneous provisions about fees, forms and the service of notices."


def test_sg_p7_i1_framework_synonyms_rank_a_gold_section_first():
    profile = load_profile(JURIS / "sg.yaml")
    indicators = load_indicators(INDICATORS)
    clauses = [_clause("13", _SG_S13), _clause("24", _SG_S24), _clause("11", _SG_S11),
               _clause("99", _DECOY)]
    assert _top_section(profile, _ind(indicators, "P7-I1"), clauses) in {"13", "24"}


def test_sg_p7_i4_dpo_synonyms_rank_designation_section_first():
    profile = load_profile(JURIS / "sg.yaml")
    indicators = load_indicators(INDICATORS)
    clauses = [_clause("13", _SG_S13), _clause("24", _SG_S24), _clause("11", _SG_S11),
               _clause("99", _DECOY)]
    assert _top_section(profile, _ind(indicators, "P7-I4"), clauses) == "11"


# MY PDPA s.5 (Principles) / s.6 (General Principle consent) wording.
_MY_S5 = ("The personal data protection principles are the General Principle, the Notice "
          "and Choice Principle, the Disclosure Principle, the Security Principle and the "
          "Retention, Data Integrity and Access Principles.")
_MY_S6 = ("A data user shall not process personal data about a data subject unless the data "
          "subject has given his consent to the processing of the personal data.")
_MY_DECOY = "The Commissioner may appoint officers and authorise the payment of fees."


def test_my_p7_i1_principles_synonyms_rank_a_gold_section_first():
    profile = load_profile(JURIS / "my.yaml")
    indicators = load_indicators(INDICATORS)
    clauses = [_clause("5", _MY_S5), _clause("6", _MY_S6), _clause("101", _MY_DECOY)]
    assert _top_section(profile, _ind(indicators, "P7-I1"), clauses) in {"5", "6"}
