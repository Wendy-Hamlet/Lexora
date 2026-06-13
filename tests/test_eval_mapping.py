"""Mapping-quality eval harness (P-4) — offline core.

Loads ``scripts/eval_mapping.py`` (evals live in scripts/, outside the package)
and pins its pure scoring on a synthetic document: section-level hit@1 / hit@3,
the gold loader, and the A/B contract that the dense channel can pull the
on-point section above a vocabulary-overlap decoy that BM25 ranks first.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np

from lexora.models.clause import CanonicalSpan, Clause
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import LegalSystem, PortalSpec, SourceProfile, SourceType

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_mapping.py"
_spec = importlib.util.spec_from_file_location("eval_mapping", _SCRIPT)
em = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(em)

_W = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Hashed bag-of-words; cosine ~= token overlap (mirrors test_semantic)."""

    DIM = 512

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for r, text in enumerate(texts):
            for tok in _W.findall(text.lower()):
                out[r, hash(tok) % self.DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def _clause(section: str, text: str) -> Clause:
    cid = f"doc::s{section}"
    return Clause(
        clause_id=cid, document_id="doc", structural_path=f"Section {section}",
        section_number=section,
        span=CanonicalSpan(span_id=cid + ".s", document_id="doc", char_start=0,
                           char_end=len(text), text=text),
    )


def _app_clause(app: str, item: str | None, text: str) -> Clause:
    """An Australian Privacy Principle clause as the Schedule-aware parser emits it
    (structural_path carries 'Australian Privacy Principle N', section_number=N)."""
    suffix = f"-{item}" if item else ""
    cid = f"doc::sch1-app{app}{suffix}"
    path = f"Schedule 1 > Australian Privacy Principle {app}" + (f".{item}" if item else "")
    return Clause(
        clause_id=cid, document_id="doc", structural_path=path,
        section_number=app, paragraph_number=item,
        span=CanonicalSpan(span_id=cid + ".s", document_id="doc", char_start=0,
                           char_end=len(text), text=text),
    )


def _profile() -> SourceProfile:
    return SourceProfile(
        jurisdiction="Singapore", iso_code="SG", primary_language="en",
        legal_system=LegalSystem.common,
        portals=[PortalSpec(name="P", url="https://e/", source_type=SourceType.primary)],
    )


def _ind(sub: str, desc: str, keywords=None) -> RDTIIIndicator:
    return RDTIIIndicator(
        rdtii_id="6.4" if sub == "P6-I4" else "7.1", submission_id=sub, pillar=6,
        name=sub, description=desc, keywords=keywords or [],
    )


def test_load_gold_parses_sections_document_aware(tmp_path):
    csv_path = tmp_path / "g.csv"
    csv_path.write_text(
        "iso,document,indicator,gold_sections,note\n"
        "SG,PDPA 2012,P6-I4,26,x\n"
        "SG,PDPA 2012,P7-I1,13;24,y\n"
        "MY,PDPA 2010,P6-I4,129,z\n",
        encoding="utf-8",
    )
    gold = em.load_gold(csv_path)
    assert gold["sg"]["PDPA 2012"]["P6-I4"] == {"26"}
    assert gold["sg"]["PDPA 2012"]["P7-I1"] == {"13", "24"}
    assert gold["my"]["PDPA 2010"]["P6-I4"] == {"129"}


def test_clause_key_distinguishes_app_from_main_body_section():
    # APP 8 (Schedule 1) and main-body s.8 both carry section_number "8"; only the
    # APP clause keys to "APP8".
    assert em._clause_key(_app_clause("8", None, "cross-border disclosure")) == "APP8"
    assert em._clause_key(_app_clause("8", "1", "before disclosing overseas")) == "APP8"
    assert em._clause_key(_clause("8", "this Act binds the Crown")) == "8"


def test_evaluate_matches_app_gold_not_the_namesake_section():
    # Gold "APP8" must be satisfied by the cross-border principle, never by the
    # unrelated main-body section 8 that shares the number. (>=3 clauses so BM25's
    # IDF is non-degenerate, as in the other eval tests.)
    indicators = [_ind("P6-I4", "cross-border disclosure of personal information to an "
                       "overseas recipient outside the country",
                       keywords=["overseas", "disclose", "cross"])]
    gold = {"P6-I4": {"APP8"}}
    decoy = _clause("99", "miscellaneous provisions about fees and forms")

    hit = em.evaluate(
        [_clause("8", "this Act binds the Crown in right of the Commonwealth"),
         _app_clause("8", "1", "before an APP entity discloses personal information to an "
                     "overseas recipient who is not in Australia it must take reasonable steps"),
         decoy],
        _profile(), indicators, gold, use_semantic=False,
    )
    assert hit[0]["hit1"] is True and "APP8" in hit[0]["retrieved"]

    # Only the namesake main-body s.8 present -> no APP8 hit (no false positive).
    miss = em.evaluate(
        [_clause("8", "this Act binds the Crown in right of the Commonwealth and the States"),
         _clause("14", "the Australian Privacy Principles are set out in Schedule 1"),
         decoy],
        _profile(), indicators, gold, use_semantic=False,
    )
    assert miss[0]["hit1"] is False and miss[0]["hit3"] is False


def test_section_targeted_synonyms_lift_app8_over_app7():
    # The lever that won P6-I4 hit@1 live: APP 8's operative wording ("overseas
    # recipient", "cross-border disclosure") — section-targeted synonyms in the AU
    # profile — must put APP 8 at BM25 rank-1, above the vocabulary-near but
    # unrelated APP 7 (direct marketing) that previously outranked it.
    clauses = [
        _app_clause("7", "1", "an APP entity must not use or disclose personal information "
                    "for the purpose of direct marketing unless an exception applies"),
        _app_clause("8", "1", "before an APP entity discloses personal information to an "
                    "overseas recipient the entity must take reasonable steps to ensure the "
                    "overseas recipient does not breach the principles"),
        _clause("99", "miscellaneous provisions about fees and forms"),
    ]
    ind = _ind("P6-I4", "conditions on cross-border transfer of personal data",
               keywords=["cross-border disclosure of personal information",
                         "overseas recipient",
                         "disclose personal information to an overseas recipient"])
    rows = em.evaluate(clauses, _profile(), [ind], {"P6-I4": {"APP8"}}, use_semantic=False)
    assert rows[0]["retrieved"][0] == "APP8"  # APP 8 wins rank-1, not APP 7
    assert rows[0]["hit1"] is True


def test_select_gold_resolves_by_substring_and_singleton():
    by_doc = {"Personal Data Protection Act 2012": {"P6-I4": {"26"}}}
    # singleton: no --doc needed
    name, block = em.select_gold(by_doc, None)
    assert name.startswith("Personal Data") and block == {"P6-I4": {"26"}}
    # substring select among several
    multi = {"Privacy Act 1988": {"P6-I4": {"16C"}}, "SOCI Act 2018": {"P7-I2": {"30"}}}
    assert em.select_gold(multi, "privacy")[0] == "Privacy Act 1988"
    # ambiguous / missing -> KeyError
    import pytest
    with pytest.raises(KeyError):
        em.select_gold(multi, None)
    with pytest.raises(KeyError):
        em.select_gold(multi, "act")


def test_evaluate_scores_hit_at_1_and_3():
    clauses = [
        _clause("26", "transfer of personal data to a country outside Singapore"),
        _clause("13", "an organisation must not collect personal data without consent"),
        _clause("99", "miscellaneous provisions about fees and forms"),
    ]
    indicators = [
        _ind("P6-I4", "cross-border transfer of personal data outside the country",
             keywords=["transfer", "outside"]),
    ]
    gold = {"P6-I4": {"26"}}
    rows = em.evaluate(clauses, _profile(), indicators, gold, use_semantic=False)
    assert len(rows) == 1
    assert rows[0]["hit1"] and rows[0]["hit3"]
    assert em.summarize(rows) == (1, 1, 1)


def test_evaluate_records_miss_when_wrong_section_ranks_first():
    # No "transfer" clause matches; the indicator should miss its gold section.
    clauses = [
        _clause("24", "an organisation must protect personal data in its possession"),
        _clause("99", "miscellaneous provisions about fees and forms"),
    ]
    indicators = [_ind("P6-I4", "cross-border transfer outside the country",
                       keywords=["transfer", "outside"])]
    rows = em.evaluate(clauses, _profile(), indicators, {"P6-I4": {"26"}}, use_semantic=False)
    assert rows[0]["hit1"] is False and rows[0]["hit3"] is False
    assert em.summarize(rows) == (0, 0, 1)


def test_evaluate_rank_records_gold_section_rank():
    # G-6.1: rank report finds the gold section's position even when it is not
    # rank 1, separating a ranking problem (gold at #2) from a recall miss (None).
    clauses = [
        _clause("24", "an organisation must protect personal data in its possession"),
        _clause("26", "transfer of personal data to a country outside Singapore"),
        _clause("99", "miscellaneous provisions about fees and forms"),
    ]
    indicators = [_ind("P6-I4", "protect personal data; transfer outside the country",
                       keywords=["protect", "transfer", "outside"])]
    rows = em.evaluate_rank(clauses, _profile(), indicators, {"P6-I4": {"26"}},
                            rank_k=10, use_semantic=False)
    assert rows[0]["rank"] is not None and rows[0]["rank"] >= 1
    assert "26" in rows[0]["retrieved"]


def test_evaluate_rank_misses_when_gold_absent():
    clauses = [_clause("24", "protect personal data"), _clause("99", "fees and forms")]
    indicators = [_ind("P6-I4", "cross-border transfer outside the country",
                       keywords=["transfer", "outside"])]
    rows = em.evaluate_rank(clauses, _profile(), indicators, {"P6-I4": {"26"}},
                            rank_k=10, use_semantic=False)
    assert rows[0]["rank"] is None


def test_summarize_rank_mrr_and_recall():
    rows = [{"rank": 1}, {"rank": 3}, {"rank": None}]
    s = em.summarize_rank(rows, ks=(1, 3, 5))
    assert s["n"] == 3
    assert abs(s["mrr"] - (1.0 + 1 / 3) / 3) < 1e-9
    assert s["recall"] == {1: 1, 3: 2, 5: 2}


def test_evaluate_threads_dense_channel_and_still_hits_gold():
    # The A/B path: passing an embedder routes retrieval through BM25+dense fusion
    # (the run the live script labels "fused"). The harness must thread it through
    # and still resolve the on-point section, so the two runs are comparable.
    clauses = [
        _clause("26", "personal data may be sent to a place outside the country only "
                      "where comparable protection for the individual is ensured"),
        _clause("99", "miscellaneous provisions about fees and forms"),
    ]
    profile = _profile()
    indicators = [_ind("P6-I4",
                       "conditions on cross-border transfer of personal data to a place "
                       "outside the country with comparable protection",
                       keywords=["transfer", "outside"])]
    gold = {"P6-I4": {"26"}}

    bm25 = em.evaluate(clauses, profile, indicators, gold, use_semantic=False)
    fused = em.evaluate(clauses, profile, indicators, gold,
                        use_semantic=True, embedder=FakeEmbedder())
    assert bm25[0]["hit1"] is True
    assert fused[0]["hit1"] is True  # dense channel threaded through, gold still found
