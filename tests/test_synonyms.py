"""G-3a LLM cross-lingual synonym generation — offline, with a fake client.

Pins the contract (query terms only, bounded, deduped, fail-safe) and the
*mechanism*: injecting LLM-drafted Chinese terms into a profile's query side lifts
BM25 coverage of a Chinese clause from zero to a hit. The live A/B against the
hand-tuned synonyms runs separately once an endpoint is configured.
"""
from __future__ import annotations

from lexora.classify.retrieval import _expand_query, build_index
from lexora.classify.synonyms import SynonymGenerator, _clean_terms, make_synonym_generator
from lexora.models.clause import CanonicalSpan, Clause
from lexora.models.indicator import RDTIIIndicator
from lexora.models.source import LegalSystem, SourceProfile


class _FakeClient:
    """Returns a canned JSON payload and records the prompt it was given."""

    def __init__(self, payload):
        self._payload = payload
        self.last_user = None

    def chat(self, system, user, json_schema=None):
        self.last_user = user
        return self._payload


class _BoomClient:
    def chat(self, system, user, json_schema=None):
        raise RuntimeError("endpoint down")


def _indicator() -> RDTIIIndicator:
    return RDTIIIndicator(
        rdtii_id="6.4",
        submission_id="P6-I4",
        pillar=6,
        name="Cross-border data transfer",
        description="Conditions on transferring personal data to another country.",
        keywords=["cross-border", "overseas recipient"],
    )


def _clause(cid: str, text: str) -> Clause:
    return Clause(
        clause_id=cid,
        document_id="d",
        structural_path=cid,
        span=CanonicalSpan(
            span_id=cid, document_id="d", char_start=0, char_end=len(text), text=text
        ),
    )


def test_clean_terms_dedupes_trims_and_bounds():
    raw = ["  跨境提供个人信息 ", "跨境提供个人信息", "", 123, "x" * 200, "·overseas recipient·"]
    out = _clean_terms(raw)
    assert "跨境提供个人信息" in out
    assert out.count("跨境提供个人信息") == 1  # deduped
    assert "overseas recipient" in out  # bullet/punct stripped
    assert all(0 < len(t) <= 80 for t in out)  # over-long dropped


def test_generate_returns_terms_and_sends_language_in_prompt():
    client = _FakeClient({"terms": ["向境外提供个人信息", "跨境提供", "标准合同"]})
    gen = SynonymGenerator(client)
    terms = gen.generate(_indicator(), "zh")
    assert terms == ["向境外提供个人信息", "跨境提供", "标准合同"]
    assert "Target language: zh" in client.last_user
    assert "P6-I4" in client.last_user


def test_generate_degrades_to_empty_on_backend_error():
    gen = SynonymGenerator(_BoomClient())
    assert gen.generate(_indicator(), "zh") == []
    assert gen.error_count == 1
    assert gen.last_error_type == "RuntimeError"


def test_make_synonym_generator_off_by_default():
    assert make_synonym_generator(use_llm=False) is None


def test_generated_terms_are_query_only_never_clause_text():
    # The generator returns a list of strings; it has no path that emits a Clause
    # or a span. Guard that the public surface stays query-side.
    gen = SynonymGenerator(_FakeClient({"terms": ["跨境提供"]}))
    out = gen.generate(_indicator(), "zh")
    assert all(isinstance(t, str) for t in out)


def test_injected_chinese_synonyms_lift_bm25_coverage_from_zero():
    # Before G-3a the English query shares no tokens with a Chinese clause, so BM25
    # coverage is 0. Injecting the LLM-drafted Chinese terms into the profile's
    # query side (where hand-tuned synonyms already live) makes the right clause
    # rank first — the whole point of G-3a.
    clauses = [
        _clause("c1", "本法所称个人信息是指可识别自然人的各种信息"),
        _clause("c2", "确需向境外提供个人信息的应当通过安全评估或者订立标准合同"),
        _clause("c3", "应当采取加密去标识化等安全技术措施防止未经授权的访问"),
    ]
    index = build_index(clauses)
    ind = _indicator()

    bare = SourceProfile(
        jurisdiction="probe", iso_code="zz", primary_language="zh",
        legal_system=LegalSystem.civil,
    )
    before = index.query(_expand_query(ind, bare, "zh"), top_k=3)
    assert all(h.score <= 0 for h in before)  # English query: no lexical match

    # Drafted by the LLM (here canned), then injected after human review.
    drafted = SynonymGenerator(
        _FakeClient({"terms": ["向境外提供个人信息", "跨境提供", "标准合同"]})
    ).generate(ind, "zh")
    tuned = bare.model_copy(update={"keywords_by_indicator": {ind.id: {"zh": drafted}}})

    after = index.query(_expand_query(ind, tuned, "zh"), top_k=3)
    assert after[0].clause_id == "c2"
    assert after[0].score > 0  # a real cross-lingual lexical hit now
