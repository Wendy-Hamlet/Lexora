"""G-1a Unicode-aware tokenizer: CJK/Thai bigrams without regressing Latin.

The non-Latin "hard zero" (scripts/eval_intrinsic.py) came from an ASCII-only
tokenizer that turned every CJK/Thai clause into an empty BM25 document. G-1a
adds character bigrams for boundary-free scripts. These tests pin two contracts:
(1) Latin text tokenizes EXACTLY as before — round-1 English retrieval must not
move; (2) a Chinese query now retrieves the right Chinese clause, where before
the index was empty.
"""
from __future__ import annotations

import re

from lexora.classify.retrieval import _tokenize, build_index
from lexora.models.clause import CanonicalSpan, Clause

# The pre-G-1a tokenizer, inlined, to prove Latin output is byte-for-byte stable.
_OLD = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", re.UNICODE)


def _old_tokenize(text: str) -> list[str]:
    return [t.lower() for t in _OLD.findall(text)]


def _clause(cid: str, text: str) -> Clause:
    return Clause(
        clause_id=cid,
        document_id="d",
        structural_path=cid,
        span=CanonicalSpan(
            span_id=cid, document_id="d", char_start=0, char_end=len(text), text=text
        ),
    )


def test_latin_tokenization_is_unchanged():
    for sample in [
        "An organisation must not transfer personal data overseas.",
        "Section 26A: cross-border data flows, retention 7-years, e-mail.",
        "BM25 + dense_fusion RRF, top-k=5.",
        "",
    ]:
        assert _tokenize(sample) == _old_tokenize(sample)


def test_cjk_run_becomes_overlapping_bigrams():
    assert _tokenize("个人信息") == ["个人", "人信", "信息"]


def test_single_cjk_char_falls_back_to_unigram():
    assert _tokenize("法") == ["法"]


def test_mixed_latin_and_cjk_keeps_both_channels():
    toks = _tokenize("APP8 跨境传输 rule")
    assert "app8" in toks
    assert "rule" in toks
    assert "跨境" in toks and "境传" in toks and "传输" in toks


def test_thai_run_is_bigram_tokenized():
    toks = _tokenize("ข้อมูลส่วนบุคคล")
    assert len(toks) >= 2
    assert all(len(t) == 2 for t in toks)


def test_chinese_query_retrieves_the_right_chinese_clause():
    # Before G-1a every clause tokenized to [] and the index was empty; now a
    # within-language BM25 query ranks the matching clause first.
    clauses = [
        _clause("c1", "本法所称个人信息是指可识别自然人的各种信息"),
        _clause("c2", "确需向境外提供个人信息的应当通过安全评估或者订立标准合同"),
        _clause("c3", "应当采取加密去标识化等安全技术措施防止未经授权的访问"),
    ]
    index = build_index(clauses)
    hits = index.query(_tokenize("向境外提供个人信息 跨境"), top_k=3)
    assert hits, "CJK query must return candidates after G-1a"
    assert hits[0].clause_id == "c2"
    assert hits[0].score > 0  # a real lexical match, not a zero-score fallback
