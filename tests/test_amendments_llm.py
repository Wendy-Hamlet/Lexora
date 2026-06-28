"""LLM-assisted amendment-instruction extraction (lexora.cite.amendments_llm).

Offline — a stub client stands in for the LLM. Exercises the inert default, the
mapping of model output onto the shared AmendmentInstruction taxonomy, and the
source-verification guard that drops a fabricated section / unverifiable rename.
"""
from __future__ import annotations

from lexora.cite.amendments import InstructionKind, Operation
from lexora.cite.amendments_llm import AmendmentInstructionExtractor

DOC = (
    "An Act to amend the Example Act 2010. The principal Act is amended by "
    'substituting for the words "data user" wherever appearing the words "data '
    'controller". Section 6 of the principal Act is amended. Section 129 of the '
    "principal Act is deleted."
)


class _StubClient:
    """Returns a canned response; records the last call for assertions."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat(self, system, user, json_schema=None):
        self.calls += 1
        return self.payload


def test_inert_extractor_returns_empty():
    ex = AmendmentInstructionExtractor(client=None)
    assert ex.extract(DOC) == []


def test_maps_operations_and_scopes():
    payload = {"instructions": [
        {"operation": "rename", "scope": "global_rename", "target_section": "",
         "old_term": "data user", "new_term": "data controller",
         "evidence": "substituting for the words"},
        {"operation": "amend", "scope": "section_op", "target_section": "6"},
        {"operation": "delete", "scope": "section_op", "target_section": "129"},
    ]}
    ex = AmendmentInstructionExtractor(client=_StubClient(payload))
    out = ex.extract(DOC)
    assert [i.kind for i in out] == [
        InstructionKind.global_rename, InstructionKind.section_op, InstructionKind.section_op,
    ]
    assert out[0].op is Operation.rename and out[0].new_term == "data controller"
    assert out[1].op is Operation.amend and out[1].target_section == "6"
    assert out[2].op is Operation.delete and out[2].target_section == "129"


def test_source_verification_drops_fabricated_section():
    payload = {"instructions": [
        {"operation": "delete", "scope": "section_op", "target_section": "999"},  # not in DOC
        {"operation": "amend", "scope": "section_op", "target_section": "6"},     # in DOC
    ]}
    ex = AmendmentInstructionExtractor(client=_StubClient(payload))
    out = ex.extract(DOC)
    assert [i.target_section for i in out] == ["6"]
    assert ex.rejected == 1


def test_unverifiable_rename_term_is_dropped():
    payload = {"instructions": [
        {"operation": "rename", "scope": "global_rename", "old_term": "nonexistent term",
         "new_term": "whatever"},
    ]}
    ex = AmendmentInstructionExtractor(client=_StubClient(payload))
    out = ex.extract(DOC)
    assert out == []  # old_term not in source -> rename unusable
    assert ex.rejected == 1


def test_malformed_item_rejected():
    payload = {"instructions": [{"operation": "bogus", "scope": "section_op",
                                 "target_section": "6"}]}
    ex = AmendmentInstructionExtractor(client=_StubClient(payload))
    assert ex.extract(DOC) == []
    assert ex.rejected == 1
