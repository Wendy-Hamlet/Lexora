"""Compare deterministic (regex) vs LLM amendment-instruction extraction on one
amending Act, to gauge how brittle the regex path is and where the LLM helps.

Usage:
    LEXORA_AMENDMENT_LLM=1 python scripts/compare_amendment_extract.py <amending_act.pdf>

The LLM path needs an OpenAI-compatible backend configured (LEXORA_LLM_* / OPENAI_*).
Without it, only the regex column is shown (the LLM column is inert -> empty).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pypdf import PdfReader

from lexora.cite.amendments import (
    classify_version,
    detect_amends_target,
    parse_amendment_instructions,
)
from lexora.cite.amendments_llm import make_amendment_extractor


def _text(pdf: str) -> str:
    return "\n".join((p.extract_text() or "") for p in PdfReader(pdf).pages)


def _key(i) -> tuple:
    return (i.op.value, i.target_section, i.old_term.lower(), i.new_term.lower())


def _fmt(i) -> str:
    term = f' {i.old_term}->{i.new_term}' if i.old_term else ""
    sec = f"s.{i.target_section}" if i.target_section else "-"
    return f"{i.kind.value:<22} {i.op.value:<10} {sec:<6}{term}"


def main(pdf: str) -> None:
    text = _text(pdf)
    print(f"document: {pdf}")
    print(f"version : {classify_version(text).value}   amends: {detect_amends_target(text)}")

    regex = parse_amendment_instructions(text)
    print(f"\n=== REGEX ({len(regex)}) ===", flush=True)
    for i in regex:
        print("  " + _fmt(i), flush=True)

    print("\n[calling LLM ...]", flush=True)
    llm = make_amendment_extractor(use_llm=True).extract(text)
    print(f"\n=== LLM ({len(llm)}) ===")
    for i in llm:
        print("  " + _fmt(i))

    rset = {_key(i) for i in regex}
    lset = {_key(i) for i in llm}
    only_regex = [i for i in regex if _key(i) not in lset]
    only_llm = [i for i in llm if _key(i) not in rset]
    print(f"\n=== DIFF ===  shared={len(rset & lset)}  regex-only={len(only_regex)}  llm-only={len(only_llm)}")
    for i in only_regex:
        print("  [regex-only] " + _fmt(i))
    for i in only_llm:
        print("  [llm-only]   " + _fmt(i))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: compare_amendment_extract.py <amending_act.pdf>")
    main(sys.argv[1])
