"""Draft cross-lingual query synonyms with the LLM (G-3a) — human-review gate.

Generates, for every in-scope RDTII indicator, local-language legal search terms
in the target language and writes them to a REVIEW YAML in the same shape as a
jurisdiction profile's ``keywords_by_indicator`` block:

    keywords_by_indicator:
      "6.4":
        zh: ["向境外提供个人信息", "跨境提供", ...]

It deliberately does NOT edit the live config: the LLM drafts, a human reviews,
then the reviewed block is pasted into configs/jurisdictions/<iso>.yaml. The model
only ever produces *query* terms — never citation text — so review is about
retrieval quality, not the verbatim contract.

Usage:
    LEXORA_ENV_FILE=/path/.env python scripts/gen_synonyms.py --language zh \
        --out outputs/synonyms_zh.review.yaml
    python scripts/gen_synonyms.py --language th --dry-run   # list prompts, no calls
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from lexora.classify.synonyms import make_synonym_generator  # noqa: E402
from lexora.indicators import load_indicators  # noqa: E402

INDICATORS = REPO / "configs" / "rdtii_indicators.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--language", required=True, help="target language tag, e.g. zh / th")
    ap.add_argument("--out", type=Path, help="write the review YAML here")
    ap.add_argument("--dry-run", action="store_true", help="show indicators, make no LLM calls")
    args = ap.parse_args()

    indicators = load_indicators(INDICATORS)

    if args.dry_run:
        print(f"Would draft {args.language} synonyms for {len(indicators)} indicators:")
        for ind in indicators:
            print(f"  - {ind.submission_id} ({ind.id}) {ind.name}")
        return

    gen = make_synonym_generator(use_llm=True)
    if gen is None:
        print("LLM synonym generator unavailable (install the [llm] extra and set "
              "LEXORA_LLM_* / OPENAI_* env vars).", file=sys.stderr)
        raise SystemExit(1)

    block: dict[str, dict[str, list[str]]] = {}
    for ind in indicators:
        terms = gen.generate(ind, args.language)
        print(f"{ind.submission_id} ({ind.id}): {len(terms)} term(s)")
        if terms:
            block[ind.id] = {args.language: terms}

    if gen.error_count:
        print(f"warning: {gen.error_count} generation(s) failed "
              f"(last: {gen.last_error_type}); those indicators were skipped.",
              file=sys.stderr)

    doc = {"keywords_by_indicator": block}
    text = yaml.safe_dump(doc, allow_unicode=True, sort_keys=True, width=100)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"\nReview draft -> {args.out}  (paste reviewed block into the profile)")
    else:
        print("\n" + text)


if __name__ == "__main__":
    main()
