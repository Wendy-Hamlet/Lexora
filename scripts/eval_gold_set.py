"""Evaluation harness against a hand-curated gold set of 50–100 clauses.

Metrics:
    - retrieval recall@k         (does the gold clause appear in top-k?)
    - indicator precision        (of validated citations, fraction labelled correctly)
    - citation exact-match rate  (validated citation char_start/char_end == gold)
    - OCR citable-page rate      (citable pages / total pages)
    - abstention quality         (precision/recall of abstain decisions)
    - conflict-detection accuracy (does it flag the right multi-source disagreements?)

Gold set lives in data/gold/<jurisdiction>.yaml — see docs for schema.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", type=Path, default=Path("data/gold"))
    parser.add_argument("--out", type=Path, default=Path("data/eval_report.json"))
    args = parser.parse_args()

    print(f"Gold dir: {args.gold_dir}")
    print(f"Report:   {args.out}")
    # TODO: load gold, run pipeline, compute metrics, write report.
    raise SystemExit("Eval harness not yet implemented.")


if __name__ == "__main__":
    main()
