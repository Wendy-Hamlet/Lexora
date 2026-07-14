# Round 1 submission — output files

The two deliverable files, exactly as uploaded to the submission portal, plus the run
summary. These are what the code in this repository produced; they are here so the results
can be inspected without running anything.

| File | What it is |
| :---- | :---- |
| `Lexora_SG-AU-MY_P6-P7.csv` | **Primary deliverable.** 669 provisions, the official 13 columns of `OUTPUT_TEMPLATE_31MAY.xlsx`, header text and order byte-for-byte. |
| `Lexora_SG-AU-MY_P6-P7.json` | **Supplementary deliverable.** Same 669 provisions, plus what CSV cannot carry: source PDF path, `pdf_is_scanned`, OCR quality, timing, retrieval method, and the raw context either side of every snippet. |
| `run_summary.json` | Per-economy run report: instruments discovered, NEW/KNOWN split, documents fetched, indicators covered, amendment-currency breakdown. |

## The run that produced them

```bash
LEXORA_LIVE=1 python scripts/run_submission.py -j all --budget 20 \
    --verify-clauses --rationale-llm --metadata-llm --amendment-llm \
    --jobs 3 --doc-workers 4 --llm-workers 16 --serial-fetch
```

2026-07-14, 09:46 → 12:12 (2 h 26 m), live crawl of the official portals. GLM-5.2 as the
relevance judge. OCR on the GPU (`rapidocr:1.4.4+cuda`).

| | Instruments | **NEW** | KNOWN | Fetched | Citations | Indicators |
| :---- | ----: | ----: | ----: | ----: | ----: | ----: |
| Singapore | 54 | **20** | 34 | 152 | 291 | 6 |
| Australia | 43 | **20** | 23 | 83 | 228 | 8 |
| Malaysia | 39 | **15** | 19 | 60 | 150 | 7 |
| **Total** | **136** | **55** | 76 | 295 | **669** | |

137 of the 669 rows are tagged `NEW` — provisions found beyond the sample kit.

## Two things to know before you compare this to your own run

**1. The relevance judgement is not reproducible run-to-run.** The LLM is not deterministic
even at `temperature=0`; the same Act judged three times gave 11 / 14 / 18 citations. This
file is one sample, not a fixed point. Everything the LLM does *not* decide — discovery,
fetching, parsing, retrieval, and the verbatim text of every snippet — reproduces exactly.
See *Determinism* in the root README.

**2. P6-I3 is empty, and that is the correct answer.** P6-I3 (infrastructure requirements)
asks whether a law mandates establishing a local data centre as a condition of service.
None of the three economies does, so no provision maps to it. An earlier top-k ranking
architecture filled it with 51 false positives — clauses about "establishing disciplinary
committees" — because a ranking cannot return an empty set. The 0/1 membership judge can.
