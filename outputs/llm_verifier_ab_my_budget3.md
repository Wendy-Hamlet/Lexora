# LLM Verifier A/B Evaluation: MY budget 3

Date: 2026-06-12

## Configuration

- Endpoint source: `.env` via `OPENAI_BASE_URL` loaded through `LEXORA_ENV_FILE`.
- API key source: `.env` via `OPENAI_API_KEY`.
- Model source: `.env.example` via `LEXORA_LLM_MODEL`.
- Model used: `gpt-5.4`.
- Secret handling: API key and endpoint URL values were not printed or written to this report.

## Commands

```bash
LEXORA_ENV_FILE=/home/ubuntu/scratch/swx/Lexora/.env .venv/bin/python scripts/run_submission.py --dry-run --verify
LEXORA_ENV_FILE=/home/ubuntu/scratch/swx/Lexora/.env .venv/bin/lexora map -j my --budget 3 --out outputs/map_my_verify_off_budget3.jsonld
LEXORA_ENV_FILE=/home/ubuntu/scratch/swx/Lexora/.env .venv/bin/lexora map -j my --verify --budget 3 --out outputs/map_my_verify_on_budget3.jsonld
```

## Result Summary

| Metric | Verify off | Verify on |
| --- | ---: | ---: |
| Citation rows | 4 | 1 |
| Covered indicators | 4 (`P7-I1`, `P7-I2`, `P7-I4`, `P7-I5`) | 1 (`P7-I1`) |
| `CONFLICT_REVIEW` rows | 0 | 1 |
| Output CSV | `outputs/map_my_verify_off_budget3.csv` | `outputs/map_my_verify_on_budget3.csv` |
| Output JSON-LD | `outputs/map_my_verify_off_budget3.jsonld` | `outputs/map_my_verify_on_budget3.jsonld` |

The verifier path is now wired and visibly enabled: both `run_submission.py --dry-run --verify` and `lexora map -j my --verify --budget 3` print `LLM verifier ON`.

The configured endpoint/model now returns parseable verifier JSON after the client supplies an explicit output-token limit and retries empty JSON-mode responses through plain chat. A minimal LLM smoke check returned parseable JSON, and the production verifier run completed without backend-error warnings.

The verify-on run retained one candidate (`P7-I1`, Section 4) but routed it to `CONFLICT_REVIEW` as uncertain, and removed three weak/wrong candidates (`P7-I2`, `P7-I4`, `P7-I5`). This is a meaningful verifier A/B result, not a backend failure.

## Sample Review

The verify-off rows changed as follows:

| Indicator | Verify-off clause | Verify-on result | Manual spot check |
| --- | --- | --- |
| `P7-I1` | `PERSONAL DATA PROTECTION ACT 2010`, Section 4 | Retained as `CONFLICT_REVIEW` / uncertain | Weak evidence for the framework-existence indicator: it is a definitions section. It hints at a horizontal personal-data framework but is not a strong framework-establishing provision by itself, so routing to review is appropriate. |
| `P7-I2` | `PERSONAL DATA PROTECTION ACT 2010`, Section 115(1) | Dropped | Wrong/weak evidence for cybersecurity framework: the clause concerns authorized-officer search access to computerized data, not a dedicated cybersecurity law. |
| `P7-I4` | `PERSONAL DATA PROTECTION ACT 2010`, Section 115(1) | Dropped | Wrong evidence for DPO/DPIA requirements: the same search-power clause does not establish appointment of a DPO or DPIA duties. |
| `P7-I5` | `PERSONAL DATA PROTECTION ACT 2010`, Section 32(1) | Dropped | Weak evidence for government access: the clause concerns refusal of a data access request, not government access to personal data without court orders. |

## Recommendation

Continue using `--verify` for A/B evaluation and review triage. The MY budget-3 result shows the verifier is doing useful tightening: it removed three weak/wrong mappings and routed the remaining weak framework evidence to `CONFLICT_REVIEW` rather than treating it as clean evidence.

Do not make `--verify` the default yet from this single small run. Next, rerun a fuller all-jurisdiction budget-20 comparison and inspect whether the verifier systematically removes weak evidence without over-dropping good evidence.
