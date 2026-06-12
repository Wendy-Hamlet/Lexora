# Findings

Thread: `019ebb0d-b2c7-7222-920c-e4538d05f70d`

## 2026-06-12 Initial Findings

- `CODEX_THREAD_ID` is `019ebb0d-b2c7-7222-920c-e4538d05f70d`; this is the only thread directory this agent will write.
- `goal.md` defines the immediate handoff goal as closing the loop on API key / OpenAI-compatible endpoint access and real LLM verifier validation, not rewriting the main pipeline.
- Current source checkout is on `main...origin/main`.
- Current source checkout has uncommitted/untracked state: `.env.example` modified, `.agent_record/` untracked, two docs under `docs/` untracked, and `goal.md` untracked.
- Existing records from another thread are present under `.agent_record/019ebaff-551a-7393-b7cc-2ae8ca937b81/`; they may be read but not modified.
- Previous thread records say the known integration gap is that root `.env` used `OPENAI_API_KEY` / `OPENAI_BASE_URL`, while code and `.env.example` expected `LEXORA_LLM_*`.
- Previous thread records say `src/lexora/config.py` used `os.environ.get` and did not automatically load `.env`.
- Previous thread records say `lexora map --verify` and `scripts/run_submission.py --verify` route through `make_verifier(use_llm=True)` and can degrade if the `openai` SDK/backend is unavailable.
- The rest of `goal.md` preserves later priorities: mapping misses, live crawl stability, G-track blind tests, multilingual infrastructure, and constrained LLM assistance. These are not required for the current verifier handoff stage unless needed by acceptance criteria.
- User clarified that API key and URL are in `.env`, model is in `.env.example`, LLM integration is mandatory, and other items are optional/skippable if blocked.
- Safe `.env` inspection printed names only: `OPENAI_API_KEY` and `OPENAI_BASE_URL`; no values were printed or recorded.
- `.env.example` contains `LEXORA_LLM_BASE_URL`, `LEXORA_LLM_API_KEY`, and `LEXORA_LLM_MODEL`.
- `src/lexora/classify/llm_client.py` currently says it reads endpoint/key/model from `LEXORA_LLM_*`, via `load_config()`.
- `make_verifier(use_llm=True)` only returns `None` when the `openai` SDK is missing or construction fails; endpoint call failures are caught inside `Verifier.verify()` and treated as abstain.
- `lexora map --verify` prints `LLM verifier ON` when `make_verifier()` returns a verifier; `scripts/run_submission.py --dry-run --verify` only prints the plan and does not instantiate/check the verifier.
- After implementation, `LEXORA_ENV_FILE=/home/ubuntu/scratch/swx/Lexora/.env .venv/bin/python scripts/run_submission.py --dry-run --verify` reports `LLM verifier ON (model: gpt-5.4)` without printing key or URL.
- `LEXORA_ENV_FILE=/home/ubuntu/scratch/swx/Lexora/.env .venv/bin/lexora map -j my --verify --budget 3` reached CLI startup and printed `LLM verifier ON`, but failed before crawl because the current environment uses a SOCKS proxy and `socksio` was not installed for `httpx`.
- The first config implementation incorrectly let `.env.example` `LEXORA_LLM_BASE_URL` override `.env` `OPENAI_BASE_URL`; this was fixed by preserving dotenv source layers and resolving aliases within each layer from highest to lowest precedence.
- Safe config structure check after the fix reports a remote/named endpoint, `/v1` path present, model present, and API key present, without printing URL/key.
- Minimal LLM smoke checks now reach the configured endpoint/model but the endpoint returns empty or non-parseable content for structured verifier requests. The code now raises `LlmResponseError` for this case so CLI/submission output reports explicit verifier backend response errors.
- Current MY budget-3 A/B artifacts are in the implementation worktree: `outputs/map_my_verify_off_budget3.csv`, `outputs/map_my_verify_off_budget3.jsonld`, `outputs/map_my_verify_on_budget3.csv`, `outputs/map_my_verify_on_budget3.jsonld`, and `outputs/llm_verifier_ab_my_budget3.md`.
- A/B result: verify-off produced 4 citation rows across 4 indicators; verify-on produced 0 rows because 3 verifier judgements failed with `LlmResponseError`. This is not yet a valid semantic tightening result.
- Independent review found a valid risk that `.env.example` should not be a full runtime config layer. The implementation was adjusted so `.env.example` only contributes `LEXORA_LLM_MODEL` / `OPENAI_MODEL`, while URL/API key must come from process env or `.env`.
- Independent review found `run_submission.py` non-dry-run should warn when `--verify` is requested but `make_verifier()` returns `None`; this warning and a regression test were added.
- Final safe config structure check with `LEXORA_ENV_FILE=/home/ubuntu/scratch/swx/Lexora/.env`: scheme present, remote/named host, `/v1` path present, model `gpt-5.4`, API key present. URL/key values were not printed.
- Implementation worktree `.env.example` was synchronized to `LEXORA_LLM_MODEL=gpt-5.4` to match the source checkout's user-provided model setting.
- JSON extraction was hardened to support fenced JSON and explanatory text around a JSON object.
- Structured LLM calls now retry empty/invalid JSON-mode output through plain chat up to `LEXORA_LLM_MAX_RETRIES` / `OPENAI_MAX_RETRIES` (default 2).
- Final MY budget-3 verify-on run completed without backend-error warnings and produced 1 citation row, routed to `CONFLICT_REVIEW`.
- Final A/B result: verify-off produced 4 citation rows across `P7-I1`, `P7-I2`, `P7-I4`, `P7-I5`; verify-on produced 1 citation row for `P7-I1`, with `CONFLICT_REVIEW`; verifier removed three weak/wrong rows.
- Final verification: `scripts/run_submission.py --dry-run --verify` prints `LLM verifier ON (model: gpt-5.4)`; full test suite result is 131 passed, 4 skipped; targeted ruff check passed.
- Secret scan found only placeholders/test fixtures such as `.env.example` local URL and test dummy keys; no real `.env` values were written to reports or records.
- Completion audit: the current user-scoped `goal.md` stage is satisfied. LLM config is wired from `.env`/`.env.example`, verifier starts and runs against the configured endpoint, endpoint errors are explicit and non-fatal, a MY verify-off/on A/B with outputs and evaluation report exists, and no secrets were recorded.
- Continued endpoint parameter probe found the current model returns content when chat completions include an explicit output-token limit: `chat_max_tokens` and `chat_max_completion_tokens` both returned `Hello`; `chat_json_max_completion` returned `{"ok":true}`. Plain chat without token limit returned 400 or empty content. This is a client-fixable issue.
- After adding default `max_completion_tokens=512` to the LLM client, a real smoke check with the configured endpoint/model returned parseable JSON: `ok=True`, keys `['ok']`.
