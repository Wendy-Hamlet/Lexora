# Task Plan

Thread: `019ebb0d-b2c7-7222-920c-e4538d05f70d`
Goal: implement `goal.md`.
Started: 2026-06-12

## Scope

Current user-scoped stage is complete when the repository proves the mandatory LLM verifier integration path:

- Runtime config reads `LEXORA_LLM_BASE_URL`, `LEXORA_LLM_API_KEY`, `LEXORA_LLM_MODEL`, with documented or coded fallback from `OPENAI_*`.
- `.env` handling is confirmed without committing secrets.
- Verifier can be enabled without silently falling back to BM25.
- Produce at least one verify-off / verify-on A/B comparison.
- Required output artifacts and a short evaluation record should be produced.
- `goal.md` itself remains actionable and does not leak secret values.

## Constraints

- User-facing reporting must be Simplified Chinese.
- Work docs are written only under this thread directory.
- Code changes must be made in a new worktree and new branch, not directly in the source checkout.
- For complex implementation, use a closed-loop workflow with independent review.
- Do not write real API keys into git, logs, `goal.md`, or `.agent_record`.

## Phases

| Phase | Status | Description |
| --- | --- | --- |
| 1 | complete | Restore context: read `goal.md`, existing records, docs, current code, and current git state. |
| 2 | complete | Create a separate implementation worktree and branch if code changes are required. |
| 3 | complete | Implement config/verifier changes needed to satisfy A/B-ready verifier behavior. |
| 4 | complete | Run focused verification commands and capture artifacts without leaking secrets. |
| 5 | complete | Run independent review/feedback loop and address findings. |
| 6 | complete | Audit every `goal.md` acceptance item and decide whether the active goal is complete. |
| 7 | in_progress | Sync the completed code, output artifacts, and conclusion reports to the remote branch without committing secrets. |

## Errors Encountered

| Error | Attempt | Resolution |
| --- | --- | --- |
| `pytest` command not found | Ran targeted tests with `pytest tests/test_config.py tests/test_verifier.py` in the implementation worktree | Next attempt will use `python -m pytest`; if missing, inspect/install project test dependencies. |
| `python` command not found | Ran targeted tests with `python -m pytest ...` | Next attempt will use `python3 -m pytest`. |
| `pytest` module missing | Ran targeted tests with `python3 -m pytest ...` | Create a local worktree virtualenv and install `.[dev,llm]` before re-running tests. |
| `python3 -m venv .venv` failed because `ensurepip` is unavailable | Tried to create the user-requested project-local `.venv` in the implementation worktree | Check alternative environment tooling (`uv`, `virtualenv`, pip entrypoints) or install the missing venv support if available. |
| Ruff import-order failures in `config.py` and `scripts/run_submission.py` | Ran targeted ruff check after tests | Apply ruff's automatic import sorting and re-run the check. |
| `lexora map -j my --verify --budget 3` failed before crawl due `ImportError: Using SOCKS proxy, but the 'socksio' package is not installed` | Ran the small verifier command with `LEXORA_ENV_FILE` pointing at the source `.env` | Add/install the `httpx` SOCKS extra so the current proxy environment can create HTTP clients. |
| `.env.example` `LEXORA_LLM_BASE_URL` overrode `.env` `OPENAI_BASE_URL` because dotenv layers were flattened before alias selection | Safe config structure check showed the runtime LLM URL was local despite `.env` containing `OPENAI_BASE_URL` | Preserve dotenv layers and search higher-precedence files before lower-precedence files for each primary/alias group. |
| Endpoint rejected `response_format=json_object` unless lower-case `json` appears in input | Minimal LLM smoke test returned `BadRequestError` 400 `param=input` | Add lower-case `json` instruction to both system and user prompts when structured output is requested. |
| Endpoint returned empty content under JSON mode and plain chat for the configured model | Minimal LLM smoke tests and verify-on map returned empty/non-parseable model responses | Add JSON-mode fallback and raise `LlmResponseError` on empty/unparseable structured responses so CLI reports explicit backend response errors. |
| Independent review found `.env.example` was treated as a full runtime config source | Subagent review of `config.py` and tests | Changed `.env.example` handling so it can only supply `LEXORA_LLM_MODEL` / `OPENAI_MODEL`; `.env` remains the source for URL/API key and non-LLM defaults no longer drift. |
| Independent review found `scripts/run_submission.py` non-dry-run lacked a warning when verifier construction returns `None` | Subagent review | Added warning in `run_one()` and an offline regression test. |
| Independent review found `LlmClient.chat()` docstring was stale | Subagent review | Updated docstring to describe JSON-mode fallback and `LlmResponseError`. |
| Endpoint returned empty content unless chat requests included an explicit output-token limit | Non-secret endpoint parameter probe | Added `llm_max_tokens` config and default `max_completion_tokens=512` in `LlmClient`; smoke check then returned parseable JSON. |
| Endpoint intermittently returned empty/non-JSON content for verifier-style prompts | Repeated verifier prompt probes | Added structured-output retries and robust JSON-object extraction from fenced/explanatory responses. Final MY verify-on run completed without backend-error warning. |
