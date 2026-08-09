# Roadmap

Written 2026-08-09, after the RDTII hackathon ended. The competition is over and Lexora is now
an open-source project developed at whatever pace it gets developed at. This document says what
it is for, what is worth keeping, and what order to build things in.

## What this project actually is

For eight weeks Lexora was "a tool that maps digital-trade law to RDTII Pillars 6 and 7 for
Singapore, Australia and Malaysia". That framing is dead. It described a submission form, not a
program.

Three things in this repo are hard, already work, and are rare in legal-tech tooling:

1. **The verbatim invariant.** Every quoted provision is byte-identical to the source document at
   a recorded character offset. Hard-verified 163/163 on the submitted corpus. Most LLM legal
   tools cannot prove a quotation came from the document they name.
2. **Record / replay.** A live run can be recorded and replayed entirely offline, and the output
   CSV is sha256-identical to the live one. Reproducibility of a network-dependent, model-
   dependent pipeline is rare anywhere, let alone here.
3. **Amendment currency.** The pipeline detects that a cited law has been amended or repealed
   since the text it quotes, live-validated across three jurisdictions with no hardcoding.

Those three are the product. RDTII is one *application* of them that happened to be a hackathon.

**So: Lexora is a verifiable legal-provision corpus builder with a pluggable framework mapper.**
RDTII becomes the reference configuration and the worked example — not the core.

Everything below follows from that sentence.

## The one structural fact that explains most of the backlog

```
src/lexora/  16,414 LOC     tests/  11,632 LOC     664 tests
├── pipeline.py            1,905 LOC   ← all orchestration lives here
├── storage/db.py             23 LOC   ← 7-table schema written in the docstring,
│                                         init_db() raises NotImplementedError
├── api/app.py                29 LOC   ← /healthz, plus one NotImplementedError
└── cli.py                             ← extract / classify / export subcommands
                                          all raise NotImplementedError
```

Every seam that was designed on day one — a corpus, staged commands, an HTTP surface — is an
unimplemented stub, and one 1,905-line function graph does the work instead. That single fact
causes symptoms we have been treating as unrelated bugs:

| Symptom | Cause |
| --- | --- |
| The rationale generator has no cache; a full Malaysia run with it costs 2,545 s | There is no corpus, so there is **nowhere to put anything**. Every run re-crawls, re-parses and re-judges, and keeps nothing but a CSV. |
| The demo command diverged from the delivery command, which is what lost the pitch | Only one command does real work, so it was tuned for a stopwatch and then shown to legal judges. |
| No review UI | Between runs there is nothing to browse. |
| Portal drift went unnoticed for weeks (Malaysia mapped 150 rows on 20 Jul, finds none today with the same code) | No persisted history to diff against. |
| Coverage stuck at 31/38 | Adding a collection route means touching the monolith. |

Fixing these one at a time is how we got here. Building the seams fixes them as a class.

## Non-goals

Stated so they stop consuming attention:

- **No accuracy-chasing on the current gold sets.** They are tiny, partly annotated against our
  own retrieval pool, and the judge is not reproducible (11 / 14 / 18 citations on identical cold
  runs). Improvement measured against them is mostly noise. Fix the measurement first (Phase 3),
  then optimise.
- **No new retrieval stack, no new models.** BM25 + the per-clause judge is the lane that works,
  and every small model we tested is more than 2x worse than the model-free BM25 fallback.
- **No scoring.** ESCAP forbade it; that reason is gone, but the reason to stay out is better: a
  tool that cites is falsifiable, a tool that scores is arguable.

## Deletions

An open-source repo is read before it is run. Competition scaffolding will mislead every reader
who arrives after us, so it goes:

- `collect/secondary/*` — seven third-party tracker adapters (UNCTAD, DLA Piper, ICLG,
  Linklaters, STRI, WorldMap). These were **never citable by rule**; they existed as a recall aid
  under a deadline. Keeping an uncitable source in a tool whose whole claim is citability is a
  trap. Delete, or move behind an explicitly-named `research/` namespace that no export path can
  reach.
- The submission CSV schema and `submission/` conventions — an artifact of one xlsx template.
- `DEFAULT_PILLARS = (6, 7)` and every other place RDTII is a default rather than a config.
- The `.agent_record` / demo scaffolding accumulated for the pitch.

---
## The safety net for a large refactor

This roadmap breaks a 1,905-line monolith apart, on a schedule with no deadline and every chance
of a month-long pause. The way that normally dies is a half-migrated repo that nobody wants to
touch when they come back.

One rule prevents it, and we already own the machinery:

> **The new path lands behind a flag, with the old path still working, until the new path
> reproduces `submission/Lexora_SG-AU-MY_P6-P7.csv` byte-for-byte from a recorded replay.**

Record/replay already produces sha256-identical output, so this is a real, automatable acceptance
test rather than a good intention. Every phase below states its own acceptance test in the same
spirit: a thing that is either true or false, not a feeling that the code got nicer.

---

## Phase 0 — Stop shipping columns that lie (days) — DONE 2026-08-09

Outcome, so the next reader does not have to reconstruct it from commits:

* **The rationale layer is cached** (`df38771`) and stores the model's RAW response, not the
  answer we accepted. That decision paid for itself the same day: retuning the length guard
  below was measured over 205 cached answers and 5 live calls.
* **Every LLM layer says ON or OFF**, in `run_submission.py` and in `cli.py` — the command
  that survives Phase 2 and was the silent one.
* **The 29% mystery was our own character cap** (`07b2f70`). Reason-level counters showed 59
  of 77 fallbacks were `too_long` and only 13 involved the copy guard we had blamed for two
  months. The model obeys its "max 280 characters" instruction (median 273 over 205
  responses) but the tail reaches 549, and the cap allowed 20 characters of slack. Raised to
  1000: **62.8% → 93.2% model-authored**, median rationale length unchanged at 271.
* **`Confidence` was already fixed** (`addead5`) and the README already documented the judge's
  non-reproducibility. Both roadmap items were written from memory and were stale.
* **The floor that replaced them** is measured: see `--min-score` below.
* **The acceptance check found one more**: `Law Number / Ref` and `Last Amended` are empty on
  all 181 rows without `--metadata-llm`, because the portal publishes no structured metadata.
  A permanently empty column lies by omission. Open as task #40, and the column-constancy
  check itself should become automated rather than something someone remembers to run.

---

### Original plan

Trust is the only asset an open-source verification tool has. Everything currently published that
overstates itself gets fixed or deleted before strangers read it.

1. **Cache the rationale generator** the way `classify/judge_cache.py` caches the judge — key on
   the rendered prompt. This is the root of the 2,545 s cost that made us drop `--rationale-llm`,
   which is what lost the pitch. Cheap version now; it becomes corpus-backed in Phase 1.
2. **Make the metadata and amendment layers speak when they are off.** `d6e4350` fixed this for
   the rationale layer only; `metadata_extractor` and `amendment_extractor` still print nothing
   when their client is absent, so `Law Number` and `Last Amended` degrade silently. Same shape,
   same fix.
3. **Read `llm_used` / `fallbacks`.** ~29% of rationales fall back to template even with the flag
   on and nobody has ever looked at why. Prime suspect: `copies_provision` rejects any 6-word run
   shared with the provision, and legal prose is formulaic ("for a period of not less than"). The
   counters tell us whether the model declined or our own guard shot it down.
4. **~~The `Confidence` column is a constant.~~ Already fixed** on 2026-07-28 in `addead5`:
   `_SCORE_SCALE` went 5.0 → 40.0, so confidence now spreads (raw 20 → 0.46, 80 → 0.96). The
   ~1.0 values are a property of the *submitted* file, not of the code. **What replaces this
   item:** the same fix made a floor live that had been provably inert for the project's whole
   life. `min_score` (default **0.35**) gates *pool entry* in the per-clause lane
   (`pipeline.py:1720`); post-rescale that is raw BM25 ≥ 14.6, a cut that could not happen
   before. It bites exactly the tail `pool_k=40` exists to reach — measured retrieval ceiling
   pool 3 → 38%, 20 → 76%, 40 → 95%. The 0.7 variant was measured (cuts 29% of judge calls);
   the live 0.35 default was not. Measure it on a replay before changing it.
5. **~~Document that the judge is not reproducible.~~ Already done** — README has a *Determinism*
   section stating the 11 / 14 / 18 result, that the submitted CSV is one sample, and which parts
   of the pipeline *are* stable. What is still missing is a properly measured spread over more
   than three runs, which is Phase 3's job, not Phase 0's.

*Acceptance: a full SG run with every LLM layer off prints, for each layer, that it is off; no
output column is a constant; the rationale layer replays at 100% cache.*

## Phase 1 — The corpus

Build `storage/db.py` for real, to the seven-table schema already written in its own docstring:
`documents`, `pages`, `clauses`, `indicators`, `claims`, `citations`, `audit_events`.

Design constraints that matter more than the choice of ORM:

- **Content-addressed identity.** A document is its sha256; a clause is `(document_sha, start,
  end)`. This makes the verbatim invariant a property of the schema instead of a property of one
  run, and makes every downstream cache key derivable rather than invented.
- **Fetching never interprets.** `lexora fetch` writes raw bytes plus provenance and stops. Parse,
  judge and export are separate passes over stored state.
- **Re-running is incremental by construction**, not by a flag.

What this dissolves: the rationale cost problem (there is finally somewhere to put a verdict);
portal drift (yesterday's corpus is a diff target — the Malaysia regression would have been a
failing diff, not a silent zero); the demo/delivery fork (see Phase 2); and the precondition for
any UI at all.

*Acceptance: a second run over an unchanged corpus makes zero network calls and zero model calls,
and produces byte-identical output.*

## Phase 2 — Make the seams real

Kill the three `NotImplementedError` subcommands by making `extract`, `classify` and `export`
genuine stages over the corpus. `pipeline.py` becomes a thin composition of them instead of the
place where everything happens.

This is the structural fix for what lost the pitch. When each stage has exactly one way to run it,
the command you demo *is* the command that produces the deliverable, because no other command
exists. The 2026-08-03 failure — a stopwatch-tuned command shown to legal judges — is not
reachable from this design.

*Acceptance: `fetch | extract | classify | export` reproduces the submitted CSV byte-for-byte from
a replay, and `run_submission.py` is deleted rather than kept alongside.*

## Phase 3 — Reproducibility as a feature

This is what makes the project worth other people's attention, and it is currently our weakest
claim: we advertise verifiability while the judge silently varies run to run.

- **`audit_events` earns its place**: every verdict records model id, prompt hash, cache hit/miss,
  cost and timestamp. Already in the schema; never built.
- **Measure judge variance instead of hoping.** N cold runs on a fixed corpus, publish the spread.
  A stated ±range is a credible claim; an unstated one is a defect waiting to be found by a
  stranger.
- **Opt-in self-consistency** (k=3 majority) for cells that matter, priced from the existing cost
  accounting so the trade is visible.
- **`lexora verify <output>`** — re-check any published artifact against the corpus with no model
  in the loop: do the quotes match at the recorded offsets, do the locators resolve, is any cited
  law now amended. This is the single most compelling thing this repo could offer a stranger, and
  it is mostly assembling parts that already exist (`cite/validator.py`, the offsets, the
  currency layer).

*Acceptance: `lexora verify` on the submitted CSV passes 163/163 quotes with no network and no
model, and README states a measured variance figure.*

## Phase 4 — Pluggability, proven with a second instance

Today "pluggable" is aspirational: SG/AU/MY are a yaml file each *plus* bespoke code
(`au_enumerate.py`, `my_inventory.py`, SSO path handling, three locator dialects).

- **Extract a `Collector` protocol** — search, enumerate, resolve full text, resolve amendments —
  and reimplement the three existing jurisdictions on top of it. If the protocol cannot express
  what AU's EPUB path or MY's 1,287-title JSON inventory already does, the protocol is wrong.
- **Frameworks become data.** RDTII moves to `frameworks/rdtii.yaml`; the code stops knowing what
  a "pillar" is.
- **Add a second framework and a fourth jurisdiction.** A pluggability claim with one instance is
  not a claim. GDPR chapters or a national DPA checklist is the obvious second framework — it
  reuses the same provisions we already collect, so it tests the seam without new collection.

*Acceptance: a new jurisdiction is added without editing anything outside `configs/` and one new
collector module, and a framework is swapped without touching `src/lexora/classify/`.*

## Phase 5 — Coverage and language

- **The 7 unreachable instruments** (coverage 31/38) are all soft law, guidance or international
  agreements. The mechanism is understood: `known_instruments` stores names without URLs and
  resolves them by searching statute databases — and soft law is not in a statute database, so it
  is structurally unreachable. Two new collection routes are needed: ministry and regulator sites,
  and treaty databases. This is the largest genuine engineering item on the list.
- **Non-English** (the G-track): CJK/Thai tokenisation, bge-m3 wiring and multi-script parsing are
  done; a real non-English run and a civil-law structural pack are not.
- **New jurisdictions** on the Phase-4 protocol, chosen for what they stress rather than for
  coverage: one civil-law, one non-English, one with a hostile portal.

## Phase 6 — Surfaces

- **The API for real** — `api/app.py` over the corpus, read-only first.
- **A review UI.** It ranks below the adapter work under an open-source priority, but it must
  exist, because reviewing 669 rows in a spreadsheet is precisely how a fully-templated rationale
  column survived two weeks of rehearsal. The data layer already exists: every run writes a JSON
  sidecar with evidence and offsets. What is missing is a page that shows a provision, its quoted
  span highlighted in the source document, the indicator, the rationale and the verdict — one
  screen where a lawyer can say "that is wrong".

## Cross-cutting: the test suite proves less than it appears to

664 green tests have coexisted with: a portal that changed under us (tests mock it), three separate
CI failures from optional extras never installed on the runner, and a `close()` call that only
works on Windows while CI hung for six hours on Linux. For a repo strangers are meant to clone,
that is disqualifying on its own.

- **A contract-test tier that hits real portals**, run on a schedule, allowed to fail loudly and
  separately from the unit suite.
- **CI installs the extras**, at least in one matrix leg — `.[dev]` alone has now hidden three
  failures.
- **CI runs on Linux and Windows**, because we have shipped platform-specific behaviour twice
  without noticing.

## Order, and why

Phase 0 first because shipped lies are cheap to fix and expensive to be caught with. Phase 1 and 2
next because they are the root of most of the backlog and everything after them is easier. Phase 3
before Phase 4 because pluggability without reproducibility just multiplies unverified output.
Phase 5 and 6 are genuinely optional and can be reordered by whatever is interesting on the day —
which, given the pace this project now runs at, is a feature.

