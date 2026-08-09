# Lexora — AI Tool for Digital Trade Regulatory Analysis

UN Global Hackathon on AI for Digital Trade Regulatory Analysis
Team: **Verbatim Trade** | Round: 1
Last updated: 2026-07-15

> A verifiable AI system for mapping digital-trade regulations to the UN ESCAP **RDTII 2.1** framework.
>
> **No canonical span, no claim.** — every machine-emitted claim is a verbatim quote bound to a specific document, page, and structural URI, and is validated before it reaches a human.

---

## What This Tool Does

Lexora automates the two tasks the RDTII requires for its **Step 1** (evidence
production). It does **not** assign scores — scoring is a downstream human step.

**Task 1 — Automated Evidence Discovery.** Given an economy and a regulatory topic,
Lexora crawls the official government legal portal (no URL is handed in), ranks the
candidates, fetches the full-text PDF/HTML (including scanned/image PDFs via OCR), and
extracts clean, structured, verbatim text.

**Task 2 — Intelligent Mapping & Citation.** The structured text is mapped to specific
RDTII indicator IDs. Each matched provision is recorded with an exact article-level
citation, a verbatim snippet, a `mapping_rationale`, and a Discovery Tag marking whether
it was found independently (**NEW**) or matched a known example (**KNOWN**).

**Mandatory scope:** Pillar 6 (Cross-Border Data Policies) and Pillar 7 (Domestic Data
Protection & Privacy) — 9 in-scope regulatory indicators (P6-I1…I4, P7-I1…I5; P6-I5 is a
non-regulatory indicator and is auto-excluded).
**Economies covered (Round 1):** Singapore, Australia, Malaysia.

---

## Quick Start

⚠ **Required for Round 1.** A reviewer with basic Python should be able to run this in
under 10 minutes, using only the steps below.

```bash
# 1. Clone
git clone https://github.com/Wendy-Hamlet/Lexora.git
cd Lexora

# 2. Environment (Python 3.10+; verified on 3.12)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # pinned; includes the LLM client AND the OCR stack
playwright install chromium       # Singapore's SSO portal 403s a plain HTTP client

# 3. Configure the LLM endpoint (any OpenAI-compatible server)
cp .env.example .env              # then edit .env — see "Swapping the LLM" below

# 4. Run it — one command, no manual steps
python main.py --economy Singapore --pillar 6
```

**Output** (all three, always):

```
outputs/Singapore_P6_<timestamp>.csv     official 13 columns, one row per provision
outputs/Singapore_P6_<timestamp>.json    same rows + OCR / timing / context metadata
outputs/Singapore_P6_<timestamp>.html    the same rows as a filterable review page
```

The HTML is the policy reviewer's path: open it in any browser — no server, no build, no
network — filter by economy / indicator / `NEW`, search the full text, and read each
provision with its quote laid out in full and its official source one click away. It is a
view of the CSV, not a second source of truth.

**No API key?** Add `--no-llm`. The engine still crawls, OCRs, parses, maps (BM25) and
writes both files — you get complete output with template rationales, so you can verify
the pipeline end-to-end before configuring any endpoint.

The economy argument is forgiving: `Singapore`, `SG`, `sg` and even `Singapre` all
resolve (fuzzy-matched, with a note); an unrecognisable one exits with the supported list
rather than a stack trace.

### Running without a network

A live crawl of a rate-limited government portal is minutes of work and needs the portals
to be reachable and behaving. Two flags let a run be reproduced without either:

```bash
python main.py --economy Singapore --pillar 6 --record    # crawl live, keep every response
python main.py --economy Singapore --pillar 6 --offline   # replay it; no socket is opened
```

Replay is not a mock and not a pre-baked file. Every byte is the byte the portal actually
sent, on a date the run reports; discovery, OCR, parsing and retrieval all execute exactly
as they do live. Measured on Singapore Pillar 6 (`--no-llm --budget 3`,
2026-07-27): **16 m 56 s live → 1 m 39 s replayed, and the submission CSV is
byte-identical (sha256 `b12390a1…`)**. The only field that differs anywhere in the output
is `retrieval_timestamp`, and it correctly reports when the portal served the bytes rather
than when they were read back.

A URL that was never recorded replays as an unreachable portal (`504`), which the pipeline
already knows how to carry on past — a recording is a snapshot with a date, never a claim
about today.

**Replaying the judge needs the verdict cache, not the recording.** Interception is at the
httpx transport, and the OpenAI SDK builds an httpx client like everything else, so in
`--offline` the judge's calls are intercepted too. But a cached verdict is served *above*
the HTTP layer — a hit makes no request at all, which is why a recording taken with a warm
cache contains no LLM traffic to replay. So an offline run answers from `data/cache/`, and a
clause it has no verdict for gets the same synthetic `504` as an unreachable portal: that
judgement fails, and past the failure-rate gate the document degrades to the key-free BM25
lane with every row marked. Correct, and loud, but not a demonstration of the engine.

Two consequences worth knowing before you rely on it. A `--record` run must therefore be
made **with the judge on**, so it fills `data/http_cache/` and `data/cache/judge.sqlite`
together; and any edit to the judge prompt voids both at once, by design. `--offline` says
which case you are in before the run starts:

```
  !! judge cache [sg]: NO verdict answers the prompt this run will send (103 stored
     under an older prompt or model).
```

**A replayed run labels itself, everywhere.** Replay exists so the engine can be shown
working in seconds on a conference network, and that convenience is only honest if nobody
can mistake it for a cold live crawl. So `--offline` marks its own output in three places
at once, none of which survives being cropped, forwarded, or opened in a spreadsheet:

- the console says so **before** the run starts, not only in a summary afterwards;
- the HTML console carries a red banner above the first number on the page;
- every artifact is written as `DEMO_<name>.csv` / `.json` / `.html` / `.jsonld`.

A `--record` run is *not* a demonstration: it fetches from the source, so its output is
current and submittable. A warm verdict cache is not one either — the judge cache is keyed
on the rendered prompt and the clause text, so a hit is the same model answering the same
question. The line is drawn at one thing only: was the network replayed. Both facts are
reported (`N clause(s) judged live, M from the verdict cache`); only replay renames files.

---

## Full Usage

```bash
python main.py \
  --economy "Malaysia" \
  --pillar 6 \              # 6, 7, or all (default)
  --output-dir outputs/ \
  --budget 20 \             # max instruments to map
  --llm-workers 8 \         # parallel LLM calls
  --doc-workers 4 \         # instruments processed concurrently
  --record                  # or --offline; see "Running without a network" above
```

The run reports its progress as it goes — one line per discovery query, then one per
mapped instrument with its clause and citation counts. `--quiet` turns that off.

`--doc-workers` defaults to the economy's own `fetch_policy` in
`configs/jurisdictions/<iso>.yaml`, because how much concurrency a portal tolerates is a
property of the portal: `legislation.gov.au` applies a cumulative per-IP limit and
answers a burst with a challenge page that looks like a `200`, so AU serialises its
downloads while still parsing and mapping in parallel.

### What the portals actually did

Every run ends with a line like:

```
portals: 48 portal quer(ies): 25 answered, 0 empty, 13 blocked, 10 unrendered
```

The distinction is load-bearing. A portal that **refuses** to answer is not an economy
without that law, and the two used to be indistinguishable: discovery read "this page
lists no instruments" as "nothing matched". Measured on 2026-07-27, that reading was
wrong every single time — of 67 Singapore query renders, 16 came back as a 923-byte
CloudFront `403 ERROR / Request blocked` page and 20 as a 39-byte empty document, and
**none** was a genuine empty result set. Three of them named known instruments (Computer
Misuse Act, Criminal Procedure Code, Banking Act 1970).

Refusals are now classified, retried once after the sweep has cooled, and reported. Raise
`--doc-workers` only with these counts in front of you: concurrency that turns answers
into refusals is not a speed-up.

### Malaysia: read the statute book, do not search it

Malaysia's search proxy is worse than slow. Measured 2026-08-01 it answered roughly three
requests in five and returned HTTP 500 for the rest — and on 20 July it silently stopped
*filtering*: the same request that used to return the Personal Data Protection Act began
returning the first page of the register, with status 200 and real Act titles. Nothing
errored anywhere in the stack.

So it is no longer the primary route. `lom.agc.gov.my` renders its own listing pages from
JSON feeds, and those are a different service: **four requests give 1,287 Acts**, 1,183
with a direct English PDF, plus the English title, the commencement remark (including
`NOT YET IN FORCE`) and a 116-entry repeal chain naming the repealing Act.

```bash
LEXORA_MY_ENUMERATE=1 LEXORA_BRUTE_JUDGE=1 \
  python scripts/run_submission.py -j my --budget 120 --verify-clauses
```

| Variable | What it does |
| :---- | :---- |
| `LEXORA_MY_ENUMERATE` | Use the portal's listings + a two-stage relevance filter instead of its search |
| `LEXORA_MY_ENUM_VERDICTS` | Where to append the verdict cache (default `outputs/cache/my_enum_verdicts.jsonl`) |
| `LEXORA_MY_ENUM_WORKERS` | Parallel title judgements (default 16) |

The listings carry no subject metadata, so relevance is decided in two stages: from each
Act's **title**, then — for the ones flagged only for retention (P7-I3) or government
access (P7-I5), which almost any statute might carry — from its **table of contents**,
OCR'ing scans twelve pages deep. 1,287 → 249 → about 110.

**A keyword filter cannot substitute for the first stage.** Over the same titles it
reached 5 of the 8 Malaysian gold statutes and scored `CYBER SECURITY ACT 2024` at
**exactly 0.00**, because those two words appear in none of our indicator phrases.

Three properties are deliberate and will not change without a reason on the record:

- **An Act we could not read is KEPT.** Over the size cap, no text layer, no English PDF —
  those mean "not judged", never "judged irrelevant".
- **OCR noise never reaches the judge.** A judge shown noise answers "nothing relevant",
  which is indistinguishable from a real verdict and would delete the Act silently.
- **A KNOWN instrument never depends on being judged relevant.** The filter cut Income Tax
  Act 1967 and Service Tax Act 2018, both gold; the backstop restored both and said so in
  the log.

The 1,287 verdicts ship in `data/reference/my_enum_verdicts.jsonl`, so a fresh clone
reaches the same working set **without an API key and without paying to re-judge**.

### All three Round-1 economies in one file

```bash
LEXORA_LIVE=1 LEXORA_OCR=1 python scripts/run_submission.py -j all --budget 20 \
    --verify-clauses --rationale-llm --metadata-llm --amendment-llm --llm-workers 16
```

| Flag | What it adds |
| :---- | :---- |
| `--verify-clauses` | **The relevance decision.** Per-clause 9-in-1 LLM judge: each candidate clause is judged yes/no against *all* indicators at once. This is what `main.py` uses by default. |
| `--verify-cells` | Older per-(clause × indicator) verifier. Tightening only. |
| `--rationale-llm` | LLM-written Mapping Rationale (the quote itself is still copied verbatim from the parsed clause, never generated). |
| `--metadata-llm` | LLM-assisted Law Number / Last Amended extraction, source-verified. |
| `--amendment-llm` | Amendment-currency adjudication (flags provisions whose cited law was since amended). |
| `--secondary` | Use third-party trackers to *guide* discovery. They are never cited. |

### Other entry points

```bash
lexora discover -j sg                                        # discovery only (search → rank)
lexora demo -j sg --url <pdf-or-html-url> --browser          # map one live URL
lexora demo -j sg --pdf path/to/act.pdf --source-url <url>   # map a local PDF (bypass crawler)

python scripts/eval_discovery.py    # discovery eval vs the official legal inventory
python scripts/eval_mapping.py      # section-level mapping eval vs the provision gold
```

---

## Architecture — six stages

```
┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌────────┐   ┌─────────┐
│ collect  │ → │ extract  │ → │ structure  │ → │ classify │ → │  cite  │ → │ export  │
└──────────┘   └──────────┘   └────────────┘   └──────────┘   └────────┘   └─────────┘
  crawl +        OCR/HTML/PDF   article/§ tree   BM25 retrieval  verbatim     CSV (13-col),
  discovery     + offsets       (schedule-aware) + LLM verifier  quote check  JSON-LD
```

| Stage | Package | Description |
| :---- | :---- | :---- |
| Collect | `src/lexora/collect/` | Portal crawler, autonomous discovery + ranking, per-portal strategies (SG SSO browser, AU OData, MY Fess), hashing; `collect/secondary/` = third-party trackers used only to *guide* discovery, never cited |
| Extract | `src/lexora/extract/` | HTML / PDF text / OCR (RapidOCR), verbatim char-offset invariant |
| Structure | `src/lexora/structure/` | Format-general legal parser (dotted SG/MY, spaced AU, civil-law articles), schedule-aware |
| Classify | `src/lexora/classify/` | Clause retrieval (BM25 default, dense opt-in) + constrained LLM verifier |
| Cite | `src/lexora/cite/` | Citation builder, verbatim validator, mapping rationale, metadata |
| Export | `src/lexora/export/` | Official 13-column CSV + JSON-LD |

See [`docs/architecture.md`](docs/architecture.md) and
[`docs/anti_hallucination.md`](docs/anti_hallucination.md) for the verbatim contract.

---

## Swapping the LLM (No Vendor Lock-in)

Lexora talks to **any OpenAI-compatible endpoint** — swap the model by changing `.env`
only, no code change. The LLM is optional: with no endpoint configured, the pipeline
degrades gracefully to BM25 + the verbatim validator.

```ini
# .env — OpenAI / vLLM / llama.cpp / Ollama all expose an OpenAI-compatible /v1
LEXORA_LLM_BASE_URL=http://localhost:11434/v1     # Ollama; or https://api.openai.com/v1
LEXORA_LLM_MODEL=llama3                            # or gpt-4o, Qwen2.5-7B-Instruct, …
LEXORA_LLM_API_KEY=not-needed-for-local            # real key for cloud
```

The client is abstracted in `src/lexora/classify/llm_client.py`. The config value reaches
**every** lane — `pick_one`, `per_cell` and the per-clause judge that produces essentially
all of our output. `LEXORA_BRUTE_MODEL` exists to pin a *different* model for the judge, but
it is an override: leave it unset and the judge follows `LEXORA_LLM_MODEL` like everything
else. (It did not always. Until 2026-07-28 it defaulted to a hardcoded vendor model name, so
following the instructions above pointed a local Ollama at a model it had never heard of and
every judgement 404'd. `tests/test_provenance.py::test_config_swap_reaches_every_llm_lane`
now pins this lane by lane.)

### Open-source fallback (if commercial API)

Swapping the endpoint works. **Swapping in a small open-weight model does not preserve the
output**, and we would rather show the measurement than imply otherwise. Same document
(MY PDPA), same 90-clause pool, same gold:

| lane | gold recall | API calls | wall | cost |
| :---- | ----: | ----: | ----: | ----: |
| GLM-5.2, 9-in-1 (default) | **85 %** | 90 | 36 s | ¥1.13 |
| **BM25 + boundary rules — no model at all** | **38 %** | **0** | **0.0 s** | **¥0** |
| GLM-4.5-Flash, one yes/no per indicator | 15 % | 810 | 317 s | ¥0 |
| GLM-4.5-Flash, 9-in-1 | 8 % | 90 | 59 s | ¥0 |
| Qwen3.5-35B-A3B, 9-in-1 | 12 % | 90 | 38 s | ¥0.65 |
| DeepSeek-V4-Flash, 9-in-1 | 6 % | 90 | 34 s | ¥0.06 |

Two things follow, and neither is a slogan.

**The 9-in-1 question is the problem, and decomposing it helps.** Asking a model to hold
nine long indicator definitions at once and return the right *subset* is a multi-label task
over a ~3.5k-token context. Small models collapse to one or two answers regardless of the
clause. Asked one indicator at a time — a yes/no with a single definition in front of it —
the same free model roughly doubles (8 % → 15 %, 1 → 8 citations). The trade is 9× the
calls, which is the wrong trade for a metered frontier model and the right one for a
self-hosted server where only wall-clock is spent. `scripts/bench_judge.py --binary`
reproduces it.

**But our open-source fallback is not a small model — it is no model.** The deterministic
lane (BM25 retrieval + per-indicator boundary rules + the verbatim validator) reaches 38 %
on the same gold with zero API calls and zero seconds, beating every small model we could
test by more than 2×. It needs no key, no server, no GPU and no weights, it is what
`--no-llm` has always run, and since 2026-07-28 the pipeline **falls back to it
automatically** when the judge is systematically unreachable — with `DEGRADED: LLM judge
unavailable` written into every affected row, so a degraded run can never be mistaken for a
judged one.

**What we did not test:** llama3 itself. We have no local Ollama host in this environment,
so the open-weight numbers above come from small *hosted* models on an OpenAI-compatible
endpoint. A 70B-class local model may well land between the free tier and GLM-5.2; we are
not claiming otherwise, only reporting what we measured. The swap mechanism is exercised by
tests; the quality claim is scoped to the models in the table.

## Swapping the OCR Engine

| Engine | Config | Notes |
| :---- | :---- | :---- |
| RapidOCR (ONNX) | `LEXORA_OCR_ENGINE=rapidocr` (default) | Bundled weights, no API key. CPU by default; **`pip install -r requirements-gpu.txt` moves it to an NVIDIA GPU (~5x)** |
| PaddleOCR | `LEXORA_OCR_ENGINE=paddleocr` | GPU-server path (`paddlepaddle-gpu`) |

OCR is enabled with `LEXORA_OCR=1` (on by default in `main.py` and `run_submission.py`) and
language-configured with `LEXORA_OCR_LANG`. `LEXORA_OCR_GPU=0` pins it to the CPU.

---

## Output Format

Every run writes **both** deliverable files, named `<Economy>_P<pillar>_<timestamp>`.

### CSV — official 13 columns, exact order

Header text and order match `OUTPUT_TEMPLATE_31MAY.xlsx` byte for byte (judges validate
programmatically — do not rename or reorder).

| # | Column | Required | Notes |
| :---- | :---- | :---- | :---- |
| 1 | `Economy` | Required | Official UN country name |
| 2 | `Law Name` | Required | Full official statute name + year |
| 3 | `Law Number / Ref` | Optional | e.g. `Act 709` |
| 4 | `Last Amended` | Required | Year; blank if never amended |
| 5 | `Indicator ID` | Required | RDTII code, e.g. `P6-I4` |
| 6 | `Article / Section` | Required | Article **and** paragraph, e.g. `s. 26(1)` |
| 7 | `Discovery Tag` | Required | `NEW` = found independently; `KNOWN` = in the sample kit |
| 8 | `Location Reference` | Optional | PDF page no. \| HTML anchor |
| 9 | `Verbatim Snippet` | Required | Copied from the parsed clause — never model-generated |
| 10 | `Mapping Rationale` | Optional | Why it maps, max 300 chars |
| 11 | `Source URL` | Required | Direct link on the official portal |
| 12 | `Confidence` | Optional | 0.00–1.00 |
| 13 | `Notes` | Optional | OCR issues, bilingual source, cross-references |

### JSON — same rows, richer metadata

One flat object per provision (JSON rows == CSV rows), carrying what CSV cannot hold:
`source_pdf_path`, `pdf_is_scanned`, `ocr_quality_cer`, `processing_time_seconds`,
`model_version`, `retrieval_method`, `raw_context_before` / `raw_context_after`.

`scripts/run_submission.py` additionally emits a JSON-LD dump and a `.summary.json` run
report (instruments discovered, NEW/KNOWN split, indicators covered).

---

## Actual Cost Per Document

*Required by the hackathon rubric for UN sustainability assessment. **Measured from real
runs, not estimated** — token counts come from the live LLM client's accounting, page counts
from the parsed PDF, wall-clock from a timer around the run. Reproduce any row:*

```bash
PYTHONPATH=src python tools/cost_logger.py \
  --pdf data/raw/my/<hash>.pdf --economy my --pillar 6 \
  --price-in 1.180 --price-cached 0.295 --price-out 4.130   # GLM-5.2 list prices: fresh / cached / output
# writes logs/cost_report.json and prints the table
```

Everything except the LLM runs locally, so **the LLM is the only metered component**. We
benchmark **two** documents, because the cost profile of a text PDF and a scanned one differ
in where the time goes (not in what they cost):

| | **Text PDF** | **Scanned PDF** |
| :---- | :---- | :---- |
| Document | Malaysia **PDPA 2010** (Act 709) | Malaysia **Computer Crimes Act 1997** (Act 563) |
| Size | 95 pages · 148,091 chars | 12 pages · 572 embedded images · **0-char text layer** |
| OCR | not needed (text layer present) | **required** — RapidOCR on GPU (`rapidocr:1.4.4+cuda`) |
| Clauses judged | 91 | 12 |
| Citations produced | 11–18 (see *Determinism*) | 1 |
| LLM calls | 103–110 | 14 |
| Input tokens (**of which cached**) | ~345,000 (**48–51%**) | ~43,600 (**21–49%**) |
| Output tokens | ~11,000 | ~1,300 |
| **Wall-clock** | **77–86 s** | **40–46 s** |
| **Cost, first run** | **$0.29 – $0.31** | **$0.038 – $0.049** |
| **Cost, re-run** (verdict cache) | **$0.031** | ~**$0.000** |
| **Cost (open-weight swap)** | **$0.000** | **$0.000** |

Ranges, not point estimates, and both spreads are real. The **cache hit rate** is best-effort
on the provider's side and varies run to run, which moves the cost. The **citation count**
varies because the model does — see below.

| Component | Engine used | Metered? | Cost |
| :---- | :---- | :---- | :---- |
| Crawling | self-hosted (httpx / Playwright) | no | $0.0000 |
| OCR | RapidOCR — PP-OCR on ONNX Runtime, GPU (`rapidocr:1.4.4+cuda`) | no | $0.0000 |
| Embedding | BAAI/bge-m3, local (dense channel off by default) | no | $0.0000 |
| Parsing / retrieval | BM25, local | no | $0.0000 |
| **LLM mapping** | **GLM-5.2** (relevance judge + rationale + metadata) | **yes** | **$0.038 – $0.31** |

**Measured on:** 2026-07-14, Python 3.12.2 in a clean venv from the pinned
`requirements.txt`, `--llm-workers 16`, verdict cache bypassed (`LEXORA_JUDGE_CACHE=0`) so
these are true cold costs. **LLM:** GLM-5.2 via an OpenAI-compatible gateway.

### The three rates behind the bill

GLM-5.2 meters input in two tiers, plus output — and the cached tier, at a quarter of the
fresh-input price, is the one worth engineering for:

| | ¥ / 1M | $ / 1M @ 6.78 |
| :---- | ----: | ----: |
| Input, fresh | ¥8 | $1.180 |
| **Input served from the prompt cache** | **¥2** | **$0.295** |
| Output | ¥28 | $4.130 |

The relevance judge asks about **one clause against all nine indicators**, so every call
carries the same 3,051-token indicator catalogue — **95% of the prompt is identical from one
call to the next**. We put that catalogue **first** and the clause **last**, so the shared
text lands in the provider's cacheable prefix and ~50% of all input tokens bill at the ¥2
rate. Reverse the order — clause first — and the prefix breaks, costing ~40% more for
byte-identical work.

`tools/cost_logger.py` reports `cached_input_tokens` and prices all three rates. Only the
*rates* are parameters (`--price-in`, `--price-cached`, `--price-out`), so a judge can
re-price our measured token counts against any model.

### Determinism: your numbers will not exactly match ours, and here is why

**Read this before comparing your output to our submitted CSV.**

The relevance decision is an LLM judgement, and **the LLM is not deterministic even at
`temperature=0`**. We ran the same 95-page Act through the same code three times, cold, and
got **11, 14 and 18 citations**. Nothing in the pipeline changed between runs — same prompt,
same retrieval pool, same clauses. Frontier MoE backends simply do not guarantee a
reproducible sample, and GLM-5.2 is one.

We are telling you this rather than quietly hoping you run it once:

* **Our submitted CSV is one sample**, not a fixed point. A re-run will produce a similar,
  not identical, set of provisions.
* **What IS stable** is everything the LLM does not decide: which laws are discovered, which
  are fetched, how they are parsed into clauses, which clauses are retrieved into the pool,
  and the verbatim text of every snippet (sliced from the source by character offset — the
  model never writes text that ships). Re-run those and they reproduce exactly.
* **Where the variance lands** is the marginal, weakly-supported provisions — the clauses a
  human annotator would also argue about. The flagship mappings (MY PDPA s.129 → P6-I4,
  AU APP 8 → P6-I4) come back every time.

**The verdict cache below removes this variance for anyone re-running our work.**

### The verdict cache

A per-clause verdict is a pure function of `(model, the rendered prompt, clause text)`.
Nothing else reaches the judge. So verdicts are cached in `data/cache/judge.sqlite`, keyed by
a hash of exactly that — **edit an indicator's long definition, reword the instructions,
reorder the prompt, swap the model, or re-parse a clause one character differently, and the
model is asked again.** Backend failures are never cached (caching a failure would turn one
outage into a permanently missing citation).

It was built to cut cost — a re-run of the text benchmark drops from **$0.31 to $0.031** and
from 86 s to 19 s. But its more important effect is the one above: **it pins the first
verdict, so a re-run reproduces the previous output exactly instead of resampling.** It is
the difference between a pipeline you can re-run and one you can only run.

- `LEXORA_JUDGE_CACHE=0` bypasses it entirely — that is how the cold costs above were measured,
  and how you reproduce them.
- Every run prints its split (`N served from cache / M judged by the LLM`) and the cost report
  carries `judge_cache`, so a warm run can never be passed off as a cold one.
- The cache ships **empty** (gitignored). Your first run pays the cold cost and judges every
  clause itself; your second run is free and identical to your first.

### The rationale cache

The Mapping Rationale column has the same shape of problem and now has the same answer. A
rationale is a pure function of `(model, system prompt, rendered user prompt)`, so it is
cached in `data/cache/rationale.sqlite` under a hash of exactly that, with the same rule:
change how you ask and the model is asked again; failures are never stored.

This mattered more than the money. Without it, `--rationale-llm` turned a full Malaysia run
from 554 s into 2,545 s — every rationale a fresh live call, with no cache to replay from.
The flag was therefore dropped from the full-economy commands, and a full Singapore run was
demonstrated with a Mapping Rationale column that was 181 of 181 deterministic template.
**A layer that cannot be replayed is a layer that gets switched off**, and then the output
column that a human actually reads is the one that quietly degrades.

- What is stored is the model's **raw answer**, not the rationale we accepted. The guards
  (length, score talk, the 6-word verbatim-copy check) re-run on every read, so tuning a
  guard can be evaluated across the whole corpus without paying for a single new call. Cache
  the accepted answer instead and a guard change would leave every stored row untouched —
  the experiment would measure nothing and look like it worked.
- `LEXORA_RATIONALE_CACHE=0` bypasses it; `LEXORA_RATIONALE_CACHE_PATH` moves it.
- Every run prints the hit/miss split, and prints `OFF` when there is no cache at all.

**Open-weight swap = $0.000 per document.** OCR, embedding, parsing, retrieval and crawling
are already self-hosted; pointing `LEXORA_LLM_BASE_URL` at a local Ollama/vLLM server (see
*Swapping the LLM*) removes the only metered call. Compute only, no API spend.

**On GPU.** OCR dominates wall-clock on scanned corpora, so it runs on the GPU when a CUDA
runtime is present — measured **5.6x** faster than CPU, with identical recognised text.
The CPU wheel is the default (see `requirements-gpu.txt` to switch). The engine reads back
the providers its sessions **actually** got and only calls itself `+cuda` when detection,
classification and recognition are all on CUDA; the `ocr_engine` field in the JSON sidecar
carries that name, so a silent CPU fallback is impossible. `LEXORA_OCR_GPU=0` forces CPU.

### Cost log excerpt (`logs/cost_report_scanned.json`)

```json
{
  "document": "MY_ComputerCrimesAct1997_Act563.pdf",
  "measured_on": "2026-07-14",
  "pages": 12,
  "ocr":       { "engine": "rapidocr:1.4.4+cuda", "pages": 12, "scanned": true, "cost_usd": 0.0 },
  "embedding": { "model": "BAAI/bge-m3", "tokens": 0, "cost_usd": 0.0 },
  "llm":       { "model": "GLM-5.2", "calls": 14, "failed_calls": 0,
                 "input_tokens": 43560, "cached_input_tokens": 8966, "cache_hit_rate": 0.206,
                 "output_tokens": 1281,
                 "price_in_per_1m_usd": 1.18, "price_cached_in_per_1m_usd": 0.295,
                 "price_out_per_1m_usd": 4.13,
                 "cost_usd": 0.0488 },
  "judge_cache": { "clauses_judged_by_llm": 12, "clauses_served_from_cache": 0, "enabled": false },
  "total_cost_usd": 0.0488,
  "total_cost_usd_open_weight_swap": 0.0,
  "citations": 1,
  "processing_time_seconds": 40.3
}
```

`clauses_served_from_cache: 0` is what makes this a cold measurement — every one of the 12
clauses was judged by the model. A re-run reports the split the other way and costs ~$0.00.
`failed_calls` is reported separately because a rejected call bills nothing and returns
nothing: a run whose every request was refused would otherwise print a tidy `$0.0000` table
and look like a bargain.

---

## Project layout

```
Lexora/
├── main.py            # ← reviewer entry point: --economy <name> --pillar <6|7|all>
├── requirements.txt   # pinned runtime deps (engine + LLM client + OCR + browser)
├── .env.example       # copy to .env; LLM endpoint + feature switches
├── src/lexora/
│   ├── models/        # Pydantic data contracts (citation, clause, source, indicator, secondary)
│   ├── collect/       # Stage 1 — crawler, discovery, per-portal strategies, secondary/ trackers
│   ├── extract/       # Stage 2 — HTML / PDF text / OCR
│   ├── structure/     # Stage 3 — legal parser
│   ├── classify/      # Stage 4 — retrieval + LLM verifier + boundary rules + lifecycle
│   ├── cite/          # Stage 5 — citation builder, validator, rationale, metadata
│   ├── export/        # Stage 6 — CSV / JSON-LD
│   ├── storage/ api/  # SQLite + object store; FastAPI app
│   └── config.py      # env-driven runtime config (LexoraConfig)
├── configs/
│   ├── jurisdictions/        # one YAML per economy (sg, au, my, _template)
│   ├── rdtii_indicators.yaml # all 12 pillars defined; runs default to the 9 P6/P7 indicators
│   ├── secondary_sources.yaml
│   └── eval/                 # gold inventory, mapping gold, intrinsic parser fixtures
├── submission/        # the CSV + JSON we submitted, and the run that produced them
├── scripts/           # run_submission.py (Round-1 entry), run_pipeline.py, eval_*.py
├── tools/             # cost_logger.py — the measured cost-per-document benchmark
├── docs/              # architecture, anti_hallucination, citation_schema, jurisdiction_profile
└── tests/
```

Add a new economy by writing one YAML under `configs/jurisdictions/` — see
[`docs/jurisdiction_profile.md`](docs/jurisdiction_profile.md).

---

## Known Limitations

Honest by design — these guide where to be cautious.

- **The relevance judgement is not reproducible run-to-run.** The same Act, judged three
  times by the same code at `temperature=0`, yielded 11 / 14 / 18 citations. This is the
  backend, not the pipeline (see *Determinism* above). The variance sits in marginal
  provisions; flagship mappings are stable. The verdict cache pins a run's verdicts so
  re-runs are exact, and a sampling-and-vote judge (union over N samples, as
  `classify/brute_judge.py` already does for discovery) would narrow it at ~N× the cost —
  **not yet done for the per-clause judge.**
- **Recall is bounded by what the judge sees, not by the parser.** Retrieval pools the top
  `LEXORA_MAP_POOL_K` (default 40) clauses per indicator and the LLM judge decides
  membership over that pool. A relevant provision ranked below the pool is never judged.
  Measured retrieval ceiling on our flagship gold: pool 3 → 38%, 20 → 76%, 40 → 95%.
- **Confidence is relative, not calibrated:** it is a normalised retrieval score, not a
  probability. Treat < 0.80 as review-flagged.
- **The model's own knowledge is a poor amendment clock.** `LEXORA_METADATA_RECALL=1`
  (opt-in, off by default) asks the model what it knows about a named law in a *separate*
  call with none of the document in front of it, and annotates a row only where that
  contradicts what was read from the text — it never becomes an answer. Measured on six
  laws: act numbers recalled correctly 5/5 and an invented Act correctly declined, but
  `last_amended` was declined on 3 of 5 real laws. So it is a useful *contradiction*
  channel and not a source of amendment dates. Agreement between the two is not evidence
  either: both once agreed on a wrong year for Malaysia's Food Act 1983.
- **Delegated legislation:** the engine retrieves principal statutes and discovers
  amendments, but does not exhaustively follow cross-references into subordinate
  regulations.
- **Australia portal anti-bot:** `legislation.gov.au` applies cumulative per-IP throttling
  and can serve an HTML decoy under load; mitigated with a per-host throttle and
  `--serial-fetch`.
- **Bilingual corpora:** Malaysia's portal mixes English and Malay; non-English provisions
  in a mixed PDF may be missed. Non-Latin scripts (CJK, Thai) are tokenised but not yet
  validated against gold.
- **Secondary sources are never citable:** third-party trackers only *guide* discovery;
  they never become a citation.
- **Windows, non-ASCII paths:** `pip` decodes `requirements.txt` with the *locale* codec,
  so we keep that file pure ASCII (a stray em dash makes `pip install -r` die with
  `UnicodeDecodeError` on a cp936/cp932 box). An editable install (`pip install -e .`) also
  breaks under a non-ASCII home directory, because the `.pth` file is written UTF-8 and read
  back as cp936. The Quick Start path (`pip install -r requirements.txt`) is unaffected.

---

## Running the Test Suite

```bash
pytest                       # offline; 449 tests, no network, no API key
LEXORA_LIVE=1 pytest -m live # live portal tests
```

---

## Team

Team **Verbatim Trade** — Bohan LIU, Chiheng JIN, Wenxiang SHI, Yiying TANG, Zijie Oscar WEI.

## License

Apache 2.0 — see [`LICENSE`](LICENSE). All required runtime dependencies are permissively
licensed.

## Key Dates

| Date | Milestone |
| :---- | :---- |
| **20 July 2026** | **Round 1 submission deadline** |
| 31 July 2026 | 20 teams shortlisted |
| 3 August 2026 | Live online pitch |
| 5 August 2026 | 5 finalists announced |
| October 2026 | Grand Finale — Bangkok |

Built for the [UN ESCAP × KMITL Global Hackathon on AI for Digital Trade Regulatory Analysis](https://unitescap.medium.com/global-hackathon-on-using-ai-for-digital-trade-regulatory-analysis-3c6213ddffa3) (2026).
