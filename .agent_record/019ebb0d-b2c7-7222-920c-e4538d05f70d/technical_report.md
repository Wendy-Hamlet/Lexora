# Lexora LLM Verifier 技术报告

日期：2026-06-12

## 1. 本阶段目标

本阶段目标不是重写 Lexora 主流程，而是把已经存在的 constrained LLM verifier 真正接入可运行的 OpenAI-compatible endpoint，并跑出至少一组可量化 A/B 结果。

验收重点：

- 运行时能从 `.env` / 环境变量读取 API key 和 URL。
- model 能按当前约定从 `.env.example` 读取。
- `--verify` 能真正启用 LLM verifier，而不是静默回退到 BM25 + verbatim。
- endpoint 异常不能崩掉 pipeline，也不能静默伪装成正常 verifier 结果。
- 至少产出一组 verify-off / verify-on 对比和简短评估记录。
- 不泄露真实 API key 或 endpoint URL。

代码实现位于独立 worktree：

`/home/ubuntu/scratch/swx/worktree_Lexora/llm-env-integration-019ebb0d`

分支：

`codex/llm-env-integration-019ebb0d`

## 2. 当前已实现内容

### 2.1 运行时配置接入

核心文件：

- `src/lexora/config.py`
- `.env.example`

已实现：

- 支持自动解析本地 `.env`。
- 支持通过 `LEXORA_ENV_FILE=/path/to/.env` 显式指定配置文件。
- 支持原生 Lexora 变量：
  - `LEXORA_LLM_BASE_URL`
  - `LEXORA_LLM_API_KEY`
  - `LEXORA_LLM_MODEL`
- 支持 OpenAI-compatible 常见变量作为 fallback：
  - `OPENAI_BASE_URL`
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL`
- 按当前项目实际约定处理优先级：
  - `.env` 提供真实 URL / API key。
  - `.env.example` 只允许提供 model 默认值，例如 `LEXORA_LLM_MODEL=gpt-5.4`。
  - `.env.example` 不作为完整运行时配置层，避免占位 URL/key 或 OCR 等其他默认值污染真实运行配置。
- 增加 LLM 运行参数：
  - `LEXORA_LLM_MAX_TOKENS` / `OPENAI_MAX_TOKENS`，默认 512。
  - `LEXORA_LLM_MAX_RETRIES` / `OPENAI_MAX_RETRIES`，默认 2。

这样做的原因是当前 endpoint/model 在没有显式 token 上限时会返回 400 或空 content；设置 `max_completion_tokens` 后可以返回正常内容。

### 2.2 OpenAI-compatible LLM client 兼容性增强

核心文件：

- `src/lexora/classify/llm_client.py`

已实现：

- 使用 OpenAI SDK 连接 OpenAI-compatible endpoint。
- 每次 chat completion 默认携带 `max_completion_tokens`，避免 endpoint 返回空 content。
- 对 structured output 做兼容处理：
  - 先尝试 `response_format={"type": "json_object"}`。
  - 如果 endpoint 返回 400 或空 content，则 fallback 到普通 chat completion。
  - 如果普通 chat 返回 fenced JSON、前后带解释文字的 JSON，client 会提取第一个 JSON object。
  - 如果多次尝试后仍为空或不可解析，抛出 `LlmResponseError`。
- `LlmResponseError` 不会让 pipeline 崩溃；verifier 会计数并报告。

这解决了当前 endpoint 的几个实际兼容问题：

- 要求 prompt 中出现小写 `json`。
- JSON mode 下可能返回空 content。
- 不设置 token 上限时可能返回空 content 或 400。
- 某些模型输出可能不是裸 JSON，而是 Markdown code fence 或带解释文本的 JSON。

### 2.3 Verifier 安全边界和错误可见性

核心文件：

- `src/lexora/classify/verifier.py`
- `src/lexora/cli.py`
- `scripts/run_submission.py`

已有 verifier 设计边界保持不变：

- verifier 只在 retriever 给出的候选 clause 中选择。
- verifier 只返回 `clause_id` / `label` / `confidence` / `rationale`。
- verifier 不生成最终 quote 文本。
- quote 仍然来自 canonical span，继续满足 “No canonical span, no claim”。

新增：

- `Verifier` 记录：
  - `error_count`
  - `last_error_type`
- `lexora map --verify` 如果遇到 verifier backend error，会打印明确 warning。
- `scripts/run_submission.py --verify` 非 dry-run 路径中，如果 verifier 构造失败，也会提示用户当前会继续走 BM25 + verbatim。
- `scripts/run_submission.py --dry-run --verify` 现在会实例化 verifier 并打印：
  - `LLM verifier ON (model: gpt-5.4)`

这避免了一个关键风险：用户以为 verifier 已经工作，实际只是静默回退。

### 2.4 当前 A/B 产物

输出目录：

`/home/ubuntu/scratch/swx/worktree_Lexora/llm-env-integration-019ebb0d/outputs`

关键文件：

- `llm_verifier_ab_my_budget3.md`
- `map_my_verify_off_budget3.csv`
- `map_my_verify_off_budget3.jsonld`
- `map_my_verify_on_budget3.csv`
- `map_my_verify_on_budget3.jsonld`

命令：

```bash
LEXORA_ENV_FILE=/home/ubuntu/scratch/swx/Lexora/.env .venv/bin/python scripts/run_submission.py --dry-run --verify
LEXORA_ENV_FILE=/home/ubuntu/scratch/swx/Lexora/.env .venv/bin/lexora map -j my --budget 3 --out outputs/map_my_verify_off_budget3.jsonld
LEXORA_ENV_FILE=/home/ubuntu/scratch/swx/Lexora/.env .venv/bin/lexora map -j my --verify --budget 3 --out outputs/map_my_verify_on_budget3.jsonld
```

结果：

| 指标 | Verify off | Verify on |
| --- | ---: | ---: |
| citation rows | 4 | 1 |
| covered indicators | 4 (`P7-I1`, `P7-I2`, `P7-I4`, `P7-I5`) | 1 (`P7-I1`) |
| `CONFLICT_REVIEW` rows | 0 | 1 |

解释：

- verify-off 输出 4 条候选，全部作为 verified citation。
- verify-on 删除了 3 条弱/错映射：
  - `P7-I2` 落到了 PDPA Section 115(1)，但该条是授权人员搜索/访问 computerized data，不是 cybersecurity framework。
  - `P7-I4` 同样落到 Section 115(1)，不支持 DPO/DPIA requirement。
  - `P7-I5` 落到 Section 32(1)，该条是拒绝 data access request，不是 government access without court order。
- verify-on 保留 `P7-I1` Section 4，但标为 `CONFLICT_REVIEW` / uncertain。这个判断合理：Section 4 是定义条款，能提示个人数据保护框架存在，但不是强证明框架完整性的核心条文。

结论：

这一组小规模 A/B 已经证明 verifier 在当前 endpoint 上真实运行，并能产生有意义的 tightening / review routing 行为。

## 3. 已完成测试

### 3.1 单元与离线测试

最终全量测试：

```text
131 passed, 4 skipped
```

覆盖范围：

- `tests/test_config.py`
  - `.env` 自动读取。
  - `OPENAI_BASE_URL` / `OPENAI_API_KEY` fallback。
  - `.env.example` 只提供 model，不污染运行时默认值。
  - `LEXORA_ENV_FILE` 显式路径。
  - `OPENAI_MAX_TOKENS` / `LEXORA_LLM_MAX_TOKENS`。
  - `OPENAI_MAX_RETRIES` / `LEXORA_LLM_MAX_RETRIES`。
- `tests/test_llm_client.py`
  - JSON mode 空响应 fallback。
  - transient empty response retry。
  - fenced JSON / 带解释文本 JSON 提取。
  - 无法解析时抛 `LlmResponseError`。
- `tests/test_verifier.py`
  - verifier 只选择已有 clause id。
  - hallucinated id 被丢弃。
  - `no_match` abstain。
  - `uncertain` 路由到 review。
  - backend failure 不崩溃。
- `tests/test_robustness.py`
  - `run_submission.py` summary 行为。
  - `--verify` 请求但 verifier 不可用时输出 warning。

### 3.2 Lint / 静态检查

定向 ruff 检查通过：

```text
All checks passed!
```

### 3.3 真实 endpoint smoke test

真实 endpoint/model 测试成功：

- model: `gpt-5.4`
- max tokens: `512`
- max retries: `2`
- minimal JSON smoke check 返回 parseable JSON。

### 3.4 小规模真实 A/B

已完成 MY budget-3：

- verify-off 输出 CSV / JSON-LD。
- verify-on 输出 CSV / JSON-LD。
- 生成 A/B 评估报告。
- 人工抽检 4 条变更样例。
- 没有写入真实 API key 或真实 endpoint URL。

## 4. 测试是否完整

当前测试对“代码接入是否正确”比较完整，但对“产品级质量是否稳定”还不完整。

已比较充分覆盖：

- 配置路径。
- endpoint 兼容性。
- verifier 安全边界。
- backend 错误显式化。
- CLI / submission dry-run 可见性。
- 小规模真实 A/B。

仍不足：

- A/B 只跑了 MY budget-3，样本太小。
- 没有完成 SG / AU / MY 全量 budget-20 比较。
- 没有多轮重复运行，无法评估 endpoint 波动、重试率、耗时、成本。
- 没有系统性人工标注集来计算 precision / recall / false drop。
- 没有把 verifier-on/off 和 mapping gold eval 结合起来看 hit@1 / hit@3。
- 没有覆盖 live crawl 长时间稳定性。
- 没有评估多语言或新经济体泛化。

## 5. 下一步大规模测试计划

### 5.1 Round 1 all-jurisdiction A/B

目标：验证 verifier 在 SG / AU / MY 全链路上的收益和风险。

建议命令：

```bash
LEXORA_ENV_FILE=/home/ubuntu/scratch/swx/Lexora/.env .venv/bin/python scripts/run_submission.py -j all --budget 20 --out outputs/submission_round1_verify_off.csv
LEXORA_ENV_FILE=/home/ubuntu/scratch/swx/Lexora/.env .venv/bin/python scripts/run_submission.py -j all --verify --budget 20 --out outputs/submission_round1_verify_on.csv
```

需要统计：

- citation 行数变化。
- 覆盖指标数变化。
- `CONFLICT_REVIEW` 行数。
- verifier 删除的行数。
- verifier 保留但改为 review 的行数。
- 每个 jurisdiction 的 discovered / fetched / citation / review。
- backend error count。
- 总耗时和平均每个 judgement 耗时。
- API 成本估算。

人工抽检建议：

- 每个 jurisdiction 至少抽 10 条 verify-off 被删除样例。
- 每个 jurisdiction 至少抽 5 条 verify-on 保留样例。
- 所有 `CONFLICT_REVIEW` 全量看一遍，样本通常不会太大。

判定标准：

- verifier 删除的大多数是弱证据或错指标映射。
- verifier 不应大量误删明显正确条文。
- `CONFLICT_REVIEW` 应集中在边界案例，而不是泛滥。
- backend error 应接近 0，且有明确日志。

### 5.2 与 mapping gold eval 结合

目标：不只看 CSV 行数，还看 section-level mapping 质量。

建议：

```bash
.venv/bin/python scripts/eval_mapping.py
```

扩展方向：

- 把 verifier-off / verifier-on 的 section 结果也纳入 hit@1 / hit@3 对比。
- 对 SG P7-I1/P7-I4、MY P7-I1、AU P7-I1 这些已知 miss 建立更明确 gold。
- 记录 verifier 是修正了错误映射，还是因为召回候选不够而无法发挥作用。

### 5.3 稳定性 / 成本测试

建议跑 3 轮相同配置：

```bash
for i in 1 2 3; do
  LEXORA_ENV_FILE=/home/ubuntu/scratch/swx/Lexora/.env .venv/bin/python scripts/run_submission.py -j all --verify --budget 20 --out outputs/submission_round1_verify_on_run${i}.csv
done
```

观察：

- 每轮输出行数是否稳定。
- LLM backend error 是否出现。
- retry 次数是否过高。
- 单次 run 耗时。
- endpoint 是否有 rate limit / timeout。

### 5.4 Live crawl 测试

目标：验证真实门户访问稳定性。

重点：

- SG SSO headless browser 是否仍有 empty shell / networkidle timeout。
- MY Fess/Solr proxy 是否稳定。
- AU OData 是否稳定。
- SOCKS proxy 环境下 `httpx[socks]` 是否稳定。

建议保留每日 summary：

- discovered instruments。
- fetched full texts。
- citation rows。
- NEW / KNOWN。
- failed portals。
- runtime。

## 6. 后续开发扩展点

### 6.1 Verifier 质量提升

- 保存每个 LLM judgement 的非敏感摘要：indicator、candidate clause id、label、confidence、error type。
- 增加 verifier audit log，便于复盘 false drop。
- 增加 prompt version 字段，避免 A/B 不可复现。
- 进一步压缩 prompt，降低 endpoint 空响应概率和成本。
- 引入 token/cost 统计。

### 6.2 Retrieval / mapping 修复

优先处理已知 miss：

- SG P7-I1 / P7-I4。
- MY P7-I1。
- AU P7-I1 fusion 行为。

方向：

- 扩 `configs/eval/mapping_sections.csv`。
- 增加 section-targeted synonyms。
- 对 BM25-only / fusion / verifier-on 做三方对比。

### 6.3 Review UI

`CONFLICT_REVIEW` 已经有数据来源，下一步可以做轻量 review 页面：

- 左侧 indicator。
- 右侧 clause 原文。
- 展示 verifier label / confidence / rationale。
- 人工标记 accept / reject / needs better source。

### 6.4 泛化和多语言

按 `goal.md` 后续计划：

- NZ blind test：英语普通法迁移。
- TH 或 CN blind test：非拉丁 / 大陆法压力测试。
- Unicode tokenizer。
- 多脚本 legal parser。
- 多语言 embedding，如 BGE-M3。

## 7. 当前建议

短期不要把 `--verify` 设为默认开启。当前 MY budget-3 结果积极，但样本太小。

建议下一步按顺序推进：

1. 跑 SG / AU / MY budget-20 verify-off/on A/B。
2. 做人工抽检，确认 verifier 删除的是弱证据而不是好证据。
3. 把 verifier judgement 结果结构化写入 summary。
4. 修已知 mapping miss。
5. 做 live crawl 稳定性测试。
6. 再决定是否在 demo 或 submission path 中默认启用 `--verify`。
