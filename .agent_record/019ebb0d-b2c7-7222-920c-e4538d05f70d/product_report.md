# Lexora 项目产品说明（全局版）

日期：2026-06-12

## 1. 一句话总结

Lexora 是一个面向 UN ESCAP × KMITL Hackathon 的数字贸易法规分析系统。它的目标不是做一个普通聊天机器人，而是自动从各国法规门户发现法律文件，抽取可审计的逐字引用，并把这些引用映射到 RDTII 2.1 指标体系中。

项目当前已经进入“可跑 Round-1 原型”的状态：

- SG / AU / MY 三个 Round-1 经济体的主链路已经基本打通。
- 系统可以自动发现法规、拉取全文、解析条文、检索相关条款、生成逐字引用，并输出官方 submission CSV / JSON-LD。
- 已经有 NEW / KNOWN 证据标记、监管机构附属文书连接器、语义检索层、mapping eval、full submission run。
- 最近新增并验证了 LLM verifier：它不是生成答案，而是审核候选证据，删除弱证据或把边界案例送到人工复核。

产品定位可以这样讲：

> Lexora 是一个“可审计法规证据引擎”。它先用确定性 pipeline 找到原文证据，再用受控 LLM 做证据审核，而不是让 LLM 直接写法规结论。

## 2. 项目要解决的问题

Hackathon 主题是用 AI 分析数字贸易法规，并映射到 RDTII 框架。普通 AI / RAG 系统在这个场景下有几个硬伤：

- 容易幻觉：生成看似合理但原文不存在的结论。
- 容易 paraphrase：不是逐字引用，评审无法回到原文核对。
- 容易引用失效法律或错误条文。
- 容易把“关键词相似”的条文错映射到指标上。
- 对亚太多法域、多语言、多门户结构的适配不足。

Lexora 的差异化策略是：

- 每条 claim 必须绑定到具体法规文件、页码/锚点、条文路径和 canonical span。
- 没有 canonical span 就不输出 claim。
- LLM 不写最终引用文本，只做候选证据审核。
- 输出保持可复查：CSV / JSON-LD / review status / NEW-KNOWN 标签。

这对 Hackathon 很关键，因为它让项目不只是“能生成答案”，而是“能证明答案来自哪里”。

## 3. 当前整体实现情况

### 3.1 数据模型和基础架构

已完成：

- 项目 Python package 和基础目录结构。
- Pydantic 数据模型：
  - source / raw document
  - clause / canonical span
  - citation / evidence claim
  - RDTII indicator
- jurisdiction profile schema。
- SG / AU / MY 三个 Round-1 economy profile。
- RDTII Pillar 6 + Pillar 7 的 regulatory indicators。
- 官方 13 列 submission CSV exporter。
- JSON-LD exporter。

产品意义：

系统已经有可复用的数据契约，不是一次性脚本。后续加国家、加门户、加 review UI 都有明确接口。

### 3.2 法规发现和全文获取

已完成：

- `lexora discover -j <iso>`：按经济体做法规发现。
- `lexora map -j <iso>`：端到端自动 map。
- SG：
  - Singapore Statutes Online。
  - headless browser 渲染。
  - 对 SSO 空壳页做检测和重试。
- AU：
  - Federal Register / OData API。
  - 过滤非 Act 子文书。
  - AU act catalogue + semantic crosswalk。
- MY：
  - Laws of Malaysia / Fess-Solr proxy。
  - Act 编号去重。
  - dropped TLS / transport error 重试。
- 两段式全文解析：
  - 先找 landing/search result。
  - 再 resolve 到真实 PDF / full-text document。
- 多文书 mapping：
  - 不再只取 top-1 法规。
  - 一个指标可以召回一组相关文书。
  - 每条输出带 NEW / KNOWN。

产品意义：

Lexora 已经能从“给一个 URL 才能跑”的 demo，升级到“给一个国家代码，系统自己找法规”的 demo。这是项目的核心自动化能力。

### 3.3 条文解析和逐字引用

已完成：

- HTML / PDF 文本抽取。
- 通用条文 parser：
  - SG / MY dotted numbering。
  - AU spaced numbering。
  - 自动识别编号风格。
- Schedule-aware 解析：
  - 解决正文 section 和 Schedule 内编号撞号问题。
  - 支持 AU Privacy Act Schedule 1 内的 APP 条文寻址。
  - 让 APP 8 这种关键条文可以被逐字引用。
- clause_id 唯一性兜底。
- citation validator：
  - quote 必须来自 canonical span。
  - 不接受 LLM paraphrase。

产品意义：

这个能力是 Lexora 区别于普通 RAG 的核心：输出不是“模型说法”，而是“可回到原文的逐字证据”。

### 3.4 检索、语义层和 mapping

已完成：

- BM25 条文检索。
- 可选 dense semantic layer。
- BM25 + dense RRF fusion。
- BM25 rank-1 anchor，避免 dense 把正确 top-1 挤掉。
- section-targeted synonym。
- mapping eval：
  - small gold set。
  - hit@1 / hit@3。
  - BM25 vs fusion A/B。
- AU P6-I4 已通过 APP 8 可寻址 + targeted synonym 拉到 rank-1。

已有阶段性结果：

- 文档记录中，生产相关 hit@1 从 2/7 提升到 3/7。
- AU P6-I4 从“APP 8 不可寻址”推进到“APP 8 可寻址且可命中”。

产品意义：

系统不是只追求“发现很多文件”，也开始衡量“映射到的条文是否是正确条文”。这让后续优化有数字依据，而不是凭感觉调 prompt 或关键词。

### 3.5 附属文书和 NEW 证据

已完成监管站连接器：

- SG PDPC advisory guidelines。
- MY PDP sectoral codes of practice。
- AU OAIC APP guidelines / PIA / data breach guidance。

产品意义：

Hackathon 评分里 NEW evidence 很重要。很多有价值证据不在成文法门户，而在监管机构站点、指南、守则、标准里。Lexora 已经开始覆盖这类资料，不只局限于旗舰法。

### 3.6 Full submission run

已完成：

- `scripts/run_submission.py`
- 支持：
  - `-j sg|au|my|all`
  - `--budget`
  - `--verify`
  - `--dry-run`
  - `--out`
- 输出：
  - 官方 13 列 submission CSV。
  - JSON-LD。
  - per-economy summary。
- 每行带 NEW / KNOWN。
- 单个经济体失败不会让整轮完全丢失。

已有文档记录：

- 此前 live `-j all` 已能落地 `submission_round1.csv`。
- 当时记录为 103 行 citation，SG / AU / MY 均能跑出结果。

产品意义：

这已经是 Round-1 deliverable prototype：不是单点 demo，而是可以生成评审需要看的 CSV 产物。

### 3.7 LLM verifier

最近阶段完成：

- LLM API 接入。
- `.env` 读取 API key / endpoint URL。
- `.env.example` 读取 model。
- OpenAI-compatible endpoint 兼容：
  - explicit output-token limit。
  - JSON mode fallback。
  - retry。
  - fenced JSON / 包裹 JSON 解析。
- `lexora map --verify` 真实启用 verifier。
- `scripts/run_submission.py --dry-run --verify` 能显示 verifier readiness。
- verifier backend error 会明确提示，不会静默回退。

设计边界：

- LLM 不写引用文本。
- LLM 不新增条文。
- LLM 只在 retriever 找到的候选中选。
- `match` 输出 verified。
- `uncertain` 进入 `CONFLICT_REVIEW`。
- `no_match` / 幻觉 id / backend error 不输出 claim。

小规模真实 A/B：

| 指标 | 不开 verifier | 开 verifier |
| --- | ---: | ---: |
| citation rows | 4 | 1 |
| covered indicators | 4 | 1 |
| `CONFLICT_REVIEW` rows | 0 | 1 |

解释：

- MY budget-3 下，verify-off 输出 4 条。
- verify-on 删除 3 条弱/错映射。
- 保留 1 条 P7-I1，但标为 `CONFLICT_REVIEW`，表示需要人工确认。

产品意义：

verifier 已经不只是“代码存在”，而是接真 endpoint 跑出了一次有效 A/B。它的定位是质量安全门，不是生成器。

## 4. 当前项目成熟度判断

### 可以对外展示的能力

- 自动发现 SG / AU / MY 法规。
- 自动获取全文。
- 自动解析条文。
- 自动映射到 RDTII Pillar 6 / 7 指标。
- 输出逐字引用。
- 输出官方 submission CSV / JSON-LD。
- 标记 NEW / KNOWN。
- 使用 regulator guidance 扩展证据来源。
- 使用 LLM verifier 审核候选证据。
- 将不确定 mapping 送到 review。

### 可以作为 Hackathon demo 主线

建议 demo 叙事：

1. 选择一个经济体，比如 Malaysia 或 Australia。
2. Lexora 自动发现相关法规。
3. 系统解析法规并抽取条文。
4. 系统生成逐字引用，不让 LLM 编写 quote。
5. 开启 LLM verifier，对候选引用做审核。
6. 展示 verifier 删除弱证据，并把边界案例送到 `CONFLICT_REVIEW`。
7. 导出 submission CSV。

这条 demo 主线比较有说服力，因为它展示的是可审计流程，而不是普通 AI 问答。

### 不建议过度承诺的能力

目前不要说：

- 已经覆盖所有亚太经济体。
- 多语言完全可用。
- OCR 已经完成。
- verifier 已经适合默认开启。
- 所有 mapping 都准确。
- Review UI 已经可用。

更准确的说法：

> Lexora 的 Round-1 三经济体主链路已打通，核心可审计机制成立；接下来要扩大 A/B 和稳定性测试，并补多语言泛化和人工 review 工具。

## 5. 当前测试情况

### 5.1 自动化测试

最新结果：

```text
131 passed, 4 skipped
```

覆盖内容包括：

- 配置读取。
- `.env` / `.env.example` 行为。
- LLM client endpoint 兼容。
- verifier 安全边界。
- backend error 不崩溃。
- SG browser shell 检测与重试。
- MY / AU transport retry。
- submission summary。
- parser / mapping / citation 等既有模块。

### 5.2 代码规范检查

ruff 检查通过：

```text
All checks passed!
```

### 5.3 真实 endpoint 测试

已完成：

- endpoint/model smoke test。
- parseable JSON 返回验证。
- `scripts/run_submission.py --dry-run --verify` readiness 验证。
- MY budget-3 verify-off / verify-on A/B。

### 5.4 已有 live / eval 记录

来自项目进展文档：

- SG / AU / MY live submission run 曾成功跑出 Round-1 CSV。
- discovery coverage 已有多轮 live 记录。
- mapping eval 已有 hit@1 / hit@3 记录。
- Schedule-aware parser 已在真实 AU Privacy Act 上验证 APP 可寻址。

## 6. 测试是否完整

不完整，但已经从“代码能不能跑”推进到了“产品主链路能不能验证”的阶段。

比较完整的部分：

- 单元测试和离线测试。
- 三经济体主链路的已有 live 记录。
- CSV / JSON-LD exporter。
- parser / verifier / config / retry 机制。
- 小规模真实 verifier A/B。

仍然不足的部分：

- LLM verifier 只跑了 MY budget-3，样本太小。
- 没有最新的 SG / AU / MY budget-20 verifier A/B。
- 没有连续多轮稳定性测试。
- 没有足够人工抽检来估算误删率。
- 没有系统统计成本、耗时、retry 次数。
- `CONFLICT_REVIEW` 还没有 review UI。
- 多语言 / 非拉丁法域还没验证。
- OCR pipeline 尚未完成。

产品层面的判断：

> 当前已达到“Hackathon 原型可演示 + 小规模真实验证”的阶段，但还没达到“默认生产化开启所有能力”的阶段。

## 7. 下一步大规模测试计划

### Step 1：三经济体 full A/B

目标：验证 LLM verifier 在 Round-1 真实规模下的收益。

建议命令：

```bash
LEXORA_ENV_FILE=/home/ubuntu/scratch/swx/Lexora/.env .venv/bin/python scripts/run_submission.py -j all --budget 20 --out outputs/submission_round1_verify_off.csv
LEXORA_ENV_FILE=/home/ubuntu/scratch/swx/Lexora/.env .venv/bin/python scripts/run_submission.py -j all --verify --budget 20 --out outputs/submission_round1_verify_on.csv
```

需要统计：

- 总 citation 行数。
- 每个国家 citation 行数。
- 覆盖指标数。
- NEW / KNOWN。
- `CONFLICT_REVIEW` 行数。
- verifier 删除的行数。
- verifier 保留的行数。
- backend error 数。
- 总耗时。

### Step 2：人工抽检

建议抽检：

- 每个国家至少 10 条被 verifier 删除的证据。
- 每个国家至少 5 条 verifier 保留的证据。
- 所有 `CONFLICT_REVIEW`。

判断问题：

- verifier 删除的是弱证据，还是误删好证据？
- verifier 标为 uncertain 的案例是否确实值得人工 review？
- 被保留的 claim 是否真正支持指标？

### Step 3：连续 3 轮稳定性测试

目标：确认 endpoint 和 crawler 不只是偶然成功。

建议：

```bash
for i in 1 2 3; do
  LEXORA_ENV_FILE=/home/ubuntu/scratch/swx/Lexora/.env .venv/bin/python scripts/run_submission.py -j all --verify --budget 20 --out outputs/submission_round1_verify_on_run${i}.csv
done
```

观察：

- 每轮输出是否稳定。
- endpoint error 是否出现。
- crawler error 是否出现。
- SG SSO 是否超时。
- 耗时和成本是否可接受。

### Step 4：更新 demo 素材

如果大规模 A/B 结果好，可以把以下内容放进 demo：

- verify-off / verify-on 对比表。
- verifier 删除弱证据的例子。
- `CONFLICT_REVIEW` 的例子。
- 逐字引用回到原文的例子。
- NEW evidence 的例子。

## 8. 后续产品路线

### 8.1 短期：Round-1 打磨

目标：把 SG / AU / MY 的 submission 原型打磨稳定。

重点：

- 跑 budget-20 full A/B。
- 做人工抽检。
- 修 SG / AU / MY 已知 mapping miss：
  - SG P7-I1 / P7-I4。
  - MY P7-I1。
  - AU P7-I1。
- 修 SG SSO `networkidle` / live crawl 稳定性。
- 形成可展示 summary：
  - discovered。
  - fetched。
  - citation rows。
  - NEW / KNOWN。
  - review rows。

### 8.2 中期：Review 和质量闭环

目标：让系统不只是自动输出，也能支持人审。

方向：

- 简单 review UI。
- `CONFLICT_REVIEW` 队列。
- verifier rationale 展示。
- 人工 accept / reject / needs better source。
- 每次 run 的质量 dashboard。
- 错误样例回流到 mapping gold。

### 8.3 后期：泛化到更多经济体

目标：从“深做三国”变成“能扩展到更多亚太经济体”。

优先验证：

- NZ：英语普通法迁移。
- TH 或 CN：非拉丁 / 大陆法压力测试。

需要补的基础能力：

- Unicode-aware tokenizer。
- 多语言 embedding，如 BGE-M3。
- 多脚本 legal parser：
  - `Article N`
  - `第N条`
  - 泰文 `มาตรา N`
  - 俄文 `Статья N`
- LLM 辅助生成跨语言同义词。
- LLM 辅助草拟 jurisdiction YAML profile。

### 8.4 OCR 和扫描件

目前 OCR pipeline 还没完成。

后续需要：

- OCR 抽取。
- OCR confidence。
- 低置信度引用送 review。
- OCR 噪声下的 verbatim fuzzy 校验。

## 9. 风险和应对

| 风险 | 当前状态 | 应对 |
| --- | --- | --- |
| LLM 幻觉 | 已通过设计限制 | LLM 不写 quote，只返回 clause_id |
| 错指标映射 | 仍存在 | verifier + mapping eval + gold 扩充 |
| 误删好证据 | 尚未大规模验证 | budget-20 A/B + 人工抽检 |
| 门户不稳定 | 部分已有 retry | SG networkidle 仍需修 |
| 非英语泛化差 | 已识别 | G-track：Unicode tokenizer / BGE-M3 / 多脚本 parser |
| OCR 缺口 | 未完成 | 后续 pipeline |
| review 成本 | UI 未完成 | 先用 CSV / JSON-LD + `CONFLICT_REVIEW` 队列 |

## 10. 当前产品判断

当前整个项目不是“只做了一个 LLM verifier 小功能”，而是已经形成了一个比较完整的可审计法规映射原型：

- 有三国法规发现。
- 有全文抓取。
- 有条文解析。
- 有检索和语义层。
- 有逐字引用。
- 有官方 CSV 输出。
- 有 NEW evidence。
- 有 mapping eval。
- 有 LLM verifier。
- 有小规模真实 A/B。

但它还处于 Hackathon prototype 阶段，不是最终产品。

接下来最重要的不是继续堆新功能，而是做三件事：

1. 扩大验证规模：SG / AU / MY budget-20 full A/B。
2. 建立质量闭环：人工抽检 + review queue + mapping gold 扩充。
3. 准备泛化：NZ + TH/CN blind test，决定多语言基础设施优先级。

如果这些跑通，Lexora 的 Hackathon 叙事会比较清楚：

> 我们不是让 LLM 直接生成法规判断；我们让系统找到逐字证据，再让 LLM 做受控审核。这样既能自动化，也能保持可审计。
