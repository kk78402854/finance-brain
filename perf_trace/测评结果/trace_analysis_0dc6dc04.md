# Trace 分析报告：finance-perf-demo（0dc6dc04）

> 数据来源：Langfuse 导出文件 `C:\Users\26833\Downloads\trace-0dc6dc049034ba5cb73df3087478e79a.json`
> 分析对象：`traceName = finance-perf-demo`，共 16 个 observation，0 个 score
> 说明：下文所有时间、token、条数均直接取自该 JSON；JSON 中没有的信息一律标注「需补充」，不做推断性赋值。

---

## 1. 整体结构

| 项 | 值 |
|---|---|
| trace 名（JSON 中 `traceName`） | `finance-perf-demo` |
| 根 span 名 | `LangGraph` |
| trace ID | `0dc6dc049034ba5cb73df3087478e79a` |
| Session ID | `class-eaf422e119e24cf48ff92943d4605e0a` |
| User ID | 空（demo 脚本未设置 `user_id`，需补充） |
| Tags | `classroom`、`single-query` |
| Environment / Version | `default` / 空 |
| 用户问题（metadata.user_query） | 易方达智造优势混合基金主要投资什么方向？ |
| 开始时间（UTC） | 2026-11-06T08:59:25.855Z（北京时间 16:59:25.855） |
| 结束时间（UTC） | 2026-11-06T08:59:58.169Z |
| 总耗时 | **32.314 s** |
| 是否有报错 | 无（16 个 span 全部 `level=DEFAULT`，无 `statusMessage`） |

### 1.1 调用树（缩进=父子关系，括号内为该 span 自身耗时）

```
LangGraph (32.314s)
├── node_item_name_confirm (8.558s)
│   ├── RunnableSequence (1.465s)
│   │   ├── ChatOpenAI / qwen-flash (1.460s)
│   │   └── JsonOutputParser (0.001s)
│   └── after_node_item_name_confirm (0.002s)
├── node_search_embedding (0.761s)          ← 三路并行
├── node_search_embedding_hyde (2.829s)     ← 三路并行
│   ├── RunnableSequence (1.334s)
│   │   ├── ChatOpenAI / qwen-flash (1.330s)
│   │   └── StrOutputParser (0.001s)
├── node_web_search_mcp (4.875s)            ← 三路并行
├── node_rrf (0.013s)
├── node_rerank (16.398s)
└── node_answer_output (2.444s)
    └── ChatOpenAI / qwen-flash (2.373s)
```

7 个业务节点全部出现，顺序与 [main_graph.py](file:///D:/dev/workspace/finance-brain/processor/query_processor/main_graph.py) 一致：主体确认 → 三路检索（并行）→ RRF → Rerank → 答案输出。

### 1.2 时序（相对 trace 开始的秒数）

| 区间(s) | 发生的事 | 耗时 |
|---|---|---|
| 0.003 – 3.397 | `node_item_name_confirm` 前半段，**无子 span**（代码上是 Mongo 取历史会话） | 3.394s |
| 3.397 – 4.862 | 主体识别 + 查询重写 LLM 调用 | 1.465s |
| 4.862 – 8.558 | 同一节点后半段，**无子 span**（代码上是 BGE-M3 向量化 + Milvus 主体名检索 + 写用户消息） | 3.696s |
| 8.566 – 13.443 | 三路检索并行：embedding 0.761s / HyDE 2.829s / 联网 4.875s | 4.877s（由最慢分支决定） |
| 13.445 – 13.458 | RRF 融合 | 0.013s |
| 13.460 – 29.858 | Rerank 重排 | 16.398s |
| 29.868 – 32.312 | 答案生成 | 2.444s |

---

## 2. 关键路径与耗时构成

按占总时长（32.314s）排序：

| 排名 | 环节 | 耗时 | 占比 | 是否在关键路径 | 说明 |
|---|---|---|---|---|---|
| 1 | `node_rerank` | 16.398s | **50.75%** | 是 | 只有一个节点 span，内部无子 span，无法区分「模型加载」与「打分」 |
| 2 | `node_item_name_confirm` 内未埋点的两段 | 3.394 + 3.696 = **7.090s** | **21.94%** | 是 | 分别是 Mongo 历史读取、BGE-M3 向量化 + Milvus 检索 |
| 3 | `node_web_search_mcp` | 4.875s | **15.09%** | 是（三路中最慢） | 拉起 RRF 必须等它结束 |
| 4 | `node_answer_output` | 2.444s | 7.56% | 是 | 其中 LLM 2.373s |
| 5 | `node_search_embedding_hyde` | 2.829s | 8.75% | **否** | 与联网检索并行，被 4.875s 完全遮盖 |
| 6 | `node_search_embedding` | 0.761s | 2.36% | 否 | 被遮盖 |
| 7 | `node_rrf` | 0.013s | 0.04% | 是 | 可忽略 |

**串并行判断**：三路检索开始时间相差 2ms（8.566 / 8.567 / 8.568），结束于 9.327 / 11.396 / 13.443，确认为真并行；并行段合计 4.877s，若串行则需 8.465s，并行已省下约 3.6s。

**空等（无 span 覆盖的时间）**：32.314s 中有 7.090s（21.94%）落在 `node_item_name_confirm` 内部但没有子 span，只能靠代码推断是 Mongo 与 Milvus/向量化，**trace 本身无法证明**。这是当前埋点最大的盲区。

---

## 3. LLM 调用明细

共 3 次 LLM 调用，全部为 `qwen-flash`，`temperature=0.1`：

| # | 所属节点 | 用途 | 输入 token | 输出 token | 合计 | cache 命中 | 耗时 |
|---|---|---|---|---|---|---|---|
| 1 | node_item_name_confirm | 提取主体名 + 重写问题（JSON 输出） | 330 | 36 | 366 | `input_cache_read=0` | 1.460s |
| 2 | node_search_embedding_hyde | HyDE 生成假设答案 | 148 | 134 | 282 | 0 | 1.330s |
| 3 | node_answer_output | 基于 6 条参考切片生成最终答案 | 2926 | 164 | 3090 | 0 | 2.373s |
| **合计** | — | — | **3404** | **334** | **3738** | **0** | **5.163s** |

- 3 次调用提示词互不相同，**没有发现同一段上下文被重复送入**。
- 单次请求 token 量约 3.7k，其中 78% 集中在最后一次答案生成。
- 全部 generation 的 `completionStartTime` 为空 → 本次是**非流式**调用（demo 脚本 `is_stream=False`），因此**首 token 延迟无法从这条 trace 得到**，标注为「需补充」。
- `costDetails` 为空、`totalCost=null` → 未配置模型价格，**金额成本无法计算**（需补充单价）。
- `input_cache_read` 三次均为 0，说明本轮没有命中提示词缓存（是否可命中取决于供应商策略，需补充确认）。

---

## 4. 异常与浪费

### 4.1 没有发现的问题（先排除）

- 无报错、无重试、无空返回；`scores` 为空数组（没有人工/自动评分）。
- 无重复 LLM 调用。

### 4.2 发现的浪费点（均有 JSON 证据）

1. **联网检索 4.875s，结果基本没用上（15.09%）**
   - `node_web_search_mcp` 返回 `web_search_docs` 共 3 条；RRF 输出的 5 条 `rrf_chunks` **全部是 `type=milvus`**，web 结果是在 rerank 阶段才被合并进来的。
   - 最终 `reranked_docs` 6 条里只有 1 条 `type=web`，排名第 6（score 0.8432，前 5 条 milvus 为 1.1974~0.9429）。
   - 最终答案的 `<ctx-meta>` 只引用了两份「向量数据库」来源，**没有引用这条 web 结果**。
   - 即：这道题上，15% 的总耗时换来的是一条没进答案的上下文。

2. **Rerank 16.398s 占比过半，但不可归因**
   - 节点内代码包含「超长切片交给 LLM 压缩」的分支（[node_rerank.py](file:///D:/dev/workspace/finance-brain/processor/query_processor/nodes/node_rerank.py#L89-L104)），而本 trace **没有任何 GENERATION 子 span**，说明 6 条切片都未超 512 token 阈值，**压缩分支未触发**，16.398s 全是本地 reranker 的 tokenize + 打分。
   - 本次是 `python perf_trace/trade_demo.py` 单次脚本运行，reranker 是进程内懒加载（[reranker_utils.py](file:///D:/dev/workspace/finance-brain/utils/lm/reranker_utils.py#L26-L38)），**16.398s 很可能包含一次性的模型冷加载**。这一点 trace 无法区分（没有子 span），必须再跑一次同进程的热调用才能确认，标注为「需验证」。

3. **同一章节被切成 3 片并同时进上下文（重复上下文）**
   - `reranked_docs` 前 3 名是同一个小节的连续切片：`…投资目标与投资策略_1 / _2 / _3`，正文分别 593 / 490 / 265 字符，合计 1348 字符。
   - 6 条参考内容总字符 2688，**其中 50.1% 来自同一小节的重复内容**；答案生成输入 2926 token 里因此有约三分之一是重复信息。

4. **上下文元信息在 rerank 阶段丢失**
   - `step_2_merge_rrf_and_web` 只保留 `title/text/score/type/url`，丢掉 `chunk_id / item_name / file_title / parent_title`（[node_rerank.py](file:///D:/dev/workspace/finance-brain/processor/query_processor/nodes/node_rerank.py#L56-L71)）。
   - 证据：`reranked_docs` 里 6 条记录的 `chunk_id` 全为 `null`；答案提示词中的来源信息显示为 `来源信息:[无]`；最终 `<ctx-meta>` 只能拼出长标题。
   - 附带影响：无法按 `chunk_id` 去重、无法在前端精确展示文件名/章节。

5. **可缓存但没缓存**
   - 答案生成的系统提示词（身份定义 + 规则 + 输出格式）每次请求都固定约 1.2k token，本轮 `input_cache_read=0`，说明没吃到任何缓存收益。

---

## 5. 结论：优化建议（按性价比排序）

### 建议 1｜先拆解并压掉 `node_rerank` 的 16.398s（占比 50.75%）

- 做法：给 rerank 节点补 3 个子 span——`tokenize`、`model_load`、`compute_score`；若确认冷加载占大头，则服务启动时预热模型；若确认是纯打分慢，再考虑换 GPU / ONNX / 减少待排文档数（当前 6 条）。
- 预期收益：若冷加载占一半，常驻服务下该节点可直接降到 8s 以内，端到端从 32.3s 降到约 24s（-25%）；若属打分本身慢，收益取决于硬件，需补充 GPU 型号后再评估。
- 验证指标（Langfuse）：同一 `traceName=finance-perf-demo` 下 `node_rerank` 的耗时，连跑两次对比「首次 / 第二次」；以及同节点新增子 span 的耗时拆分。

### 建议 2｜给联网检索加超时/按需触发（占比 15.09%，且本轮基本无用）

- 做法：`node_web_search_mcp` 设 2s 超时并降级为空结果；或改成「知识库召回不足（如 rrf 条数 < 阈值 或 top1 分数低于阈值）才触发」。
- 预期收益：最坏情况下端到端减少约 2.9s（4.875 → 2.0），并省掉一次 MCP 调用；本题场景可直接省 4.875s（-15%）。
- 验证指标（Langfuse）：`node_web_search_mcp` 的 p95 耗时；`reranked_docs` 中 `type=web` 的条数与占比；答案 `<ctx-meta>` 引用 web 来源的比例。

### 建议 3｜修掉 rerank 阶段的元信息丢失 + 同章节切片合并

- 做法：`step_2_merge_rrf_and_web` 透传 `chunk_id / item_name / file_title / parent_title`；对 `parent_title` 相同的相邻切片做合并/去重后再送入 rerank 与答案生成。
- 预期收益：上下文从 6 条（2688 字符）压到 4 条左右（约 1700 字符），答案生成输入 token 从 2926 降到约 2000（-30%），同时恢复可追溯的来源展示。这既省钱又改善答案质量（减少重复内容干扰）。
- 验证指标（Langfuse）：`node_answer_output` 那次 generation 的 `usageDetails.input`；`reranked_docs` 的唯一 `parent_title` 数量；`chunk_id` 非空率。

### 建议 4｜补齐非 LLM 环节的子 span（当前 21.94% 时间不可解释）

- 做法：给 Mongo 历史读取、BGE-M3 向量化、Milvus 检索、reranker 打分分别加 observation（`openinference`/`@observe` 或手动 span）。
- 预期收益：本身不降时延，但把 7.09s 的黑盒变成可归因项，是后续所有优化的前提；否则每次只能猜。
- 验证指标（Langfuse）：`node_item_name_confirm` 内子 span 覆盖率（有子 span 的时间 / 节点总耗时），目标 >90%。

### 建议 5｜降低 LLM 侧的固定开销（1.46s + 1.33s，约 500 token）

- 做法：开启供应商侧提示词缓存并确认命中（当前 `input_cache_read=0`）；HyDE 分支对「问题已含完整产品名」的场景跳过（本题 `item_names` 已精确命中「易方达智造优势混合基金」，HyDE 的边际收益低）。
- 预期收益：HyDE 分支被 4.875s 的联网分支遮盖，跳过它**不减少端到端时间**，只省 148+134 token 与一次调用；真正的时间收益要靠建议 2 把联网分支压下来后才体现。缓存命中的收益需补充供应商缓存规则后才能估算。
- 验证指标（Langfuse）：`input_cache_read` > 0 的比例；`node_search_embedding_hyde` 调用次数。

> 排序理由：建议 1、2 直接砍关键路径上的时间（合计 65.8% 的时间）；建议 3 同时降 token 与提升可追溯性；建议 4 是让后续优化「有的放矢」的前提；建议 5 收益最小且依赖外部策略。

---

## 6. 与 Locust 压测的衔接

这条 trace 是**单用户、非流式**脚本调用，只能作为基线，不能代表并发表现。压测时需要盯的字段：

| 观察项 | 从哪里取 | 关注的指标 |
|---|---|---|
| 端到端时延 | Locust `03 complete answer`（或 trace 根 span `LangGraph`） | P50 / P95 / P99；本基线 32.314s |
| 首段答案延迟 | Locust `02 first answer delta`（流式时才有） | P95；本 trace 为非流式，**基线缺失，需补充** |
| 提交接口时延 | Locust `01 submit /query` | P95（应远小于端到端，仅代表入队） |
| 节点耗时漂移 | Langfuse 各 `node_*` span | p95；重点看 `node_rerank`、`node_web_search_mcp`、`node_answer_output` |
| 吞吐 | Locust RPS + Langfuse trace 数/分钟 | 稳态 QPS、失败率 |
| token 消耗 | generation `usageDetails.input/output` | 每请求 token（基线 3404 in / 334 out / 3738 total）、总 token/s |
| 缓存命中 | `usageDetails.input_cache_read` | 命中率（基线 0） |
| 错误率 | Langfuse `level=ERROR` + `statusMessage`；Locust failures | 失败率、超时数 |
| 检索质量 | `reranked_docs` 条数与 `type` 分布 | web 占比（基线 1/6）、top1 分数（基线 1.1974） |

压测注意：
- 查询服务端口是 **8002**（不是文档里的 8001）；`/query` 响应体只有 `session_id`、**没有 `task_id`**，文档 4.1 的脚本需按本项目实际接口改写。
- 流式链路（`is_stream=True`）才能产出首 token 指标，压测应走流式，否则「首段答案延迟」无从统计。
- Langfuse 上报是异步的，压测结束后要等几十秒再做统计，且高并发下 SDK 有 `flush_at / flush_interval` 批量发送，trace 条数可能出现短暂滞后。

---

## 7. 与接入预期的差异 & 排查方向

| 现象 | 判断 | 排查方向 |
|---|---|---|
| `traceName=finance-perf-demo`，不是 `finance-query-api` | **符合预期但覆盖不全**：这条是 [trade_demo.py](file:///D:/dev/workspace/finance-brain/perf_trace/trade_demo.py) 脚本链路 | Web 链路需另起服务后调 `/query` 验证，应出现 `finance-query-api`；导入链路应出现 `finance-import-api`（本次完全没有，因未跑导入） |
| 根 span 名是 `LangGraph` 而不是 trace 名 | 正常：Langfuse 用 LangGraph 根 run 命名根 span，`traceName` 单独覆盖 trace 列表展示名 | 看 UI 时以 Trace 列表的 `finance-perf-demo` 为准，展开后根 span 叫 `LangGraph` 属预期 |
| 节点内大量时间无子 span | **与本次最小化接入一致**：只挂了 `CallbackHandler`，只覆盖 LangChain/LangGraph 调用 | 需要补自定义 span（见建议 4） |
| 无 ERROR、无 score | 正常：本轮是成功路径，且未做评分 | 压测时按 `level` 过滤错误，评分需另行接入 |
| 无成本、无 TTFT、无 user_id | 数据缺失，非接入错误 | 成本需在 Langfuse 配置模型价格；TTFT 需流式调用；user_id 需在 `propagate_attributes` 传入 |

---

## 8. 待补充信息（JSON 中没有，不做臆测）

1. 模型单价与计费口径（`costDetails` 为空、`totalCost=null`）。
2. reranker 运行设备型号与算力（JSON 只有节点耗时，无硬件信息）。
3. reranker 冷加载耗时（需在常驻进程中再跑一次对比）。
4. 首 token 延迟（需流式调用）。
5. Langfuse 项目内的 trace 截图/页面数据（本报告完全基于导出 JSON）。
