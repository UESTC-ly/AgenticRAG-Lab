# AgenticRAG-Lab 学习指南

一份给新人的渐进式学习路径。整个代码库约 2100 行 Python，按本指南节奏，**4~6 小时**可以完整讲清楚；**1~2 天**可以上手做扩展。

---

## 0. 学习目标

读完这份指南并完成练习后,你应该能:

1. 独立画出 Router → Planner → Executor Loop → Critic → Synthesizer 的完整控制流,并说清每一跳的输入/输出类型
2. 解释 Hybrid Retrieval 里 lexical / semantic / RRF / rerank 各自的作用
3. 说清 agent 如何判断证据是否充足,以及 max_iterations 保护机制
4. 看懂 benchmark 的 EM / F1 / citation rate / latency 是怎么算的
5. 在不破坏原有测试的前提下,自己加一个 Retriever 或替换一个 Critic 实现

---

## 1. 前置知识

| 类别 | 要求 |
|---|---|
| Python | ≥ 3.11,熟悉 `dataclass` / `Enum` / `Protocol` 风格 |
| NLP 基础 | 知道什么是 BM25、向量检索、Reranker(不用会实现) |
| RAG 概念 | 知道 Retrieval-Augmented Generation 的基本动机即可 |
| 工具链 | `unittest`(本项目**不用 pytest**)、命令行基本操作 |

**一句话背景**: Agentic RAG = 普通 RAG + 一个会自己决定「要不要再检索一次」的 agent loop。这个项目用最小代价实现了这个 loop,方便阅读学习和做 benchmark。

---

## 2. 学习路径(五个阶段)

### Stage 1: 把它跑起来(15 分钟)

目标: 看到输出,建立直觉。

```bash
# 1. 跑单元测试,确保环境 OK
PYTHONPATH=src python3 -m unittest discover -s tests -v

# 2. 跑一个多跳问题,观察 trace 输出
PYTHONPATH=src python3 -m agentic_rag_lab "Which university did the author of The Hobbit attend before becoming a professor?"

# 3. 跑内置 demo benchmark,看到指标
PYTHONPATH=src python3 -m agentic_rag_lab --benchmark
```

**观察点**:
- 多跳问题的回答里是否带了 citation
- benchmark 表格里的 EM / F1 / latency 数值
- CLI 输出里 `TraceEvent` 的 stage 顺序(这就是 agent loop 的心跳)

---

### Stage 2: 理解数据契约(30 分钟)

目标: 在读控制流之前,先把「组件之间传什么」弄清。

**必读文件**: `src/agentic_rag_lab/models.py` (72 行)

核心类型速查表:

| 类型 | 角色 | 关键字段 |
|---|---|---|
| `RouteTarget` (Enum) | 路由三选一 | `DIRECT_ANSWER` / `AGENTIC_RAG` / `CALCULATOR` |
| `RouteDecision` | Router 输出 | `target`, `rationale` |
| `PlanStep` / `QueryPlan` | Planner 输出 | `steps: list[PlanStep]` |
| `Document` | 原始文档 | `doc_id`, `title`, `content` |
| `EvidenceItem` | 检索结果单位 | `document`, `score`, `query`, `citations` |
| `CritiqueResult` | Critic 输出 | `is_sufficient`, `follow_up_queries`, `confidence` |
| `TraceEvent` | 可观测性事件 | `stage`, `detail` |
| `RunResult` | 整个 `run()` 的最终返回 | `answer`, `citations`, `iterations`, `trace` |

**记忆锚点**: 所有 stage 的 IO 都是这些 dataclass。想扩展任何组件,先问自己:「我的 IO 是否还能用这些类型表达?」如果要改字段,**整个 pipeline + 测试**都要一起动。

---

### Stage 3: 读主循环(45 分钟,最重要)

目标: 彻底吃透 `AgenticRAG.run()`。这是整个项目的骨架。

**必读文件**: `src/agentic_rag_lab/agent.py` (118 行)

推荐方式: 打开文件,**跟着 `TraceEvent(stage=...)` 一路读**,对照 Stage 1 里的 CLI 输出。

#### 主循环骨架(伪代码)

```text
run(query):
    decision = router.route(query)          # → RouteDecision
    emit TraceEvent("router", ...)

    if decision 是 DIRECT_ANSWER:
        return synthesizer.direct_answer(query)
    if decision 是 CALCULATOR:
        return _safe_calculate(query)       # AST 安全执行

    # 进入 agentic_rag 分支
    plan = planner.plan(query)              # → QueryPlan (多条 sub-query)
    evidence_pool: dict[doc_id, EvidenceItem] = {}
    active_queries = [step.query for step in plan.steps]

    for iteration in 1..max_iterations:
        for sub_query in active_queries:
            for item in retriever.search(sub_query, top_k=5):
                # 按 doc_id 去重,保留更高分的证据
                if 更好: evidence_pool[doc_id] = item
        emit TraceEvent("retriever", ...)

        critique = critic.evaluate(query, evidence_pool.values())
        emit TraceEvent("critic", ...)

        if critique.is_sufficient:
            break
        active_queries = critique.follow_up_queries  # 下一轮用新 query

    answer, citations = synthesizer.answer(query, evidence_pool.values())
    return RunResult(...)
```

#### 关键设计点

| 设计 | 在哪一行 | 为什么这样 |
|---|---|---|
| `evidence_pool` 用 `dict[doc_id]` | `agent.py:58, 67-69` | 跨 iteration 去重,保留最高分版本 |
| 按 `score` 比较决定是否替换 | `agent.py:68` | 同一 doc 在不同 sub-query 下分数不同,保留最好命中 |
| `max_iterations` 硬限制 | `agent.py:63` | 防止 critic 一直说「不够」导致死循环 |
| `critique.follow_up_queries` 覆盖 `active_queries` | `agent.py:85` | 下一轮只追新问题,不重跑老 sub-query |
| `_safe_calculate` 用 `ast` 而不是 `eval` | `agent.py:97-117` | 避免任意代码执行;只支持加减乘除 |

#### 自测问题

读完后试着不看代码回答:

1. 如果 Planner 返回 0 个 step,循环会发生什么?
2. 如果同一个文档在两次 iteration 里都被命中,evidence_pool 里是几份?
3. `max_iterations=3` 时,retriever.search 最多被调用几次?(提示: 取决于 plan.steps 和每轮 follow_up_queries 的长度)
4. Critic 返回 `is_sufficient=True` 但 `follow_up_queries` 非空,会发生什么?

---

### Stage 4: 读四个 Stub 组件 + LLM Synthesizer(75 分钟)

按 agent 调用顺序读,每个都很短:

| 顺序 | 文件 | 行数 | 读的重点 |
|---|---|---|---|
| 1 | `router.py` | 56 | 看它用什么规则区分 direct / calculator / agentic。关注 `route()` 方法签名 |
| 2 | `planner.py` | 41 | 看它如何把一个复杂问题拆成 sub-queries。关注 `plan()` 返回结构 |
| 3 | `critic.py` | 103 | **最值得细读**。`evaluate()` 里如何判断证据充足;如何生成 follow-up |
| 4 | `synthesizer.py` | 77 | 规则版:两个入口 `direct_answer()` / `answer()`,带 citation |
| 5 | `llm_synthesizer.py` | ~120 | **真实 LLM 版**:Ollama HTTP 调用、prompt 设计、citation 白名单、三层降级(空证据 / LLM 异常 / 规则 fallback) |

**重要**: Router/Planner/Critic/synthesizer 四个是 **rule-based stub**,`llm_synthesizer.py` 是接入本地 Ollama 的真实 LLM 版本。两者并存,通过 `build_demo_agent()` vs `build_real_agent()` 选择。rule 版保留作 baseline 方便回归 + 做消融 A/B;真实版是生产路径。

保持它们的**公开方法签名稳定**,这样 `AgenticRAG` 永远不用改:

```python
router.route(query) -> RouteDecision
planner.plan(query) -> QueryPlan
critic.evaluate(query, evidence) -> CritiqueResult
synthesizer.direct_answer(query) -> str
synthesizer.answer(query, evidence) -> tuple[str, list[str]]
```

---

### Stage 5: 读检索层(90 分钟)

目标: 理解 Hybrid Retrieval 的每一层,包括真实模型栈。

| 文件 | 行数 | 角色 |
|---|---|---|
| `retrieval/lexical.py` | 47 | BM25-like 词频打分 |
| `retrieval/semantic.py` | 35 | 离线语义代理 baseline(没有真 embedding,用 token 展开近似) |
| `retrieval/embedding_semantic.py` | ~80 | **真实 dense retriever**:`sentence-transformers` + `bge-small-en-v1.5`,预计算 doc embedding,numpy 点积打分 |
| `retrieval/reranker.py` | ~65 | **真实 cross-encoder reranker**:`bge-reranker-base`,对 (query, doc) 成对打分 |
| `retrieval/hybrid.py` | ~75 | **核心**: RRF 融合 + 可选 reranker 注入 |
| `text.py` | 84 | 公共工具: tokenize、RRF、token expand |

#### HybridRetriever 两种模式(对照 `hybrid.py`)

**模式 A — 无 reranker(默认,离线):**

1. 双路召回:`lexical.search(q, top_k*2)` + `semantic.search(q, top_k*2)`
2. RRF 融合:`reciprocal_rank_fusion(rank)` 按名次给分
3. 加权合并:lexical 贡献原分 1%,semantic 贡献 50%
4. 轻量 rerank:query 和 doc 做 token overlap 加分

**模式 B — 有 reranker(生产):**

1. 双路召回 top_k × `recall_multiplier`(默认 5 倍,给 rerank 更宽的候选池)
2. RRF 融合生成候选列表
3. **Cross-encoder 精排**:`reranker.rerank(query, candidates, top_k=top_k)`

**共同契约**:

```python
retriever.search(query: str, top_k: int) -> list[EvidenceItem]
reranker.rerank(query: str, items: list[EvidenceItem], *, top_k: int | None) -> list[EvidenceItem]
```

任何新 retriever / reranker 只要实现这个方法,就能塞进 `HybridRetriever` 或直接替换给 `AgenticRAG`。

#### Bi-encoder vs Cross-encoder(必掌握)

| | Bi-encoder (embedding) | Cross-encoder (reranker) |
|---|---|---|
| 做什么 | 把 query、doc 独立编码成向量,做内积 | 把 (query, doc) 拼起来一起过模型,直接输出相关性分数 |
| 何时用 | 召回(对全库) | 精排(对 top_k 候选) |
| 成本 | O(N) 前向(一次 encode 全库可缓存) | O(k) 前向每次查询,无法缓存 |
| 精度 | 一般 | 明显更高 |
| 项目中的模型 | `bge-small-en-v1.5` (384 维) | `bge-reranker-base` |

---

## 3. 评测闭环(30 分钟)

目标: 看懂 benchmark 是怎么打指标的。

| 文件 | 职责 |
|---|---|
| `evaluation/metrics.py` (23) | EM、token F1 算法 |
| `evaluation/runner.py` (78) | 遍历样本、聚合 latency / citation rate |
| `evaluation/hotpotqa.py` (163) | HotpotQA 上的 retrieval baseline / ablation |
| `demo.py` | 连线示例: 构造 agent + 调 runner |

**动手验证**:

```bash
# 内置 demo 数据的 benchmark
PYTHONPATH=src python3 scripts/run_eval.py

# 规则版 retrieval 对比(lexical / semantic / hybrid)
PYTHONPATH=src python3 scripts/prepare_hotpotqa_slice.py --limit 500
PYTHONPATH=src python3 scripts/run_hotpotqa_retrieval_ablation.py --top-ks 1,3,5 --format markdown

# 真实模型完整消融(bge-small + bge-reranker,MPS 加速)
pip install -r requirements-real.txt
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python3 scripts/run_hotpotqa_real_ablation.py \
  --device mps --top-ks 1,3,5,10 --format markdown
```

完整结果看 `doc/BENCHMARK_RESULTS.md`。

**看结果时的问题**:

- 为什么 hybrid 不一定在所有 top_k 下都赢?
- citation rate 低意味着什么?(答案生成时没有引用证据 → synthesizer 有问题,或证据不相关)
- latency 瓶颈在 retriever 还是 critic?

---

## 4. 产品层(可选,90 分钟)

只在你想做 UI / API 改动时读。

| 文件 | 行数 | 内容 |
|---|---|---|
| `mvp.py` | 469 | Service 层: 用户文档、知识库、历史、运行日志 |
| `web.py` | 464 | 用标准库写的 HTTP server + HTML 前端 |
| `chunking.py` | 29 | 长文档自动切片 |

启动本地 MVP:

```bash
PYTHONPATH=src python3 -m agentic_rag_lab serve \
  --slice data/processed/hotpotqa/dev_slice.jsonl \
  --user-store data/product/user_documents.jsonl \
  --kb-store data/product/knowledge_bases.json \
  --port 8000
```

浏览器打开 `http://127.0.0.1:8000`,界面上的每个面板都对应 `web.py` 里的一条路由。

---

## 5. 动手练习(从易到难)

完成前三个就算真正上手了。

### Level 1 — 改参数(10 分钟/题)

1. 把 `AgenticRAG` 的 `max_iterations` 从 3 改成 1,跑 benchmark,看 F1 下降多少
2. 在 `HybridRetriever` 里把 lexical 权重从 `0.01` 改成 `0.5`,观察对 HotpotQA ablation 的影响
3. 给 `demo.py` 的 corpus 加一条新文档,加一道相关多跳问题,观察能否正确回答

### Level 2 — 加新组件(30 分钟/题)

4. 实现一个 `RandomRetriever`: `search()` 随机返回 `top_k` 个文档。跑对比,看 F1 跌多少 —— **这是你的 baseline**
5. 在 `critic.py` 里加一个 `min_evidence_count` 参数,证据数 < 阈值就一定判不足

### Level 3 — 替换 stub(1~2 小时/题)

6. 把 `router.py` 改成基于关键词长度 + 数字占比的启发式,跑 benchmark 对比
7. 在 `synthesizer.answer()` 里加一个规则: 如果答案里没出现任何 citation 里的 doc_id,就标记为 `low_confidence`
8. 实现一个新 CLI: `python -m agentic_rag_lab trace <query>` 只打 trace 不打答案

### Level 4 — 接真组件(半天起步)

9. 用 `rank_bm25` 重写 `LexicalRetriever`,保持 `.search()` 契约不变
10. 接入 `sentence-transformers` 做真 embedding,重写 `SemanticRetriever`
11. 接一个本地 Ollama / LM Studio,把 `Critic.evaluate()` 换成 LLM 调用

---

## 6. 关键设计决策回顾

把这些点串起来,就是整个系统的设计逻辑:

1. **问题**: 单轮 RAG 在多跳问题上召回不全 → 需要 agent loop
2. **Router**: 简单问题不走 agent,省资源
3. **Planner**: 把复杂 query 拆成 sub-queries(为什么拆而不是直接检索? → 单查询召回有限)
4. **Hybrid Retrieval**: lexical 抓关键词、semantic 抓语义、RRF 融合名次、rerank 微调
5. **Critic**: 证据不够就追问,给 agent 自我纠错能力;`max_iterations` 防死循环
6. **Synthesizer**: 带 citation 输出,便于验证

**刻意为之的 trade-off**:

- 为什么不用 LangChain? → 可解释性优先,手写状态机看得见每一跳
- 为什么 Critic 是规则? → 保证离线可跑,也让 loop 逻辑和 LLM 调用解耦
- 为什么 RRF 而不是 score weighted sum? → 不同检索器的 score 不可比,名次才可比
- 为什么 evidence_pool 按 doc_id 去重? → 同一文档可能被多条 sub-query 命中,要保留最佳分数

---

## 7. 推荐阅读顺序总清单

```text
Day 1 (核心,3~4 小时):
  README.md
  PROJECT_DESCRIPTION.md (设计动机)
  CLAUDE.md (工程约定)
  → Stage 1: 跑起来
  → Stage 2: models.py
  → Stage 3: agent.py (反复读)
  → Stage 4: router/planner/critic/synthesizer
  → Stage 5: retrieval/*
  → tests/test_agentic_rag.py 对照看

Day 2 (评测 + 扩展,3~4 小时):
  → evaluation/*
  → scripts/run_hotpotqa_retrieval_*
  → 练习 Level 1 + Level 2
  → 如有时间: mvp.py + web.py

Day 3+ (深入):
  → 练习 Level 3 / Level 4
  → 阅读 ARCHITECTURE.md 补全大图
  → 考虑加新 benchmark(MuSiQue / 2Wiki)
```

---

## 8. 常见困惑 FAQ

**Q: 为什么用 `unittest` 不用 `pytest`?**
A: 零依赖原则。`unittest` 是标准库,`pytest` 是第三方。不要因为习惯改写测试。

**Q: `PYTHONPATH=src` 是必须的吗?**
A: 是。包没装到 site-packages,用 src-layout 让 import 找得到 `agentic_rag_lab`。也可以 `pip install -e .` 但项目刻意不这么做。

**Q: 为什么有的文件叫 `mvp.py` 有的叫 `web.py`?**
A: `mvp.py` 是 service 层(业务逻辑、存储);`web.py` 是 HTTP 层(路由、HTML)。分层是为了以后换 FastAPI 只动 `web.py`。

**Q: `semantic.py` 没有真 embedding,怎么还叫 semantic?**
A: 它是「离线代理」—— 用 token 扩展 + 词形近似模拟语义相似,零依赖。真实场景会换成 bge-m3 之类。契约不变,所以换起来很便宜。

**Q: HotpotQA 数据在哪?**
A: 首次跑 `scripts/prepare_hotpotqa_slice.py` 会下载到 `data/raw/hotpotqa/`,切片输出到 `data/processed/hotpotqa/`。两个目录都 gitignore。

---

## 9. 下一步方向(学完之后)

按 README 的 **Next Steps** 推进:

1. `LexicalRetriever` → 真 BM25(`rank_bm25`)
2. `SemanticRetriever` → 真 dense embedding
3. 加 reranker 适配层
4. 运行日志 → Langfuse 观测
5. 更多 benchmark: MuSiQue、2WikiMultihopQA
6. 更强的 synthesis / critic(LLM 替换)

每一步都是「替换一个 stub,不改主循环」—— 这就是这个项目架构的**最大好处**。
