# AgenticRAG-Lab (Lite)

一个面向复杂多跳问答的轻量 Agentic RAG 原型，强调四个可讲清楚的核心点：

- `Router`：简单问题直答，复杂问题进入 agent loop
- `Planner`：把多跳问题拆成可执行的 sub-queries
- `Hybrid Retrieval`：lexical + semantic + RRF + 轻量 rerank
- `Critic`：证据不够时自动追问，避免一次检索定生死

这个仓库刻意保持 **手写 orchestration + 离线可跑**，方便在 2 周窗口里做 demo、做 benchmark、做面试讲解。

更详细的项目背景、设计取舍与扩展路线见 `doc/PROJECT_DESCRIPTION.md`。
项目架构图与工作流沉淀见 `doc/ARCHITECTURE.md`。
真实模型 retrieval 消融结果见 `doc/BENCHMARK_RESULTS.md`。

## Architecture

```text
User Query
   ↓
Router
   ├─ direct answer
   ├─ calculator
   └─ agentic_rag
         ↓
       Planner
         ↓
   Executor Loop
     ├─ LexicalRetriever              (BM25-like)
     ├─ SemanticRetriever             (offline proxy / bge-small / bge-m3)
     └─ HybridRetriever               (RRF fusion + optional CrossEncoderReranker)
         ↓
       Critic
         ├─ sufficient → Synthesizer  (rule-based CitationSynthesizer / LLMSynthesizer via Ollama)
         └─ insufficient → follow-up queries
```

## Project Layout

```text
src/agentic_rag_lab/
  agent.py                    # main orchestration loop
  mvp.py                      # product service layer
  web.py                      # local web app and JSON API
  router.py                   # direct / calculator / agentic routing
  planner.py                  # rule-based query decomposition
  critic.py                   # sufficiency checks + follow-up generation
  synthesizer.py              # rule-based cited answer synthesis
  llm_synthesizer.py          # LLM-backed synthesizer (Ollama via stdlib HTTP)
  data/
    hotpotqa.py               # dataset download / slicing
  retrieval/
    lexical.py                # BM25-like lexical scoring
    semantic.py               # offline semantic proxy (zero-dependency baseline)
    embedding_semantic.py     # real dense retriever (sentence-transformers / bge-m3)
    reranker.py               # cross-encoder reranker (bge-reranker-v2-m3)
    hybrid.py                 # RRF fusion + optional reranker
  evaluation/
    metrics.py                # EM / token F1
    runner.py                 # benchmark aggregation
    hotpotqa.py               # retrieval baseline / ablation
  demo.py                     # demo corpus + build_demo_agent / build_real_agent
tests/                        # 48 offline unit tests (fake-model injection, zero GPU needed)
scripts/                      # data prep / baseline / ablation helpers
```

## Quickstart

### 纯离线(零依赖,默认规则版组件)

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/prepare_hotpotqa_slice.py --limit 500
PYTHONPATH=src python3 -m agentic_rag_lab ask "Which university did the author of The Hobbit attend before becoming a professor?"
PYTHONPATH=src python3 -m agentic_rag_lab benchmark
PYTHONPATH=src python3 -m agentic_rag_lab serve --slice data/processed/hotpotqa/dev_slice.jsonl --user-store data/product/user_documents.jsonl --kb-store data/product/knowledge_bases.json --port 8000
PYTHONPATH=src python3 scripts/run_hotpotqa_retrieval_baseline.py --top-k 5
PYTHONPATH=src python3 scripts/run_hotpotqa_retrieval_ablation.py --top-ks 1,3,5 --format markdown
```

### 启用真实模型(bge embedding + cross-encoder reranker + Ollama LLM)

```bash
pip install -r requirements-real.txt        # sentence-transformers, numpy
ollama pull gemma4:e2b                       # 或 qwen2.5:7b
ollama serve                                 # 另开一个终端

# Agent 走真实组件
PYTHONPATH=src python3 -m agentic_rag_lab ask --real --llm-model gemma4:e2b \
  "Which university did the author of The Hobbit attend before becoming a professor?"

# retrieval 端到端消融(含真实 embedding + reranker)
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python3 scripts/run_hotpotqa_real_ablation.py \
  --device mps --top-ks 1,3,5,10 --format markdown
```

## Current Scope

已实现：

- 手写 `AgenticRAG` 主循环
- 离线 `Router / Planner / Critic / Synthesizer`
- 轻量 hybrid retrieval(无外部依赖,作为 baseline)
- **真实 dense retrieval**: `EmbeddingSemanticRetriever` + `sentence-transformers`(bge-small / bge-m3)
- **真实 reranker**: `CrossEncoderReranker`(bge-reranker-base / bge-reranker-v2-m3)
- **真实 LLM synthesizer**: `LLMSynthesizer` 通过 Ollama 调用本地模型(gemma4:e2b / qwen2.5:7b),带 citation 白名单校验与规则版自动降级
- HotpotQA `dev_distractor` 500 条切片接入
- benchmark runner(EM / F1 / citation rate / latency)
- HotpotQA retrieval baseline / ablation(含真实模型 6 配置对比,见 `doc/BENCHMARK_RESULTS.md`)
- 基于 HotpotQA slice 的本地 MVP Web 应用
- 自定义知识库文档持久化、检索调试、历史记录与浏览器工作台

暂未实现:

- MuSiQue / 2Wiki 等更多数据集 ingestion
- 端到端 EM / F1 benchmark 接入 LLM synthesizer(目前仅 retrieval 阶段有真实模型评测)
- Langfuse tracing
- RAGAS 指标

## Benchmark 核心结果

**Retrieval** (HotpotQA dev, `all_supporting_docs_hit_rate@5`):

| 配置 | @5 all-hit |
|---|---|
| hybrid-rule (规则版 baseline, 500 样本) | 0.302 |
| semantic-bge (bge-small-en-v1.5, 500 样本) | 0.798 |
| hybrid-bge-rerank (+ bge-reranker-base, 100 样本) | **0.930** |

**End-to-End** (HotpotQA dev 20 样本, Ollama `gemma4:e2b`):

| 配置 | F1 | Citation Rate |
|---|---|---|
| rule-retr + rule-synth (baseline) | 0.085 | 1.000 |
| real-retr + rule-synth | 0.100 | 1.000 |
| rule-retr + llm-synth | 0.250 | 0.350 |
| **real-retr + llm-synth** | **0.434** | **0.850** |

完整分析见 `doc/BENCHMARK_RESULTS.md`;RRF 退化发现的技术博客见 `doc/BLOG_RRF_RETROGRADE.md`。

## HotpotQA Slice

仓库现在已经支持把 `HotpotQA dev` 转成可直接用于后续评测的本地 JSONL 小切片。

```bash
PYTHONPATH=src python3 scripts/prepare_hotpotqa_slice.py --limit 100
```

默认行为：

- 优先使用官方 `dev_distractor` 数据
- 默认保留 `medium,hard`
- 默认包含 `bridge,comparison`
- 输出到 `data/processed/hotpotqa/dev_slice.jsonl`

在切片生成后，可以直接运行离线 retrieval baseline：

```bash
PYTHONPATH=src python3 scripts/run_hotpotqa_retrieval_baseline.py --top-k 5
```

如果要看 `lexical / semantic / hybrid` 的对比实验，可以运行：

```bash
PYTHONPATH=src python3 scripts/run_hotpotqa_retrieval_ablation.py --top-ks 1,3,5 --format markdown
```

## MVP Product

当前仓库已经具备一个本地可运行的 MVP 产品形态：

- HotpotQA slice 驱动的多跳问答演示
- 浏览器界面
- `/api/ask` 问答接口
- `/api/documents` 文档创建/列表接口
- `/api/retrieve` 检索调试接口
- `/api/history` 历史记录接口
- `/api/knowledge-bases` 知识库创建/列表接口
- `/api/import-paths` 本地文件导入接口
- `/api/runs` 运行日志接口
- 示例问题列表
- 自定义知识库文档持久化
- 多知识库管理
- 本地知识库文档浏览
- 长文档自动 chunking
- retrieval inspector
- trace 展示
- 本地持久化运行日志
- baseline / ablation 结果可视化

启动方式：

```bash
PYTHONPATH=src python3 -m agentic_rag_lab serve --slice data/processed/hotpotqa/dev_slice.jsonl --user-store data/product/user_documents.jsonl --kb-store data/product/knowledge_bases.json --port 8000
```

启动后打开：

```text
http://127.0.0.1:8000
```

典型工作流：

1. 启动本地服务
2. 在页面里直接提问 HotpotQA 示例问题
3. 先创建一个新的知识库，再通过 `Add custom knowledge` 粘贴自己的文本知识
4. 用 `Inspect Retrieval` 查看当前命中的文档和分数
5. 通过 `/api/import-paths` 或文件选择器导入本地文档
6. 在 `History` / `Knowledge Base` / `Knowledge Bases` 面板里追踪最近使用情况
7. 通过 `/api/runs` 查看持久化运行日志

## Why This Shape

- **面试可解释**：核心状态机和 retry loop 都是自己写的
- **离线可验证**：没有向量库、没有外部模型也能先把控制流跑通
- **便于替换**：后面只需要把 retriever / planner / critic 的实现从 rule-based 换成真实服务

## Next Steps

1. 换更大的 `bge-m3` (2.3GB) + `bge-reranker-v2-m3` (2GB) 重跑 500 样本消融,作为正式数字
2. 接入 `gemma4:e2b` / `qwen2.5:7b` 到 synthesizer,跑端到端 EM / F1 / citation rate
3. 加入 MuSiQue(2-hop / 3-hop / 4-hop 分层)做跨数据集验证
4. 重调 hybrid 的 RRF 权重 —— 真实 embedding 下 `semantic-bge > hybrid-bge`,融合需要重新 tune
5. `LexicalRetriever` 升级为真正的 `rank_bm25` / `Tantivy`
6. 把本地运行日志升级成 Langfuse / 更完整请求观测
7. 增加多会话管理与更完整的用户工作区
