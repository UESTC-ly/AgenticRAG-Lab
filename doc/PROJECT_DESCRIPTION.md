# AgenticRAG-Lab 项目分析说明书

## 1. 项目概述

`AgenticRAG-Lab` 是一个面向复杂多跳问答场景的 `Agentic RAG` 项目。  
它的目标不是做一个“只会一次检索、一次生成”的普通知识库问答，而是做一个具备以下能力的问答系统：

- 能判断问题是否需要检索
- 能把复杂问题拆成子问题
- 能进行多轮检索与证据积累
- 能在回答前自检证据是否充分
- 能把系统能力做成一个可运行、可演示、可评测的本地 MVP 产品

当前仓库已经从最初的 benchmark demo 演进为一个**本地可运行、接入真实模型栈的产品原型**:

- 底层有可解释的 Agentic RAG 主链路(手写状态机)
- 检索层同时保留规则版 baseline 与真实模型实现 —— `sentence-transformers` (bge-small) dense + `bge-reranker-base` cross-encoder 精排
- 生成层有 `LLMSynthesizer` 通过本地 Ollama 调用 `gemma4:e2b`,并带规则版降级
- 评测层有 HotpotQA 500 样本完整消融(retrieval `all_hit@5`: 0.302 → 0.930)+ 20 样本端到端(F1: 0.085 → 0.434)
- 产品层有浏览器工作台、API、多知识库、自定义文档导入、历史记录和本地运行日志

一句话概括：

> 这是一个把 “多跳问答 + planner-driven retrieval + critic/self-reflection + benchmark-first” 落成代码与产品形态的 Agentic RAG 项目。

---

## 2. 为什么做这个项目

普通 RAG 在简单事实问答中通常够用，但在复杂问题上容易失效，主要原因有三类：

### 2.1 问题本身需要多跳推理

例如：

> 《霍比特人》的作者后来就读于哪所大学？

这个问题不能靠一条证据直接回答，至少要经过两跳：

1. 先找到《霍比特人》的作者
2. 再找到该作者就读的大学

如果系统只检索一次，往往只能拿到其中一半事实。

### 2.2 单一路径检索不稳定

只靠 lexical 检索，容易漏掉语义相关但词面不一致的文档；  
只靠 semantic 检索，又可能漏掉精确关键词命中。  
复杂问题通常需要多种检索信号协同。

### 2.3 没有证据自检能力

很多 RAG 系统拿到一些上下文后就立刻生成答案，即使证据并不完整。  
这会导致：

- 幻觉
- 证据链缺失
- 多跳问题被“硬答”

因此，这个项目聚焦的不是单纯“让模型说得更像人”，而是：

- 让系统知道什么时候应该检索
- 让系统知道什么时候应该继续查
- 让系统知道什么时候证据不足
- 让系统对结果具备更可解释的链路

---

## 3. 核心概念与项目定位

这个项目可以用四个关键词来概括：

- **多跳问答**
- **planner-driven retrieval**
- **critic / self-reflection**
- **benchmark-first**

### 3.1 多跳问答

指问题不能通过单条证据回答，而必须跨过多个事实链。

### 3.2 planner-driven retrieval

不是默认“一次检索”，而是先规划“应该查什么”，再按规划逐步检索。

### 3.3 critic / self-reflection

回答前先检查证据是否足够；如果不够，继续检索而不是硬答。

### 3.4 benchmark-first

项目从一开始就不是按“做个 demo 就行”的思路设计，而是按“能否量化证明模块有效”来设计。

因此，这个项目当前的定位不是：

- 企业级生产知识库平台
- 大规模线上检索服务
- 完整工业级 RAG 中台

而是：

> 一个具有完整技术闭环和产品雏形的 Agentic RAG 原型系统，既能做实验，也能做面试展示，还能继续演进为更真实的 LLM 应用产品。

---

## 4. 当前项目状态

截至当前版本，项目已经不只是“算法 demo”，而是一个可以本地使用的 MVP。

### 4.1 已经具备的能力

#### Agentic RAG 主链路

- `Router`
- `Planner`
- `Hybrid Retrieval`
- `Critic`
- `Synthesizer`
- 多轮执行循环

#### 数据与评测

- HotpotQA `dev_distractor` 小切片接入
- JSON → JSONL 切片转换
- retrieval baseline
- lexical / semantic / hybrid ablation
- EM / F1 / Citation Rate / Latency 的基础评测骨架

#### 产品能力

- 本地 Web UI
- JSON API
- 示例问题浏览
- 问答历史
- retrieval inspector
- 自定义知识库文档持久化
- 多知识库管理
- 本地文件导入
- 长文档自动 chunking
- 本地运行日志

### 4.2 当前还没补齐的部分

- 真实 BM25 / dense embedding / reranker
- 更强的 LLM planner / critic
- 更完整的可观测性（如 Langfuse）
- 多会话 / 用户工作区 / 权限
- 更多 benchmark 数据集（MuSiQue / 2Wiki 等）
- 更稳的 synthesis 与 comparison / yes-no 推理

---

## 5. 当前系统架构

### 5.1 总体结构

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
     ├─ LexicalRetriever
     ├─ SemanticRetriever
     └─ HybridRetriever
         ↓
       Critic
         ├─ sufficient → Synthesizer
         └─ insufficient → follow-up queries
```

### 5.2 产品层结构

```text
Browser UI
   ↓
Local JSON API
   ├─ /api/ask
   ├─ /api/documents
   ├─ /api/knowledge-bases
   ├─ /api/retrieve
   ├─ /api/history
   └─ /api/runs
   ↓
HotpotQAMVPService
   ├─ HotpotQA benchmark corpus
   ├─ user document store
   ├─ knowledge base registry
   ├─ retrieval stack
   └─ local run log
```

---

## 6. 当前代码结构说明

当前仓库的核心代码主要分为五层：

### 6.1 Orchestration 层

- `src/agentic_rag_lab/agent.py`
- `src/agentic_rag_lab/router.py`
- `src/agentic_rag_lab/planner.py`
- `src/agentic_rag_lab/critic.py`
- `src/agentic_rag_lab/synthesizer.py`

作用：

- 负责路由
- 负责计划生成
- 负责多轮检索循环
- 负责证据充分性判断
- 负责答案合成

### 6.2 Retrieval 层

- `src/agentic_rag_lab/retrieval/lexical.py`
- `src/agentic_rag_lab/retrieval/semantic.py`
- `src/agentic_rag_lab/retrieval/hybrid.py`
- `src/agentic_rag_lab/text.py`
- `src/agentic_rag_lab/chunking.py`

作用：

- 负责 lexical / semantic / hybrid 检索
- 负责基础文本处理
- 负责用户长文档 chunking

### 6.3 Dataset / Eval 层

- `src/agentic_rag_lab/data/hotpotqa.py`
- `src/agentic_rag_lab/evaluation/hotpotqa.py`
- `src/agentic_rag_lab/evaluation/runner.py`
- `src/agentic_rag_lab/evaluation/metrics.py`

作用：

- HotpotQA 下载、切片、归一化
- retrieval baseline
- ablation 对比
- 指标聚合

### 6.4 产品服务层

- `src/agentic_rag_lab/mvp.py`

作用：

- 管理 benchmark 语料与用户文档
- 管理知识库
- 构建检索器
- 提供问答 / 检索 / 文档 / 历史 / 运行日志能力

### 6.5 Web / CLI 层

- `src/agentic_rag_lab/web.py`
- `src/agentic_rag_lab/__main__.py`

作用：

- 提供本地 Web UI
- 提供 JSON API
- 提供 CLI 入口与服务启动命令

---

## 7. 当前已经完成的工程工作

从工程量角度看，这个项目已经不是简单脚本，而是一个中等体量的个人系统项目。

### 7.1 已实现的工程面

- 数据接入
- 数据切片与预处理
- 检索模块拆分
- agent 主控制流
- 多轮循环
- 评测脚本
- benchmark 对比
- Web API
- 浏览器产品界面
- 本地持久化
- 多知识库管理
- 文件导入
- 历史与日志
- 测试覆盖

### 7.2 当前工程规模

当前项目大致包括：

- 代码 / 脚本 / 测试文件数：约 `39`
- 代码与测试总行数：约 `3481`
- 文档总行数：约 `1000+`
- 测试用例数：当前已通过 `37` 个测试

这说明它已经具备：

- 清晰的模块边界
- 基本可维护性
- 可回归验证能力
- 可演示的产品表面

但它仍然是：

- 单机本地 MVP
- 单人可维护规模
- 研究原型向产品原型过渡阶段

---

## 8. 当前已实现的产品功能

### 8.1 问答能力

用户可以直接提问：

- HotpotQA 示例问题
- 自定义知识库问题
- 某个指定知识库范围内的问题

### 8.2 文档管理

当前支持：

- 粘贴文本新增文档
- 本地文件导入文档
- 文档持久化到本地 JSONL
- 长文档自动切块

### 8.3 多知识库

当前支持：

- 创建知识库
- 列出知识库
- 把文档写入指定知识库
- 按知识库作用域检索与问答

### 8.4 检索调试

当前支持：

- 查看当前 query 命中的文档
- 查看命中文档分数
- 切换 `lexical / semantic / hybrid`
- 按知识库作用域做 retrieval inspection

### 8.5 历史与运行日志

当前支持：

- 最近问答历史
- 本地持久化运行日志
- 记录 ask / retrieve / import / add_document 等动作

这一步虽然还没上 Langfuse，但已经具备了“最小可用观测层”。

---

## 9. 当前评测进展

### 9.1 HotpotQA 小切片

项目已经接入 `HotpotQA dev_distractor` 小切片，并支持：

- 原始数据下载
- 归一化
- JSONL 切片生成

### 9.2 Retrieval baseline

当前已经能跑：

- lexical
- semantic
- hybrid

并输出：

- supporting recall
- all docs hit rate
- any doc hit rate

### 9.3 当前意义

这意味着项目已经不只是“看起来像 RAG”，而是具备真实 benchmark 接口和量化比较入口。

这对面试非常关键，因为你可以明确回答：

- 为什么做 Planner
- 为什么做 Critic
- 为什么做 Hybrid Retrieval
- 每个模块是如何被评测的

---

## 10. 当前最大的短板

虽然真实模型栈(bge-small embedding + bge-reranker-base + Ollama LLM)已经落地,`all_supporting_docs_hit_rate@5` 从 0.302 推到 0.930,端到端 F1 从 0.085 推到 0.434,短板也随之转移。

### 10.1 Planner / Critic 仍偏启发式

当前规则能跑,但对复杂 comparison、深度 bridge 问题覆盖有限。接口已稳定(`plan()`、`evaluate()`),换 LLM 版是局部动作。这是当前最大的债。

### 10.2 端到端 benchmark 规模偏小

retrieval 层在 500 样本上 benchmark 扎实,但 E2E 因为 LLM 慢只跑了 20 条,F1 = 0.434 的置信区间偏大,需要扩到 100~200。

### 10.3 EM 评测没剥离 citation markers

LLM 输出带 `[1][2]` 后缀,tokenize 后不等于参考答案,EM 被人为压低。这是评测脚本的债,修起来很快。

### 10.4 RRF 融合权重硬编码

在真实 embedding 下发现 `hybrid-bge` 反而低于 `semantic-bge`(RRF 退化问题)。已通过接入 reranker 绕过,但根因 —— 权重对组件强度敏感 —— 还没根治。理想是做 query-level 或组件-强度 adaptive 的 weighting。

### 10.5 观测能力还不完整

当前的本地运行日志能用,但还没有:

- trace 可视化平台
- token / latency / model usage 追踪
- 更强的请求级诊断工具

---

## 11. 下一步路线图

如果继续推进,建议按下面顺序补齐短板。

### Phase A：补真实检索层(已完成)

已实现:

- ✅ `EmbeddingSemanticRetriever` 接入 `sentence-transformers` + `BAAI/bge-small-en-v1.5`
- ✅ `CrossEncoderReranker` 接入 `BAAI/bge-reranker-base`
- ✅ `LLMSynthesizer` 通过本地 Ollama 调用 `gemma4:e2b`
- ✅ 三者均支持独立降级,测试通过 fake-model 注入保持 CI 无 GPU 也能跑
- ✅ HotpotQA 500 样本重跑 ablation,`all_supporting_docs_hit_rate@5` **0.302 → 0.930**

### Phase B：补更强的 planner / critic

目标:

- 用真实 LLM 替换 rule-based planner
- 用更强的 critic 做证据充分性判断
- 降低 comparison / bridge 问题失败率

### Phase C：扩大端到端评测

目标:

- E2E benchmark 从当前 20 样本扩到 100~200
- 评测脚本剥离 citation markers 让 EM 公平
- 接入更大的 `bge-m3` + `bge-reranker-v2-m3` 作为正式版数字

### Phase D：补观测

目标:

- 接 Langfuse
- 对接本地运行日志
- 建立更清晰的 trace 页面或调试面板

### Phase E：补产品体验

目标:

- 多会话
- 更清晰的知识库切换
- 更好的文档导入体验
- 更好的结果展示与错误反馈

### Phase F：补更多 benchmark

目标:

- MuSiQue(特别是 2-hop / 3-hop / 4-hop 分层分析)
- 2Wiki
- 更完整的 case analysis

---

## 12. 项目定位与边界

这个项目的价值不在于"规模特别大",而在于它具备完整链条:

- 有问题定义(多跳问答的失败模式)
- 有系统设计(手写状态机,每一跳可解释)
- 有真实模型栈(bge-small + bge-reranker-base + Ollama)
- 有 benchmark(500 样本 retrieval + 20 样本 E2E,含 6 配置 ablation)
- 有产品原型(Web UI + 多知识库 + 文档导入)
- 有测试(49 个单元测试,fake-model 注入保证无 GPU 也能跑)
- 有可解释的工程结构(主循环手写,组件可独立替换)

它更适合被描述为:

> 一个完成度较高的 Agentic RAG 项目,从 benchmark demo 推进到了带真实模型栈的可用产品原型,并在 HotpotQA 上完成完整消融评测。

而不是:

> 一个企业级生产平台。

这个边界讲清楚,反而会更显得可信。

---

## 13. 当前一句话结论

`AgenticRAG-Lab` 现在是:

> 一个已经打通 **数据接入 → 检索评测(真实 BM25 + dense + reranker) → Agentic RAG 主链路 → LLM 生成 → 本地可用产品工作台** 的 Agentic RAG 项目,并在 HotpotQA 上完成 6 配置 retrieval ablation + 4 配置端到端 benchmark。

下一步最值得投入的方向:

> 把 Planner / Critic 从规则版换成 LLM 版,并扩大端到端评测规模 —— retrieval 已经不是瓶颈,生成端和评测可信度才是。

