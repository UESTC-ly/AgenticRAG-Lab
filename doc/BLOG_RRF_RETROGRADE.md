# 我把检索器换成真 embedding 后,hybrid 融合反而退化了

> 一个在 HotpotQA 上做消融实验时撞到的反直觉现象,以及它教我的事。

## TL;DR

我把 `SemanticRetriever` 从 token 扩展近似的 proxy 换成了 `bge-small-en-v1.5`,单纯 dense 的 `all_supporting_docs_hit_rate@5` 从 **0.372** 跳到 **0.798**。信心十足地把它塞进 `HybridRetriever` 里做 RRF 融合 —— 期望的剧情是 hybrid 比单路更高。

结果是 `hybrid-bge@5 = 0.444`,**不仅没涨,反而比纯 `semantic-bge` 低了 35 个百分点**。

本文记录这个现象背后的机制、验证过程,以及一个 2 小时的修复方案。

---

## 1. 背景:一个手写的 Agentic RAG loop

[AgenticRAG-Lab](https://github.com/) 是我从零写的多跳问答系统。主循环是 `Router → Planner → Executor Loop (Retriever + Critic) → Synthesizer`,没有用 LangChain 之类的框架,纯手写状态机,每一跳都能解释。

检索层之前是三个组件:

- `LexicalRetriever` — BM25-like 词频打分
- `SemanticRetriever` — 离线代理,用 token 扩展近似语义相似
- `HybridRetriever` — 用 Reciprocal Rank Fusion (RRF) 把两路融合

这套配置在 HotpotQA dev_distractor 500 样本上的 baseline 是:

| 配置 | `all_hit@5` |
|---|---|
| lexical | 0.226 |
| semantic-rule | 0.372 |
| hybrid-rule | 0.302 |

> 顺便一提:规则版 hybrid 已经比 lexical 低,但比 semantic-rule 也低。这个现象当时被解释为"融合的代价",没深究。**这其实是同一个问题的早期信号,只是差距没那么夸张。**

---

## 2. 升级 dense:单次 +50 个百分点

我把 `SemanticRetriever` 替换成基于 `sentence-transformers + bge-small-en-v1.5` 的真实 dense retriever(`EmbeddingSemanticRetriever`)。

实现大约 60 行,关键点:

- `sentence-transformers` 加载模型,一次性编码 corpus
- numpy 暴力点积打分(4858 passages 不值得上 faiss)
- 归一化 embedding,cosine similarity = dot product
- `argpartition` 取 top-k

跑相同的 500 样本消融:

| 配置 | `all_hit@5` |
|---|---|
| lexical | 0.226 |
| semantic-rule | 0.372 |
| hybrid-rule | 0.302 |
| **semantic-bge** | **0.798** |

单纯把假 semantic 换成真 embedding,带来 +42.6 个百分点。这是 RAG 栈里我见过的**最大单次杠杆**。

在我换 dense 之前,我以为最大的瓶颈会是 synthesizer 或 critic。错了。**RAG 的瓶颈几乎总是在检索**。

---

## 3. 意外:hybrid 融合后**反而退化**

自信地把 `EmbeddingSemanticRetriever` 塞进 `HybridRetriever`:

```python
retriever = HybridRetriever(
    lexical=LexicalRetriever(corpus),
    semantic=EmbeddingSemanticRetriever(corpus),  # 新的
)
```

预期:hybrid 应该比 semantic 单独还高,因为"两路召回 + 融合"总是比单路好。

实际:

| 配置 | `all_hit@5` |
|---|---|
| semantic-bge | 0.798 |
| **hybrid-bge** | **0.444** |

融合之后**掉了 35 个百分点**。

---

## 4. 为什么会退化?

先看 `HybridRetriever.search` 的核心逻辑:

```python
# 双路各召回 top_k * 2
lexical_results = self.lexical.search(query, top_k=top_k * 2)
semantic_results = self.semantic.search(query, top_k=top_k * 2)

# RRF + 原始分数的加权组合
for rank, item in enumerate(lexical_results, start=1):
    score = reciprocal_rank_fusion(rank) + (item.score * 0.01)
    fused[item.document.doc_id] = ...

for rank, item in enumerate(semantic_results, start=1):
    addition = reciprocal_rank_fusion(rank) + (item.score * 0.5)
    ...
```

几个观察:

1. **RRF 按名次打分**(`1/(k+rank)`),不是按原始相似度
2. **两路给的名次权重是等价的** —— 无论哪一路返回的第一个,都拿到相同的 RRF 基础分
3. 我的 hack 是给 semantic 乘 0.5 给 lexical 乘 0.01,作为原始分数的"补偿"

问题来了:

**原规则版里,lexical 和 semantic-rule 的质量差不多**(recall@5 分别 0.459 和 0.630)。RRF 等权融合是合理的,因为两路都能贡献有用信息。

**换成 bge 之后,dense 的 recall@5 是 0.896,lexical 只有 0.459**。两路的质量差了**近一倍**。

这时候 RRF 等权融合做了一件特别愚蠢的事:**它把 lexical 排名前几的那些低质量候选,也拉进了候选池的上半部分**。

举个例子:一个多跳问题"哪个大学是 Hobbit 作者在成为教授前读的?"
- semantic-bge 返回的 top-5 正好命中两个金证据(Tolkien 传记 + Exeter College 条目)
- lexical 返回的 top-5 只是包含 "hobbit" "author" "university" 这些词的其他文章
- RRF 融合:lexical 排名第 1 的文章拿到 `1/(60+1) ≈ 0.016` 的 RRF 分,和 semantic 第 1 的那篇**完全等价**
- 最终 top-5 里混进了 lexical 的 2~3 个噪声,挤掉了 semantic 的第 4、5 位真命中

数字上:
- recall@5:semantic-bge 0.896 → hybrid-bge **0.659**(-0.237)
- all_hit@5:semantic-bge 0.798 → hybrid-bge **0.444**(-0.354)

---

## 5. 这不是 bug,是 **RRF 的固有假设被打破了**

RRF 的经典论文 [Cormack et al., 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) 的核心假设是:

> When two retrieval systems both rank a document highly, that document is very likely relevant.

换句话说:RRF 假设**所有融合的系统都是"大致靠谱"的贡献者**。当一路系统明显强于另一路时,RRF 会变成一个"投票被低质量声音稀释"的失衡机器。

生产级 RAG 里 hybrid 依然流行,是因为实践中 BM25 和 dense embedding 的质量**接近**(都是~0.5 recall 量级)。我这里的情况是极端失衡(0.459 vs 0.896),RRF 因此失效。

---

## 6. 三种修复方向

### 方案 A:重调 RRF 的 k

RRF 公式 `1/(k + rank)` 里的 `k`(典型值 60),控制了衰减速度。大 k 让名次权重更平均(弱路也能上位),小 k 让 top-1 占绝对优势。

当 dense 远强于 lexical 时,应该用更小的 k 让 dense 的 top-1 天然锁死位置。
缺点:需要数据集调参,换一个 query 分布就得重调。

### 方案 B:给强路更高权重

直接把 RRF 分乘以经验权重:`semantic_rrf × 2.0 + lexical_rrf × 0.5`。
这是 LangChain 的 `EnsembleRetriever` 的做法。
缺点:和方案 A 一样需要调。

### 方案 C(我的选择):加一个 cross-encoder reranker 做精排

- 让 hybrid 广召(比如 top_k × 5 = 25 个候选)
- 用 cross-encoder 对 (query, doc) 逐对打分
- 返回精排后的 top_k

cross-encoder 的 query-document interaction 能力远强于 bi-encoder,它可以**把 hybrid 混进来的噪声淘汰掉**,同时还可能救回 dense 漏掉的题。

实现很简单(70 行):

```python
class CrossEncoderReranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name)

    def rerank(self, query, items, *, top_k=None):
        pairs = [[query, f"{it.document.title}. {it.document.content}"] for it in items]
        scores = self._model.predict(pairs)
        rescored = [EvidenceItem(document=it.document, score=float(s), ...) for it, s in zip(items, scores)]
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k] if top_k else rescored
```

然后 `HybridRetriever` 接受一个可选的 `reranker` 参数注入。

---

## 7. 结果:reranker 不仅救了 hybrid,还反超了单路

在 100 样本上跑:

| 配置 | `all_hit@5` |
|---|---|
| hybrid-rule (原 baseline) | 0.340 |
| semantic-bge | 0.820 |
| hybrid-bge(未救) | 0.520 |
| **hybrid-bge-rerank** | **0.930** |

两个观察:

1. **reranker 把 hybrid 的 0.520 拉回到 0.930** —— 它有能力从混杂的候选池里挑出正确答案
2. **hybrid-bge-rerank (0.930) > semantic-bge (0.820)** —— 当 reranker 存在时,hybrid 重新变成有用的架构,因为它提供了更宽的召回面,reranker 负责精度

这就是生产级 RAG 栈"bi-encoder 广撒网 + cross-encoder 精挑"的**理论依据**和**实验证据**。

---

## 8. 学到的几件事

**1. 组件之间的质量差距决定了融合策略。**
两路质量接近 → RRF 等权有效;一路远强于另一路 → RRF 反而有害,需要 reranker 或重调权重。

**2. 做消融的时候,每次只动一个变量。**
如果我把 dense 和 reranker 一起加,就永远发现不了 RRF 退化这个现象。单独跑 `semantic-bge`、单独跑 `hybrid-bge`,对比才能暴露问题。

**3. 反直觉的数字值得深入,而不是回避。**
看到 hybrid-bge 低于 semantic-bge 的时候,我的第一反应是"可能我实现有 bug"。但实际上代码是对的,是 RRF 的假设不成立。**工程里最有价值的发现,通常藏在不符合预期的数字后面。**

**4. 框架的默认行为可能在你的数据上是错的。**
LangChain 的 `EnsembleRetriever` 默认 `weights=[0.5, 0.5]`,对等融合两路。如果你的 dense 和 BM25 质量差距大,这个默认值就是陷阱。

---

## 9. 项目链接

- GitHub: (填入链接)
- 完整消融数据:`doc/BENCHMARK_RESULTS.md`
- 架构说明:`doc/ARCHITECTURE.md`

整个系统包含:

- 手写 `Router / Planner / Hybrid Retrieval / Critic / Synthesizer` 主链路
- 真实 retrieval 栈:`bge-small` + `bge-reranker-base`
- LLM synthesizer:本地 Ollama (`gemma4:e2b`),citation 强制白名单校验
- HotpotQA dev 500 样本完整 ablation
- 48 个单元测试(真实模型通过 fake 注入,无 GPU 也能跑)

---

## 附录:复现命令

```bash
# 500 样本,5 配置(无 reranker,~2 分钟)
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python3 scripts/run_hotpotqa_real_ablation.py \
  --device mps --top-ks 1,3,5,10 --no-reranker --format markdown

# 100 样本,含 reranker(~4 分钟)
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python3 scripts/run_hotpotqa_real_ablation.py \
  --slice data/processed/hotpotqa/dev_slice_rerank100.jsonl \
  --device mps --top-ks 5,10 --format markdown
```
