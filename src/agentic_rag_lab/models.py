from __future__ import annotations

"""
项目核心数据模型定义。

这个文件的作用是把 Agentic RAG 系统里的“状态”和“数据交换格式”统一下来，
避免不同模块之间通过随意的 dict 或字符串约定来通信。

当前主要覆盖五类对象：

1. 路由层对象
   - 用户问题应该走哪条执行路径
2. 规划层对象
   - 一个复杂问题被拆成哪些步骤
3. 检索/证据层对象
   - 检索到了什么文档、分数是多少
4. Critic 层对象
   - 当前证据是否足够继续回答
5. 最终执行结果对象
   - 系统最后给用户返回什么

这些模型本身不包含业务逻辑，它们只负责定义结构。
真正的业务逻辑分布在 router / planner / retriever / critic / synthesizer / agent 等模块中。
"""

from dataclasses import dataclass, field
from enum import Enum


class RouteTarget(str, Enum):
    """
    路由层的目标类型。

    这里用 `str + Enum` 的组合，而不是普通 Enum，有两个好处：

    1. 序列化更方便
       - 写日志、返回 API、输出 JSON 时可以直接得到字符串值
    2. 可读性更强
       - `direct_answer` / `agentic_rag` / `calculator` 能直接体现语义

    当前系统支持三种执行路径：

    - DIRECT_ANSWER
      适合特别简单、无需检索的定义类问题
    - AGENTIC_RAG
      适合需要规划、检索、批判和合成的复杂问题
    - CALCULATOR
      适合安全可解析的算术表达式
    """
    DIRECT_ANSWER = "direct_answer"
    AGENTIC_RAG = "agentic_rag"
    CALCULATOR = "calculator"


@dataclass(slots=True)
class Document:
    """
    文档对象，是检索层和知识库层的基础单位。

    无论这个文档来自：
    - HotpotQA benchmark 语料
    - 用户上传的自定义知识库
    - 文件导入后的 chunk

    最终都会统一映射成这个结构，方便后面的检索器和合成器复用。
    """
    # 文档唯一 ID。
    # 例如：
    # - title::The Hobbit
    # - user::pricing-1::chunk-1
    doc_id: str

    # 文档标题。
    # 在 HotpotQA 中通常对应 wiki title；
    # 在用户知识库中通常对应文档名或用户输入的标题。
    title: str

    # 文档正文内容。
    # 检索器主要会基于 title + content 做 tokenization、召回和排序。
    content: str

    # 附加元信息。
    # 当前常见字段包括：
    # - source: hotpotqa / user / demo
    # - knowledge_base: 所属知识库
    # - created_at: 创建时间或占位标记
    # - parent_doc_id: 原始文档 ID（chunk 场景下）
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RouteDecision:
    """
    Router 的输出结果。

    它回答三个问题：
    1. 当前问题应该走哪条主路径？
    2. 是否需要进入规划阶段？
    3. 这么判断的原因是什么？
    """
    # 路由目标。
    target: RouteTarget

    # 是否需要 Planner 继续拆问题。
    # 一般 direct_answer / calculator 为 False，
    # agentic_rag 为 True。
    needs_planning: bool

    # 路由理由，用于 trace、调试、产品展示和面试解释。
    rationale: str


@dataclass(slots=True)
class PlanStep:
    """
    规划器拆出来的单个步骤。

    一个复杂问题通常不会被直接回答，而是先拆成若干子问题。
    `PlanStep` 就是这些子问题的标准表示形式。
    """
    # 当前步骤的唯一标识。
    # 例如：identify-bridge-entity / resolve-destination
    step_id: str

    # 当前步骤真正执行的检索查询。
    query: str

    # 当前步骤的意图说明。
    # 用于解释这个步骤存在的原因，便于 trace 与后续调试。
    purpose: str

    # 依赖步骤列表。
    # 未来如果要升级成 DAG 执行器，这个字段可以支持更复杂的调度。
    depends_on: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QueryPlan:
    """
    Planner 的完整输出。

    它把“原始用户问题”映射成“按顺序/依赖关系组织的多个 PlanStep”。
    """
    # 用户原始问题。
    original_query: str

    # 拆解后的步骤列表。
    steps: list[PlanStep]


@dataclass(slots=True)
class EvidenceItem:
    """
    单条证据对象，是检索器的标准输出。

    检索器并不直接返回裸文档，而是返回 “文档 + 分数 + 查询上下文” 的证据对象，
    这样后续 Critic 和 Synthesizer 才能知道：
    - 这条证据是由哪个 query 检索出来的
    - 它的重要性/相关性分数是多少
    - 最终应该如何引用它
    """
    # 命中的文档本体。
    document: Document

    # 当前文档相对于 query 的检索分数。
    # 不同 retriever 的分数物理意义可能不同，
    # 但在同一 retriever 内可用于排序。
    score: float

    # 触发本次检索的 query。
    # 在多轮、多子问题检索时，这个字段很重要。
    query: str

    # 对应的引用标记列表。
    # 当前实现通常直接使用 doc_id。
    citations: list[str]


@dataclass(slots=True)
class CritiqueResult:
    """
    Critic 的输出结果。

    Critic 不是负责生成答案，而是负责判断：
    - 当前证据够不够？
    - 如果不够，下一轮应该继续查什么？
    - 当前判断的大致置信度如何？
    """
    # 是否认为当前证据已经足够支撑回答。
    is_sufficient: bool

    # 解释为什么 sufficiency / insufficiency 成立。
    # 用于 trace 展示和开发时调试。
    reasoning: str

    # 如果证据不足，下一轮推荐检索的 follow-up queries。
    follow_up_queries: list[str] = field(default_factory=list)

    # Critic 当前判断的置信度。
    # 当前是启发式值，未来可替换成更真实的模型打分。
    confidence: float = 0.0


@dataclass(slots=True)
class TraceEvent:
    """
    一条结构化执行轨迹事件。

    这个对象用于记录系统在一次请求里经历了哪些阶段，
    方便：
    - 本地 UI 展示 trace
    - 调试错误
    - 后续接入更正式的 tracing/observability 系统
    """
    # 当前事件属于哪个阶段。
    # 例如：router / planner / retriever / critic / synthesizer
    stage: str

    # 当前阶段的文字说明。
    detail: str


@dataclass(slots=True)
class RunResult:
    """
    一次完整执行的最终返回结果。

    这是 `AgenticRAG.run()` 的标准输出对象，也是产品层最终会暴露给
    Web UI 或 API 的核心信息来源。
    """
    # 用户原始 query。
    query: str

    # 最终生成的答案文本。
    answer: str

    # 最终答案对应的引用列表。
    citations: list[str]

    # 最终走的是哪条主链路。
    # 当前通常是 RouteTarget 的字符串值。
    route: str

    # 实际执行了多少轮迭代。
    # 对 direct_answer / calculator 一般为 0。
    iterations: int

    # 本次执行过程中采集到的结构化 trace。
    trace: list[TraceEvent] = field(default_factory=list)
