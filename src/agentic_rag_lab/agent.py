from __future__ import annotations

"""
Agentic RAG 主执行器。

这个文件定义了项目里最核心的控制流对象：`AgenticRAG`。

它的职责不是负责某一个具体能力，而是把多个模块按固定顺序编排起来：

1. Router
   - 判断问题该走哪条主路径
2. Planner
   - 把复杂问题拆成子问题
3. Retriever
   - 针对子问题分轮检索证据
4. Critic
   - 判断证据是否足够
5. Synthesizer
   - 生成最终答案

从系统角度看，这个类相当于整个 Agentic RAG 的“调度器”或“主状态机”。

当前实现刻意保持轻量：
- 不依赖外部 workflow 框架
- 不做复杂 DAG 调度
- 不做异步执行

这样做的好处是：
- 控制流清晰、可解释
- 容易写测试
- 面试时能准确讲清每一步是怎么触发的
"""

import ast
import operator

from .models import CritiqueResult, EvidenceItem, RunResult, RouteTarget, TraceEvent


class AgenticRAG:
    """
    Agentic RAG 主执行器。

    设计上采用依赖注入，而不是在类内部直接实例化各模块。
    这样做有三个好处：

    1. 更容易测试
       - 不同测试可以注入不同的 router / planner / retriever
    2. 更容易替换实现
       - 例如未来把 rule-based planner 换成 LLM planner
    3. 更符合“模块化 Agent 系统”的设计方式

    当前这个类本身不关心具体实现细节，只关心调用顺序和状态流转。
    """
    def __init__(
        self,
        *,
        router,
        planner,
        retriever,
        critic,
        synthesizer,
        max_iterations: int = 3,
    ) -> None:
        # Router：决定问题应该走哪条路径。
        self.router = router

        # Planner：把复杂问题拆成多个可执行检索步骤。
        self.planner = planner

        # Retriever：真正执行检索，返回 EvidenceItem 列表。
        self.retriever = retriever

        # Critic：判断当前证据是否足够支撑回答。
        self.critic = critic

        # Synthesizer：根据最终证据生成答案。
        self.synthesizer = synthesizer

        # 最大迭代轮数，用于防止 Critic / follow-up query 导致死循环。
        self.max_iterations = max_iterations

    def run(self, query: str) -> RunResult:
        """
        执行一次完整的 Agentic RAG 问答流程。

        输入：
        - query：用户原始问题

        输出：
        - RunResult：包含最终答案、引用、路由信息、迭代次数和 trace

        当前整体流程如下：

        1. Router 判断路径
        2. 如果是 direct_answer / calculator，直接返回
        3. 如果是 agentic_rag：
           - 先由 Planner 生成 plan
           - 再进入多轮检索循环
           - 每轮执行：retrieve -> critique
           - 直到 Critic 认为 sufficiency 成立，或者达到最大轮数
        4. 用 Synthesizer 合成最终答案
        """
        # trace 用于记录整次执行的关键阶段事件。
        # 这既方便调试，也方便在产品 UI 里展示给用户。
        trace: list[TraceEvent] = []

        # 第一步：由 Router 判断这次 query 应该走哪条路径。
        decision = self.router.route(query)
        trace.append(TraceEvent(stage="router", detail=decision.rationale))

        # 路由分支 1：
        # 对非常简单的问题，直接调用 direct_answer，不经过检索链路。
        if decision.target is RouteTarget.DIRECT_ANSWER:
            answer = self.synthesizer.direct_answer(query)
            trace.append(TraceEvent(stage="synthesizer", detail="Answered without retrieval."))
            return RunResult(
                query=query,
                answer=answer,
                citations=[],
                route=decision.target.value,
                iterations=0,
                trace=trace,
            )

        # 路由分支 2：
        # 对纯算术表达式，走安全计算路径，不使用检索和生成。
        if decision.target is RouteTarget.CALCULATOR:
            answer = str(self._safe_calculate(query))
            trace.append(TraceEvent(stage="calculator", detail="Evaluated arithmetic expression."))
            return RunResult(
                query=query,
                answer=answer,
                citations=[],
                route=decision.target.value,
                iterations=0,
                trace=trace,
            )

        # 路由分支 3：
        # 进入完整的 Agentic RAG 工作流。

        # Planner 先对复杂问题做拆解。
        plan = self.planner.plan(query)
        trace.append(TraceEvent(stage="planner", detail=f"Generated {len(plan.steps)} plan steps."))

        # evidence_pool 是整个执行过程中的“证据池”。
        # 这里按 doc_id 去重，避免同一文档在多轮、多子问题检索中被重复累计。
        evidence_pool: dict[str, EvidenceItem] = {}

        # 当前轮次正在执行的检索 query 集合。
        # 初始值来自 Planner 的 step.query。
        active_queries = [step.query for step in plan.steps]

        # 保存 Critic 的最近一次判断结果。
        # 当前代码里后面没有直接使用它做返回，但保留这个变量便于未来扩展。
        critique: CritiqueResult | None = None

        # 记录实际执行了多少轮迭代。
        iteration_count = 0

        # 进入多轮检索-批判循环。
        for iteration in range(1, self.max_iterations + 1):
            iteration_count = iteration

            # 对当前轮的所有 active query 分别做检索。
            for sub_query in active_queries:
                for item in self.retriever.search(sub_query, top_k=5):
                    existing = evidence_pool.get(item.document.doc_id)

                    # 如果这条文档之前没出现过，直接加入证据池。
                    # 如果已经出现过，则保留分数更高的那条证据。
                    # 这样能避免重复文档污染 Critic 判断。
                    if existing is None or item.score > existing.score:
                        evidence_pool[item.document.doc_id] = item

            # 记录当前轮检索完成后的证据规模。
            trace.append(
                TraceEvent(
                    stage="retriever",
                    detail=f"Iteration {iteration} gathered {len(evidence_pool)} unique evidence items.",
                )
            )

            # Critic 根据“原始 query + 当前所有证据”判断是否足够回答。
            critique = self.critic.evaluate(query, list(evidence_pool.values()))
            trace.append(
                TraceEvent(
                    stage="critic",
                    detail=f"Iteration {iteration}: {critique.reasoning}",
                )
            )

            # 如果 Critic 认为证据已经足够，提前结束循环。
            if critique.is_sufficient:
                break

            # 否则使用 Critic 给出的 follow-up queries 进入下一轮检索。
            active_queries = critique.follow_up_queries

        # 无论是提前满足 sufficiency，还是达到最大轮数，
        # 最终都由 Synthesizer 基于当前 evidence_pool 生成答案。
        answer, citations = self.synthesizer.answer(query, list(evidence_pool.values()))
        return RunResult(
            query=query,
            answer=answer,
            citations=citations,
            route=decision.target.value,
            iterations=iteration_count,
            trace=trace,
        )

    def _safe_calculate(self, query: str) -> int | float:
        """
        对简单算术问题做安全求值。

        当前支持的流程：
        1. 从自然语言 query 中剥离 `what is` 和问号
        2. 用 `ast.parse(..., mode='eval')` 解析表达式
        3. 只递归求值白名单中的算术节点

        这样做的目的，是避免直接使用 `eval()` 带来的执行风险。
        """
        expression = query.lower().replace("what is", "").replace("?", "").strip()
        node = ast.parse(expression, mode="eval")
        return self._eval_node(node.body)

    def _eval_node(self, node):  # noqa: ANN001
        """
        递归计算 AST 表达式节点。

        当前只允许：
        - 常量
        - 二元运算：+ - * /
        - 一元负号

        任何超出白名单的节点都会抛出异常，
        这是 `_safe_calculate()` 安全边界的一部分。
        """
        # 白名单运算符映射。
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.USub: operator.neg,
        }

        # 常量节点，例如数字 1、2、3。
        if isinstance(node, ast.Constant):
            return node.value

        # 二元运算节点，例如 1 + 2、3 * 4。
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](self._eval_node(node.left), self._eval_node(node.right))

        # 一元运算节点，例如 -5。
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](self._eval_node(node.operand))

        # 所有未显式允许的 AST 节点一律拒绝。
        raise ValueError("Unsupported calculation.")
