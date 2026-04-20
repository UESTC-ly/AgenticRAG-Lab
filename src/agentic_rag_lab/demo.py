from __future__ import annotations

import sys

from .agent import AgenticRAG
from .critic import EvidenceCritic
from .evaluation.runner import BenchmarkCase, BenchmarkMethod, format_markdown_table, run_benchmark
from .models import Document
from .planner import RuleBasedPlanner
from .retrieval.hybrid import HybridRetriever
from .retrieval.lexical import LexicalRetriever
from .retrieval.semantic import SemanticRetriever
from .router import Router
from .synthesizer import CitationSynthesizer


def build_demo_corpus() -> list[Document]:
    return [
        Document(
            doc_id="hobbit-author",
            title="The Hobbit",
            content="The Hobbit was written by J. R. R. Tolkien.",
            metadata={"source": "demo"},
        ),
        Document(
            doc_id="tolkien-oxford",
            title="Tolkien at Oxford",
            content="J. R. R. Tolkien studied at Exeter College, Oxford.",
            metadata={"source": "demo"},
        ),
        Document(
            doc_id="rag-definition",
            title="RAG definition",
            content="Retrieval augmented generation combines retrieval with LLM generation.",
            metadata={"source": "demo"},
        ),
    ]


def build_demo_agent() -> AgenticRAG:
    corpus = build_demo_corpus()
    retriever = HybridRetriever(
        lexical=LexicalRetriever(corpus),
        semantic=SemanticRetriever(corpus),
    )
    return AgenticRAG(
        router=Router(),
        planner=RuleBasedPlanner(),
        retriever=retriever,
        critic=EvidenceCritic(),
        synthesizer=CitationSynthesizer(),
    )


def build_real_agent(
    *,
    embedding_model: str = "BAAI/bge-m3",
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    llm_model: str = "qwen2.5:7b",
    ollama_url: str = "http://localhost:11434",
    fallback_on_error: bool = True,
) -> AgenticRAG:
    """Build an agent using real embedding + reranker + LLM components.

    Each upgrade is independent and falls back gracefully if the required
    dependency or service is unavailable (so the demo still runs).
    """
    from .llm_synthesizer import LLMSynthesizer

    corpus = build_demo_corpus()
    lexical = LexicalRetriever(corpus)

    try:
        from .retrieval.embedding_semantic import EmbeddingSemanticRetriever

        semantic = EmbeddingSemanticRetriever(corpus, model_name=embedding_model)
    except ImportError as exc:
        if not fallback_on_error:
            raise
        print(
            f"[warn] embedding model unavailable ({exc}); falling back to offline semantic retriever.",
            file=sys.stderr,
        )
        semantic = SemanticRetriever(corpus)

    reranker = None
    try:
        from .retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name=reranker_model)
    except ImportError as exc:
        if not fallback_on_error:
            raise
        print(
            f"[warn] reranker unavailable ({exc}); using fusion-only retrieval.",
            file=sys.stderr,
        )

    retriever = HybridRetriever(lexical=lexical, semantic=semantic, reranker=reranker)

    synthesizer = LLMSynthesizer(
        model=llm_model,
        base_url=ollama_url,
        fallback=CitationSynthesizer() if fallback_on_error else None,
    )

    return AgenticRAG(
        router=Router(),
        planner=RuleBasedPlanner(),
        retriever=retriever,
        critic=EvidenceCritic(),
        synthesizer=synthesizer,
    )


def build_demo_benchmark_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            query="Which university did the author of The Hobbit attend before becoming a professor?",
            reference_answer="Exeter College Oxford",
        ),
        BenchmarkCase(
            query="What is retrieval augmented generation?",
            reference_answer="Retrieval augmented generation augments an LLM with retrieved external context before synthesis.",
        ),
    ]


def run_demo_benchmark() -> str:
    agent = build_demo_agent()
    synthesizer = CitationSynthesizer()
    retriever = agent.retriever

    methods = [
        BenchmarkMethod(
            name="Direct Answer",
            run=lambda query: {"answer": synthesizer.direct_answer(query), "citations": []},
        ),
        BenchmarkMethod(
            name="Hybrid Retrieve + Synthesize",
            run=lambda query: (
                lambda results: {
                    "answer": synthesizer.answer(query, results)[0],
                    "citations": synthesizer.answer(query, results)[1],
                }
            )(retriever.search(query, top_k=3)),
        ),
        BenchmarkMethod(
            name="Full AgenticRAG",
            run=lambda query: (
                lambda result: {"answer": result.answer, "citations": result.citations}
            )(agent.run(query)),
        ),
    ]

    rows = run_benchmark(build_demo_benchmark_cases(), methods)
    return format_markdown_table(rows)
