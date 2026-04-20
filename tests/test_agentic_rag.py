import unittest

from agentic_rag_lab.agent import AgenticRAG
from agentic_rag_lab.critic import EvidenceCritic
from agentic_rag_lab.models import Document
from agentic_rag_lab.planner import RuleBasedPlanner
from agentic_rag_lab.retrieval.hybrid import HybridRetriever
from agentic_rag_lab.retrieval.lexical import LexicalRetriever
from agentic_rag_lab.retrieval.semantic import SemanticRetriever
from agentic_rag_lab.router import Router
from agentic_rag_lab.synthesizer import CitationSynthesizer


def build_agent() -> AgenticRAG:
    corpus = [
        Document(
            doc_id="hobbit-author",
            title="The Hobbit",
            content="The Hobbit was written by J. R. R. Tolkien.",
            metadata={"source": "sample"},
        ),
        Document(
            doc_id="tolkien-oxford",
            title="Tolkien at Oxford",
            content="J. R. R. Tolkien studied at Exeter College, Oxford.",
            metadata={"source": "sample"},
        ),
    ]
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
        max_iterations=2,
    )


class AgenticRAGTests(unittest.TestCase):
    def test_agentic_rag_returns_cited_answer(self) -> None:
        agent = build_agent()

        result = agent.run(
            "Which university did the author of The Hobbit attend before becoming a professor?"
        )

        self.assertIn("Oxford", result.answer)
        self.assertTrue(result.citations)
        self.assertGreaterEqual(result.iterations, 1)

    def test_agentic_rag_uses_direct_answer_route_for_simple_queries(self) -> None:
        agent = build_agent()

        result = agent.run("What is retrieval augmented generation?")

        self.assertEqual(result.route, "direct_answer")
        self.assertTrue(result.answer)


if __name__ == "__main__":
    unittest.main()
