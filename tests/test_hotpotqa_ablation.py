import unittest

from agentic_rag_lab.evaluation.hotpotqa import (
    HotpotQACase,
    benchmark_retrievers_on_hotpotqa,
    build_hotpotqa_corpus,
)
from agentic_rag_lab.retrieval.hybrid import HybridRetriever
from agentic_rag_lab.retrieval.lexical import LexicalRetriever
from agentic_rag_lab.retrieval.semantic import SemanticRetriever


class HotpotQAAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "id": "q1",
                "question": "Which university did the author of The Hobbit attend?",
                "answer": "Exeter College Oxford",
                "type": "bridge",
                "level": "medium",
                "supporting_facts": [
                    {"title": "The Hobbit", "sent_id": 0},
                    {"title": "Tolkien at Oxford", "sent_id": 0},
                ],
                "contexts": [
                    {"title": "The Hobbit", "sentences": ["The Hobbit was written by J. R. R. Tolkien."]},
                    {
                        "title": "Tolkien at Oxford",
                        "sentences": ["J. R. R. Tolkien studied at Exeter College, Oxford."],
                    },
                ],
            },
            {
                "id": "q2",
                "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
                "answer": "yes",
                "type": "comparison",
                "level": "hard",
                "supporting_facts": [
                    {"title": "Scott Derrickson", "sent_id": 0},
                    {"title": "Ed Wood", "sent_id": 0},
                ],
                "contexts": [
                    {"title": "Scott Derrickson", "sentences": ["Scott Derrickson is an American director."]},
                    {"title": "Ed Wood", "sentences": ["Ed Wood was an American filmmaker."]},
                ],
            },
        ]
        self.cases = [
            HotpotQACase(
                case_id="q1",
                question="Which university did the author of The Hobbit attend?",
                answer="Exeter College Oxford",
                level="medium",
                question_type="bridge",
                supporting_titles={"The Hobbit", "Tolkien at Oxford"},
            ),
            HotpotQACase(
                case_id="q2",
                question="Were Scott Derrickson and Ed Wood of the same nationality?",
                answer="yes",
                level="hard",
                question_type="comparison",
                supporting_titles={"Scott Derrickson", "Ed Wood"},
            ),
        ]

    def test_benchmark_retrievers_returns_rows_for_each_method_and_k(self) -> None:
        corpus = build_hotpotqa_corpus(self.records)
        retrievers = {
            "lexical": LexicalRetriever(corpus),
            "semantic": SemanticRetriever(corpus),
            "hybrid": HybridRetriever(
                lexical=LexicalRetriever(corpus),
                semantic=SemanticRetriever(corpus),
            ),
        }

        rows = benchmark_retrievers_on_hotpotqa(self.cases, retrievers, top_ks=[1, 2])

        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["top_k"], 1)
        self.assertIn("method", rows[0])
        self.assertIn("supporting_doc_recall", rows[0])
        self.assertEqual({row["method"] for row in rows}, {"lexical", "semantic", "hybrid"})


if __name__ == "__main__":
    unittest.main()
