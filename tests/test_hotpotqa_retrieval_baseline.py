import tempfile
import unittest
from pathlib import Path

from agentic_rag_lab.data.hotpotqa import export_slice_jsonl
from agentic_rag_lab.evaluation.hotpotqa import (
    HotpotQACase,
    build_hotpotqa_corpus,
    build_hotpotqa_cases,
    evaluate_retriever_on_hotpotqa,
    load_hotpotqa_jsonl_slice,
)
from agentic_rag_lab.retrieval.hybrid import HybridRetriever
from agentic_rag_lab.retrieval.lexical import LexicalRetriever
from agentic_rag_lab.retrieval.semantic import SemanticRetriever


class HotpotQARetrievalBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.slice_records = [
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

    def test_load_jsonl_and_build_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "slice.jsonl"
            export_slice_jsonl(self.slice_records, path)
            loaded = load_hotpotqa_jsonl_slice(path)

        cases = build_hotpotqa_cases(loaded)

        self.assertEqual(len(cases), 2)
        self.assertIsInstance(cases[0], HotpotQACase)
        self.assertEqual(cases[0].supporting_titles, {"The Hobbit", "Tolkien at Oxford"})

    def test_build_hotpotqa_corpus_deduplicates_titles(self) -> None:
        duplicated = self.slice_records + [
            {
                "id": "q3",
                "question": "dup",
                "answer": "dup",
                "type": "bridge",
                "level": "medium",
                "supporting_facts": [{"title": "The Hobbit", "sent_id": 0}],
                "contexts": [{"title": "The Hobbit", "sentences": ["The Hobbit was written by Tolkien."]}],
            }
        ]

        corpus = build_hotpotqa_corpus(duplicated)
        doc_ids = {doc.doc_id for doc in corpus}

        self.assertEqual(len(corpus), 4)
        self.assertIn("title::The Hobbit", doc_ids)

    def test_evaluate_retriever_on_hotpotqa_reports_hit_metrics(self) -> None:
        corpus = build_hotpotqa_corpus(self.slice_records)
        cases = build_hotpotqa_cases(self.slice_records)
        retriever = HybridRetriever(
            lexical=LexicalRetriever(corpus),
            semantic=SemanticRetriever(corpus),
        )

        metrics = evaluate_retriever_on_hotpotqa(cases, retriever, top_k=2)

        self.assertEqual(metrics["case_count"], 2)
        self.assertGreater(metrics["supporting_doc_recall@2"], 0)
        self.assertGreater(metrics["all_supporting_docs_hit_rate@2"], 0)


if __name__ == "__main__":
    unittest.main()
