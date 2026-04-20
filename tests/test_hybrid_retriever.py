import unittest

from agentic_rag_lab.models import Document
from agentic_rag_lab.retrieval.hybrid import HybridRetriever
from agentic_rag_lab.retrieval.lexical import LexicalRetriever
from agentic_rag_lab.retrieval.semantic import SemanticRetriever


class HybridRetrieverTests(unittest.TestCase):
    def test_hybrid_retriever_merges_and_reranks_results(self) -> None:
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
            Document(
                doc_id="noise",
                title="Unrelated",
                content="Bananas are yellow and unrelated to Tolkien.",
                metadata={"source": "sample"},
            ),
        ]
        retriever = HybridRetriever(
            lexical=LexicalRetriever(corpus),
            semantic=SemanticRetriever(corpus),
        )

        results = retriever.search("author of The Hobbit university", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertIn(results[0].document.doc_id, {"hobbit-author", "tolkien-oxford"})
        self.assertTrue(all(result.score > 0 for result in results))


if __name__ == "__main__":
    unittest.main()
