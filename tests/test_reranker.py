import unittest

from agentic_rag_lab.models import Document, EvidenceItem
from agentic_rag_lab.retrieval.hybrid import HybridRetriever
from agentic_rag_lab.retrieval.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    """Scores pairs by keyword hit count so rerank behavior is deterministic."""

    def predict(self, pairs, *, batch_size=16, show_progress_bar=False):
        scores = []
        for query, doc in pairs:
            query_tokens = set(query.lower().split())
            doc_tokens = set(doc.lower().split())
            scores.append(float(len(query_tokens & doc_tokens)))
        return scores


def evidence(doc_id: str, content: str, score: float) -> EvidenceItem:
    doc = Document(doc_id=doc_id, title=doc_id, content=content)
    return EvidenceItem(document=doc, score=score, query="", citations=[doc_id])


class RerankerTests(unittest.TestCase):
    def test_reorders_by_cross_encoder_score(self) -> None:
        reranker = CrossEncoderReranker(model=FakeCrossEncoder())
        items = [
            evidence("a", "apple banana", 0.9),
            evidence("b", "tolkien oxford university", 0.1),
        ]

        result = reranker.rerank("where did tolkien study", items, top_k=2)

        self.assertEqual([item.document.doc_id for item in result], ["b", "a"])

    def test_top_k_truncates(self) -> None:
        reranker = CrossEncoderReranker(model=FakeCrossEncoder())
        items = [evidence(f"d{i}", f"word{i} tolkien", 0.1) for i in range(5)]

        result = reranker.rerank("tolkien", items, top_k=2)

        self.assertEqual(len(result), 2)


class FakeLexical:
    def __init__(self, corpus):
        self.corpus = corpus

    def search(self, query, top_k=5):
        return [EvidenceItem(document=d, score=0.5, query=query, citations=[d.doc_id]) for d in self.corpus[:top_k]]


class FakeSemantic(FakeLexical):
    pass


class HybridWithRerankerTests(unittest.TestCase):
    def test_hybrid_uses_reranker_when_provided(self) -> None:
        corpus = [
            Document(doc_id="a", title="A", content="apple banana"),
            Document(doc_id="b", title="B", content="tolkien oxford"),
            Document(doc_id="c", title="C", content="nothing relevant"),
        ]
        hybrid = HybridRetriever(
            lexical=FakeLexical(corpus),
            semantic=FakeSemantic(corpus),
            reranker=CrossEncoderReranker(model=FakeCrossEncoder()),
            recall_multiplier=3,
        )

        results = hybrid.search("tolkien oxford", top_k=1)

        self.assertEqual(results[0].document.doc_id, "b")


if __name__ == "__main__":
    unittest.main()
