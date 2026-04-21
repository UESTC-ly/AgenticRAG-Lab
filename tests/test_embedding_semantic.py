import unittest

try:
    import numpy as np  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(
        "numpy not available; install requirements-real.txt to enable "
        "EmbeddingSemanticRetriever tests."
    ) from exc

from agentic_rag_lab.models import Document
from agentic_rag_lab.retrieval.embedding_semantic import EmbeddingSemanticRetriever


class FakeEmbeddingModel:
    """Deterministic stand-in for a sentence-transformers model.

    Maps any text to a 3-D unit vector based on keyword presence, so
    similarity comparisons are predictable without loading a real model.
    """

    def encode(self, texts, *, batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True):
        import numpy as np

        vectors = []
        for text in texts:
            lower = text.lower()
            vec = np.array(
                [
                    1.0 if "hobbit" in lower else 0.0,
                    1.0 if "tolkien" in lower or "oxford" in lower else 0.0,
                    1.0 if "python" in lower else 0.0,
                ],
                dtype=float,
            )
            norm = np.linalg.norm(vec)
            if normalize_embeddings and norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.vstack(vectors)


def corpus():
    return [
        Document(doc_id="hobbit", title="The Hobbit", content="The Hobbit was written by Tolkien."),
        Document(doc_id="oxford", title="Oxford", content="Tolkien studied at Oxford."),
        Document(doc_id="python", title="Python", content="Python is a programming language."),
    ]


class EmbeddingSemanticRetrieverTests(unittest.TestCase):
    def test_ranks_by_semantic_similarity(self) -> None:
        retriever = EmbeddingSemanticRetriever(corpus(), model=FakeEmbeddingModel())

        results = retriever.search("Where did Tolkien study?", top_k=2)

        self.assertEqual([item.document.doc_id for item in results], ["oxford", "hobbit"])
        self.assertGreater(results[0].score, 0)

    def test_filters_zero_similarity(self) -> None:
        retriever = EmbeddingSemanticRetriever(corpus(), model=FakeEmbeddingModel())

        results = retriever.search("Is there a Python library?", top_k=3)

        doc_ids = [item.document.doc_id for item in results]
        self.assertEqual(doc_ids, ["python"])

    def test_empty_corpus_returns_empty(self) -> None:
        retriever = EmbeddingSemanticRetriever([], model=FakeEmbeddingModel())

        self.assertEqual(retriever.search("anything", top_k=5), [])


if __name__ == "__main__":
    unittest.main()
