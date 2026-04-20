from __future__ import annotations

from typing import Protocol

from ..models import Document, EvidenceItem


class EmbeddingModel(Protocol):
    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = ...,
        normalize_embeddings: bool = ...,
        show_progress_bar: bool = ...,
        convert_to_numpy: bool = ...,
    ): ...


class EmbeddingSemanticRetriever:
    """Dense semantic retriever backed by a real embedding model.

    Default model is ``BAAI/bge-m3`` via ``sentence-transformers``. The model
    can be injected for testing. Embeddings are pre-computed once at
    construction and cached in memory.
    """

    def __init__(
        self,
        corpus: list[Document],
        *,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        batch_size: int = 32,
        model: EmbeddingModel | None = None,
    ) -> None:
        self.corpus = corpus
        self.model_name = model_name
        self.batch_size = batch_size

        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "EmbeddingSemanticRetriever requires `sentence-transformers`. "
                    "Install with: pip install sentence-transformers"
                ) from exc
            model = SentenceTransformer(model_name, device=device)
        self._model = model

        if corpus:
            texts = [f"{doc.title}. {doc.content}" for doc in corpus]
            self.doc_embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        else:
            self.doc_embeddings = None

    def search(self, query: str, top_k: int = 5) -> list[EvidenceItem]:
        if not self.corpus:
            return []

        import numpy as np

        query_emb = self._model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        scores = self.doc_embeddings @ query_emb  # cosine (normalized dot product)

        n = len(self.corpus)
        k = min(top_k, n)
        if k <= 0:
            return []

        if k == n:
            order = np.argsort(-scores)
        else:
            partitioned = np.argpartition(-scores, k)[:k]
            order = partitioned[np.argsort(-scores[partitioned])]

        results: list[EvidenceItem] = []
        for idx in order:
            score = float(scores[int(idx)])
            if score <= 0:
                continue
            doc = self.corpus[int(idx)]
            results.append(
                EvidenceItem(
                    document=doc,
                    score=score,
                    query=query,
                    citations=[doc.doc_id],
                )
            )
        return results
