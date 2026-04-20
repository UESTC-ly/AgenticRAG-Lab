from __future__ import annotations

from typing import Protocol

from ..models import EvidenceItem


class RerankModel(Protocol):
    def predict(
        self,
        pairs: list[list[str]],
        *,
        batch_size: int = ...,
        show_progress_bar: bool = ...,
    ): ...


class CrossEncoderReranker:
    """Cross-encoder reranker backed by ``BAAI/bge-reranker-v2-m3``.

    Applied after the fusion step: takes a candidate list from
    ``HybridRetriever`` and rescores each (query, document) pair with a
    cross-encoder, which is much more accurate than bi-encoder similarity
    at the cost of O(N) model calls per query.
    """

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int = 16,
        model: RerankModel | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "CrossEncoderReranker requires `sentence-transformers`. "
                    "Install with: pip install sentence-transformers"
                ) from exc
            model = CrossEncoder(model_name, device=device)
        self._model = model

    def rerank(
        self,
        query: str,
        items: list[EvidenceItem],
        *,
        top_k: int | None = None,
    ) -> list[EvidenceItem]:
        if not items:
            return []

        pairs = [
            [query, f"{item.document.title}. {item.document.content}"]
            for item in items
        ]
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        rescored = [
            EvidenceItem(
                document=item.document,
                score=float(score),
                query=query,
                citations=item.citations,
            )
            for item, score in zip(items, scores)
        ]
        rescored.sort(
            key=lambda it: (it.score, it.document.doc_id),
            reverse=True,
        )
        if top_k is not None:
            rescored = rescored[:top_k]
        return rescored
