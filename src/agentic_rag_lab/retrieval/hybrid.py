from __future__ import annotations

from ..models import EvidenceItem
from ..text import reciprocal_rank_fusion, token_counts


class HybridRetriever:
    """Hybrid retriever: lexical + semantic, fused with RRF.

    If a ``reranker`` is provided, the retriever recalls
    ``top_k * recall_multiplier`` candidates from fusion and asks the
    reranker (typically a cross-encoder) to produce the final top_k.

    Without a reranker, a lightweight token-overlap bonus is applied as a
    cheap rerank proxy — preserving the original offline behavior.
    """

    def __init__(
        self,
        lexical,
        semantic,
        *,
        reranker=None,
        recall_multiplier: int = 5,
    ) -> None:
        self.lexical = lexical
        self.semantic = semantic
        self.reranker = reranker
        self.recall_multiplier = recall_multiplier

    def search(self, query: str, top_k: int = 5) -> list[EvidenceItem]:
        candidate_k = (
            top_k * self.recall_multiplier if self.reranker is not None else top_k * 2
        )
        lexical_results = self.lexical.search(query, top_k=candidate_k)
        semantic_results = self.semantic.search(query, top_k=candidate_k)
        fused: dict[str, EvidenceItem] = {}

        for rank, item in enumerate(lexical_results, start=1):
            score = reciprocal_rank_fusion(rank) + (item.score * 0.01)
            fused[item.document.doc_id] = EvidenceItem(
                document=item.document,
                score=score,
                query=query,
                citations=item.citations,
            )

        for rank, item in enumerate(semantic_results, start=1):
            addition = reciprocal_rank_fusion(rank) + (item.score * 0.5)
            existing = fused.get(item.document.doc_id)
            if existing is None:
                fused[item.document.doc_id] = EvidenceItem(
                    document=item.document,
                    score=addition,
                    query=query,
                    citations=item.citations,
                )
                continue
            existing.score += addition

        if self.reranker is not None:
            candidates = sorted(
                fused.values(),
                key=lambda it: (it.score, it.document.doc_id),
                reverse=True,
            )[:candidate_k]
            return self.reranker.rerank(query, candidates, top_k=top_k)

        query_vector = token_counts(query, expand=True)
        reranked = []
        for item in fused.values():
            doc_vector = token_counts(
                f"{item.document.title} {item.document.content}",
                expand=True,
            )
            overlap_bonus = sum(1 for token in query_vector if token in doc_vector) * 0.02
            reranked.append(
                EvidenceItem(
                    document=item.document,
                    score=item.score + overlap_bonus,
                    query=query,
                    citations=item.citations,
                )
            )

        reranked.sort(key=lambda item: (item.score, item.document.doc_id), reverse=True)
        return reranked[:top_k]
