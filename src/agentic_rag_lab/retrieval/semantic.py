from __future__ import annotations

from collections import Counter

from ..models import Document, EvidenceItem
from ..text import cosine_like, token_counts


class SemanticRetriever:
    def __init__(self, corpus: list[Document]) -> None:
        self.corpus = corpus
        self.doc_vectors = {
            document.doc_id: token_counts(f"{document.title} {document.content}", expand=True)
            for document in corpus
        }

    def search(self, query: str, top_k: int = 5) -> list[EvidenceItem]:
        query_vector = token_counts(query, expand=True)
        scored: list[EvidenceItem] = []

        for document in self.corpus:
            score = cosine_like(query_vector, self.doc_vectors[document.doc_id])
            if score <= 0:
                continue
            scored.append(
                EvidenceItem(
                    document=document,
                    score=score,
                    query=query,
                    citations=[document.doc_id],
                )
            )

        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

