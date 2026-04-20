from __future__ import annotations

import math
from collections import Counter

from ..models import Document, EvidenceItem
from ..text import tokenize


class LexicalRetriever:
    def __init__(self, corpus: list[Document]) -> None:
        self.corpus = corpus
        self.doc_tokens = {
            document.doc_id: Counter(tokenize(f"{document.title} {document.content}"))
            for document in corpus
        }
        self.doc_freq = Counter()
        for tokens in self.doc_tokens.values():
            for token in tokens:
                self.doc_freq[token] += 1

    def search(self, query: str, top_k: int = 5) -> list[EvidenceItem]:
        query_tokens = Counter(tokenize(query))
        scored: list[EvidenceItem] = []
        total_docs = max(len(self.corpus), 1)

        for document in self.corpus:
            doc_tokens = self.doc_tokens[document.doc_id]
            score = 0.0
            for token, q_tf in query_tokens.items():
                if token not in doc_tokens:
                    continue
                idf = math.log((1 + total_docs) / (1 + self.doc_freq[token])) + 1.0
                score += q_tf * doc_tokens[token] * idf
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

