from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "by",
    "did",
    "do",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "what",
    "which",
    "who",
}

SYNONYMS = {
    "attend": {"study", "studied", "education", "college", "university"},
    "attended": {"study", "studied", "education", "college", "university"},
    "university": {"college", "campus", "school", "oxford"},
    "college": {"university", "school", "campus"},
    "author": {"writer", "written", "wrote"},
    "wrote": {"author", "written", "writer"},
    "written": {"author", "wrote", "writer"},
    "professor": {"academic", "scholar", "faculty"},
    "founded": {"founded", "started", "created"},
}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, *, expand: bool = False) -> list[str]:
    tokens = [token for token in normalize(text).split() if token and token not in STOPWORDS]
    if not expand:
        return tokens

    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(sorted(SYNONYMS.get(token, set())))
    return expanded


def token_counts(text: str, *, expand: bool = False) -> Counter[str]:
    return Counter(tokenize(text, expand=expand))


def reciprocal_rank_fusion(rank: int, *, k: int = 60) -> float:
    return 1.0 / (k + rank)


def cosine_like(query_tokens: Counter[str], doc_tokens: Counter[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    overlap = sum(query_tokens[token] * doc_tokens.get(token, 0) for token in query_tokens)
    if overlap <= 0:
        return 0.0
    q_norm = math.sqrt(sum(value * value for value in query_tokens.values()))
    d_norm = math.sqrt(sum(value * value for value in doc_tokens.values()))
    return overlap / (q_norm * d_norm)

