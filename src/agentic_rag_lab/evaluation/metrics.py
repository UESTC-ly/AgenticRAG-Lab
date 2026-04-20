from __future__ import annotations

from collections import Counter

from ..text import tokenize


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if " ".join(tokenize(prediction)) == " ".join(tokenize(reference)) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred = Counter(tokenize(prediction))
    gold = Counter(tokenize(reference))
    if not pred or not gold:
        return 0.0
    overlap = sum((pred & gold).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(gold.values())
    return 2 * precision * recall / (precision + recall)

