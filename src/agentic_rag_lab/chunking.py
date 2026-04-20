from __future__ import annotations

import re


def chunk_text(text: str, *, max_chars: int = 240, overlap_chars: int = 40) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            chunks.append(current)
            overlap = current[-overlap_chars:].strip()
            current = f"{overlap} {sentence}".strip() if overlap else sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
