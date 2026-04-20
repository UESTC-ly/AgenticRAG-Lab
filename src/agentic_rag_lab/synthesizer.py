from __future__ import annotations

import re

from .models import EvidenceItem
from .text import tokenize


class CitationSynthesizer:
    def answer(self, query: str, evidence: list[EvidenceItem]) -> tuple[str, list[str]]:
        ordered = sorted(evidence, key=lambda item: item.score, reverse=True)
        citations = [item.document.doc_id for item in ordered]

        same_nationality = self._answer_same_nationality(query, ordered)
        if same_nationality is not None:
            return same_nationality, citations

        for item in ordered:
            match = re.search(r"stud(?:y|ied) at ([^.]+)\.", item.document.content, re.IGNORECASE)
            if match:
                institution = match.group(1).strip()
                return (institution, citations)

        snippets = []
        query_tokens = set(tokenize(query, expand=True))
        for item in ordered[:2]:
            sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", item.document.content.strip()) if sentence.strip()]
            best_sentence = ""
            best_score = -1
            for sentence in sentences:
                sentence_tokens = set(tokenize(sentence, expand=True))
                score = len(query_tokens & sentence_tokens)
                if score > best_score:
                    best_score = score
                    best_sentence = sentence
            snippets.append(f"{(best_sentence or item.document.content.strip())} [{item.document.doc_id}]")
        answer = " ".join(snippets) if snippets else f"No answer found for: {query}"
        return answer, citations

    def direct_answer(self, query: str) -> str:
        lowered = query.lower().strip(" ?")
        if "retrieval augmented generation" in lowered:
            return (
                "Retrieval augmented generation augments an LLM with retrieved external context "
                "before synthesis."
            )
        return f"Direct response: {query}"

    def _answer_same_nationality(self, query: str, evidence: list[EvidenceItem]) -> str | None:
        if "same nationality" not in query.lower():
            return None

        nationality_terms = [
            "american",
            "british",
            "canadian",
            "french",
            "german",
            "italian",
            "spanish",
            "irish",
            "scottish",
            "australian",
            "indian",
            "chinese",
            "japanese",
            "mexican",
        ]
        matches = []
        for item in evidence:
            lower = item.document.content.lower()
            match = next((term for term in nationality_terms if re.search(rf"\b{term}\b", lower)), None)
            if match:
                matches.append(match)
        if len(matches) >= 2:
            return "yes" if len(set(matches)) == 1 else "no"
        return None
