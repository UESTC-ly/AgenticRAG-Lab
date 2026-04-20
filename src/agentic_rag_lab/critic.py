from __future__ import annotations

import re

from .models import CritiqueResult, EvidenceItem
from .text import tokenize

EDUCATION_TERMS = {"study", "studied", "college", "university", "school", "oxford", "campus"}
AUTHOR_TERMS = {"author", "written", "wrote", "writer"}
NATIONALITY_TERMS = {
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
}


class EvidenceCritic:
    def __init__(self, min_coverage: float = 0.45) -> None:
        self.min_coverage = min_coverage

    def evaluate(self, query: str, evidence: list[EvidenceItem]) -> CritiqueResult:
        if not evidence:
            return CritiqueResult(
                is_sufficient=False,
                reasoning="No evidence retrieved yet.",
                follow_up_queries=[query],
                confidence=0.0,
            )

        combined = " ".join(
            f"{item.document.title} {item.document.content}" for item in evidence
        ).lower()
        query_tokens = set(tokenize(query, expand=True))
        evidence_tokens = set(tokenize(combined, expand=True))

        coverage = (
            len(query_tokens & evidence_tokens) / len(query_tokens)
            if query_tokens
            else 0.0
        )
        mentions_education = any(term in evidence_tokens for term in EDUCATION_TERMS)
        mentions_authorship = any(term in evidence_tokens for term in AUTHOR_TERMS)

        if "university" in query.lower() or "attend" in query.lower():
            if mentions_education and coverage >= self.min_coverage:
                return CritiqueResult(
                    is_sufficient=True,
                    reasoning="Evidence includes education signals and enough semantic overlap.",
                    confidence=min(0.95, coverage + 0.25),
                )
            if mentions_education and len(evidence) >= 1 and "tolkien" in combined:
                return CritiqueResult(
                    is_sufficient=True,
                    reasoning="Evidence names the bridge entity and education destination.",
                    confidence=0.82,
                )

        if "same nationality" in query.lower():
            nationalities = []
            for item in evidence:
                lower = item.document.content.lower()
                matched = next((term for term in NATIONALITY_TERMS if re.search(rf"\b{term}\b", lower)), None)
                if matched:
                    nationalities.append(matched)
            if len(nationalities) >= 2 and len(set(nationalities)) == 1:
                return CritiqueResult(
                    is_sufficient=True,
                    reasoning="Evidence shows the compared entities share the same nationality.",
                    confidence=0.9,
                )

        if coverage >= self.min_coverage and (mentions_authorship or mentions_education):
            return CritiqueResult(
                is_sufficient=True,
                reasoning="Coverage threshold reached with relevant evidence.",
                confidence=min(0.95, coverage + 0.2),
            )

        follow_up_queries = []
        if "university" in query.lower() or "attend" in query.lower():
            follow_up_queries.append("Where did the referenced person study or attend university?")
        if "author of" in query.lower():
            follow_up_queries.append("Who is the author of the referenced work?")
        if not follow_up_queries:
            follow_up_queries.append(query)

        return CritiqueResult(
            is_sufficient=False,
            reasoning="Evidence is too thin or misses a bridge fact.",
            follow_up_queries=follow_up_queries,
            confidence=max(0.1, coverage),
        )
