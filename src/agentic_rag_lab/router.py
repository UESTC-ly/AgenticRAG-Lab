from __future__ import annotations

import re

from .models import RouteDecision, RouteTarget

MULTI_HOP_HINTS = (
    "before",
    "after",
    "author of",
    "founded by",
    "acquired",
    "which university did",
    "which company",
    "same nationality",
    "were ",
    "are ",
    "did ",
)


class Router:
    def route(self, query: str) -> RouteDecision:
        stripped = query.strip()
        lowered = stripped.lower()

        if self._looks_like_math(stripped):
            return RouteDecision(
                target=RouteTarget.CALCULATOR,
                needs_planning=False,
                rationale="Detected arithmetic expression.",
            )

        if any(hint in lowered for hint in MULTI_HOP_HINTS):
            return RouteDecision(
                target=RouteTarget.AGENTIC_RAG,
                needs_planning=True,
                rationale="Detected multi-hop reasoning cues.",
            )

        if "retrieval augmented generation" in lowered:
            return RouteDecision(
                target=RouteTarget.DIRECT_ANSWER,
                needs_planning=False,
                rationale="Known lightweight definition query routed to direct answer.",
            )

        return RouteDecision(
            target=RouteTarget.AGENTIC_RAG,
            needs_planning=True,
            rationale="Defaulting to retrieval-backed answering for non-trivial natural-language questions.",
        )

    @staticmethod
    def _looks_like_math(query: str) -> bool:
        return bool(re.fullmatch(r"(?i)\s*what is\s+[\d\s\+\-\*\/\(\)\.]+\??\s*", query))
