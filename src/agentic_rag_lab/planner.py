from __future__ import annotations

import re

from .models import PlanStep, QueryPlan


class RuleBasedPlanner:
    def plan(self, query: str) -> QueryPlan:
        cleaned = query.strip()
        lower = cleaned.lower()

        if "author of" in lower and "which university" in lower:
            work_match = re.search(r"author of ([^?]+?)(?: attend| before| after|\?)", cleaned, re.IGNORECASE)
            work = work_match.group(1).strip() if work_match else "the referenced work"
            steps = [
                PlanStep(
                    step_id="identify-bridge-entity",
                    query=f"Who is the author of {work}?",
                    purpose="Find the bridge entity needed for the next hop.",
                ),
                PlanStep(
                    step_id="resolve-destination",
                    query=cleaned,
                    purpose="Find the final institution once the bridge entity is known.",
                    depends_on=["identify-bridge-entity"],
                ),
            ]
            return QueryPlan(original_query=cleaned, steps=steps)

        return QueryPlan(
            original_query=cleaned,
            steps=[
                PlanStep(
                    step_id="single-hop",
                    query=cleaned,
                    purpose="Answer directly or retrieve once.",
                )
            ],
        )

