import unittest

from agentic_rag_lab.planner import RuleBasedPlanner


class PlannerTests(unittest.TestCase):
    def test_planner_keeps_simple_query_single_step(self) -> None:
        planner = RuleBasedPlanner()

        plan = planner.plan("What is retrieval augmented generation?")

        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].query, "What is retrieval augmented generation?")

    def test_planner_splits_multi_hop_query_into_subqueries(self) -> None:
        planner = RuleBasedPlanner()

        plan = planner.plan(
            "Which university did the author of The Hobbit attend before becoming a professor?"
        )

        self.assertGreaterEqual(len(plan.steps), 2)
        self.assertTrue(any("author of The Hobbit" in step.query for step in plan.steps))
        self.assertTrue(any("which university" in step.query.lower() for step in plan.steps))


if __name__ == "__main__":
    unittest.main()
