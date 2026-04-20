import unittest

from agentic_rag_lab.models import RouteTarget
from agentic_rag_lab.router import Router


class RouterTests(unittest.TestCase):
    def test_routes_simple_question_to_direct_answer(self) -> None:
        router = Router()

        decision = router.route("What is retrieval augmented generation?")

        self.assertIs(decision.target, RouteTarget.DIRECT_ANSWER)
        self.assertFalse(decision.needs_planning)

    def test_routes_multi_hop_question_to_agent_loop(self) -> None:
        router = Router()

        decision = router.route(
            "Which university did the author of The Hobbit attend before becoming a professor?"
        )

        self.assertIs(decision.target, RouteTarget.AGENTIC_RAG)
        self.assertTrue(decision.needs_planning)

    def test_routes_math_queries_to_calculator(self) -> None:
        router = Router()

        decision = router.route("What is (17 + 5) * 3?")

        self.assertIs(decision.target, RouteTarget.CALCULATOR)

    def test_routes_comparison_question_to_agent_loop(self) -> None:
        router = Router()

        decision = router.route("Were Scott Derrickson and Ed Wood of the same nationality?")

        self.assertIs(decision.target, RouteTarget.AGENTIC_RAG)
        self.assertTrue(decision.needs_planning)


if __name__ == "__main__":
    unittest.main()
