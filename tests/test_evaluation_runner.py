import unittest

from agentic_rag_lab.evaluation.runner import BenchmarkCase, BenchmarkMethod, run_benchmark


class EvaluationRunnerTests(unittest.TestCase):
    def test_run_benchmark_aggregates_scores(self) -> None:
        cases = [
            BenchmarkCase(query="q1", reference_answer="Oxford"),
            BenchmarkCase(query="q2", reference_answer="Tolkien"),
        ]
        methods = [
            BenchmarkMethod(
                name="perfect",
                run=lambda query: {"answer": "Oxford" if query == "q1" else "Tolkien", "citations": ["doc-1"]},
            ),
            BenchmarkMethod(
                name="partial",
                run=lambda query: {"answer": "Oxford", "citations": []},
            ),
        ]

        rows = run_benchmark(cases, methods)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].name, "perfect")
        self.assertEqual(rows[0].exact_match, 1.0)
        self.assertGreater(rows[0].token_f1, rows[1].token_f1)


if __name__ == "__main__":
    unittest.main()
