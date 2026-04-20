import unittest

from agentic_rag_lab.evaluation.metrics import exact_match, token_f1


class MetricsTests(unittest.TestCase):
    def test_exact_match_normalizes_case_and_punctuation(self) -> None:
        self.assertEqual(exact_match("Exeter College, Oxford", "exeter college oxford"), 1.0)

    def test_token_f1_handles_partial_overlap(self) -> None:
        score = token_f1("Exeter College Oxford", "Oxford")

        self.assertGreater(score, 0)
        self.assertLess(score, 1)


if __name__ == "__main__":
    unittest.main()
