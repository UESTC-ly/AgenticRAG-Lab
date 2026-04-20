import unittest

from agentic_rag_lab.critic import EvidenceCritic
from agentic_rag_lab.models import Document, EvidenceItem


class CriticTests(unittest.TestCase):
    def test_critic_accepts_sufficient_evidence(self) -> None:
        critic = EvidenceCritic()
        evidence = [
            EvidenceItem(
                document=Document(
                    doc_id="tolkien-oxford",
                    title="Tolkien at Oxford",
                    content="J. R. R. Tolkien studied at Exeter College, Oxford.",
                    metadata={"source": "sample"},
                ),
                score=0.92,
                query="Where did Tolkien study?",
                citations=["tolkien-oxford"],
            )
        ]

        critique = critic.evaluate(
            "Which university did the author of The Hobbit attend?",
            evidence,
        )

        self.assertTrue(critique.is_sufficient)
        self.assertEqual(critique.follow_up_queries, [])

    def test_critic_requests_follow_up_when_evidence_is_thin(self) -> None:
        critic = EvidenceCritic(min_coverage=0.7)
        evidence = [
            EvidenceItem(
                document=Document(
                    doc_id="hobbit-author",
                    title="The Hobbit",
                    content="The Hobbit was written by J. R. R. Tolkien.",
                    metadata={"source": "sample"},
                ),
                score=0.55,
                query="author of The Hobbit",
                citations=["hobbit-author"],
            )
        ]

        critique = critic.evaluate(
            "Which university did the author of The Hobbit attend?",
            evidence,
        )

        self.assertFalse(critique.is_sufficient)
        self.assertTrue(critique.follow_up_queries)

    def test_critic_accepts_same_nationality_comparison_when_evidence_aligns(self) -> None:
        critic = EvidenceCritic()
        evidence = [
            EvidenceItem(
                document=Document(
                    doc_id="scott",
                    title="Scott Derrickson",
                    content="Scott Derrickson is an American director.",
                    metadata={"source": "sample"},
                ),
                score=0.9,
                query="Scott Derrickson nationality",
                citations=["scott"],
            ),
            EvidenceItem(
                document=Document(
                    doc_id="ed",
                    title="Ed Wood",
                    content="Ed Wood was an American filmmaker.",
                    metadata={"source": "sample"},
                ),
                score=0.88,
                query="Ed Wood nationality",
                citations=["ed"],
            ),
        ]

        critique = critic.evaluate(
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            evidence,
        )

        self.assertTrue(critique.is_sufficient)


if __name__ == "__main__":
    unittest.main()
