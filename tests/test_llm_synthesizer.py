import unittest

from agentic_rag_lab.llm_synthesizer import LLMSynthesizer
from agentic_rag_lab.models import Document, EvidenceItem
from agentic_rag_lab.synthesizer import CitationSynthesizer


def evidence():
    return [
        EvidenceItem(
            document=Document(doc_id="hobbit", title="The Hobbit", content="Written by Tolkien."),
            score=0.9,
            query="",
            citations=["hobbit"],
        ),
        EvidenceItem(
            document=Document(doc_id="oxford", title="Oxford", content="Tolkien studied at Oxford."),
            score=0.7,
            query="",
            citations=["oxford"],
        ),
    ]


class LLMSynthesizerTests(unittest.TestCase):
    def test_answer_uses_transport_and_extracts_citations(self) -> None:
        calls = []

        def fake_transport(prompt: str) -> str:
            calls.append(prompt)
            # Evidence is numbered [1]=hobbit (score 0.9), [2]=oxford (score 0.7).
            return "Tolkien studied at Oxford [2] after writing The Hobbit [1]."

        synth = LLMSynthesizer(transport=fake_transport)

        answer, citations = synth.answer("Where did the author of The Hobbit study?", evidence())

        self.assertIn("Oxford", answer)
        self.assertEqual(citations, ["oxford", "hobbit"])
        self.assertEqual(len(calls), 1)
        self.assertIn("[1] Title: The Hobbit", calls[0])
        self.assertIn("[2] Title: Oxford", calls[0])

    def test_falls_back_when_transport_raises(self) -> None:
        def broken_transport(prompt: str) -> str:
            raise RuntimeError("ollama down")

        synth = LLMSynthesizer(transport=broken_transport, fallback=CitationSynthesizer())

        answer, citations = synth.answer("Where did Tolkien study?", evidence())

        self.assertTrue(answer)
        self.assertTrue(citations)

    def test_empty_evidence_short_circuits_to_fallback(self) -> None:
        def transport_should_not_run(prompt: str) -> str:
            raise AssertionError("transport should not be called when evidence is empty")

        synth = LLMSynthesizer(transport=transport_should_not_run)

        answer, citations = synth.answer("Anything?", [])

        self.assertIsInstance(answer, str)
        self.assertEqual(citations, [])

    def test_direct_answer_uses_transport(self) -> None:
        synth = LLMSynthesizer(transport=lambda prompt: "A concise direct reply.")

        self.assertEqual(synth.direct_answer("What is RAG?"), "A concise direct reply.")

    def test_invalid_citations_are_discarded(self) -> None:
        # [1] is valid (maps to hobbit); [99] and [unknown-doc] are invalid.
        synth = LLMSynthesizer(transport=lambda prompt: "Answer [99] [unknown-doc] [1].")

        _, citations = synth.answer("Q", evidence())

        self.assertEqual(citations, ["hobbit"])

    def test_legacy_doc_id_citation_still_works(self) -> None:
        # Backwards-compat: if the model outputs raw doc_ids, still map them.
        synth = LLMSynthesizer(transport=lambda prompt: "Answer [hobbit].")

        _, citations = synth.answer("Q", evidence())

        self.assertEqual(citations, ["hobbit"])


if __name__ == "__main__":
    unittest.main()
