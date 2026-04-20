import tempfile
import unittest
from pathlib import Path

from agentic_rag_lab.data.hotpotqa import export_slice_jsonl
from agentic_rag_lab.mvp import HotpotQAMVPService


class MVPServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "id": "q1",
                "question": "Which university did the author of The Hobbit attend?",
                "answer": "Exeter College Oxford",
                "type": "bridge",
                "level": "medium",
                "supporting_facts": [
                    {"title": "The Hobbit", "sent_id": 0},
                    {"title": "Tolkien at Oxford", "sent_id": 0},
                ],
                "contexts": [
                    {"title": "The Hobbit", "sentences": ["The Hobbit was written by J. R. R. Tolkien."]},
                    {"title": "Tolkien at Oxford", "sentences": ["J. R. R. Tolkien studied at Exeter College, Oxford."]},
                ],
            },
            {
                "id": "q2",
                "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
                "answer": "yes",
                "type": "comparison",
                "level": "hard",
                "supporting_facts": [
                    {"title": "Scott Derrickson", "sent_id": 0},
                    {"title": "Ed Wood", "sent_id": 0},
                ],
                "contexts": [
                    {"title": "Scott Derrickson", "sentences": ["Scott Derrickson is an American director."]},
                    {"title": "Ed Wood", "sentences": ["Ed Wood was an American filmmaker."]},
                ],
            },
        ]

    def test_service_builds_stats_and_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "slice.jsonl"
            export_slice_jsonl(self.records, path)
            service = HotpotQAMVPService.from_slice(path)

        stats = service.stats()
        examples = service.example_questions(limit=2)

        self.assertEqual(stats["case_count"], 2)
        self.assertEqual(len(examples), 2)
        self.assertIn("query_modes", stats)

    def test_service_answers_query_with_trace_and_reference_when_known(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "slice.jsonl"
            export_slice_jsonl(self.records, path)
            service = HotpotQAMVPService.from_slice(path)

        result = service.ask("Which university did the author of The Hobbit attend?")

        self.assertIn("answer", result)
        self.assertIn("trace", result)
        self.assertTrue(result["citations"])
        self.assertEqual(result["reference_answer"], "Exeter College Oxford")

    def test_service_answers_same_nationality_question_concisely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "slice.jsonl"
            export_slice_jsonl(self.records, path)
            service = HotpotQAMVPService.from_slice(path)

        result = service.ask("Were Scott Derrickson and Ed Wood of the same nationality?")

        self.assertEqual(result["route"], "agentic_rag")
        self.assertIn(result["answer"].lower(), {"yes", "yes."})

    def test_service_can_add_and_persist_custom_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            slice_path = Path(tmp_dir) / "slice.jsonl"
            user_doc_path = Path(tmp_dir) / "user_docs.jsonl"
            export_slice_jsonl(self.records, slice_path)
            service = HotpotQAMVPService.from_slice(slice_path, user_document_store=user_doc_path)

            created = service.add_document(
                title="Internal FAQ",
                content="Our refund window is 30 days from the date of purchase.",
                source="user",
            )
            service_reloaded = HotpotQAMVPService.from_slice(slice_path, user_document_store=user_doc_path)
            result = service_reloaded.ask("What is the refund window?")
            store_exists = user_doc_path.exists()

        self.assertEqual(created["title"], "Internal FAQ")
        self.assertTrue(store_exists)
        self.assertIn("30 days", result["answer"])
        self.assertTrue(any("user::" in citation for citation in result["citations"]))

    def test_service_lists_documents_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            slice_path = Path(tmp_dir) / "slice.jsonl"
            export_slice_jsonl(self.records, slice_path)
            service = HotpotQAMVPService.from_slice(slice_path)
            service.ask("Which university did the author of The Hobbit attend?")

        documents = service.list_documents(limit=3)
        history = service.history(limit=5)

        self.assertTrue(documents)
        self.assertTrue(history)
        self.assertEqual(history[0]["query"], "Which university did the author of The Hobbit attend?")

    def test_service_can_create_knowledge_base_and_query_with_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            slice_path = Path(tmp_dir) / "slice.jsonl"
            user_doc_path = Path(tmp_dir) / "user_docs.jsonl"
            kb_store = Path(tmp_dir) / "knowledge_bases.json"
            export_slice_jsonl(self.records, slice_path)
            service = HotpotQAMVPService.from_slice(
                slice_path,
                user_document_store=user_doc_path,
                knowledge_base_store=kb_store,
            )

            kb = service.create_knowledge_base("product-docs", description="Internal product handbook")
            service.add_document(
                title="Pricing",
                content="The Pro plan costs 99 dollars per month.",
                source="user",
                knowledge_base="product-docs",
            )
            result = service.ask("How much does the Pro plan cost?", knowledge_base="product-docs")

        self.assertEqual(kb["name"], "product-docs")
        self.assertIn("99 dollars", result["answer"])
        self.assertEqual(result["knowledge_base"], "product-docs")

    def test_service_chunks_long_document_for_retrieval(self) -> None:
        long_content = (
            "The Pro plan includes priority support and onboarding. "
            "It also includes audit logs and SSO. "
            "The Enterprise plan includes custom SLAs and dedicated success managers."
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            slice_path = Path(tmp_dir) / "slice.jsonl"
            user_doc_path = Path(tmp_dir) / "user_docs.jsonl"
            export_slice_jsonl(self.records, slice_path)
            service = HotpotQAMVPService.from_slice(slice_path, user_document_store=user_doc_path)
            created = service.add_document(title="Plan Matrix", content=long_content, source="user")
            retrieve = service.retrieve("Which plan includes audit logs?", top_k=5, method="hybrid")

        self.assertEqual(created["title"], "Plan Matrix")
        self.assertTrue(any(item["doc_id"].startswith("user::plan-matrix-") for item in retrieve["results"]))
        self.assertTrue(any("audit logs" in item["preview"].lower() for item in retrieve["results"]))

    def test_service_can_import_local_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            slice_path = Path(tmp_dir) / "slice.jsonl"
            user_doc_path = Path(tmp_dir) / "user_docs.jsonl"
            source_path = Path(tmp_dir) / "pricing.md"
            source_path.write_text("# Pricing\\n\\nStarter costs 19 dollars. Pro costs 99 dollars.", encoding="utf-8")
            export_slice_jsonl(self.records, slice_path)
            service = HotpotQAMVPService.from_slice(slice_path, user_document_store=user_doc_path)

            imported = service.import_files([source_path], knowledge_base="workspace")
            answer = service.ask("How much does Pro cost?", knowledge_base="workspace")

        self.assertEqual(len(imported), 1)
        self.assertEqual(imported[0]["title"], "pricing.md")
        self.assertIn("99 dollars", answer["answer"])

    def test_service_persists_run_log_for_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            slice_path = Path(tmp_dir) / "slice.jsonl"
            user_doc_path = Path(tmp_dir) / "user_docs.jsonl"
            run_log_path = Path(tmp_dir) / "runs.jsonl"
            export_slice_jsonl(self.records, slice_path)
            service = HotpotQAMVPService.from_slice(
                slice_path,
                user_document_store=user_doc_path,
                run_log_store=run_log_path,
            )
            service.ask("Which university did the author of The Hobbit attend?")
            service.retrieve("Tolkien Oxford", top_k=3, method="hybrid")
            runs = service.runs(limit=10)
            store_exists = run_log_path.exists()

        self.assertTrue(store_exists)
        self.assertGreaterEqual(len(runs), 2)
        self.assertIn(runs[0]["event_type"], {"ask", "retrieve"})


if __name__ == "__main__":
    unittest.main()
