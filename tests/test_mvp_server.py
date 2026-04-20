import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_lab.data.hotpotqa import export_slice_jsonl
from agentic_rag_lab.mvp import HotpotQAMVPService
from agentic_rag_lab.web import dispatch_request


class MVPServerTests(unittest.TestCase):
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
            }
        ]

    def test_server_serves_health_and_ask_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "slice.jsonl"
            export_slice_jsonl(self.records, path)
            service = HotpotQAMVPService.from_slice(path)
            _, _, health = dispatch_request(service, method="GET", path="/api/health")
            _, _, answer = dispatch_request(
                service,
                method="POST",
                path="/api/ask",
                body=json.dumps({"query": "Which university did the author of The Hobbit attend?"}).encode("utf-8"),
            )

        self.assertEqual(json.loads(health.decode("utf-8"))["status"], "ok")
        self.assertIn("answer", json.loads(answer.decode("utf-8")))

    def test_server_supports_document_create_list_and_retrieve_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "slice.jsonl"
            user_doc_path = Path(tmp_dir) / "user_docs.jsonl"
            kb_store = Path(tmp_dir) / "knowledge_bases.json"
            export_slice_jsonl(self.records, path)
            service = HotpotQAMVPService.from_slice(
                path,
                user_document_store=user_doc_path,
                knowledge_base_store=kb_store,
            )

            _, _, created = dispatch_request(
                service,
                method="POST",
                path="/api/documents",
                body=json.dumps(
                    {
                        "title": "Support Playbook",
                        "content": "Enterprise plans include priority support and onboarding.",
                    }
                ).encode("utf-8"),
            )
            _, _, listed = dispatch_request(service, method="GET", path="/api/documents?limit=5")
            _, _, retrieved = dispatch_request(
                service,
                method="POST",
                path="/api/retrieve",
                body=json.dumps({"query": "Which plans include onboarding?", "top_k": 3}).encode("utf-8"),
            )

        created_payload = json.loads(created.decode("utf-8"))
        listed_payload = json.loads(listed.decode("utf-8"))
        retrieved_payload = json.loads(retrieved.decode("utf-8"))
        self.assertEqual(created_payload["title"], "Support Playbook")
        self.assertTrue(listed_payload["documents"])
        self.assertTrue(retrieved_payload["results"])

    def test_server_supports_knowledge_base_creation_and_scoped_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "slice.jsonl"
            user_doc_path = Path(tmp_dir) / "user_docs.jsonl"
            kb_store = Path(tmp_dir) / "knowledge_bases.json"
            export_slice_jsonl(self.records, path)
            service = HotpotQAMVPService.from_slice(
                path,
                user_document_store=user_doc_path,
                knowledge_base_store=kb_store,
            )

            _, _, created_kb = dispatch_request(
                service,
                method="POST",
                path="/api/knowledge-bases",
                body=json.dumps({"name": "product-docs", "description": "Product docs"}).encode("utf-8"),
            )
            dispatch_request(
                service,
                method="POST",
                path="/api/documents",
                body=json.dumps(
                    {
                        "title": "Pricing",
                        "content": "The Pro plan costs 99 dollars per month.",
                        "knowledge_base": "product-docs",
                    }
                ).encode("utf-8"),
            )
            _, _, answer = dispatch_request(
                service,
                method="POST",
                path="/api/ask",
                body=json.dumps(
                    {
                        "query": "How much does the Pro plan cost?",
                        "knowledge_base": "product-docs",
                    }
                ).encode("utf-8"),
            )

        self.assertEqual(json.loads(created_kb.decode("utf-8"))["name"], "product-docs")
        self.assertIn("99 dollars", json.loads(answer.decode("utf-8"))["answer"])

    def test_server_supports_importing_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "slice.jsonl"
            user_doc_path = Path(tmp_dir) / "user_docs.jsonl"
            kb_store = Path(tmp_dir) / "knowledge_bases.json"
            source_path = Path(tmp_dir) / "faq.txt"
            source_path.write_text("Refunds are available for 30 days after purchase.", encoding="utf-8")
            export_slice_jsonl(self.records, path)
            service = HotpotQAMVPService.from_slice(
                path,
                user_document_store=user_doc_path,
                knowledge_base_store=kb_store,
            )

            _, _, imported = dispatch_request(
                service,
                method="POST",
                path="/api/import-paths",
                body=json.dumps({"paths": [str(source_path)], "knowledge_base": "workspace"}).encode("utf-8"),
            )

        imported_payload = json.loads(imported.decode("utf-8"))
        self.assertEqual(len(imported_payload["documents"]), 1)
        self.assertEqual(imported_payload["documents"][0]["title"], "faq.txt")

    def test_server_exposes_run_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "slice.jsonl"
            user_doc_path = Path(tmp_dir) / "user_docs.jsonl"
            run_log_path = Path(tmp_dir) / "runs.jsonl"
            export_slice_jsonl(self.records, path)
            service = HotpotQAMVPService.from_slice(
                path,
                user_document_store=user_doc_path,
                run_log_store=run_log_path,
            )
            dispatch_request(
                service,
                method="POST",
                path="/api/ask",
                body=json.dumps({"query": "Which university did the author of The Hobbit attend?"}).encode("utf-8"),
            )
            _, _, runs = dispatch_request(service, method="GET", path="/api/runs?limit=5")

        runs_payload = json.loads(runs.decode("utf-8"))
        self.assertTrue(runs_payload["runs"])


if __name__ == "__main__":
    unittest.main()
