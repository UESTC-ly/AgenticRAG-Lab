from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import time

from .agent import AgenticRAG
from .chunking import chunk_text
from .critic import EvidenceCritic
from .evaluation.hotpotqa import (
    benchmark_retrievers_on_hotpotqa,
    build_hotpotqa_cases,
    build_hotpotqa_corpus,
    load_hotpotqa_jsonl_slice,
)
from .models import Document
from .planner import RuleBasedPlanner
from .retrieval.hybrid import HybridRetriever
from .retrieval.lexical import LexicalRetriever
from .retrieval.semantic import SemanticRetriever
from .router import Router
from .synthesizer import CitationSynthesizer
from .text import normalize


class HotpotQAMVPService:
    def __init__(
        self,
        *,
        records: list[dict[str, object]],
        user_document_store: str | Path | None = None,
        knowledge_base_store: str | Path | None = None,
        run_log_store: str | Path | None = None,
    ) -> None:
        self.records = records
        self.user_document_store = Path(user_document_store) if user_document_store else None
        if knowledge_base_store:
            self.knowledge_base_store = Path(knowledge_base_store)
        elif self.user_document_store:
            self.knowledge_base_store = self.user_document_store.with_name("knowledge_bases.json")
        else:
            self.knowledge_base_store = None
        if run_log_store:
            self.run_log_store = Path(run_log_store)
        elif self.user_document_store:
            self.run_log_store = self.user_document_store.with_name("runs.jsonl")
        else:
            self.run_log_store = None
        self.cases = build_hotpotqa_cases(records)
        self.case_by_normalized_question = {
            normalize(case.question): case for case in self.cases if case.question
        }
        self.base_corpus = build_hotpotqa_corpus(records)
        self.knowledge_bases = self._load_knowledge_bases()
        self.user_documents = self._load_user_documents()
        self._history: list[dict[str, object]] = []
        self._rebuild_indexes()

    @classmethod
    def from_slice(
        cls,
        path: str | Path,
        *,
        user_document_store: str | Path | None = None,
        knowledge_base_store: str | Path | None = None,
        run_log_store: str | Path | None = None,
    ) -> "HotpotQAMVPService":
        return cls(
            records=load_hotpotqa_jsonl_slice(path),
            user_document_store=user_document_store,
            knowledge_base_store=knowledge_base_store,
            run_log_store=run_log_store,
        )

    def _append_run_log(self, event_type: str, payload: dict[str, object]) -> None:
        entry = {
            "event_type": event_type,
            "timestamp": time.time(),
            **payload,
        }
        self._history.append(entry)
        if not self.run_log_store:
            return
        self.run_log_store.parent.mkdir(parents=True, exist_ok=True)
        with self.run_log_store.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def runs(self, *, limit: int = 20) -> list[dict[str, object]]:
        if self.run_log_store and self.run_log_store.exists():
            items = []
            for line in self.run_log_store.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                items.append(json.loads(line))
            return list(reversed(items[-limit:]))
        return list(reversed(self._history[-limit:]))

    def _load_knowledge_bases(self) -> list[dict[str, str]]:
        bases = [
            {"name": "hotpotqa", "description": "Built-in HotpotQA benchmark slice", "source": "system"},
            {"name": "workspace", "description": "Default user knowledge base", "source": "user"},
        ]
        if not self.knowledge_base_store or not self.knowledge_base_store.exists():
            return bases

        payload = json.loads(self.knowledge_base_store.read_text(encoding="utf-8"))
        existing_names = {item["name"] for item in bases}
        for item in payload:
            name = str(item.get("name", "")).strip()
            if not name or name in existing_names:
                continue
            bases.append(
                {
                    "name": name,
                    "description": str(item.get("description", "")),
                    "source": str(item.get("source", "user")),
                }
            )
            existing_names.add(name)
        return bases

    def _persist_knowledge_bases(self) -> None:
        if not self.knowledge_base_store:
            return
        self.knowledge_base_store.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            item
            for item in self.knowledge_bases
            if item.get("source") == "user"
        ]
        self.knowledge_base_store.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_user_documents(self) -> list[Document]:
        if not self.user_document_store or not self.user_document_store.exists():
            return []

        documents: list[Document] = []
        for line in self.user_document_store.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            documents.append(
                Document(
                    doc_id=str(payload["doc_id"]),
                    title=str(payload["title"]),
                    content=str(payload["content"]),
                    metadata={
                        "source": str(payload.get("source", "user")),
                        "created_at": str(payload.get("created_at", "")),
                        "knowledge_base": str(payload.get("knowledge_base", "workspace")),
                    },
                )
            )
        return documents

    def _persist_user_document(self, document: Document) -> None:
        if not self.user_document_store:
            return
        self.user_document_store.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "doc_id": document.doc_id,
            "title": document.title,
            "content": document.content,
            "source": document.metadata.get("source", "user"),
            "created_at": document.metadata.get("created_at", ""),
            "knowledge_base": document.metadata.get("knowledge_base", "workspace"),
        }
        with self.user_document_store.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _rebuild_indexes(self) -> None:
        self.corpus = [*self.base_corpus, *self.user_documents]
        self.corpora_by_knowledge_base: dict[str, list[Document]] = {
            "hotpotqa": list(self.base_corpus),
            "all": list(self.corpus),
        }
        for kb in self.knowledge_bases:
            name = kb["name"]
            if name == "hotpotqa":
                continue
            kb_docs = [
                document
                for document in self.user_documents
                if document.metadata.get("knowledge_base", "workspace") == name
            ]
            self.corpora_by_knowledge_base[name] = kb_docs

        self.retrievers_by_knowledge_base: dict[str, dict[str, object]] = {}
        for kb_name, corpus in self.corpora_by_knowledge_base.items():
            lexical = LexicalRetriever(corpus)
            semantic = SemanticRetriever(corpus)
            hybrid = HybridRetriever(lexical=lexical, semantic=semantic)
            self.retrievers_by_knowledge_base[kb_name] = {
                "lexical": lexical,
                "semantic": semantic,
                "hybrid": hybrid,
            }

    def _next_user_document_id(self, title: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", normalize(title))[:32].strip("-") or "document"
        return f"user::{slug}-{len(self.user_documents) + 1}"

    def _get_retriever_set(self, knowledge_base: str) -> dict[str, object]:
        return self.retrievers_by_knowledge_base.get(knowledge_base, self.retrievers_by_knowledge_base["all"])

    def _build_agent(self, knowledge_base: str) -> AgenticRAG:
        retriever = self._get_retriever_set(knowledge_base)["hybrid"]
        return AgenticRAG(
            router=Router(),
            planner=RuleBasedPlanner(),
            retriever=retriever,
            critic=EvidenceCritic(),
            synthesizer=CitationSynthesizer(),
            max_iterations=3,
        )

    def stats(self) -> dict[str, object]:
        level_counts = Counter(case.level for case in self.cases)
        type_counts = Counter(case.question_type for case in self.cases)
        return {
            "case_count": len(self.cases),
            "corpus_document_count": len(self.corpus),
            "user_document_count": len(self.user_documents),
            "knowledge_base_count": len(self.knowledge_bases),
            "level_distribution": dict(level_counts),
            "type_distribution": dict(type_counts),
            "query_modes": ["direct_answer", "calculator", "agentic_rag"],
        }

    def example_questions(self, *, limit: int = 8) -> list[dict[str, str]]:
        examples = []
        for case in self.cases[:limit]:
            examples.append(
                {
                    "id": case.case_id,
                    "question": case.question,
                    "answer": case.answer,
                    "type": case.question_type,
                    "level": case.level,
                }
            )
        return examples

    def list_knowledge_bases(self) -> list[dict[str, object]]:
        counts = Counter(
            document.metadata.get("knowledge_base", "workspace")
            for document in self.user_documents
        )
        listed = []
        for kb in self.knowledge_bases:
            name = kb["name"]
            listed.append(
                {
                    "name": name,
                    "description": kb.get("description", ""),
                    "source": kb.get("source", "user"),
                    "document_count": len(self.base_corpus) if name == "hotpotqa" else counts.get(name, 0),
                }
            )
        listed.append(
            {
                "name": "all",
                "description": "Union of benchmark and custom knowledge bases",
                "source": "system",
                "document_count": len(self.corpus),
            }
        )
        return listed

    def create_knowledge_base(self, name: str, *, description: str = "") -> dict[str, object]:
        normalized_name = re.sub(r"[^a-z0-9_-]+", "-", normalize(name)).strip("-") or "workspace"
        for kb in self.knowledge_bases:
            if kb["name"] == normalized_name:
                return {
                    "name": kb["name"],
                    "description": kb.get("description", ""),
                    "source": kb.get("source", "user"),
                }
        created = {"name": normalized_name, "description": description.strip(), "source": "user"}
        self.knowledge_bases.append(created)
        self._persist_knowledge_bases()
        self._rebuild_indexes()
        return created

    def list_documents(
        self,
        *,
        limit: int = 20,
        source: str | None = None,
        knowledge_base: str = "all",
    ) -> list[dict[str, object]]:
        documents = self.corpora_by_knowledge_base.get(knowledge_base, self.corpus)
        if source:
            documents = [
                document for document in documents if document.metadata.get("source", "hotpotqa") == source
            ]
        listed = []
        for document in documents[:limit]:
            listed.append(
                {
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "source": document.metadata.get("source", "hotpotqa"),
                    "knowledge_base": document.metadata.get("knowledge_base", "hotpotqa"),
                    "preview": document.content[:220],
                }
            )
        return listed

    def history(self, *, limit: int = 20) -> list[dict[str, object]]:
        return [
            item
            for item in self.runs(limit=limit)
            if item.get("event_type") == "ask"
        ]

    def add_document(
        self,
        *,
        title: str,
        content: str,
        source: str = "user",
        knowledge_base: str = "workspace",
    ) -> dict[str, object]:
        if knowledge_base not in {kb["name"] for kb in self.knowledge_bases}:
            self.create_knowledge_base(knowledge_base)
        parent_doc_id = self._next_user_document_id(title)
        title = title.strip()
        chunks = chunk_text(content.strip())
        if not chunks:
            chunks = [content.strip()]

        for index, chunk in enumerate(chunks, start=1):
            document = Document(
                doc_id=f"{parent_doc_id}::chunk-{index}",
                title=title,
                content=chunk,
                metadata={
                    "source": source,
                    "created_at": "local",
                    "knowledge_base": knowledge_base,
                    "parent_doc_id": parent_doc_id,
                },
            )
            self.user_documents.append(document)
            self._persist_user_document(document)
        self._rebuild_indexes()
        self._append_run_log(
            "add_document",
            {
                "title": title,
                "knowledge_base": knowledge_base,
                "chunk_count": len(chunks),
                "doc_id": parent_doc_id,
            },
        )
        return {
            "doc_id": parent_doc_id,
            "title": title,
            "source": source,
            "knowledge_base": knowledge_base,
            "chunk_count": len(chunks),
            "preview": chunks[0][:220],
        }

    def import_files(self, paths: list[str | Path], *, knowledge_base: str = "workspace") -> list[dict[str, object]]:
        imported: list[dict[str, object]] = []
        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists() or not path.is_file():
                continue
            imported.append(
                self.add_document(
                    title=path.name,
                    content=path.read_text(encoding="utf-8"),
                    source="user",
                    knowledge_base=knowledge_base,
                )
            )
        self._append_run_log(
            "import_files",
            {"knowledge_base": knowledge_base, "imported_count": len(imported), "paths": [str(item) for item in paths]},
        )
        return imported

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        method: str = "hybrid",
        knowledge_base: str = "all",
    ) -> dict[str, object]:
        retriever_map = self._get_retriever_set(knowledge_base)
        retriever = retriever_map.get(method, retriever_map["hybrid"])
        results = retriever.search(query, top_k=top_k)
        payload = {
            "query": query,
            "method": method,
            "top_k": top_k,
            "knowledge_base": knowledge_base,
            "results": [
                {
                    "doc_id": item.document.doc_id,
                    "title": item.document.title,
                    "score": item.score,
                    "source": item.document.metadata.get("source", "hotpotqa"),
                    "knowledge_base": item.document.metadata.get("knowledge_base", "hotpotqa"),
                    "preview": item.document.content[:260],
                }
                for item in results
            ],
        }
        self._append_run_log(
            "retrieve",
            {
                "query": query,
                "knowledge_base": knowledge_base,
                "method": method,
                "top_k": top_k,
                "result_count": len(payload["results"]),
            },
        )
        return payload

    def ask(self, query: str, *, knowledge_base: str = "all") -> dict[str, object]:
        result = self._build_agent(knowledge_base).run(query)
        matched_case = None
        if knowledge_base in {"hotpotqa", "all"}:
            matched_case = self.case_by_normalized_question.get(normalize(query))
        payload = {
            "query": query,
            "answer": result.answer,
            "citations": result.citations,
            "route": result.route,
            "iterations": result.iterations,
            "knowledge_base": knowledge_base,
            "trace": [{"stage": item.stage, "detail": item.detail} for item in result.trace],
            "reference_answer": matched_case.answer if matched_case else None,
            "matched_case_id": matched_case.case_id if matched_case else None,
            "matched_case_type": matched_case.question_type if matched_case else None,
            "matched_case_level": matched_case.level if matched_case else None,
        }
        self._append_run_log(
            "ask",
            {
                "query": query,
                "answer": result.answer,
                "route": result.route,
                "knowledge_base": knowledge_base,
                "reference_answer": payload["reference_answer"],
                "citations": result.citations,
            },
        )
        return payload

    def benchmark_summary(self, *, top_ks: list[int] | None = None) -> list[dict[str, object]]:
        top_ks = top_ks or [1, 3, 5]
        retrievers = {
            "lexical": self.lexical,
            "semantic": self.semantic,
            "hybrid": self.hybrid,
        }
        return benchmark_retrievers_on_hotpotqa(self.cases, retrievers, top_ks=top_ks)
