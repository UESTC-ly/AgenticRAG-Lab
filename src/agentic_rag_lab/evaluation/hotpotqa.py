from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..models import Document


@dataclass(slots=True)
class HotpotQACase:
    case_id: str
    question: str
    answer: str
    level: str
    question_type: str
    supporting_titles: set[str]


def load_hotpotqa_jsonl_slice(path: str | Path) -> list[dict[str, object]]:
    slice_path = Path(path)
    records: list[dict[str, object]] = []
    for line in slice_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def build_hotpotqa_cases(records: list[dict[str, object]]) -> list[HotpotQACase]:
    cases: list[HotpotQACase] = []
    for record in records:
        supporting_titles = {
            str(fact.get("title", ""))
            for fact in list(record.get("supporting_facts", []))
            if fact.get("title")
        }
        cases.append(
            HotpotQACase(
                case_id=str(record.get("id", "")),
                question=str(record.get("question", "")),
                answer=str(record.get("answer", "")),
                level=str(record.get("level", "")),
                question_type=str(record.get("type", "")),
                supporting_titles=supporting_titles,
            )
        )
    return cases


def build_hotpotqa_corpus(records: list[dict[str, object]]) -> list[Document]:
    grouped_sentences: dict[str, list[str]] = defaultdict(list)
    doc_metadata: dict[str, dict[str, str]] = {}

    for record in records:
        question_id = str(record.get("id", ""))
        for context in list(record.get("contexts", [])):
            title = str(context.get("title", "")).strip()
            sentences = [str(item).strip() for item in list(context.get("sentences", [])) if str(item).strip()]
            if not title or not sentences:
                continue

            grouped_sentences[title].extend(sentences)
            metadata = doc_metadata.setdefault(title, {"source": "hotpotqa", "question_ids": ""})
            existing_ids = [item for item in metadata["question_ids"].split(",") if item]
            if question_id and question_id not in existing_ids:
                existing_ids.append(question_id)
            metadata["question_ids"] = ",".join(existing_ids)

    corpus: list[Document] = []
    for title in sorted(grouped_sentences):
        unique_sentences = []
        seen = set()
        for sentence in grouped_sentences[title]:
            if sentence in seen:
                continue
            seen.add(sentence)
            unique_sentences.append(sentence)
        corpus.append(
            Document(
                doc_id=f"title::{title}",
                title=title,
                content=" ".join(unique_sentences),
                metadata=doc_metadata.get(title, {"source": "hotpotqa"}),
            )
        )
    return corpus


def evaluate_retriever_on_hotpotqa(
    cases: list[HotpotQACase],
    retriever,
    *,
    top_k: int = 5,
) -> dict[str, float]:
    if not cases:
        return {
            "case_count": 0,
            "supporting_doc_recall@k": 0.0,
            "all_supporting_docs_hit_rate@k": 0.0,
            "any_supporting_doc_hit_rate@k": 0.0,
        }

    total_supporting = 0
    total_hits = 0
    all_hit_cases = 0
    any_hit_cases = 0

    for case in cases:
        results = retriever.search(case.question, top_k=top_k)
        retrieved_titles = {item.document.title for item in results}
        hits = len(case.supporting_titles & retrieved_titles)
        total_hits += hits
        total_supporting += len(case.supporting_titles)
        if hits > 0:
            any_hit_cases += 1
        if case.supporting_titles and hits == len(case.supporting_titles):
            all_hit_cases += 1

    case_count = len(cases)
    return {
        "case_count": case_count,
        "supporting_doc_recall@2" if top_k == 2 else f"supporting_doc_recall@{top_k}": total_hits / max(total_supporting, 1),
        "all_supporting_docs_hit_rate@2" if top_k == 2 else f"all_supporting_docs_hit_rate@{top_k}": all_hit_cases / case_count,
        "any_supporting_doc_hit_rate@2" if top_k == 2 else f"any_supporting_doc_hit_rate@{top_k}": any_hit_cases / case_count,
    }


def benchmark_retrievers_on_hotpotqa(
    cases: list[HotpotQACase],
    retrievers: dict[str, object],
    *,
    top_ks: list[int],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for top_k in top_ks:
        for method_name, retriever in retrievers.items():
            metrics = evaluate_retriever_on_hotpotqa(cases, retriever, top_k=top_k)
            rows.append(
                {
                    "method": method_name,
                    "top_k": top_k,
                    "case_count": int(metrics["case_count"]),
                    "supporting_doc_recall": float(metrics[f"supporting_doc_recall@{top_k}"]),
                    "all_supporting_docs_hit_rate": float(metrics[f"all_supporting_docs_hit_rate@{top_k}"]),
                    "any_supporting_doc_hit_rate": float(metrics[f"any_supporting_doc_hit_rate@{top_k}"]),
                }
            )
    return rows


def format_hotpotqa_ablation_markdown(rows: list[dict[str, float | int | str]]) -> str:
    header = (
        "| Method | Top-K | Supporting Recall | All Docs Hit Rate | Any Doc Hit Rate |\n"
        "| --- | --- | --- | --- | --- |"
    )
    lines = [
        f"| {row['method']} | {row['top_k']} | {float(row['supporting_doc_recall']):.3f} | "
        f"{float(row['all_supporting_docs_hit_rate']):.3f} | {float(row['any_supporting_doc_hit_rate']):.3f} |"
        for row in rows
    ]
    return "\n".join([header, *lines])
