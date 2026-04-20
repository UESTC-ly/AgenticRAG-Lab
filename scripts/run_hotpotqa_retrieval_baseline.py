#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_lab.evaluation.hotpotqa import (
    build_hotpotqa_cases,
    build_hotpotqa_corpus,
    evaluate_retriever_on_hotpotqa,
    load_hotpotqa_jsonl_slice,
)
from agentic_rag_lab.retrieval.hybrid import HybridRetriever
from agentic_rag_lab.retrieval.lexical import LexicalRetriever
from agentic_rag_lab.retrieval.semantic import SemanticRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an offline retrieval baseline on a HotpotQA slice.")
    parser.add_argument(
        "--slice",
        type=Path,
        default=Path("data/processed/hotpotqa/dev_slice.jsonl"),
        help="Path to the processed HotpotQA JSONL slice.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval cutoff.")
    args = parser.parse_args()

    records = load_hotpotqa_jsonl_slice(args.slice)
    cases = build_hotpotqa_cases(records)
    corpus = build_hotpotqa_corpus(records)
    retriever = HybridRetriever(
        lexical=LexicalRetriever(corpus),
        semantic=SemanticRetriever(corpus),
    )
    metrics = evaluate_retriever_on_hotpotqa(cases, retriever, top_k=args.top_k)

    summary = {
        "slice_path": str(args.slice),
        "case_count": len(cases),
        "corpus_document_count": len(corpus),
        "top_k": args.top_k,
        "metrics": metrics,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
