#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_lab.evaluation.hotpotqa import (
    benchmark_retrievers_on_hotpotqa,
    build_hotpotqa_cases,
    build_hotpotqa_corpus,
    format_hotpotqa_ablation_markdown,
    load_hotpotqa_jsonl_slice,
)
from agentic_rag_lab.retrieval.hybrid import HybridRetriever
from agentic_rag_lab.retrieval.lexical import LexicalRetriever
from agentic_rag_lab.retrieval.semantic import SemanticRetriever


def parse_top_ks(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values or [1, 3, 5]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare lexical / semantic / hybrid retrieval on a HotpotQA slice.")
    parser.add_argument(
        "--slice",
        type=Path,
        default=Path("data/processed/hotpotqa/dev_slice.jsonl"),
        help="Path to the processed HotpotQA JSONL slice.",
    )
    parser.add_argument(
        "--top-ks",
        default="1,3,5",
        help="Comma-separated top-k values to evaluate.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format.",
    )
    args = parser.parse_args()

    top_ks = parse_top_ks(args.top_ks)
    records = load_hotpotqa_jsonl_slice(args.slice)
    cases = build_hotpotqa_cases(records)
    corpus = build_hotpotqa_corpus(records)

    retrievers = {
        "lexical": LexicalRetriever(corpus),
        "semantic": SemanticRetriever(corpus),
        "hybrid": HybridRetriever(
            lexical=LexicalRetriever(corpus),
            semantic=SemanticRetriever(corpus),
        ),
    }
    rows = benchmark_retrievers_on_hotpotqa(cases, retrievers, top_ks=top_ks)

    if args.format == "markdown":
        print(format_hotpotqa_ablation_markdown(rows))
        return

    summary = {
        "slice_path": str(args.slice),
        "case_count": len(cases),
        "corpus_document_count": len(corpus),
        "top_ks": top_ks,
        "rows": rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
