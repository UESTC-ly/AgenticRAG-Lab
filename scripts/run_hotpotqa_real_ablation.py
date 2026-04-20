#!/usr/bin/env python3
"""Ablation on a HotpotQA slice comparing rule-based vs. real retrieval stacks.

Compares up to six configurations:
  1. lexical            — BM25-like lexical only
  2. semantic-rule      — token-expansion proxy (offline baseline)
  3. semantic-bge       — real dense embedding (sentence-transformers)
  4. hybrid-rule        — fusion of #1 + #2
  5. hybrid-bge         — fusion of #1 + #3
  6. hybrid-bge-rerank  — fusion of #1 + #3 + cross-encoder rerank

Heavy configurations (#3, #5, #6) are only constructed when the required
dependencies are importable, so the script degrades gracefully.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Must be set before torch import: avoids libomp double-init on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    return values or [1, 3, 5]


def build_real_retrievers(
    corpus,
    lexical,
    embedding_model: str,
    reranker_model: str,
    device: str | None,
    include_reranker: bool,
) -> dict:
    """Return dict of retriever_name -> retriever, or empty dict if deps missing."""
    retrievers: dict = {}

    try:
        from agentic_rag_lab.retrieval.embedding_semantic import (
            EmbeddingSemanticRetriever,
        )
    except ImportError as exc:
        print(f"[warn] embedding unavailable ({exc}); skipping real configurations.")
        return retrievers

    print(f"[info] loading embedding model: {embedding_model} (device={device or 'auto'})")
    t0 = time.time()
    dense = EmbeddingSemanticRetriever(
        corpus, model_name=embedding_model, device=device
    )
    print(f"[info] embedded {len(corpus)} passages in {time.time() - t0:.1f}s")

    retrievers["semantic-bge"] = dense
    retrievers["hybrid-bge"] = HybridRetriever(lexical=lexical, semantic=dense)

    if include_reranker:
        try:
            from agentic_rag_lab.retrieval.reranker import CrossEncoderReranker

            print(f"[info] loading reranker: {reranker_model}")
            t0 = time.time()
            reranker = CrossEncoderReranker(
                model_name=reranker_model, device=device
            )
            print(f"[info] reranker ready in {time.time() - t0:.1f}s")
            retrievers["hybrid-bge-rerank"] = HybridRetriever(
                lexical=lexical, semantic=dense, reranker=reranker
            )
        except ImportError as exc:
            print(f"[warn] reranker unavailable ({exc}); skipping rerank config.")

    return retrievers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full retrieval ablation (rule-based vs. real embedding + reranker).",
    )
    parser.add_argument(
        "--slice",
        type=Path,
        default=Path("data/processed/hotpotqa/dev_slice.jsonl"),
    )
    parser.add_argument("--top-ks", default="1,3,5,10")
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-en-v1.5",
        help="sentence-transformers model (default: bge-small-en-v1.5).",
    )
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-base",
        help="cross-encoder reranker (default: bge-reranker-base).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (e.g. mps, cuda, cpu). Auto if omitted.",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip the cross-encoder rerank configuration.",
    )
    parser.add_argument(
        "--skip-real",
        action="store_true",
        help="Only run rule-based configurations (lexical / semantic-rule / hybrid-rule).",
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write full JSON results.",
    )
    args = parser.parse_args()

    top_ks = parse_top_ks(args.top_ks)

    print(f"[info] loading slice: {args.slice}")
    records = load_hotpotqa_jsonl_slice(args.slice)
    cases = build_hotpotqa_cases(records)
    corpus = build_hotpotqa_corpus(records)
    print(f"[info] {len(cases)} cases, {len(corpus)} corpus passages")

    lexical = LexicalRetriever(corpus)
    rule_semantic = SemanticRetriever(corpus)
    retrievers: dict = {
        "lexical": lexical,
        "semantic-rule": rule_semantic,
        "hybrid-rule": HybridRetriever(lexical=lexical, semantic=rule_semantic),
    }

    if not args.skip_real:
        retrievers.update(
            build_real_retrievers(
                corpus,
                lexical=lexical,
                embedding_model=args.embedding_model,
                reranker_model=args.reranker_model,
                device=args.device,
                include_reranker=not args.no_reranker,
            )
        )

    print(f"[info] running ablation on configs: {list(retrievers.keys())}")
    t0 = time.time()
    rows = benchmark_retrievers_on_hotpotqa(cases, retrievers, top_ks=top_ks)
    print(f"[info] ablation finished in {time.time() - t0:.1f}s")

    summary = {
        "slice_path": str(args.slice),
        "case_count": len(cases),
        "corpus_document_count": len(corpus),
        "embedding_model": args.embedding_model if not args.skip_real else None,
        "reranker_model": (
            args.reranker_model
            if (not args.skip_real and not args.no_reranker)
            else None
        ),
        "top_ks": top_ks,
        "rows": rows,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"[info] wrote results to {args.output}")

    if args.format == "markdown":
        print()
        print(format_hotpotqa_ablation_markdown(rows))
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
