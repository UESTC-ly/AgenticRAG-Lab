#!/usr/bin/env python3
"""End-to-end EM / F1 / citation-rate / latency benchmark on a HotpotQA slice.

Compares up to four agent configurations:
  1. rule-retr + rule-synth   — lexical + rule-semantic + rule synthesizer
  2. real-retr + rule-synth   — bge-small + bge-reranker + rule synthesizer
  3. rule-retr + llm-synth    — lexical + rule-semantic + Ollama LLM synthesizer
  4. real-retr + llm-synth    — full real stack

Heavy configs (#2, #3, #4) are constructed only when their dependencies are
importable / reachable, so the script degrades gracefully.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from agentic_rag_lab.agent import AgenticRAG
from agentic_rag_lab.critic import EvidenceCritic
from agentic_rag_lab.evaluation.hotpotqa import (
    build_hotpotqa_cases,
    build_hotpotqa_corpus,
    load_hotpotqa_jsonl_slice,
)
from agentic_rag_lab.evaluation.metrics import exact_match, token_f1
from agentic_rag_lab.planner import RuleBasedPlanner
from agentic_rag_lab.retrieval.hybrid import HybridRetriever
from agentic_rag_lab.retrieval.lexical import LexicalRetriever
from agentic_rag_lab.retrieval.semantic import SemanticRetriever
from agentic_rag_lab.router import Router
from agentic_rag_lab.synthesizer import CitationSynthesizer


def build_real_retriever(corpus, lexical, embedding_model, reranker_model, device, include_reranker):
    from agentic_rag_lab.retrieval.embedding_semantic import EmbeddingSemanticRetriever

    dense = EmbeddingSemanticRetriever(corpus, model_name=embedding_model, device=device)
    if include_reranker:
        from agentic_rag_lab.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name=reranker_model, device=device)
        return HybridRetriever(lexical=lexical, semantic=dense, reranker=reranker)
    return HybridRetriever(lexical=lexical, semantic=dense)


def build_llm_synthesizer(model_name, base_url, timeout):
    from agentic_rag_lab.llm_synthesizer import LLMSynthesizer

    return LLMSynthesizer(
        model=model_name,
        base_url=base_url,
        timeout=timeout,
        fallback=CitationSynthesizer(),
    )


def run_config(name, cases, *, retriever, synthesizer, max_iterations):
    agent = AgenticRAG(
        router=Router(),
        planner=RuleBasedPlanner(),
        retriever=retriever,
        critic=EvidenceCritic(),
        synthesizer=synthesizer,
        max_iterations=max_iterations,
    )

    total_em = 0.0
    total_f1 = 0.0
    citation_hits = 0
    total_latency = 0.0
    iteration_sum = 0
    rows = []

    for idx, case in enumerate(cases, start=1):
        t0 = time.time()
        result = agent.run(case.question)
        elapsed = time.time() - t0

        em = exact_match(result.answer, case.answer)
        f1 = token_f1(result.answer, case.answer)
        has_citation = 1 if result.citations else 0

        total_em += em
        total_f1 += f1
        citation_hits += has_citation
        total_latency += elapsed
        iteration_sum += result.iterations

        rows.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "reference": case.answer,
                "prediction": result.answer,
                "citations": result.citations,
                "em": em,
                "f1": f1,
                "iterations": result.iterations,
                "latency_sec": round(elapsed, 3),
            }
        )

        if idx % 10 == 0 or idx == len(cases):
            print(
                f"[{name}] {idx}/{len(cases)}  running EM={total_em/idx:.3f} "
                f"F1={total_f1/idx:.3f} cite={citation_hits/idx:.3f}"
            )

    n = max(len(cases), 1)
    summary = {
        "name": name,
        "case_count": len(cases),
        "em": round(total_em / n, 4),
        "f1": round(total_f1 / n, 4),
        "citation_rate": round(citation_hits / n, 4),
        "avg_latency_sec": round(total_latency / n, 3),
        "avg_iterations": round(iteration_sum / n, 2),
    }
    return summary, rows


def format_markdown(summaries):
    header = (
        "| Configuration | EM | F1 | Citation Rate | Avg Latency (s) | Avg Iters |\n"
        "| --- | --- | --- | --- | --- | --- |"
    )
    lines = [
        f"| {s['name']} | {s['em']:.3f} | {s['f1']:.3f} | "
        f"{s['citation_rate']:.3f} | {s['avg_latency_sec']:.2f} | {s['avg_iterations']:.1f} |"
        for s in summaries
    ]
    return "\n".join([header, *lines])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end EM/F1/citation-rate/latency benchmark across 4 agent configurations.",
    )
    parser.add_argument(
        "--slice",
        type=Path,
        default=Path("data/processed/hotpotqa/dev_slice.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=100, help="Only use first N cases.")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--device", default=None)
    parser.add_argument("--llm-model", default="gemma4:e2b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument(
        "--skip-real-retr", action="store_true", help="Skip configs that need sentence-transformers."
    )
    parser.add_argument(
        "--skip-llm", action="store_true", help="Skip configs that need Ollama."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/hotpotqa/e2e_benchmark.json"),
    )
    parser.add_argument(
        "--per-case-output",
        type=Path,
        default=None,
        help="Optional JSONL path to write per-case predictions for every config.",
    )
    args = parser.parse_args()

    print(f"[info] loading slice: {args.slice} (limit={args.limit})")
    records = load_hotpotqa_jsonl_slice(args.slice)
    all_cases = build_hotpotqa_cases(records)
    cases = all_cases[: args.limit]
    corpus = build_hotpotqa_corpus(records)
    print(f"[info] {len(cases)} eval cases, corpus has {len(corpus)} passages")

    lexical = LexicalRetriever(corpus)
    rule_semantic = SemanticRetriever(corpus)
    rule_retriever = HybridRetriever(lexical=lexical, semantic=rule_semantic)

    real_retriever = None
    if not args.skip_real_retr:
        try:
            print(
                f"[info] building real retriever (embedding={args.embedding_model}, reranker={args.reranker_model})"
            )
            t0 = time.time()
            real_retriever = build_real_retriever(
                corpus,
                lexical=lexical,
                embedding_model=args.embedding_model,
                reranker_model=args.reranker_model,
                device=args.device,
                include_reranker=True,
            )
            print(f"[info] real retriever ready in {time.time() - t0:.1f}s")
        except (ImportError, Exception) as exc:
            print(f"[warn] real retriever unavailable ({exc}); skipping real-retr configs.")

    rule_synth = CitationSynthesizer()
    llm_synth = None
    if not args.skip_llm:
        try:
            llm_synth = build_llm_synthesizer(
                model_name=args.llm_model,
                base_url=args.ollama_url,
                timeout=args.llm_timeout,
            )
            print(f"[info] LLM synthesizer ready (model={args.llm_model})")
        except ImportError as exc:
            print(f"[warn] LLM synthesizer unavailable ({exc}); skipping LLM configs.")

    configs = [("rule-retr + rule-synth", rule_retriever, rule_synth)]
    if real_retriever is not None:
        configs.append(("real-retr + rule-synth", real_retriever, rule_synth))
    if llm_synth is not None:
        configs.append(("rule-retr + llm-synth", rule_retriever, llm_synth))
    if real_retriever is not None and llm_synth is not None:
        configs.append(("real-retr + llm-synth", real_retriever, llm_synth))

    print(f"[info] running {len(configs)} configurations × {len(cases)} cases")

    summaries = []
    per_case_rows = []
    for name, retriever, synth in configs:
        print(f"\n=== {name} ===")
        t0 = time.time()
        summary, rows = run_config(
            name,
            cases,
            retriever=retriever,
            synthesizer=synth,
            max_iterations=args.max_iterations,
        )
        summary["total_time_sec"] = round(time.time() - t0, 1)
        print(
            f"[done] {name}  EM={summary['em']:.3f}  F1={summary['f1']:.3f}  "
            f"cite={summary['citation_rate']:.3f}  lat={summary['avg_latency_sec']:.2f}s  "
            f"total={summary['total_time_sec']:.1f}s"
        )
        summaries.append(summary)
        per_case_rows.extend({"config": name, **row} for row in rows)

    result_obj = {
        "slice_path": str(args.slice),
        "case_count": len(cases),
        "corpus_document_count": len(corpus),
        "embedding_model": args.embedding_model if real_retriever is not None else None,
        "reranker_model": args.reranker_model if real_retriever is not None else None,
        "llm_model": args.llm_model if llm_synth is not None else None,
        "max_iterations": args.max_iterations,
        "summaries": summaries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result_obj, ensure_ascii=False, indent=2))
    print(f"\n[info] wrote summary to {args.output}")

    if args.per_case_output:
        args.per_case_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_case_output.open("w", encoding="utf-8") as handle:
            for row in per_case_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[info] wrote per-case rows to {args.per_case_output}")

    print()
    print(format_markdown(summaries))


if __name__ == "__main__":
    main()
