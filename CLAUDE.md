# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AgenticRAG-Lab (Lite) — a lightweight, offline-runnable Agentic RAG prototype for multi-hop QA, built as a hand-written orchestration loop (no LangChain/LlamaIndex). Python ≥ 3.11, zero runtime dependencies by design.

See `PROJECT_DESCRIPTION.md` for design rationale and `EVALUATION.md` for the benchmarking plan.

## Commands

All commands assume the repo root and `PYTHONPATH=src` (the package is not installed; imports resolve via src-layout).

```bash
# Run all unit tests
PYTHONPATH=src python3 -m unittest discover -s tests -v

# Run a single test module / case
PYTHONPATH=src python3 -m unittest tests.test_agentic_rag -v
PYTHONPATH=src python3 -m unittest tests.test_agentic_rag.AgenticRAGTests.test_direct_route -v

# Run the demo agent on a query
PYTHONPATH=src python3 -m agentic_rag_lab "Which university did the author of The Hobbit attend before becoming a professor?"

# Print the built-in demo benchmark table (EM / F1 / citation rate / latency)
PYTHONPATH=src python3 -m agentic_rag_lab --benchmark
PYTHONPATH=src python3 scripts/run_eval.py

# Build a local HotpotQA dev slice (downloads on first run into data/raw/)
PYTHONPATH=src python3 scripts/prepare_hotpotqa_slice.py --limit 100
```

Tests use `unittest`, not `pytest`. The global rule suggesting pytest does not apply here — do not rewrite tests unless asked.

## Architecture

The system is a state machine, not a graph framework. Follow control flow through `agent.py` to understand everything.

```
User Query → Router → {direct_answer | calculator | agentic_rag}
                                                      ↓
                                                   Planner (decompose)
                                                      ↓
                                        Executor Loop (≤ max_iterations)
                                          ├─ HybridRetriever (lexical + semantic + RRF + rerank)
                                          └─ Critic (sufficient? else produce follow-up queries)
                                                      ↓
                                                  Synthesizer (cited answer)
```

Key shape:

- **`src/agentic_rag_lab/agent.py`** — `AgenticRAG.run()` is the single orchestration loop. It owns the `evidence_pool` dedup-by-doc_id, iteration bookkeeping, and `TraceEvent` emission. New control-flow behavior goes here, not in sub-components.
- **`models.py`** — all shared dataclasses (`RouteDecision`, `QueryPlan`, `EvidenceItem`, `CritiqueResult`, `RunResult`, `TraceEvent`) and the `RouteTarget` enum. Every component communicates via these; changing a field ripples across router/planner/critic/synthesizer and their tests.
- **`router.py` / `planner.py` / `critic.py` / `synthesizer.py`** — rule-based stubs by design. They are interface stand-ins meant to be swapped for LLM-backed implementations later. Keep the public method signatures (`route`, `plan`, `evaluate`, `answer` / `direct_answer`) stable so `AgenticRAG` does not need to change.
- **`retrieval/`** — `LexicalRetriever` (BM25-like), `SemanticRetriever` (offline proxy, no embeddings), and `HybridRetriever` (RRF fusion + lightweight rerank). All three return `list[EvidenceItem]` from a `.search(query, top_k)` method — that's the retriever contract the agent depends on.
- **`evaluation/`** — `metrics.py` (EM, token F1) and `runner.py` (benchmark aggregation with latency/citation rate). `demo.py::run_demo_benchmark` wires these against the in-repo demo corpus.
- **`data/hotpotqa.py`** — HotpotQA download + slice builder. `scripts/prepare_hotpotqa_slice.py` is the CLI wrapper; the real logic lives in the module.
- **`__main__.py`** — thin CLI. Real demo wiring (corpus, agent construction) lives in `demo.py::build_demo_agent`.

## Working in this codebase

- The agent loop in `agent.py` is deliberately hand-written for interview explainability — do not replace it with a framework.
- When adding a new retriever, match the existing `.search(query, top_k) -> list[EvidenceItem]` contract so `HybridRetriever` and `AgenticRAG` keep working unchanged.
- When changing a `models.py` dataclass, grep tests and all four pipeline stages (router/planner/critic/synthesizer) — they all share these types.
- The demo corpus and demo benchmark in `demo.py` are the smoke test for end-to-end behavior; keep them runnable offline (no network, no API keys).
- HotpotQA tooling writes to `data/raw/hotpotqa/` (downloaded splits) and `data/processed/hotpotqa/` (JSONL slices). Both are gitignored.
