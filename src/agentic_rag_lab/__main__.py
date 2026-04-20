from __future__ import annotations

import argparse
import json
import sys

from .demo import build_demo_agent, build_real_agent, run_demo_benchmark
from .mvp import HotpotQAMVPService
from .web import create_server


def _normalize_legacy_query(argv: list[str]) -> list[str]:
    if not argv:
        return argv

    known_commands = {"ask", "benchmark", "serve"}
    first = argv[0]
    if first in known_commands or first.startswith("-"):
        return argv

    return ["ask", " ".join(argv)]


def main() -> None:
    normalized_argv = _normalize_legacy_query(sys.argv[1:])
    parser = argparse.ArgumentParser(description="Run the AgenticRAG-Lab Lite demo and MVP.")
    subparsers = parser.add_subparsers(dest="command")

    ask_parser = subparsers.add_parser("ask", help="Run a single query against the demo corpus")
    ask_parser.add_argument("query", help="Query to answer")
    ask_parser.add_argument(
        "--real",
        action="store_true",
        help="Use real embedding + reranker + LLM components (requires sentence-transformers + Ollama).",
    )
    ask_parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    ask_parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    ask_parser.add_argument("--llm-model", default="qwen2.5:7b")
    ask_parser.add_argument("--ollama-url", default="http://localhost:11434")

    benchmark_parser = subparsers.add_parser("benchmark", help="Print demo benchmark table")
    benchmark_parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown")

    serve_parser = subparsers.add_parser("serve", help="Run the HotpotQA MVP web app")
    serve_parser.add_argument(
        "--slice",
        default="data/processed/hotpotqa/dev_slice.jsonl",
        help="Path to the processed HotpotQA JSONL slice.",
    )
    serve_parser.add_argument(
        "--user-store",
        default="data/product/user_documents.jsonl",
        help="Path to the persistent user document JSONL store.",
    )
    serve_parser.add_argument(
        "--kb-store",
        default="data/product/knowledge_bases.json",
        help="Path to the persistent knowledge base registry JSON file.",
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind.")

    parser.add_argument("legacy_query", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--benchmark", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(normalized_argv)

    if args.command == "benchmark" or args.benchmark:
        output = run_demo_benchmark()
        if getattr(args, "json", False):
            print(json.dumps({"markdown": output}, ensure_ascii=False, indent=2))
        else:
            print(output)
        return

    if args.command == "serve":
        service = HotpotQAMVPService.from_slice(
            args.slice,
            user_document_store=args.user_store,
            knowledge_base_store=args.kb_store,
        )
        server, thread = create_server(service, host=args.host, port=args.port)
        actual_host, actual_port = server.server_address
        print(f"AgenticRAG-Lab MVP running at http://{actual_host}:{actual_port}")
        try:
            thread.join()
        except KeyboardInterrupt:
            server.shutdown()
            server.server_close()
        return

    query = args.legacy_query or getattr(args, "query", None)
    if not query:
        parser.print_help()
        return

    if getattr(args, "real", False):
        agent = build_real_agent(
            embedding_model=args.embedding_model,
            reranker_model=args.reranker_model,
            llm_model=args.llm_model,
            ollama_url=args.ollama_url,
        )
    else:
        agent = build_demo_agent()

    result = agent.run(query)
    print(result.answer)
    if result.citations:
        print(f"Citations: {', '.join(result.citations)}")


if __name__ == "__main__":
    main()
