from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .metrics import exact_match, token_f1


@dataclass(slots=True)
class BenchmarkCase:
    query: str
    reference_answer: str


@dataclass(slots=True)
class BenchmarkMethod:
    name: str
    run: Callable[[str], dict[str, object]]


@dataclass(slots=True)
class BenchmarkRow:
    name: str
    exact_match: float
    token_f1: float
    citation_rate: float
    average_latency_ms: float


def run_benchmark(
    cases: list[BenchmarkCase],
    methods: list[BenchmarkMethod],
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []

    for method in methods:
        em_total = 0.0
        f1_total = 0.0
        citation_total = 0.0
        latency_total_ms = 0.0

        for case in cases:
            started = time.perf_counter()
            result = method.run(case.query)
            latency_total_ms += (time.perf_counter() - started) * 1000
            answer = str(result.get("answer", ""))
            citations = list(result.get("citations", []))
            em_total += exact_match(answer, case.reference_answer)
            f1_total += token_f1(answer, case.reference_answer)
            citation_total += 1.0 if citations else 0.0

        total_cases = max(len(cases), 1)
        rows.append(
            BenchmarkRow(
                name=method.name,
                exact_match=em_total / total_cases,
                token_f1=f1_total / total_cases,
                citation_rate=citation_total / total_cases,
                average_latency_ms=latency_total_ms / total_cases,
            )
        )

    return rows


def format_markdown_table(rows: list[BenchmarkRow]) -> str:
    header = (
        "| Method | EM | F1 | Citation Rate | Avg Latency (ms) |\n"
        "| --- | --- | --- | --- | --- |"
    )
    lines = [
        f"| {row.name} | {row.exact_match:.3f} | {row.token_f1:.3f} | "
        f"{row.citation_rate:.3f} | {row.average_latency_ms:.2f} |"
        for row in rows
    ]
    return "\n".join([header, *lines])

