from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Callable

from .models import EvidenceItem
from .synthesizer import CitationSynthesizer


OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "qwen2.5:7b"

SYSTEM_INSTRUCTIONS = (
    "You are a careful question-answering assistant for multi-hop factoid questions. "
    "Answer ONLY using the provided evidence. "
    "Each evidence passage is numbered like [1], [2], .... "
    "RULES:\n"
    "- Every non-refusal answer MUST include at least one citation in square brackets, e.g. [1] or [2][3].\n"
    "- Only cite numbers that appear in the evidence list. Do not invent.\n"
    "- For factoid questions, answer with just the entity plus a citation, e.g. \"Exeter College, Oxford [2]\".\n"
    "- For yes/no questions, answer with just \"yes\" or \"no\" plus a citation, e.g. \"yes [1][2]\".\n"
    "- Do NOT write full sentences or restate the question.\n"
    "- If the evidence does not contain the answer, reply exactly: "
    "\"I don't know based on the provided evidence.\" (no citations)."
)


Transport = Callable[[str], str]


class LLMSynthesizer:
    """Synthesizer backed by a local Ollama LLM.

    Uses the stdlib HTTP client to avoid adding a runtime dependency. A
    ``transport`` callable can be injected for testing.

    If the LLM call fails (Ollama not running, model missing, timeout),
    falls back to ``CitationSynthesizer`` so the agent loop still completes.
    """

    def __init__(
        self,
        *,
        model: str = OLLAMA_DEFAULT_MODEL,
        base_url: str = OLLAMA_DEFAULT_URL,
        timeout: int = 120,
        temperature: float = 0.2,
        fallback: CitationSynthesizer | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.fallback = fallback if fallback is not None else CitationSynthesizer()
        self._transport: Transport = transport if transport is not None else self._ollama_generate

    def answer(self, query: str, evidence: list[EvidenceItem]) -> tuple[str, list[str]]:
        if not evidence:
            return self.fallback.answer(query, evidence)

        ordered = sorted(evidence, key=lambda item: item.score, reverse=True)
        prompt = self._build_answer_prompt(query, ordered)
        try:
            raw = self._transport(prompt).strip()
        except Exception:
            return self.fallback.answer(query, evidence)

        if not raw:
            return self.fallback.answer(query, evidence)

        citations = self._extract_citations(raw, ordered)
        return raw, citations

    def direct_answer(self, query: str) -> str:
        prompt = (
            "Answer the question concisely in one or two sentences. "
            "If you do not know, say so.\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        try:
            raw = self._transport(prompt).strip()
        except Exception:
            return self.fallback.direct_answer(query)
        return raw or self.fallback.direct_answer(query)

    # --- prompt construction -------------------------------------------------

    def _build_answer_prompt(self, query: str, evidence: list[EvidenceItem]) -> str:
        context_blocks = []
        for idx, item in enumerate(evidence, start=1):
            context_blocks.append(
                f"[{idx}] Title: {item.document.title}\nContent: {item.document.content}"
            )
        context = "\n\n".join(context_blocks)
        return (
            f"{SYSTEM_INSTRUCTIONS}\n\n"
            f"Evidence:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    def _extract_citations(self, text: str, evidence: list[EvidenceItem]) -> list[str]:
        """Map numeric citation tokens like [1], [2] back to evidence doc_ids.

        Also preserves the legacy behavior of accepting a raw doc_id in brackets
        (used by some of the unit tests).
        """
        tokens = set(re.findall(r"\[([A-Za-z0-9_\-./:]+)\]", text))
        indexed = {str(idx): item for idx, item in enumerate(evidence, start=1)}

        ordered_cites: list[str] = []
        seen: set[str] = set()

        # Pass 1: numeric citations, in the order the model emitted them.
        for raw in re.findall(r"\[(\d+)\]", text):
            item = indexed.get(raw)
            if item is None:
                continue
            doc_id = item.document.doc_id
            if doc_id in seen:
                continue
            ordered_cites.append(doc_id)
            seen.add(doc_id)

        # Pass 2: raw doc_id citations (legacy format).
        for item in evidence:
            doc_id = item.document.doc_id
            if doc_id in tokens and doc_id not in seen:
                ordered_cites.append(doc_id)
                seen.add(doc_id)
        return ordered_cites

    # --- transport -----------------------------------------------------------

    def _ollama_generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama call failed: {exc}") from exc
        return body.get("response", "")
