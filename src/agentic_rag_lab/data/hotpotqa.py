from __future__ import annotations

import json
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

HOTPOTQA_URLS = {
    "dev_distractor": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    "dev_fullwiki": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
    "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
}


def _normalize_supporting_facts(value: object) -> list[dict[str, object]]:
    if isinstance(value, dict):
        titles = list(value.get("title", []))
        sent_ids = list(value.get("sent_id", []))
        return [
            {"title": title, "sent_id": sent_id}
            for title, sent_id in zip(titles, sent_ids, strict=False)
        ]

    normalized: list[dict[str, object]] = []
    for item in value or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            normalized.append({"title": item[0], "sent_id": item[1]})
    return normalized


def _normalize_contexts(value: object) -> list[dict[str, object]]:
    if isinstance(value, dict):
        titles = list(value.get("title", []))
        sentences = list(value.get("sentences", []))
        return [
            {"title": title, "sentences": sentence_list}
            for title, sentence_list in zip(titles, sentences, strict=False)
        ]

    normalized: list[dict[str, object]] = []
    for item in value or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            normalized.append({"title": item[0], "sentences": item[1]})
    return normalized


def convert_record(record: dict[str, object]) -> dict[str, object]:
    record_id = record.get("_id", record.get("id", ""))
    return {
        "id": str(record_id),
        "question": str(record.get("question", "")),
        "answer": str(record.get("answer", "")),
        "type": str(record.get("type", "")),
        "level": str(record.get("level", "")),
        "supporting_facts": _normalize_supporting_facts(record.get("supporting_facts", [])),
        "contexts": _normalize_contexts(record.get("context", [])),
    }


def load_hotpotqa_records(path: str | Path) -> list[dict[str, object]]:
    source_path = Path(path)
    data = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("HotpotQA source file must be a JSON list.")
    return [convert_record(record) for record in data]


def build_slice(
    records: Iterable[dict[str, object]],
    *,
    limit: int,
    levels: set[str] | None = None,
    question_types: set[str] | None = None,
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []

    for record in records:
        if levels and str(record.get("level", "")) not in levels:
            continue
        if question_types and str(record.get("type", "")) not in question_types:
            continue
        filtered.append(record)

    if len(filtered) <= limit:
        return filtered

    buckets: dict[tuple[str, str], deque[dict[str, object]]] = defaultdict(deque)
    for record in filtered:
        key = (str(record.get("level", "")), str(record.get("type", "")))
        buckets[key].append(record)

    ordered_keys = sorted(buckets.keys())
    sliced: list[dict[str, object]] = []
    while len(sliced) < limit and ordered_keys:
        next_keys: list[tuple[str, str]] = []
        for key in ordered_keys:
            bucket = buckets[key]
            if bucket:
                sliced.append(bucket.popleft())
                if len(sliced) >= limit:
                    break
            if bucket:
                next_keys.append(key)
        ordered_keys = next_keys

    return sliced


def export_slice_jsonl(records: Iterable[dict[str, object]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def download_hotpotqa_split(
    split_name: str,
    destination: str | Path,
    *,
    timeout_seconds: int = 120,
) -> Path:
    if split_name not in HOTPOTQA_URLS:
        raise ValueError(f"Unsupported HotpotQA split: {split_name}")

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(HOTPOTQA_URLS[split_name], destination_path)
    return destination_path
