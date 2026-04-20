#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag_lab.data.hotpotqa import (
    HOTPOTQA_URLS,
    build_slice,
    download_hotpotqa_split,
    export_slice_jsonl,
    load_hotpotqa_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a HotpotQA dev slice for local experiments.")
    parser.add_argument(
        "--source",
        type=Path,
        help="Path to an existing HotpotQA JSON file. If omitted, the script downloads the split.",
    )
    parser.add_argument(
        "--split",
        default="dev_distractor",
        choices=sorted(HOTPOTQA_URLS.keys()),
        help="HotpotQA split to download when --source is not provided.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of examples to keep in the slice.",
    )
    parser.add_argument(
        "--levels",
        default="medium,hard",
        help="Comma-separated level filter. Empty string keeps all levels.",
    )
    parser.add_argument(
        "--types",
        default="bridge,comparison",
        help="Comma-separated question type filter. Empty string keeps all types.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/raw/hotpotqa"),
        help="Where to store downloaded raw files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/hotpotqa/dev_slice.jsonl"),
        help="Where to write the processed JSONL slice.",
    )
    args = parser.parse_args()

    source_path = args.source
    if source_path is None:
        file_name = f"{args.split}.json"
        source_path = args.download_dir / file_name
        if not source_path.exists():
            download_hotpotqa_split(args.split, source_path)

    level_filter = {item.strip() for item in args.levels.split(",") if item.strip()}
    type_filter = {item.strip() for item in args.types.split(",") if item.strip()}

    records = load_hotpotqa_records(source_path)
    sliced = build_slice(
        records,
        limit=args.limit,
        levels=level_filter or None,
        question_types=type_filter or None,
    )
    export_slice_jsonl(sliced, args.output)

    summary = {
        "source_path": str(source_path),
        "output_path": str(args.output),
        "requested_limit": args.limit,
        "actual_count": len(sliced),
        "levels": sorted(level_filter),
        "types": sorted(type_filter),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
