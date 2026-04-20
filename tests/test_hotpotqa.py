import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_lab.data.hotpotqa import (
    build_slice,
    convert_record,
    export_slice_jsonl,
    load_hotpotqa_records,
)


class HotpotQATests(unittest.TestCase):
    def test_convert_record_normalizes_hotpot_example(self) -> None:
        record = {
            "_id": "example-1",
            "question": "Which university did the author of The Hobbit attend?",
            "answer": "Exeter College, Oxford",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": {
                "title": ["The Hobbit", "Tolkien at Oxford"],
                "sent_id": [0, 0],
            },
            "context": {
                "title": ["The Hobbit", "Tolkien at Oxford"],
                "sentences": [
                    ["The Hobbit was written by J. R. R. Tolkien."],
                    ["J. R. R. Tolkien studied at Exeter College, Oxford."],
                ],
            },
        }

        normalized = convert_record(record)

        self.assertEqual(normalized["id"], "example-1")
        self.assertEqual(normalized["question"], record["question"])
        self.assertEqual(normalized["supporting_facts"][0]["title"], "The Hobbit")
        self.assertEqual(normalized["contexts"][1]["title"], "Tolkien at Oxford")

    def test_build_slice_filters_and_limits_examples(self) -> None:
        records = [
            {"id": "1", "level": "easy", "type": "bridge", "question": "q1"},
            {"id": "2", "level": "medium", "type": "bridge", "question": "q2"},
            {"id": "3", "level": "hard", "type": "comparison", "question": "q3"},
        ]

        sliced = build_slice(records, limit=1, levels={"medium", "hard"}, question_types={"bridge"})

        self.assertEqual(len(sliced), 1)
        self.assertEqual(sliced[0]["id"], "2")

    def test_build_slice_round_robins_across_available_buckets(self) -> None:
        records = [
            {"id": "1", "level": "medium", "type": "bridge", "question": "q1"},
            {"id": "2", "level": "medium", "type": "bridge", "question": "q2"},
            {"id": "3", "level": "hard", "type": "comparison", "question": "q3"},
            {"id": "4", "level": "hard", "type": "comparison", "question": "q4"},
        ]

        sliced = build_slice(records, limit=3)

        self.assertEqual(len(sliced), 3)
        self.assertEqual({row["level"] for row in sliced}, {"medium", "hard"})
        self.assertEqual({row["type"] for row in sliced}, {"bridge", "comparison"})

    def test_load_and_export_slice_round_trip(self) -> None:
        source_records = [
            {
                "_id": "example-1",
                "question": "q1",
                "answer": "a1",
                "type": "bridge",
                "level": "medium",
                "supporting_facts": {"title": ["Doc"], "sent_id": [0]},
                "context": {"title": ["Doc"], "sentences": [["Sentence one."]]},
            },
            {
                "_id": "example-2",
                "question": "q2",
                "answer": "a2",
                "type": "comparison",
                "level": "hard",
                "supporting_facts": {"title": ["Doc2"], "sent_id": [1]},
                "context": {"title": ["Doc2"], "sentences": [["Sentence two."]]},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "hotpot_dev.json"
            output_path = Path(tmp_dir) / "slice.jsonl"
            source_path.write_text(json.dumps(source_records), encoding="utf-8")

            records = load_hotpotqa_records(source_path)
            sliced = build_slice(records, limit=2)
            export_slice_jsonl(sliced, output_path)

            exported_lines = output_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(len(exported_lines), 2)
        self.assertEqual(json.loads(exported_lines[0])["id"], "example-1")


if __name__ == "__main__":
    unittest.main()
