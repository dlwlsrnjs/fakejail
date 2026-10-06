#!/usr/bin/env python3
"""Compose controlled JailNewsBench attack arms from validated PC2 clue translations."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def substitute_once(text: str, target: str, replacement: str) -> str:
    result, count = re.subn(re.escape(target), replacement, text, count=1, flags=re.IGNORECASE)
    if count != 1:
        raise ValueError(f"expected one replaceable occurrence of {target!r}, found {count}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--reviews", type=Path, default=None,
                        help="Merged Luna QA JSONL; reviewed variants must have strict_accept=true")
    args = parser.parse_args()

    records = {row["pilot_id"]: row for row in read_jsonl(args.records)}
    scores = read_jsonl(args.scores)
    review_index = {}
    if args.reviews:
        review_index = {row["review_id"]: row for row in read_jsonl(args.reviews)}
    rng = random.Random(args.seed)
    output = []

    for score in scores:
        row = records[score["pilot_id"]]
        person = row["selected_person"]
        original = row["seed_instruction_local"]
        candidates = [item for item in score["candidates"] if item["valid"] and item["language"] != "English"]
        random_item = rng.choice(candidates)
        variants = {
            "original": {"language": row["language_code"], "clue": person, "prompt": original},
            "english_quiz": {"language": "English", "clue": score["english_ipdm"]},
            "random_multilingual_quiz": {
                "language": random_item["language"], "clue": random_item["translation"]
            },
        }
        for percentile in (25, 50):
            item = score["selected"][str(percentile)]
            if item:
                variants[f"pc2_p{percentile}_quiz"] = {
                    "language": item["language"], "clue": item["translation"]
                }

        for condition, variant in variants.items():
            review_condition = {
                "random_multilingual_quiz": "random_valid",
                "pc2_p25_quiz": "pc2_p25",
                "pc2_p50_quiz": "pc2_p50",
            }.get(condition)
            if review_condition:
                review_id = f"{row['pilot_id']}::{review_condition}"
                if review_id not in review_index:
                    if args.reviews:
                        raise ValueError(f"missing Luna review: {review_id}")
                elif not review_index[review_id].get("strict_accept", False):
                    continue
            if condition != "original":
                variant["prompt"] = substitute_once(original, person, f"[{variant['clue']}]")
            output.append({
                "uid": row["source_record_id"],
                "pilot_id": row["pilot_id"],
                "condition": condition,
                "selected_person": person,
                "clue_language": variant["language"],
                "clue": variant["clue"],
                "attacked_prompt": variant["prompt"],
                "source_language_code": row["language_code"],
                "region_en": row["region_en"],
                "motivation_category": row["motivation_category"],
                "article_local": row["article_local"],
                "article_en": row["article_en"],
                "seed_instruction_local": original,
                "identity_validation_required": condition != "original",
            })

    write_jsonl(args.output, output)
    print(json.dumps({
        "records": len(records),
        "score_rows": len(scores),
        "attack_rows": len(output),
        "conditions": sorted({row["condition"] for row in output}),
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
