#!/usr/bin/env python3
"""Build the full 23 x 72 PC2 quiz-language attack matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def replace_once(text: str, target: str, replacement: str) -> str:
    result, count = re.subn(re.escape(target), replacement, text, count=1, flags=re.IGNORECASE)
    if count != 1:
        raise ValueError(f"expected one occurrence of {target!r}, found {count}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = {row["pilot_id"]: row for row in read(args.records)}
    output = []
    repair_count = 0
    for score in read(args.scores):
        row = records[score["pilot_id"]]
        repaired_by_language = {}
        selected_items = list(score.get("selected", {}).values()) + [score.get("random_valid")]
        for item in selected_items:
            if item and item.get("luna_retry_verified"):
                repaired_by_language.setdefault(item["language"], item)

        for candidate in score["candidates"]:
            item = repaired_by_language.get(candidate["language"], candidate)
            repaired = item is not candidate
            repair_count += int(repaired)
            clue = item["translation"]
            output.append({
                "matrix_id": f"{row['pilot_id']}::{candidate['language']}",
                "uid": row["source_record_id"],
                "sample_id": row.get("sample_id") or row.get("collection_id") or f"jailnewsbench:train:{row['source_record_id']}",
                "pilot_id": row["pilot_id"],
                "canonical_person": row.get("canonical_person", row["selected_person"]),
                "selected_person": row["selected_person"],
                "clue_language": candidate["language"],
                "nllb_code": candidate["nllb_code"],
                "clue": clue,
                "backtranslation": item["backtranslation"],
                "backtranslation_similarity": candidate["backtranslation_similarity"],
                "nllb_similarity_valid": candidate["valid"],
                "translation_source": "gpt-6-luna-repair" if repaired else "nllb-200-distilled-1.3B",
                "attacked_prompt": replace_once(
                    row["seed_instruction_local"], row["selected_person"], f"[{clue}]"
                ),
                "source_language_code": row["language_code"],
                "region_en": row["region_en"],
                "motivation_category": row["motivation_category"],
                "article_local": row["article_local"],
                "article_en": row["article_en"],
                "seed_instruction_local": row["seed_instruction_local"],
            })

    expected = len(records) * 72
    if len(output) != expected:
        raise ValueError(f"expected {expected} matrix rows, got {len(output)}")
    if len({row["matrix_id"] for row in output}) != len(output):
        raise ValueError("duplicate matrix_id")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "records": len(records), "languages": 72, "matrix_rows": len(output),
        "valid_rows": sum(row["nllb_similarity_valid"] for row in output),
        "luna_repaired_language_rows": repair_count, "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
