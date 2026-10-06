#!/usr/bin/env python3
"""Build the 501-row input expected by the audited behavior translator."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/jailnews_bandit_20260930/runtime/official_baseline_501/base_arms.jsonl"
OUTPUT = ROOT / "artifacts/jailnews_bandit_20260930/runtime/english_article_named_baseline_501/translation"


def main() -> None:
    rows = {}
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("attack_type") != "original":
                continue
            person = str(row["victim_person_id"])
            rows[person] = {
                "quiz_id": person,
                "sample_id": row["sample_id"],
                "source_language_code": row["language_code"],
                "behavior_local_template": row["seed_instruction_local"],
                "article_local_template": row["article_local"],
                "article_english_template": row["article_en"],
            }
    if len(rows) != 501:
        raise RuntimeError(f"expected 501 people, found {len(rows)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "cohort.jsonl"
    temporary = target.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for person in sorted(rows):
            handle.write(json.dumps(rows[person], ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    (OUTPUT / "empty_controls.jsonl").write_text("", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(target.resolve())}, indent=2))


if __name__ == "__main__":
    main()
