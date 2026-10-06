#!/usr/bin/env python3
"""Join rejected translation candidates with independent reviewer feedback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    judgments = {row["key"]: row for row in read(args.judgments)}
    output = []
    missing = []
    for row in read(args.candidates):
        key = row["translation_id"]
        judgment = judgments.get(key)
        if judgment is None:
            missing.append(key)
            continue
        parsed = judgment.get("parsed") or {}
        accepted = (
            parsed.get("equivalent") is True
            and parsed.get("target_language_valid") is True
            and parsed.get("needs_retranslation") is False
            and row.get("numbers_preserved") is True
        )
        if accepted:
            continue
        repaired = dict(row)
        feedback = dict(parsed)
        if row.get("numbers_preserved") is not True:
            feedback["error_types"] = list(feedback.get("error_types", [])) + ["number_not_preserved"]
            feedback["reason"] = (
                str(feedback.get("reason", "")).strip()
                + " The previous attempt did not preserve every source number exactly."
            ).strip()
        repaired["repair_feedback"] = feedback
        repaired["repair_feedback_model"] = "google/gemma-3-27b-it"
        repaired["repair_feedback_raw"] = judgment.get("raw_judgment")
        output.append(repaired)

    if missing:
        raise RuntimeError(f"missing judgments for {len(missing)} candidates")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(output), "accepted_removed": len(judgments) - len(output)}, indent=2))


if __name__ == "__main__":
    main()
