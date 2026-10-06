#!/usr/bin/env python3
"""Apply independently verified Luna repairs to selected PC2 translations."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def strict(row: dict) -> bool:
    return (
        row["semantic_fidelity"] in {"pass", "minor"}
        and row["identity_clue_preserved"] == "yes"
        and row["hidden_name_leak"] == "no"
        and row["grammar_quality"] in {"good", "acceptable"}
        and row["quiz_form_preserved"] == "yes"
    )


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--retry-glob", required=True)
    parser.add_argument("--verify-glob", required=True)
    parser.add_argument("--output-scores", type=Path, required=True)
    parser.add_argument("--output-reviews", type=Path, required=True)
    args = parser.parse_args()

    corrections = {}
    for filename in sorted(glob.glob(args.retry_glob)):
        for row in read(Path(filename)):
            corrections[row["review_id"]] = row
    verifies = {}
    for filename in sorted(glob.glob(args.verify_glob)):
        for row in read(Path(filename)):
            if not strict(row):
                raise ValueError(f"retry failed independent verification: {row['review_id']}")
            row["strict_accept"] = True
            row["luna_retry"] = True
            verifies[row["review_id"]] = row
    if set(corrections) != set(verifies):
        raise ValueError("correction/verification ID mismatch")

    scores = read(args.scores)
    score_index = {row["pilot_id"]: row for row in scores}
    for review_id, correction in corrections.items():
        pilot_id, condition = review_id.split("::", 1)
        score = score_index[pilot_id]
        if condition == "pc2_p25":
            item = score["selected"]["25"]
        elif condition == "pc2_p50":
            item = score["selected"]["50"]
        elif condition == "random_valid":
            item = score["random_valid"]
        else:
            raise ValueError(f"unknown retry condition: {condition}")
        if item["language"] != correction["language"]:
            raise ValueError(f"language mismatch for {review_id}")
        item["nllb_translation_original"] = item["translation"]
        item["nllb_backtranslation_original"] = item["backtranslation"]
        item["translation"] = correction["corrected_translation"]
        item["backtranslation"] = correction["corrected_backtranslation"]
        item["translation_backend"] = "gpt-6-luna-repair"
        item["luna_retry_verified"] = True

    reviews = {row["review_id"]: row for row in read(args.reviews)}
    reviews.update(verifies)
    for row in reviews.values():
        row["strict_accept"] = strict(row)
    if not all(row["strict_accept"] for row in reviews.values()):
        raise ValueError("final review set still contains rejected rows")

    write(args.output_scores, scores)
    write(args.output_reviews, list(reviews.values()))
    print(json.dumps({
        "score_rows": len(scores),
        "repaired": len(corrections),
        "final_reviews": len(reviews),
        "strict_accepted": sum(row["strict_accept"] for row in reviews.values()),
    }))


if __name__ == "__main__":
    main()
