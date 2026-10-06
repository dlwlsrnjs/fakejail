#!/usr/bin/env python3
"""Analyze which clue languages work for each person in the PC2 experiment."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def read_latest(path: Path) -> dict[str, dict]:
    latest = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("judge_error") is None:
                    latest[row["matrix_id"]] = row
    return latest


def pct(n: int, d: int) -> float:
    return round(100 * n / d, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--translated-judgments", type=Path, required=True)
    parser.add_argument("--direct-judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix = {}
    with args.matrix.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                matrix[row["matrix_id"]] = row
    translated = read_latest(args.translated_judgments)
    direct = {row["pilot_id"]: row for row in read_latest(args.direct_judgments).values()}
    if set(matrix) != set(translated):
        raise ValueError("matrix and translated judgments do not align")

    grouped = defaultdict(list)
    for matrix_id, judgment in translated.items():
        source = matrix[matrix_id]
        j = judgment["judgment"]
        grouped[source["pilot_id"]].append({
            "language": source["clue_language"],
            "strict_success": bool(j["strict_success"]),
            "non_refusal": not j["refused"],
            "entity_match": bool(j["entity_match"]),
            "core_claim_match": bool(j["core_claim_match"]),
            "substantive_news_article": bool(j["substantive_news_article"]),
            "adherence_score": j["adherence_score"],
            "confidence": j["confidence"],
            "nllb_similarity_valid": source["nllb_similarity_valid"],
        })

    people = []
    for pilot_id, cells in sorted(grouped.items()):
        sample = next(row for row in matrix.values() if row["pilot_id"] == pilot_id)
        direct_j = direct[pilot_id]["judgment"]
        cells.sort(key=lambda x: x["language"])
        successes = [x for x in cells if x["strict_success"]]
        failures = [x for x in cells if not x["strict_success"]]
        rank_key = lambda x: (
            x["strict_success"], x["adherence_score"], x["entity_match"],
            x["core_claim_match"], x["non_refusal"], x["confidence"],
        )
        best = sorted(cells, key=rank_key, reverse=True)
        worst = sorted(cells, key=rank_key)
        direct_success = bool(direct_j["strict_success"])
        people.append({
            "pilot_id": pilot_id,
            "uid": sample["uid"],
            "person": sample["selected_person"],
            "region_en": sample["region_en"],
            "source_language_code": sample["source_language_code"],
            "motivation_category": sample["motivation_category"],
            "direct_strict_success": direct_success,
            "direct_non_refusal": not direct_j["refused"],
            "direct_adherence_score": direct_j["adherence_score"],
            "translated_successes": len(successes),
            "translated_success_rate_pct": pct(len(successes), len(cells)),
            "translated_non_refusals": sum(x["non_refusal"] for x in cells),
            "translated_non_refusal_rate_pct": pct(sum(x["non_refusal"] for x in cells), len(cells)),
            "mean_adherence_score": round(sum(x["adherence_score"] for x in cells) / len(cells), 3),
            "successful_languages": [x["language"] for x in successes],
            "failed_languages": [x["language"] for x in failures],
            "gain_languages": [x["language"] for x in successes] if not direct_success else [],
            "loss_languages": [x["language"] for x in failures] if direct_success else [],
            "best_languages": [x["language"] for x in best[:10]],
            "worst_languages": [x["language"] for x in worst[:10]],
            "original_english_target": sample["article_en"],
            "direct_prompt": sample["seed_instruction_local"],
            "cells": cells,
        })

    people.sort(key=lambda x: (-x["translated_success_rate_pct"], x["person"]))
    result = {
        "people": people,
        "person_count": len(people),
        "language_count": len({cell["language"] for p in people for cell in p["cells"]}),
        "most_susceptible": [
            {"person": p["person"], "success_rate_pct": p["translated_success_rate_pct"]}
            for p in people[:5]
        ],
        "least_susceptible": [
            {"person": p["person"], "success_rate_pct": p["translated_success_rate_pct"]}
            for p in people[-5:]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "people"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
