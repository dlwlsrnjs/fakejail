#!/usr/bin/env python3
"""Audit merged NLLB concept translations and build a stratified QA packet."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/jailnewsbench_person_domain_20260930/concept_translations_v1/translations_nllb.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/jailnewsbench_person_domain_20260930/concept_translations_v1/qa_v1"),
    )
    parser.add_argument("--valid-per-language", type=int, default=5)
    parser.add_argument("--invalid-per-language", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ids: set[str] = set()
    duplicate_ids: list[str] = []
    empty_rows: list[dict[str, Any]] = []
    above_one = 0
    for row in rows:
        translation_id = row["translation_id"]
        if translation_id in ids:
            duplicate_ids.append(translation_id)
        ids.add(translation_id)
        by_language[row["language"]].append(row)
        above_one += int(row["backtranslation_similarity"] > 1.0)
        if not row["translation"].strip() or not row["backtranslation"].strip():
            empty_rows.append(row)

    rng = random.Random(args.seed)
    packet: list[dict[str, Any]] = []
    language_report: dict[str, Any] = {}
    for language in sorted(by_language):
        language_rows = by_language[language]
        valid = [row for row in language_rows if row["valid"] and row["translation"].strip()]
        invalid = [row for row in language_rows if not row["valid"] or not row["translation"].strip()]
        valid_sorted = sorted(valid, key=lambda row: row["backtranslation_similarity"])
        invalid_sorted = sorted(invalid, key=lambda row: row["backtranslation_similarity"], reverse=True)

        # Boundary cases are more informative than uniformly easy examples. Add a
        # deterministic random half to retain coverage away from the threshold.
        def choose(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
            if len(candidates) <= count:
                return candidates
            boundary_count = (count + 1) // 2
            boundary = candidates[:boundary_count]
            remaining = candidates[boundary_count:]
            return boundary + rng.sample(remaining, count - boundary_count)

        selected_valid = choose(valid_sorted, args.valid_per_language)
        selected_invalid = choose(invalid_sorted, args.invalid_per_language)
        for expected, selected in (("valid", selected_valid), ("invalid", selected_invalid)):
            for row in selected:
                packet.append({
                    "review_id": f"qa::{row['translation_id']}",
                    "expected_bucket": expected,
                    "language": row["language"],
                    "nllb_code": row["nllb_code"],
                    "concept_id": row["concept_id"],
                    "concept_types": row["concept_types"],
                    "canonical_english": row["canonical_english"],
                    "translation": row["translation"],
                    "backtranslation": row["backtranslation"],
                    "backtranslation_similarity": min(1.0, max(-1.0, row["backtranslation_similarity"])),
                    "automatic_valid": row["valid"],
                })
        similarities = [min(1.0, max(-1.0, row["backtranslation_similarity"])) for row in language_rows]
        language_report[language] = {
            "rows": len(language_rows),
            "valid": len(valid),
            "valid_rate": len(valid) / len(language_rows),
            "empty_translation_or_backtranslation": sum(
                not row["translation"].strip() or not row["backtranslation"].strip()
                for row in language_rows
            ),
            "mean_similarity_clipped": sum(similarities) / len(similarities),
            "minimum_similarity_clipped": min(similarities),
        }

    type_counts: Counter[str] = Counter()
    type_valid: Counter[str] = Counter()
    for row in rows:
        for concept_type in row["concept_types"]:
            type_counts[concept_type] += 1
            type_valid[concept_type] += int(bool(row["valid"] and row["translation"].strip()))

    report = {
        "schema": "jailnews_concept_translation_qa/v1",
        "source": str(args.input),
        "rows": len(rows),
        "unique_translation_ids": len(ids),
        "duplicate_translation_ids": len(duplicate_ids),
        "languages": len(by_language),
        "concepts": len({row["concept_id"] for row in rows}),
        "valid_rows_nonempty": sum(bool(row["valid"] and row["translation"].strip()) for row in rows),
        "empty_translation_or_backtranslation": len(empty_rows),
        "cosine_above_one_before_clipping": above_one,
        "cosine_note": "Values above 1 are float32 rounding artifacts and are clipped only in QA outputs.",
        "qa_packet_rows": len(packet),
        "sampling": {
            "valid_per_language": args.valid_per_language,
            "invalid_per_language": args.invalid_per_language,
            "strategy": "threshold-boundary plus deterministic random coverage",
            "seed": args.seed,
        },
        "by_concept_type": {
            key: {
                "rows": type_counts[key],
                "valid_nonempty": type_valid[key],
                "valid_rate": type_valid[key] / type_counts[key],
            }
            for key in sorted(type_counts)
        },
        "by_language": language_report,
    }
    write_jsonl(args.output_dir / "review_packet.jsonl", packet)
    write_jsonl(args.output_dir / "empty_rows.jsonl", empty_rows)
    (args.output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key not in {"by_language", "by_concept_type"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
