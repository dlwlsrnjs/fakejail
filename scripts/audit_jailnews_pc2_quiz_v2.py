#!/usr/bin/env python3
"""Fail-closed integrity audit for the rebuilt multilingual quiz matrix."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from jailnews_pc2_languages import LANGUAGES


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = args.base

    people = read(base / "person_cohort.jsonl")
    controls = read(base / "no_person_controls.jsonl")
    behaviors = read(base / "behavior_translations_merged.jsonl")
    inventory = read(base / "translation_inventory.jsonl")
    translations = read(base / "translations_merged.jsonl")
    matrix = read(base / "quiz_matrix.jsonl")

    require(len(people) == 100, f"person cohort has {len(people)} rows")
    require(len(controls) == 100, f"no-person cohort has {len(controls)} rows")
    require(len({row['canonical_person'] for row in people}) == 100, "people are not unique")
    require(all(row["sample_entity_state"] == "verified_person_present" for row in people),
            "person cohort contains a non-verified-person sample")
    require(all(row["person_count"] == 1 for row in people), "person count mismatch")
    require(all(row["sample_entity_state"] == "verified_no_person" for row in controls),
            "control cohort contains a non-verified-no-person sample")
    require(all(row["person_count"] == 0 and row["canonical_person"] is None for row in controls),
            "no-person control has a person")

    sources = {row["quiz_id"]: row for row in people + controls}
    require(len(sources) == 200, "duplicate cohort quiz IDs")
    require(len(behaviors) == 200, f"behavior rows={len(behaviors)}")
    for behavior in behaviors:
        source = sources[behavior["quiz_id"]]
        require(behavior.get("article_segment_protected") is True,
                f"unprotected article: {behavior['quiz_id']}")
        require(source["article_english_template"] in behavior["behavior_english_template"],
                f"official English article missing: {behavior['quiz_id']}")
        expected = source["behavior_local_template"].count("[[PERSON_1]]")
        actual = behavior["behavior_english_template"].count("[[PERSON_1]]")
        require(expected == actual, f"placeholder mismatch: {behavior['quiz_id']}")

    expected_translation_keys = {
        (item["item_id"], language) for item in inventory for language in LANGUAGES
    }
    actual_translation_keys = {(row["item_id"], row["language"]) for row in translations}
    require(actual_translation_keys == expected_translation_keys,
            f"translation grid mismatch: actual={len(actual_translation_keys)} "
            f"expected={len(expected_translation_keys)}")
    require(len(translations) == len(actual_translation_keys), "duplicate translation rows")
    require(all(str(row.get("translation", "")).strip() for row in translations),
            "translation grid contains an empty target string")
    require(all(str(row.get("backtranslation", "")).strip() for row in translations),
            "translation grid contains an empty backtranslation")

    require(len(matrix) == 14_600, f"matrix rows={len(matrix)}")
    require(len({row["matrix_id"] for row in matrix}) == len(matrix), "duplicate matrix IDs")
    require(all(str(row.get("prompt", "")).strip() for row in matrix),
            "matrix contains an empty prompt")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in matrix:
        grouped[row["quiz_id"]].append(row)
        source = sources[row["quiz_id"]]
        require(row["sample_entity_state"] == source["sample_entity_state"],
                f"entity-state mismatch: {row['matrix_id']}")
        if row["condition"] == "person_identity_quiz":
            require(source["person_count"] == 1 and "IDENTITY QUIZ" in row["prompt"],
                    f"person prompt mismatch: {row['matrix_id']}")
        elif row["condition"] == "concept_only_quiz":
            require(source["person_count"] == 0 and "IDENTITY QUIZ" not in row["prompt"],
                    f"no-person prompt mismatch: {row['matrix_id']}")
    for quiz_id, rows in grouped.items():
        require(len(rows) == 73, f"{quiz_id} has {len(rows)} settings, expected 73")
        require(sum(row["condition"] == "direct_original" for row in rows) == 1,
                f"{quiz_id} direct-control mismatch")
        languages = {row["language"] for row in rows if row["condition"] != "direct_original"}
        require(languages == set(LANGUAGES), f"{quiz_id} language grid mismatch")

    summary = {
        "status": "PASS",
        "person_cases": len(people),
        "verified_no_person_controls": len(controls),
        "behavior_valid": sum(row["valid"] for row in behaviors),
        "behavior_total": len(behaviors),
        "translations": len(translations),
        "translations_valid": sum(row["valid"] for row in translations),
        "translations_filled": sum(bool(str(row.get("translation", "")).strip()) for row in translations),
        "backtranslations_filled": sum(bool(str(row.get("backtranslation", "")).strip()) for row in translations),
        "unverified_translations_filled": sum(
            not row["valid"] and bool(str(row.get("translation", "")).strip())
            for row in translations
        ),
        "matrix_rows": len(matrix),
        "matrix_prompts_filled": sum(bool(str(row.get("prompt", "")).strip()) for row in matrix),
        "matrix_eligible": sum(row["eligible_for_model_run"] for row in matrix),
        "conditions": dict(Counter(row["condition"] for row in matrix)),
    }
    write_json(args.output or (base / "integrity_audit.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
