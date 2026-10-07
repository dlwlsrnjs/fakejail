#!/usr/bin/env python3
"""Fail-closed invariant audit for the English-controlled V5 arm matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/jailnews_bandit_20260930/runtime/stochastic_360_identity_v5_english"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional report path. Omit to validate without modifying the source snapshot.",
    )
    args = parser.parse_args()

    report = {"schema": "jailnews_stochastic_identity_v5_english_audit/v1", "modes": {}}
    for mode in ("a_literal", "canonical_en"):
        path = args.base / mode / "base_arms.jsonl"
        counts = Counter()
        languages = defaultdict(set)
        normalized_seed = defaultdict(set)
        normalized_attacked = defaultdict(set)
        trial_ids = set()
        cells = set()
        rows = 0
        failures = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows += 1
                person = str(row["victim_person_id"])
                method = str(row["attack_type"])
                language = str(row["language"])
                description = str(row["a_description"])
                trial_id = str(row["trial_id"])
                prompt = str(row["prompt"])
                attacked = str(row["attacked_prompt"])
                cell = (person, method, language)
                if trial_id in trial_ids:
                    failures.append(f"duplicate_trial:{trial_id}")
                if cell in cells:
                    failures.append(f"duplicate_cell:{cell}")
                trial_ids.add(trial_id)
                cells.add(cell)
                counts[method] += 1
                languages[(person, method)].add(language)
                marker = f"A: {description}"
                if prompt.count(marker) != 1 or attacked.count(marker) != 1:
                    failures.append(f"descriptor_marker:{row['trial_id']}")
                    continue
                normalized_seed[(person, method)].add(digest(prompt.replace(marker, "A: <DESC>")))
                normalized_attacked[(person, method)].add(
                    digest(attacked.replace(marker, "A: <DESC>"))
                )
                required = (
                    row.get("english_controlled") is True
                    and row.get("source_language_code") == "en"
                    and row.get("expected_output_language_code") == "en"
                    and row.get("instruction_translation_valid") is True
                    and prompt.startswith("ARTICLE\n")
                    and "\n\nIDENTITY BINDING\n" in prompt
                    and "\n\nINSTRUCTION\n" in prompt
                    and "[ARTICLE]" in str(row.get("instruction", ""))
                    and "Write the final news article in English." in prompt
                    and attacked.endswith(prompt)
                )
                if not required:
                    failures.append(f"contract:{row['trial_id']}")

        expected_rows = 501 * 72 * 5
        if rows != expected_rows:
            failures.append(f"rows:{rows}!={expected_rows}")
        if len(trial_ids) != expected_rows:
            failures.append(f"unique_trials:{len(trial_ids)}!={expected_rows}")
        if len(cells) != expected_rows:
            failures.append(f"unique_cells:{len(cells)}!={expected_rows}")
        if len(languages) != 501 * 5:
            failures.append(f"person_method_groups:{len(languages)}!={501 * 5}")
        bad_language_groups = sum(len(value) != 72 for value in languages.values())
        bad_seed_groups = sum(len(value) != 1 for value in normalized_seed.values())
        bad_attacked_groups = sum(len(value) != 1 for value in normalized_attacked.values())
        if bad_language_groups:
            failures.append(f"language_groups:{bad_language_groups}")
        if bad_seed_groups:
            failures.append(f"non_descriptor_seed_differences:{bad_seed_groups}")
        if bad_attacked_groups:
            failures.append(f"non_descriptor_attack_differences:{bad_attacked_groups}")
        if failures:
            raise RuntimeError(f"V5 audit failed for {mode}: {failures[:20]}")
        report["modes"][mode] = {
            "rows": rows,
            "people_method_groups": len(languages),
            "languages_per_group": 72,
            "method_counts": dict(counts),
            "only_descriptor_varies_within_person_method": True,
            "english_article_instruction_output_contract": True,
            "base_arms_sha256": file_digest(path),
        }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
