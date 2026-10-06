#!/usr/bin/env python3
"""Fail-closed invariant audit for the English-controlled V5 arm matrices."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/jailnews_bandit_20260930/runtime/stochastic_360_identity_v5_english"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    report = {"schema": "jailnews_stochastic_identity_v5_english_audit/v1", "modes": {}}
    for mode in ("a_literal", "canonical_en"):
        path = BASE / mode / "base_arms.jsonl"
        counts = Counter()
        languages = defaultdict(set)
        normalized_seed = defaultdict(set)
        normalized_attacked = defaultdict(set)
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
                prompt = str(row["prompt"])
                attacked = str(row["attacked_prompt"])
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
        }

    target = BASE / "integrity_audit.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
