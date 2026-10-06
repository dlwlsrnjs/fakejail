#!/usr/bin/env python3
"""Create a person-disjoint, metadata-diverse selector validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def labels(row: dict[str, Any]) -> set[tuple[str, str]]:
    values = {
        ("domain", row["luna_political_domain"]),
        ("role", row["luna_primary_person_role"]),
        ("event", row["luna_event_type"]),
        ("flag", "election" if row["luna_election_related"] else "not_election"),
        ("flag", "war_security" if row["luna_war_or_security_related"] else "not_war_security"),
    }
    values.update(("sensitivity", value) for value in row["luna_sensitive_concepts"])
    values.update(("conflict", value) for value in row["luna_conflicts_named"])
    return values


def stable_tie(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def diverse_samples(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    remaining = list(rows)
    selected: list[dict[str, Any]] = []
    covered: set[tuple[str, str]] = set()
    languages: set[str] = set()
    splits: set[str] = set()
    while remaining and len(selected) < count:
        def score(row: dict[str, Any]) -> tuple[float, int]:
            novelty = sum(1.0 for label in labels(row) if label not in covered)
            novelty += 0.6 * int(row["language_code"] not in languages)
            novelty += 0.4 * int(row["source_split"] not in splits)
            novelty += 0.5 * len(row["luna_sensitive_concepts"])
            novelty += 0.5 * int(row["luna_war_or_security_related"] or row["luna_election_related"])
            return novelty, stable_tie(row["sample_id"], seed)
        picked = max(remaining, key=score)
        remaining.remove(picked)
        selected.append(picked)
        covered.update(labels(picked))
        languages.add(picked["language_code"])
        splits.add(picked["source_split"])
    return selected


def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "people": len({row["luna_primary_person"] for row in rows}),
        "domains": dict(Counter(row["luna_political_domain"] for row in rows).most_common()),
        "roles": dict(Counter(row["luna_primary_person_role"] for row in rows).most_common()),
        "events": dict(Counter(row["luna_event_type"] for row in rows).most_common()),
        "sensitivities": dict(Counter(
            value for row in rows for value in row["luna_sensitive_concepts"]
        ).most_common()),
        "election_rows": sum(row["luna_election_related"] for row in rows),
        "war_security_rows": sum(row["luna_war_or_security_related"] for row in rows),
        "source_languages": len({row["language_code"] for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/jailnewsbench_person_domain_20260930/luna_review_v1/analysis_ready_samples.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/jailnewsbench_person_domain_20260930/selector_validation_split_v2"),
    )
    parser.add_argument("--validation-people", type=int, default=120)
    parser.add_argument("--samples-per-person", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_person[row["luna_primary_person"]].append(row)
    eligible = {
        person: person_rows for person, person_rows in by_person.items()
        if len(person_rows) >= args.samples_per_person
    }
    if len(eligible) < args.validation_people:
        raise ValueError(f"only {len(eligible)} people have enough samples")

    selected_people: list[str] = []
    coverage: Counter[tuple[str, str]] = Counter()
    remaining = set(eligible)
    while remaining and len(selected_people) < args.validation_people:
        def person_score(person: str) -> tuple[float, int]:
            person_labels = set().union(*(labels(row) for row in eligible[person]))
            coverage_gain = sum(1.0 / (1.0 + coverage[label]) for label in person_labels)
            diversity = len({row["luna_political_domain"] for row in eligible[person]})
            diversity += len({row["luna_event_type"] for row in eligible[person]})
            sensitive = len({value for row in eligible[person] for value in row["luna_sensitive_concepts"]})
            volume = math.log1p(len(eligible[person]))
            return coverage_gain + 0.5 * diversity + 0.7 * sensitive + 0.1 * volume, stable_tie(person, args.seed)
        picked = max(remaining, key=person_score)
        remaining.remove(picked)
        selected_people.append(picked)
        coverage.update(set().union(*(labels(row) for row in eligible[picked])))

    holdout = set(selected_people)
    validation: list[dict[str, Any]] = []
    for person in selected_people:
        chosen = diverse_samples(eligible[person], args.samples_per_person, args.seed)
        for rank, row in enumerate(chosen, 1):
            validation.append({
                **row,
                "selector_validation_person": person,
                "selector_validation_rank": rank,
                "selector_split": "person_disjoint_validation",
                "selector_selection_method": "greedy_domain_role_event_sensitivity_conflict_diversity_v2",
            })

    # Remove every training sample that mentions any held-out person, including
    # secondary people, to avoid identity leakage into Wikipedia/archetype banks.
    train = []
    removed_secondary_overlap = 0
    for row in rows:
        mentioned = {person["canonical_person"] for person in row["luna_people"]}
        if mentioned & holdout:
            if row["luna_primary_person"] not in holdout:
                removed_secondary_overlap += 1
            continue
        train.append({**row, "selector_split": "archetype_train"})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "archetype_train_samples.jsonl", train)
    write_jsonl(args.output_dir / "validation_samples.jsonl", validation)
    (args.output_dir / "validation_people.json").write_text(
        json.dumps(selected_people, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema": "jailnews_selector_person_disjoint_split/v2",
        "source_rows": len(rows),
        "source_people": len(by_person),
        "eligible_people_with_required_samples": len(eligible),
        "validation_people": len(selected_people),
        "samples_per_validation_person": args.samples_per_person,
        "validation": counts(validation),
        "archetype_train": counts(train),
        "secondary_person_overlap_rows_removed_from_train": removed_secondary_overlap,
        "asr_labels_used": False,
        "intended_experiment": {
            "main_cohort": "120 held-out people x 5 domain/event-diverse prompts",
            "language_conditions_per_prompt": [
                "routed top-1", "MoE expert union top-3", "PC2 p50",
                "English", "person-country language", "seeded random valid language",
            ],
            "sentinel_full_matrix": "a smaller stratified subset should retain all 72 languages for rank-correlation auditing",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
