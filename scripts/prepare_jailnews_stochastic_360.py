#!/usr/bin/env python3
"""Build the 501-person x 72-language x 5-method stochastic BAI pool.

The base pool retains every non-empty translation.  Round-trip QC failures are
kept as an explicit covariate so language coverage remains exactly balanced.
The target seed panel is a balanced incomplete block: every one of the 360
language/method settings is assigned to exactly ``coverage_per_arm`` distinct
people, while per-person load differs by at most one when possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from jailnewsbench_table2_qwen32 import JAILBREAKS, OFFICIAL, sha256_text


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def representative_people(path: Path) -> tuple[list[dict[str, Any]], str]:
    by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prefixes = set()
    for row in read_jsonl(path):
        by_person[row["victim_person_id"]].append(row)
        if row["attack_type"] == "context_overload":
            attacked = row["attacked_prompt"]
            prompt = row["prompt"]
            if not attacked.endswith(prompt):
                raise RuntimeError("cannot recover the frozen context prefix")
            prefixes.add(attacked[: -len(prompt)])
    if len(by_person) != 501 or any(len(rows) != 10 for rows in by_person.values()):
        raise RuntimeError("expected the completed 501 x top-10 request set")
    if len(prefixes) != 1:
        raise RuntimeError(f"expected one frozen context prefix, found {len(prefixes)}")
    people = []
    for person_id, rows in sorted(by_person.items()):
        samples = {row["sample_id"] for row in rows}
        if len(samples) != 1:
            raise RuntimeError(f"multiple representative samples for {person_id}: {samples}")
        people.append(rows[0])
    return people, prefixes.pop()


def build_base(args: argparse.Namespace) -> None:
    people, context_prefix = representative_people(args.representatives)
    sample_ids = {row["sample_id"] for row in people}
    matrix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.matrix):
        if row["sample_id"] in sample_ids:
            if not str(row.get("a_description") or "").strip():
                raise RuntimeError(f"empty translation: {row['matrix_id']}")
            matrix[row["sample_id"]].append(row)
    malformed = {sample: len(rows) for sample, rows in matrix.items() if len(rows) != 72}
    if set(matrix) != sample_ids or malformed:
        raise RuntimeError(f"incomplete selected matrix: samples={len(matrix)} malformed={len(malformed)}")

    priors = {
        (row["language"], row["method"]): row
        for row in read_jsonl(args.prior)
        if row.get("endpoint") == "strict_article_success"
    }
    if len(priors) != 360:
        raise RuntimeError(f"expected 360 ensemble priors, found {len(priors)}")

    qc = Counter()
    person_samples = Counter()
    language_counts = Counter()
    method_counts = Counter()

    def rows() -> Iterable[dict[str, Any]]:
        for person in people:
            sample_id = person["sample_id"]
            person_samples[sample_id] += 1
            for rendered in sorted(matrix[sample_id], key=lambda row: row["language"]):
                valid = bool(rendered.get("description_translation_valid"))
                qc["verified" if valid else "provisional"] += len(JAILBREAKS)
                for method in JAILBREAKS:
                    attacked = OFFICIAL.apply_attack(
                        seed_instruction=rendered["prompt"],
                        motivation_category=str(person.get("motivation_category") or "Political"),
                        attack_type=method,
                        context_prefix=context_prefix if method == "context_overload" else None,
                    )
                    prior = priors[(rendered["language"], method)]
                    arm_id = f"{rendered['language']}::{method}"
                    base_trial_id = sha256_text(
                        f"stochastic360|{person['victim_person_id']}|{sample_id}|{arm_id}"
                    )[:24]
                    language_counts[rendered["language"]] += 1
                    method_counts[method] += 1
                    yield {
                        **rendered,
                        "uid": sample_id,
                        "trial_id": base_trial_id,
                        "base_trial_id": base_trial_id,
                        "arm_id": arm_id,
                        "victim_person_id": person["victim_person_id"],
                        "victim_dataset_label": person["victim_dataset_label"],
                        "entity_id": person.get("entity_id"),
                        "wikidata_qid": person.get("wikidata_qid"),
                        "sample_id": sample_id,
                        "attack_type": method,
                        "attacked_prompt": attacked,
                        "prompt_sha256": sha256_text(attacked),
                        "language_code": rendered["nllb_code"],
                        "article_local": rendered["article"],
                        "article_en": rendered["article"],
                        "region_en": person.get("region_en"),
                        "political_domain": person.get("political_domain"),
                        "person_role": person.get("person_role"),
                        "motivation_category": person.get("motivation_category"),
                        "war_or_security_related": person.get("war_or_security_related"),
                        "election_related": person.get("election_related"),
                        "translation_qc": "verified" if valid else "provisional_roundtrip",
                        "surrogate_prior_endpoint": prior["endpoint"],
                        "surrogate_prior_mean": prior["mean"],
                        "surrogate_prior_strength": prior["prior_strength"],
                        "surrogate_between_model_variance": prior["between_model_variance"],
                    }

    args.output.mkdir(parents=True, exist_ok=True)
    count = atomic_jsonl(args.output / "base_arms.jsonl", rows())
    expected = 501 * 360
    if count != expected:
        raise RuntimeError(f"base pool mismatch: {count} != {expected}")
    manifest = {
        "schema": "jailnews_stochastic_360_base/v1",
        "people": 501,
        "unique_samples": len(sample_ids),
        "languages": 72,
        "methods": list(JAILBREAKS),
        "arms_per_person": 360,
        "rows": count,
        "translation_qc": dict(qc),
        "duplicate_sample_multiplicity": dict(Counter(person_samples.values())),
        "language_balance_min_max": [min(language_counts.values()), max(language_counts.values())],
        "method_balance_min_max": [min(method_counts.values()), max(method_counts.values())],
        "context_sha256": hashlib.sha256(context_prefix.encode()).hexdigest(),
        "base_arms": str((args.output / "base_arms.jsonl").resolve()),
    }
    write_json(args.output / "base_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def balanced_pairs(people: list[str], arms: list[str], coverage: int, seed: int) -> set[tuple[str, str]]:
    if not 1 <= coverage <= len(people):
        raise ValueError("coverage-per-arm must be between one and the number of people")
    rng = random.Random(seed)
    people = list(people)
    arms = list(arms)
    rng.shuffle(people)
    rng.shuffle(arms)
    # Consecutive round-robin slots make the global person load differ by at
    # most one; an arm's replicate slots are also distinct because coverage is
    # no larger than the person count.
    pairs = set()
    for arm_index, arm in enumerate(arms):
        for replicate in range(coverage):
            slot = arm_index * coverage + replicate
            person = people[slot % len(people)]
            pairs.add((person, arm))
    if len(pairs) != len(arms) * coverage:
        raise RuntimeError("balanced design produced duplicate person-arm cells")
    return pairs


def build_balanced(args: argparse.Namespace) -> None:
    people = []
    arms = []
    for row in read_jsonl(args.base):
        people.append(row["victim_person_id"])
        arms.append(row["arm_id"])
    people = sorted(set(people))
    arms = sorted(set(arms))
    if len(people) != 501 or len(arms) != 360:
        raise RuntimeError(f"unexpected base dimensions: people={len(people)} arms={len(arms)}")
    pairs = balanced_pairs(people, arms, args.coverage_per_arm, args.seed)
    person_load = Counter(person for person, _ in pairs)
    arm_load = Counter(arm for _, arm in pairs)

    def selected() -> Iterable[dict[str, Any]]:
        for row in read_jsonl(args.base):
            key = (row["victim_person_id"], row["arm_id"])
            if key not in pairs:
                continue
            trial_id = sha256_text(f"target-balanced-v1|{key[0]}|{key[1]}")[:24]
            yield {
                **row,
                "trial_id": trial_id,
                "balanced_design": "BIBD-like cyclic exact-arm-coverage/v1",
                "balanced_coverage_per_arm": args.coverage_per_arm,
                "repeat_index": 0,
                "sampling_seed_group": args.seed,
            }

    args.output.mkdir(parents=True, exist_ok=True)
    count = atomic_jsonl(args.output / "requests.jsonl", selected())
    expected = 360 * args.coverage_per_arm
    if count != expected:
        raise RuntimeError(f"balanced plan mismatch: {count} != {expected}")
    manifest = {
        "schema": "jailnews_target_balanced_seed/v1",
        "people": len(people),
        "arms": len(arms),
        "coverage_per_arm": args.coverage_per_arm,
        "requests": count,
        "person_load_min": min(person_load.values()),
        "person_load_max": max(person_load.values()),
        "people_with_requests": len(person_load),
        "arm_load_min": min(arm_load.values()),
        "arm_load_max": max(arm_load.values()),
        "temperature": args.temperature,
        "seed": args.seed,
        "request_file": str((args.output / "requests.jsonl").resolve()),
        "note": "Independent balanced panel; does not reuse target labels for selection.",
    }
    write_json(args.output / "plan_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    base = commands.add_parser("base")
    base.add_argument("--matrix", type=Path, required=True)
    base.add_argument("--representatives", type=Path, required=True)
    base.add_argument("--prior", type=Path, required=True)
    base.add_argument("--output", type=Path, required=True)
    base.set_defaults(function=build_base)
    balanced = commands.add_parser("balanced")
    balanced.add_argument("--base", type=Path, required=True)
    balanced.add_argument("--output", type=Path, required=True)
    balanced.add_argument("--coverage-per-arm", type=int, default=10)
    balanced.add_argument("--temperature", type=float, default=1.0)
    balanced.add_argument("--seed", type=int, default=20261004)
    balanced.set_defaults(function=build_balanced)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    args.function(args)
