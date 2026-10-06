#!/usr/bin/env python3
"""Rebuild the frozen 501x360 arm pool with explicit A identity resolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from jailnewsbench_table2_qwen32 import OFFICIAL, sha256_text


ROOT = Path(__file__).resolve().parents[1]
OLD_BASE = ROOT / "artifacts/jailnews_bandit_20260930/runtime/stochastic_360_v1/base_arms.jsonl"
MATRIX = ROOT / "data/jailnewsbench_person_domain_20260930/a_middle_resolution_v3/prompt_matrix.jsonl"
OUTPUT = ROOT / "artifacts/jailnews_bandit_20260930/runtime/stochastic_360_resolution_v3"


def read(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    count = 0
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return count


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=OLD_BASE)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    matrix_ids = set()
    people = set()
    context_prefixes = set()
    base_rows = 0
    for row in read(args.base):
        base_rows += 1
        matrix_ids.add(str(row["matrix_id"]))
        people.add(str(row["victim_person_id"]))
        if row["attack_type"] == "context_overload":
            attacked = str(row["attacked_prompt"])
            prompt = str(row["prompt"])
            if not attacked.endswith(prompt):
                raise RuntimeError(f"cannot recover context prefix: {row['trial_id']}")
            context_prefixes.add(attacked[: -len(prompt)])
    if base_rows != 501 * 360 or len(people) != 501 or len(matrix_ids) % 72:
        raise RuntimeError(
            f"unexpected base dimensions: rows={base_rows} people={len(people)} "
            f"matrix_ids={len(matrix_ids)}"
        )
    if len(context_prefixes) != 1:
        raise RuntimeError(f"expected one context prefix, found {len(context_prefixes)}")
    context_prefix = context_prefixes.pop()

    rendered = {}
    for row in read(args.matrix):
        matrix_id = str(row["matrix_id"])
        if matrix_id in matrix_ids:
            rendered[matrix_id] = row
    if set(rendered) != matrix_ids:
        raise RuntimeError(f"missing corrected matrix rows: {len(matrix_ids - set(rendered))}")

    stats = Counter()
    seen_trials = set()

    def upgraded() -> Iterable[dict[str, Any]]:
        for old in read(args.base):
            replacement = rendered[str(old["matrix_id"])]
            method = str(old["attack_type"])
            seed_prompt = str(replacement["prompt"])
            attacked = OFFICIAL.apply_attack(
                seed_instruction=seed_prompt,
                motivation_category=str(old.get("motivation_category") or "Political"),
                attack_type=method,
                context_prefix=context_prefix if method == "context_overload" else None,
            )
            trial_id = sha256_text(
                "stochastic360-resolution-v3|"
                f"{old['victim_person_id']}|{old['sample_id']}|{old['arm_id']}"
            )[:24]
            if trial_id in seen_trials:
                raise RuntimeError(f"duplicate upgraded trial: {trial_id}")
            seen_trials.add(trial_id)
            if "IDENTITY RESOLUTION\n" not in seed_prompt or not attacked.endswith(seed_prompt):
                raise RuntimeError(f"resolution prompt lost: {trial_id}")
            stats[method] += 1
            stats[f"language:{replacement['language']}"] += 1
            yield {
                **old,
                **replacement,
                "uid": old["sample_id"],
                "trial_id": trial_id,
                "base_trial_id": trial_id,
                "arm_id": old["arm_id"],
                "victim_person_id": old["victim_person_id"],
                "victim_dataset_label": old["victim_dataset_label"],
                "entity_id": old.get("entity_id"),
                "wikidata_qid": old.get("wikidata_qid"),
                "sample_id": old["sample_id"],
                "attack_type": method,
                "attacked_prompt": attacked,
                "prompt_sha256": sha256_text(attacked),
                "seed_prompt_sha256": sha256_text(seed_prompt),
                "language_code": replacement["nllb_code"],
                "article_local": replacement["article"],
                "article_en": replacement["article"],
                "region_en": old.get("region_en"),
                "political_domain": old.get("political_domain"),
                "person_role": old.get("person_role"),
                "motivation_category": old.get("motivation_category"),
                "war_or_security_related": old.get("war_or_security_related"),
                "election_related": old.get("election_related"),
                "translation_qc": old.get("translation_qc"),
                "surrogate_prior_endpoint": old.get("surrogate_prior_endpoint"),
                "surrogate_prior_mean": old.get("surrogate_prior_mean"),
                "surrogate_prior_strength": old.get("surrogate_prior_strength"),
                "surrogate_between_model_variance": old.get("surrogate_between_model_variance"),
            }

    count = atomic_jsonl(args.output / "base_arms.jsonl", upgraded())
    if count != 501 * 360 or len(seen_trials) != count:
        raise RuntimeError(f"upgraded output mismatch: rows={count} trials={len(seen_trials)}")
    manifest = {
        "schema": "jailnews_stochastic_360_base/v3-explicit-resolution",
        "source_base": str(args.base.resolve()),
        "source_base_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
        "corrected_matrix": str(args.matrix.resolve()),
        "corrected_matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
        "people": 501,
        "unique_samples": len(matrix_ids) // 72,
        "languages": 72,
        "methods": 5,
        "arms_per_person": 360,
        "rows": count,
        "identity_resolution_explicit": True,
        "expected_output_language_source": "original JailNews source_language_code",
        "method_counts": {key: stats[key] for key in sorted(stats) if not key.startswith("language:")},
        "language_count_min_max": [
            min(value for key, value in stats.items() if key.startswith("language:")),
            max(value for key, value in stats.items() if key.startswith("language:")),
        ],
        "output": str((args.output / "base_arms.jsonl").resolve()),
    }
    atomic_json(args.output / "base_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
