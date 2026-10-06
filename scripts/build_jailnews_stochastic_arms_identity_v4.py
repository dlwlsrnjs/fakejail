#!/usr/bin/env python3
"""Build separate 501x360 stochastic arm pools for both V4 identity modes."""

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
MATRIX_ROOT = ROOT / "data/jailnewsbench_person_domain_20260930/a_middle_identity_v4"
OUTPUT = ROOT / "artifacts/jailnews_bandit_20260930/runtime/stochastic_360_identity_v4"
PEOPLE = ROOT / "artifacts/jailnews_bandit_20260930/runtime/person_localizations_resolved.jsonl"
MODES = ("a_literal", "canonical_en")


def read(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
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
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=OLD_BASE)
    parser.add_argument("--matrix-root", type=Path, default=MATRIX_ROOT)
    parser.add_argument("--people", type=Path, default=PEOPLE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    canonical_names: dict[str, str] = {}
    for row in read(args.people):
        person_id = str(row["person_id"])
        name = str(row.get("canonical_name") or "").strip()
        if not name:
            continue
        previous = canonical_names.setdefault(person_id, name)
        if previous != name:
            raise RuntimeError(f"conflicting canonical names for {person_id}: {previous!r} vs {name!r}")

    base_count = 0
    people: set[str] = set()
    matrix_ids: set[str] = set()
    context_prefixes: set[str] = set()
    for row in read(args.base):
        base_count += 1
        people.add(str(row["victim_person_id"]))
        matrix_ids.add(str(row["matrix_id"]))
        if row["attack_type"] == "context_overload":
            attacked = str(row["attacked_prompt"])
            seed = str(row["prompt"])
            if not attacked.endswith(seed):
                raise RuntimeError(f"cannot recover context prefix: {row['trial_id']}")
            context_prefixes.add(attacked[: -len(seed)])
    if base_count != 501 * 360:
        raise RuntimeError(f"unexpected V2 base size: {base_count}")
    if len(people) != 501 or len(context_prefixes) != 1:
        raise RuntimeError(
            f"unexpected V2 dimensions: people={len(people)} prefixes={len(context_prefixes)}"
        )
    missing_names = people - set(canonical_names)
    if missing_names:
        raise RuntimeError(f"missing canonical English names: {len(missing_names)}")
    context_prefix = context_prefixes.pop()
    manifests: dict[str, Any] = {}

    for mode in MODES:
        matrix_path = args.matrix_root / f"prompt_matrix_{mode}.jsonl"
        replacements = {
            str(row["matrix_id"]): row
            for row in read(matrix_path)
            if str(row["matrix_id"]) in matrix_ids
        }
        if set(replacements) != matrix_ids:
            raise RuntimeError(f"missing {mode} matrix rows: {len(matrix_ids - set(replacements))}")
        seen: set[str] = set()
        stats: Counter[str] = Counter()

        def upgraded() -> Iterable[dict[str, Any]]:
            for old in read(args.base):
                replacement = replacements[str(old["matrix_id"])]
                method = str(old["attack_type"])
                seed_prompt = str(replacement["prompt"])
                attacked = OFFICIAL.apply_attack(
                    seed_instruction=seed_prompt,
                    motivation_category=str(old.get("motivation_category") or "Political"),
                    attack_type=method,
                    context_prefix=context_prefix if method == "context_overload" else None,
                )
                trial_id = sha256_text(
                    f"stochastic360-identity-v4-{mode}|{old['victim_person_id']}|"
                    f"{old['sample_id']}|{old['arm_id']}"
                )[:24]
                if trial_id in seen:
                    raise RuntimeError(f"duplicate V4 trial: {trial_id}")
                seen.add(trial_id)
                if "IDENTITY BINDING\n" not in seed_prompt or not attacked.endswith(seed_prompt):
                    raise RuntimeError(f"V4 identity contract lost: {trial_id}")
                stats[method] += 1
                stats[f"language:{replacement['language']}"] += 1
                yield {
                    **old,
                    **replacement,
                    "uid": old["sample_id"],
                    "trial_id": trial_id,
                    "base_trial_id": trial_id,
                    "arm_id": old["arm_id"],
                    "full_arm_id": f"{mode}::{old['arm_id']}",
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
                    "identity_rendering_mode": mode,
                    # Evaluation-only reference. It is not interpolated into
                    # either seed_prompt or attacked_prompt.
                    "expected_canonical_english_name": canonical_names[str(old["victim_person_id"])],
                }

        output_file = args.output / mode / "base_arms.jsonl"
        count = atomic_jsonl(output_file, upgraded())
        expected = 501 * 360
        if count != expected or len(seen) != expected:
            raise RuntimeError(f"V4 {mode} size mismatch: rows={count} trials={len(seen)}")
        manifest = {
            "schema": f"jailnews_stochastic_360_base/v4-{mode}",
            "identity_rendering_mode": mode,
            "source_base": str(args.base.resolve()),
            "source_base_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
            "matrix": str(matrix_path.resolve()),
            "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
            "canonical_name_source": str(args.people.resolve()),
            "canonical_name_exposed_in_prompt": False,
            "people": 501,
            "languages": 72,
            "methods": 5,
            "arms_per_person": 360,
            "rows": count,
            "output": str(output_file.resolve()),
            "method_counts": {
                key: stats[key] for key in sorted(stats) if not key.startswith("language:")
            },
        }
        atomic_json(args.output / mode / "base_manifest.json", manifest)
        manifests[mode] = manifest

    combined = {
        "schema": "jailnews_stochastic_identity/v4",
        "modes": manifests,
        "people": 501,
        "languages": 72,
        "methods": 5,
        "identity_modes": 2,
        "arms_per_person_per_mode": 360,
        "combined_arms_per_person": 720,
        "combined_rows": sum(item["rows"] for item in manifests.values()),
    }
    atomic_json(args.output / "manifest.json", combined)
    print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
