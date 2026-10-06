#!/usr/bin/env python3
"""Rebind selected implicit-A arms to the explicit-resolution v3 prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED = (
    ROOT
    / "artifacts/jailnews_bandit_20260930/runtime/stochastic_360_v1"
    / "gpt4omini_surrogate32k_top1/requests.jsonl"
)
DEFAULT_ARMS = (
    ROOT
    / "artifacts/jailnews_bandit_20260930/runtime/stochastic_360_resolution_v3"
    / "base_arms.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/jailnews_bandit_20260930/runtime/stochastic_360_resolution_v3"
    / "gpt4omini_implicit_prior_top1"
)


def read(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--arms", type=Path, default=DEFAULT_ARMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    arm_index: dict[tuple[str, str], dict[str, Any]] = {}
    for arm in read(args.arms):
        key = (str(arm["victim_person_id"]), str(arm["arm_id"]))
        if key in arm_index:
            raise RuntimeError(f"duplicate v3 person-arm: {key}")
        arm_index[key] = arm
    if len(arm_index) != 501 * 360:
        raise RuntimeError(f"expected 180360 v3 person-arms, found {len(arm_index)}")

    selected = list(read(args.selected))
    if len(selected) != 501 or len({row["victim_person_id"] for row in selected}) != 501:
        raise RuntimeError("expected exactly one selected arm for each of 501 people")

    rows: list[dict[str, Any]] = []
    missing: list[tuple[str, str]] = []
    for old in selected:
        key = (str(old["victim_person_id"]), str(old["arm_id"]))
        v3 = arm_index.get(key)
        if v3 is None:
            missing.append(key)
            continue
        prompt = str(v3["attacked_prompt"])
        if "IDENTITY RESOLUTION\n" not in prompt or not bool(v3.get("identity_resolution_explicit")):
            raise RuntimeError(f"explicit-resolution contract missing for {key}")
        trial_id = sha256_text(
            f"gpt4omini-resolution-v3-implicit-prior|{key[0]}|{v3['sample_id']}|{key[1]}"
        )[:24]
        row = dict(v3)
        for field, value in old.items():
            if field.startswith("surrogate_") or field == "target_evaluation_split":
                row[field] = value
        row.update(
            {
                "trial_id": trial_id,
                "base_trial_id": trial_id,
                "source_implicit_request_trial_id": old["trial_id"],
                "source_implicit_base_trial_id": old.get("base_trial_id"),
                "surrogate_prompt_condition": old.get("condition"),
                "surrogate_prior_prompt_contract": "implicit-a-v2",
                "target_prompt_condition": v3["condition"],
                "target_prompt_contract": v3.get("prompt_contract"),
                "target_evaluation_split": "explicit-resolution-v3-implicit-prior-transfer",
                "prompt_sha256": sha256_text(prompt),
            }
        )
        rows.append(row)

    if missing or len(rows) != 501 or len({row["trial_id"] for row in rows}) != 501:
        raise RuntimeError(f"v3 request coverage failed: rows={len(rows)} missing={len(missing)}")
    if any(
        row["language"] != old["language"]
        or row["attack_type"] != old["attack_type"]
        or row["arm_id"] != old["arm_id"]
        for row, old in zip(rows, selected)
    ):
        raise RuntimeError("language/method selection changed during v3 rebinding")

    output_file = args.output / "requests.jsonl"
    count = atomic_jsonl(output_file, rows)
    manifest = {
        "schema": "jailnews_gpt4omini_501_transfer_plan/v3-explicit-resolution-implicit-prior",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_model": "gpt-4o-mini-2024-07-18",
        "persons": 501,
        "requests": count,
        "identity_resolution_explicit": True,
        "surrogate_prior_prompt_contract": "implicit-a-v2",
        "target_prompt_contract": "explicit-resolution-v3",
        "languages": dict(Counter(row["language"] for row in rows)),
        "methods": dict(Counter(row["attack_type"] for row in rows)),
        "source_selected": str(args.selected.resolve()),
        "source_selected_sha256": hashlib.sha256(args.selected.read_bytes()).hexdigest(),
        "v3_arms": str(args.arms.resolve()),
        "v3_arms_sha256": hashlib.sha256(args.arms.read_bytes()).hexdigest(),
        "request_file": str(output_file.resolve()),
        "request_file_sha256": hashlib.sha256(output_file.read_bytes()).hexdigest(),
    }
    atomic_json(args.output / "selection_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
