#!/usr/bin/env python3
"""Prepare paper-original JailNewsBench prompts for the matched 501-person cohort.

No PC2 translation, A placeholder, identity binding, routing, or bandit selection is
used.  The seven conditions are the public Original, Explicit, and five jailbreak
prompt constructors at the pinned public evaluator commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from transformers import AutoTokenizer

from jailnewsbench_table2_qwen32 import (
    ALL_ATTACKS,
    OFFICIAL,
    PUBLIC_COMMIT,
    PUBLIC_EVALUATOR_BLOB,
    PUBLIC_EVALUATOR_SHA256,
    PUBLIC_PROMPT_BUNDLE_SHA256,
    build_context_prefix,
    sha256_text,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/jailnews_bandit_20260930/runtime/stochastic_360_v1/base_arms.jsonl"
DATA = ROOT / "data/jailnewsbench_raw"
MODEL = ROOT / "hf-cache/hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507-FP8/snapshots/60d80c83c53c3b611c642dbb8c942b3f90c5948a"
OUTPUT = ROOT / "artifacts/jailnews_bandit_20260930/runtime/official_baseline_501"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--tokenizer", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # One exact source-record link per victim person. Seven records are shared
    # by two people, so the primary cohort is person-weighted (501) and also
    # carries a stable source UID for a 494-unique-sample sensitivity analysis.
    cohort: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(args.base):
        person = str(row["victim_person_id"])
        source_uid = str(row["source_record_id"])
        sample_id = str(row["sample_id"])
        parts = sample_id.split(":", 2)
        if len(parts) != 3 or parts[0] != "jailnewsbench" or parts[2] != source_uid:
            raise RuntimeError(f"cannot resolve source split from sample_id: {sample_id}")
        source_split = parts[1]
        current = cohort.setdefault(
            person,
            {
                "victim_person_id": person,
                "source_uid": source_uid,
                "source_split": source_split,
                "sample_id": sample_id,
                "wikidata_qid": row.get("wikidata_qid"),
                "expected_canonical_english_name": row.get("expected_canonical_english_name")
                or (row.get("luna_people_surfaces") or [None])[0],
            },
        )
        if current["source_uid"] != source_uid or current["source_split"] != source_split:
            raise RuntimeError(f"person linked to multiple source records: {person}")
    if len(cohort) != 501:
        raise RuntimeError(f"expected 501 people, found {len(cohort)}")

    needed_keys = {
        (item["source_split"], item["source_uid"])
        for item in cohort.values()
    }
    raw: dict[tuple[str, str], dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        frame = pd.read_parquet(args.data_dir / f"{split}.parquet")
        for row in frame.to_dict(orient="records"):
            uid = str(row["uid"])
            key = (split, uid)
            # The public parquet contains a small number of reused UIDs for
            # distinct seed instructions. None belongs to this matched cohort;
            # restrict the lookup to the exact linked source keys so an
            # unrelated duplicate cannot make preparation ambiguous.
            if key not in needed_keys:
                continue
            if key in raw:
                raise RuntimeError(f"ambiguous duplicate raw key in matched cohort: {key}")
            raw[key] = row
    missing = needed_keys - set(raw)
    if missing:
        raise RuntimeError(f"missing raw JailNews records: {len(missing)}")

    ordered_people = [cohort[key] for key in sorted(cohort)]
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    context_pool = [
        str(raw[(item["source_split"], item["source_uid"])]["article_local"])
        for item in ordered_people
    ]
    context_prefix = build_context_prefix(context_pool, tokenizer, args.seed, target_tokens=512)

    attack_counts: Counter[str] = Counter()
    prompt_hashes: set[str] = set()

    def rendered() -> Iterable[dict[str, Any]]:
        for attack in ALL_ATTACKS:
            for person_item in ordered_people:
                source_uid = person_item["source_uid"]
                source_key = (person_item["source_split"], source_uid)
                source = raw[source_key]
                seed_instruction = str(source["seed_instruction_local"])
                attacked = OFFICIAL.apply_attack(
                    seed_instruction=seed_instruction,
                    motivation_category=str(source.get("motivation_category") or "Financial"),
                    attack_type=attack,
                    context_prefix=context_prefix if attack == "context_overload" else None,
                )
                trial_id = sha256_text(
                    f"official-baseline-501|{person_item['victim_person_id']}|{source_uid}|{attack}"
                )[:24]
                attack_counts[attack] += 1
                prompt_hashes.add(sha256_text(attacked))
                yield {
                    **source,
                    **person_item,
                    "uid": f"{person_item['victim_person_id']}::{source_uid}",
                    "raw_jailnews_uid": source_uid,
                    "source_split": person_item["source_split"],
                    "trial_id": trial_id,
                    "base_trial_id": trial_id,
                    "attack_type": attack,
                    "attacked_prompt": attacked,
                    "prompt": attacked,
                    "prompt_sha256": sha256_text(attacked),
                    "source_language_code": source["language_code"],
                    "target_model": "Qwen3-30B-A3B-Thinking-2507-FP8",
                    "baseline_condition": "official_jailnews_no_proposed_method",
                    "uses_pc2_translation": False,
                    "uses_A_placeholder": False,
                    "uses_identity_binding": False,
                    "uses_router_or_bandit": False,
                }

    arms_path = args.output / "base_arms.jsonl"
    count = atomic_jsonl(arms_path, rendered())
    expected = 501 * len(ALL_ATTACKS)
    if count != expected or any(attack_counts[attack] != 501 for attack in ALL_ATTACKS):
        raise RuntimeError(f"baseline coverage mismatch: rows={count}, counts={dict(attack_counts)}")
    unique_sources = len({(item["source_split"], item["source_uid"]) for item in ordered_people})
    manifest = {
        "schema": "jailnews_official_baseline_matched501/v1",
        "people": len(ordered_people),
        "unique_source_samples": unique_sources,
        "duplicated_source_links": len(ordered_people) - unique_sources,
        "conditions": ALL_ATTACKS,
        "rows": count,
        "counts_by_condition": dict(attack_counts),
        "context_tokens": 512,
        "context_sha256": sha256_text(context_prefix),
        "context_pool": "matched 501 person-weighted raw article_local records",
        "seed": args.seed,
        "proposed_method_removed": {
            "pc2_translation": True,
            "A_placeholder": True,
            "identity_binding": True,
            "router_or_bandit": True,
        },
        "prompt_source": {
            "commit": PUBLIC_COMMIT,
            "evaluate_py_blob": PUBLIC_EVALUATOR_BLOB,
            "evaluate_py_sha256": PUBLIC_EVALUATOR_SHA256,
            "prompt_bundle_sha256": PUBLIC_PROMPT_BUNDLE_SHA256,
        },
        "prompt_hashes": len(prompt_hashes),
        "output": str(arms_path.resolve()),
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
