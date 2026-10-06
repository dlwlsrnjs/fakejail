#!/usr/bin/env python3
"""Select a deterministic JailNewsBench-only pilot for POLY/BAI evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


MOTIVATIONS = ("Financial", "Political", "Social", "Psychological")


def read_person_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(str(json.loads(line)["source_record_id"]))
    return ids


def stable(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def select_split(
    parquet: Path,
    person_view: Path,
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows = pq.read_table(parquet).to_pylist()
    person_ids = read_person_ids(person_view)
    by_motivation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_motivation[str(row["motivation_category"])].append(row)

    selected: list[dict[str, Any]] = []
    used_regions: set[str] = set()
    used_languages: set[str] = set()
    # Alternate person and non-person rows. GPT later verifies whether the
    # automatic PERSON candidate is actually a public figure.
    desired_person = {
        "Financial": True,
        "Political": True,
        "Social": False,
        "Psychological": False,
    }
    for motivation in MOTIVATIONS:
        pool = by_motivation[motivation]
        want_person = desired_person[motivation]

        def score(row: dict[str, Any]) -> tuple[int, int, int, int]:
            uid = str(row["uid"])
            is_person = uid in person_ids
            return (
                int(is_person == want_person),
                int(str(row["region_en"]) not in used_regions),
                int(str(row["language_code"]) not in used_languages),
                -stable(f"{seed}:{split}:{uid}"),
            )

        picked = max(pool, key=score)
        uid = str(picked["uid"])
        record = {
            "item_id": f"jnb:{split}:{uid}",
            "source_record_id": uid,
            "source_split": split,
            "region_en": picked["region_en"],
            "language_code": picked["language_code"],
            "motivation_category": motivation,
            "selection_track": (
                "automatic_person_candidate" if uid in person_ids else "concept_candidate"
            ),
            "article_local": picked["article_local"],
            "article_en": picked["article_en"],
            "seed_instruction_local": picked["seed_instruction_local"],
        }
        selected.append(record)
        used_regions.add(str(picked["region_en"]))
        used_languages.add(str(picked["language_code"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root", type=Path,
        default=Path("/home/ljk98/POLY/data/jailnewsbench_raw"),
    )
    parser.add_argument(
        "--person-root", type=Path,
        default=Path("/home/ljk98/POLY/data/jailnewsbench_person_views"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/home/ljk98/POLY/data/jailnewsbench_poly_bai_pilot_20260929"),
    )
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for split in ("train", "test"):
        rows = select_split(
            args.raw_root / f"{split}.parquet",
            args.person_root / split / "person_records_all_motivations.jsonl",
            split,
            args.seed,
        )
        all_rows.extend(rows)
        with (args.output / f"pilot_{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "schema": "jailnews_poly_bai_pilot/v1",
        "dataset": "JailNewsBench only",
        "seed": args.seed,
        "train_items": 4,
        "test_items": 4,
        "motivation_coverage_per_split": list(MOTIVATIONS),
            "conditions": ["direct", "poly", "qwen7b_selective_ipdm_poly"],
        "registered_arms": 160,
        "arm_factorization": {
            "fragments_per_language": [3, 5, 8, 12],
            "order": ["ordered", "shuffled"],
            "number_of_languages": [2, 4, 6, 8],
            "jailnewsbench_appendix_b_attack": [
                "role_play", "system_override", "research_front",
                "neg_prompting", "context_overload"
            ]
        },
        "replaced_poly_frame_axis": [
            "plain", "persona", "fiction", "pap", "persona+fiction"
        ],
        "official_attack_source_commit": "d89690316ad55f7c74baaf63a32b0f915c2e16c1",
        "budgets": [1, 2, 4, 8, 16, 30],
        "source_grid_calls_per_condition": 4 * 160,
        "heldout_oracle_calls_per_condition": 4 * 160,
        "selection_note": (
            "Automatic person labels control sampling only. Qwen2.5-7B annotation must "
            "verify named persons and select either a person or a specific concept."
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
