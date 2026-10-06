#!/usr/bin/env python3
"""Merge NLLB-3.3B shards and apply only prior Luna-verified concept repairs."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from jailnews_pc2_languages import LANGUAGES


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/jailnewsbench_pc2_quiz_v2_20260930"
CATALOG = ROOT / "data/jailnewsbench_person_domain_20260930/concept_translations_v1/translations_verified.jsonl"
LOCALIZATIONS = ROOT / "artifacts/jailnews_bandit_20260930/runtime/person_localizations_resolved.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
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


def atomic_json(path: Path, value: Any) -> None:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, nargs="*", default=[])
    parser.add_argument("--identity-shards", type=Path, nargs="*", default=[])
    parser.add_argument("--identity-fallback-shards", type=Path, nargs="*", default=[])
    parser.add_argument("--general-fallback-shards", type=Path, nargs="*", default=[])
    parser.add_argument("--delta-shards", type=Path, nargs="*", default=[])
    parser.add_argument("--inventory", type=Path, default=BASE / "translation_inventory.jsonl")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--localizations", type=Path, default=LOCALIZATIONS)
    parser.add_argument(
        "--semantic-repairs", type=Path, nargs="*", default=None,
    )
    parser.add_argument("--output", type=Path, default=BASE / "translations_merged.jsonl")
    args = parser.parse_args()
    shards = args.shards or sorted((BASE / "translation_shards").glob("shard_*.jsonl"))
    if len(shards) != 8:
        raise RuntimeError(f"expected 8 shards, found {len(shards)}")
    rows = [row for path in shards for row in read_jsonl(path)]
    identity_shards = args.identity_shards or sorted(
        (BASE / "identity_translation_shards").glob("shard_*.jsonl")
    )
    if identity_shards:
        if len(identity_shards) != 8:
            raise RuntimeError(f"expected 8 identity override shards, found {len(identity_shards)}")
        overrides = {
            (row["item_id"], row["language"]): row
            for path in identity_shards for row in read_jsonl(path)
        }
        if len(overrides) != 100 * 72:
            raise RuntimeError(f"identity override grid has {len(overrides)} rows, expected 7200")
        rows = [overrides.get((row["item_id"], row["language"]), row) for row in rows]
    fallback_shards = args.identity_fallback_shards or sorted(
        (BASE / "identity_translation_fallback_nllb13").glob("shard_*.jsonl")
    )
    fallback_selected = 0
    if fallback_shards:
        if len(fallback_shards) != 8:
            raise RuntimeError(f"expected 8 identity fallback shards, found {len(fallback_shards)}")
        fallbacks = {
            (row["item_id"], row["language"]): row
            for path in fallback_shards for row in read_jsonl(path)
        }
        if len(fallbacks) != 100 * 72:
            raise RuntimeError(f"identity fallback grid has {len(fallbacks)} rows, expected 7200")
        selected = []
        for row in rows:
            fallback = fallbacks.get((row["item_id"], row["language"]))
            if fallback and (
                (fallback["valid"] and not row["valid"])
                or (
                    fallback["valid"] == row["valid"]
                    and fallback["backtranslation_similarity"] > row["backtranslation_similarity"] + 0.02
                )
            ):
                fallback["translation_backend"] = (
                    fallback["translation_backend"] + "+selected_over_nllb_3.3b"
                )
                selected.append(fallback)
                fallback_selected += 1
            else:
                selected.append(row)
        rows = selected
    delta_shards = args.delta_shards or sorted(
        (BASE / "control_concept_delta_shards").glob("shard_*.jsonl")
    ) + sorted((BASE / "current_concept_delta_shards").glob("shard_*.jsonl"))
    if delta_shards:
        deltas = {
            (row["item_id"], row["language"]): row
            for path in delta_shards for row in read_jsonl(path)
        }
        rows_by_key = {(row["item_id"], row["language"]): row for row in rows}
        rows_by_key.update(deltas)
        rows = list(rows_by_key.values())
    inventory = {row["item_id"]: row for row in read_jsonl(args.inventory)}
    rows = [
        row for row in rows
        if row["item_id"] in inventory
        and row["canonical_english"] == inventory[row["item_id"]]["canonical_english"]
    ]
    expected_keys = {(item_id, language) for item_id in inventory for language in LANGUAGES}
    actual_keys = {(row["item_id"], row["language"]) for row in rows}
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"current-inventory grid incomplete: actual={len(actual_keys)} expected={len(expected_keys)} "
            f"missing={len(expected_keys - actual_keys)} extra={len(actual_keys - expected_keys)}"
        )
    keys = [(row["item_id"], row["language"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate item/language translation")
    general_fallback_shards = args.general_fallback_shards or sorted(
        (BASE / "all_translation_fallback_nllb13").glob("shard_*.jsonl")
    )
    general_fallback_selected = 0
    if general_fallback_shards:
        if len(general_fallback_shards) != 8:
            raise RuntimeError(
                f"expected 8 general fallback shards, found {len(general_fallback_shards)}"
            )
        general_fallbacks = {
            (row["item_id"], row["language"]): row
            for path in general_fallback_shards for row in read_jsonl(path)
        }
        if set(general_fallbacks) != expected_keys:
            raise RuntimeError(
                f"general fallback grid has {len(general_fallbacks)} rows, "
                f"expected {len(expected_keys)}"
            )
        selected = []
        for row in rows:
            fallback = general_fallbacks[(row["item_id"], row["language"])]
            if (
                (fallback["valid"] and not row["valid"])
                or (
                    fallback["valid"] == row["valid"]
                    and fallback["backtranslation_similarity"]
                    > row["backtranslation_similarity"] + 0.02
                )
            ):
                fallback["translation_backend"] += "+selected_over_current_candidate"
                selected.append(fallback)
                general_fallback_selected += 1
            else:
                selected.append(row)
        rows = selected
    verified = {}
    for row in read_jsonl(args.catalog):
        if row.get("final_status") == "verified_luna" and row.get("usable_for_rendering"):
            verified[(row["canonical_english"].casefold(), row["language"])] = row
    localized_names = {
        (row["canonical_name"].casefold(), row["target_language"]): row
        for row in read_jsonl(args.localizations)
        if row.get("status") == "verified_localized" and row.get("localized_name")
    }
    repaired = 0
    wikipedia_name_repairs = 0
    for row in rows:
        if not row["valid"] and row["item_type"] == "person_name":
            localization = localized_names.get((row["canonical_english"].casefold(), row["language"]))
            if localization:
                row["machine_translation_original"] = row["translation"]
                row["machine_backtranslation_original"] = row["backtranslation"]
                row["machine_similarity_original"] = row["backtranslation_similarity"]
                row["translation"] = localization["localized_name"]
                row["backtranslation"] = row["canonical_english"]
                row["backtranslation_similarity"] = 1.0
                row["numbers_preserved"] = True
                row["valid"] = True
                row["translation_backend"] = "wikipedia_verified_localized_name"
                row["final_translation_status"] = "wikipedia_verified_localized_name"
                wikipedia_name_repairs += 1
                continue
        if row["valid"] or row["item_type"] != "concept":
            row["final_translation_status"] = "nllb_roundtrip_valid" if row["valid"] else "needs_review"
            continue
        correction = verified.get((row["canonical_english"].casefold(), row["language"]))
        if not correction:
            row["final_translation_status"] = "needs_review"
            continue
        row["nllb_3_3_translation_original"] = row["translation"]
        row["nllb_3_3_backtranslation_original"] = row["backtranslation"]
        row["nllb_3_3_similarity_original"] = row["backtranslation_similarity"]
        row["translation"] = correction["translation"]
        row["backtranslation"] = correction["backtranslation"]
        row["backtranslation_similarity"] = correction["backtranslation_similarity"]
        row["valid"] = True
        row["translation_backend"] = correction["translation_backend"] + "+luna_verified_catalog"
        row["final_translation_status"] = "luna_verified_catalog_repair"
        repaired += 1
    semantic_repairs = 0
    semantic_paths = args.semantic_repairs or [
        BASE / "semantic_rereview_repairs.jsonl",
        BASE / "specialized_semantic_repairs.jsonl",
        BASE / "madlad10b_semantic_repairs.jsonl",
        BASE / "gemma27b_semantic_repairs.jsonl",
        BASE / "qwen32_gemma_semantic_repairs.jsonl",
        BASE / "qwen32_feedback_gemma_semantic_repairs.jsonl",
    ]
    for semantic_path in semantic_paths:
        if not semantic_path.exists():
            continue
        repair_rows = read_jsonl(semantic_path)
        repairs_by_key = {
            (row["item_id"], row["language"]): row for row in repair_rows
        }
        if len(repairs_by_key) != len(repair_rows):
            raise RuntimeError("duplicate semantic repair keys")
        merged = []
        for row in rows:
            repair = repairs_by_key.get((row["item_id"], row["language"]))
            if repair:
                if (
                    repair["canonical_english"] != row["canonical_english"]
                    or not repair["valid"]
                ):
                    raise RuntimeError("invalid semantic repair provenance")
                merged.append(repair)
                semantic_repairs += 1
            else:
                merged.append(row)
        applied = sum(
            (row["item_id"], row["language"]) in repairs_by_key for row in rows
        )
        if applied != len(repair_rows):
            raise RuntimeError(f"orphan semantic repairs in {semantic_path}")
        rows = merged
    rows.sort(key=lambda row: (row["item_id"], row["language"]))
    count = atomic_jsonl(args.output, rows)
    summary = {
        "schema": "jailnews_pc2_translations_merged/v2",
        "rows": count,
        "valid": sum(row["valid"] for row in rows),
        "needs_review": sum(not row["valid"] for row in rows),
        "luna_catalog_repairs": repaired,
        "wikipedia_localized_name_repairs": wikipedia_name_repairs,
        "qwen32_semantic_rereview_repairs": semantic_repairs,
        "nllb_1.3b_identity_fallbacks_selected": fallback_selected,
        "nllb_1.3b_general_fallbacks_selected": general_fallback_selected,
        "delta_files_applied": len(delta_shards),
        "by_type_valid": {
            item_type: {
                "valid": sum(row["valid"] for row in rows if row["item_type"] == item_type),
                "total": sum(row["item_type"] == item_type for row in rows),
            }
            for item_type in sorted({row["item_type"] for row in rows})
        },
        "statuses": dict(Counter(row["final_translation_status"] for row in rows)),
    }
    atomic_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
