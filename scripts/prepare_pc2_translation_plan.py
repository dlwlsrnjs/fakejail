#!/usr/bin/env python3
"""Create deduplicated translation units and a PC2 adaptation cost/scale plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--languages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    language_config = json.loads(args.languages.read_text(encoding="utf-8"))
    languages = language_config["languages"]
    if len(languages) != 72 or len({x["name"] for x in languages}) != 72:
        raise ValueError("PC2 language configuration must contain 72 unique languages")
    non_english = [x for x in languages if x["name"] != "English"]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        payload = (row.get("system_prompt", "") + "\u241f" + row["prompt"]).encode("utf-8")
        unit_hash = hashlib.sha256(payload).hexdigest()
        grouped[unit_hash].append(row)

    units: list[dict[str, Any]] = []
    for unit_hash, group in grouped.items():
        first = group[0]
        tracks = sorted({row["evaluation_track"] for row in group})
        routes = sorted({row["pc2_adaptation_route"] for row in group})
        units.append({
            "translation_unit_id": f"TR_{unit_hash[:20]}",
            "text_sha256": unit_hash,
            "system_prompt": first.get("system_prompt", ""),
            "prompt": first["prompt"],
            "collection_ids": [row["collection_id"] for row in group],
            "source_datasets": sorted({row["source_dataset"] for row in group}),
            "evaluation_tracks": tracks,
            "pc2_adaptation_routes": routes,
            "language_set": "pc2_72_fixed_appendix_order",
            "translation_status": "not_started",
        })
    units.sort(key=lambda row: row["translation_unit_id"])
    harmful = [row for row in units if "harmful_generation" in row["evaluation_tracks"]]

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "translation_units_all.jsonl", units)
    write_jsonl(out / "pc2_harmful_units.jsonl", harmful)

    route_counts = Counter(
        route for row in records for route in [row["pc2_adaptation_route"]]
    )
    plan = {
        "status": "planned_not_translated",
        "source_rows": len(records),
        "unique_translation_units": len(units),
        "harmful_source_rows": sum(row["evaluation_track"] == "harmful_generation" for row in records),
        "unique_harmful_units": len(harmful),
        "languages_total": len(languages),
        "non_english_languages": len(non_english),
        "direct_multilingual_dataset": {
            "stored_language_variants": len(units) * len(languages),
            "minimum_forward_translations": len(units) * len(non_english),
            "minimum_back_translations_for_full_qc": len(units) * len(non_english),
        },
        "pc2_harmful_base_prompts": {
            "stored_language_variants": len(harmful) * len(languages),
            "minimum_forward_translations": len(harmful) * len(non_english),
            "minimum_back_translations": len(harmful) * len(non_english),
            "additional_ipdm_work": "72 forward and 71 back translations per detected sensitive expression, plus detection and description calls",
        },
        "source_row_routes": dict(sorted(route_counts.items())),
        "required_quality_gates": [
            "exact placeholder preservation",
            "language-key completeness",
            "language identification",
            "English back-translation",
            "embedding cosine similarity >= 0.9",
            "entity and requested-action preservation audit",
            "cache keyed by source text, model snapshot, prompt version, and language",
        ],
        "execution_note": (
            "No paid model calls are made by this planner. Full execution should be staged "
            "after a stratified pilot because the minimum candidate volume is large."
        ),
    }
    (out / "translation_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
