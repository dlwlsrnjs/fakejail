#!/usr/bin/env python3
"""Render the repaired 4,091-case A-middle matrix from dual NLLB grids.

The source article and instruction are kept fixed.  Only the neutral public-
figure description placed after ``A:`` varies across the 72 PC2 languages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from jailnews_pc2_languages import LANGUAGES


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/jailnewsbench_person_domain_20260930/a_middle_surface_v2"


def read(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    tmp.replace(path)
    return count


def load_grid(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for row in read(path):
            key = (row["item_id"], row["language"])
            if key in output:
                raise RuntimeError(f"duplicate translation: {key}")
            output[key] = row
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--fallback", type=Path, required=True)
    args = ap.parse_args()

    cases = list(read(args.base / "person_cases.jsonl"))
    inventory = list(read(args.base / "description_translation_inventory.jsonl"))
    primary = load_grid(sorted(args.primary.glob("shard_*.jsonl")))
    fallback = load_grid(sorted(args.fallback.glob("shard_*.jsonl")))
    expected = {(row["item_id"], language) for row in inventory for language in LANGUAGES}
    if set(primary) != expected or set(fallback) != expected:
        raise RuntimeError(
            f"translation grid mismatch: expected={len(expected)} "
            f"primary={len(primary)} fallback={len(fallback)}"
        )

    merged = []
    selected_backends: Counter[str] = Counter()
    for key in sorted(expected):
        candidates = [primary[key], fallback[key]]
        candidates.sort(
            key=lambda row: (
                bool(row.get("valid")),
                float(row.get("backtranslation_similarity") or -1.0),
            ),
            reverse=True,
        )
        winner = dict(candidates[0])
        winner["selected_backend"] = winner.get("translation_backend")
        winner["alternate_candidate"] = {
            field: candidates[1].get(field)
            for field in (
                "translation_backend", "translation", "backtranslation",
                "backtranslation_similarity", "valid",
            )
        }
        selected_backends[str(winner["selected_backend"])] += 1
        merged.append(winner)
    write(args.base / "description_translations_merged.jsonl", merged)

    index = {(row["item_id"], row["language"]): row for row in merged}
    matrix = []
    by_language: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        for language, nllb_code in LANGUAGES.items():
            tr = index[(case["description_item_id"], language)]
            description = str(tr.get("translation") or "").strip()
            prompt = case["prompt_template"].format(
                article=case["article"],
                translated_person_description=description,
                instruction=case["instruction"],
            )
            expected_prompt = (
                f"ARTICLE\n{case['article']}\n\nA: {description}"
                f"\n\nINSTRUCTION\n{case['instruction']}"
            )
            if prompt != expected_prompt:
                raise RuntimeError(f"prompt contract failed: {case['sample_id']}::{language}")
            valid = bool(tr.get("valid") and description)
            by_language[language]["rows"] += 1
            by_language[language]["valid" if valid else "invalid"] += 1
            matrix.append({
                "matrix_id": f"{case['sample_id']}::{language}",
                "sample_id": case["sample_id"],
                "source_record_id": case["source_record_id"],
                # The descriptor/attack language may differ from the language
                # in which the seed instruction asks for the news article.
                # Preserve both so fluency is judged against the requested
                # output language rather than the PC2 descriptor language.
                "source_language_code": case.get("source_language_code"),
                "entity_id": case["entity_id"],
                "wikidata_qid": case.get("wikidata_qid"),
                "primary_dataset_label": case.get("primary_dataset_label"),
                "luna_people_surfaces": case["luna_people_surfaces"],
                "language": language,
                "nllb_code": nllb_code,
                "condition": "pc2_a_middle_surface_v2",
                "article": case["article"],
                "a_description": description,
                "instruction": case["instruction"],
                "prompt": prompt,
                "description_item_id": case["description_item_id"],
                "description_backtranslation": tr.get("backtranslation"),
                "description_backtranslation_similarity": tr.get("backtranslation_similarity"),
                "description_translation_valid": valid,
                "description_translation_backend": tr.get("selected_backend"),
                "eligible_for_model_run": valid,
                "article_sha256": case["article_sha256"],
                "instruction_sha256": case["instruction_sha256"],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            })
    matrix.sort(key=lambda row: (row["sample_id"], list(LANGUAGES).index(row["language"])))
    write(args.base / "prompt_matrix.jsonl", matrix)
    manifest = {
        "schema": "jailnews_a_middle_surface_matrix/v2",
        "cases": len(cases),
        "entities": len({row["entity_id"] for row in cases}),
        "description_items": len(inventory),
        "languages": len(LANGUAGES),
        "rows": len(matrix),
        "eligible_rows": sum(row["eligible_for_model_run"] for row in matrix),
        "ineligible_rows": sum(not row["eligible_for_model_run"] for row in matrix),
        "selected_backends": dict(selected_backends),
        "qc_by_language": {key: dict(value) for key, value in by_language.items()},
        "prompt_order": ["ARTICLE", "A", "INSTRUCTION"],
        "translated_span": "neutral public-figure description only",
        "methods_added": False,
    }
    (args.base / "prompt_matrix.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
