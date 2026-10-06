#!/usr/bin/env python3
"""Build a deduplicated, sample-linked concept translation queue.

This extracts metadata concepts only; it never copies the generation request.
Names and time expressions are deliberately excluded because they need entity
preservation/localization rules rather than ordinary concept translation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FIELD_SPECS = {
    "political_domain": "luna_political_domain",
    "political_subdomain": "luna_political_subdomains",
    "event_type": "luna_event_type",
    "sensitive_concept": "luna_sensitive_concepts",
    "named_conflict": "luna_conflicts_named",
    "geography": "geographies_detected",
    "person_role": "luna_primary_person_role",
}


def values(row: dict[str, Any], field: str) -> list[str]:
    value = row.get(field)
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


def display_text(value: str, concept_type: str) -> str:
    if concept_type in {"political_domain", "event_type", "sensitive_concept", "person_role"}:
        return value.replace("_", " ")
    return " ".join(value.split())


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("–", "-").replace("—", "-")
    value = re.sub(r"[^\w\s-]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def concept_id(text: str) -> str:
    return "concept:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples", type=Path,
        default=Path("data/jailnewsbench_person_domain_20260930/luna_review_v1/analysis_ready_samples.jsonl"),
    )
    parser.add_argument(
        "--language-scores", type=Path,
        default=Path("data/jailnewsbench_pc2_contextual_20260929/language_scores/language_scores.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/jailnewsbench_person_domain_20260930/concept_translations_v1"),
    )
    args = parser.parse_args()

    samples = read_jsonl(args.samples)
    language_seed = read_jsonl(args.language_scores)[0]["candidates"]
    languages = [
        {"language": row["language"], "nllb_code": row["nllb_code"]}
        for row in language_seed
    ]

    inventory: dict[str, dict[str, Any]] = {}
    mentions: Counter[str] = Counter()
    sample_links = []
    type_counts: Counter[str] = Counter()
    source_forms: dict[str, Counter[str]] = defaultdict(Counter)
    samples_by_concept: dict[str, set[str]] = defaultdict(set)

    for row in samples:
        sample_id = row["source_record_id"]
        linked = []
        for concept_type, field in FIELD_SPECS.items():
            for source_value in values(row, field):
                text = display_text(source_value, concept_type)
                key = normalize(text)
                if not key or key in {"unclear", "other or unclear", "unknown or not explicit"}:
                    continue
                cid = concept_id(key)
                if cid not in inventory:
                    inventory[cid] = {
                        "concept_id": cid,
                        "canonical_english": text,
                        "normalized_key": key,
                        "concept_types": set(),
                    }
                inventory[cid]["concept_types"].add(concept_type)
                source_forms[cid][source_value] += 1
                mentions[cid] += 1
                type_counts[concept_type] += 1
                samples_by_concept[cid].add(sample_id)
                linked.append({"concept_id": cid, "concept_type": concept_type})
        deduped = list({(item["concept_id"], item["concept_type"]): item for item in linked}.values())
        sample_links.append({
            "source_record_id": sample_id,
            "selector_validation_person": row.get("selector_validation_person"),
            "primary_person": row.get("luna_primary_person"),
            "concepts": sorted(deduped, key=lambda item: (item["concept_type"], item["concept_id"])),
        })

    records = []
    for cid, item in inventory.items():
        records.append({
            "concept_id": cid,
            "canonical_english": item["canonical_english"],
            "normalized_key": item["normalized_key"],
            "concept_types": sorted(item["concept_types"]),
            "mention_count": mentions[cid],
            "sample_count": len(samples_by_concept[cid]),
            "source_forms": [
                {"text": text, "count": count}
                for text, count in source_forms[cid].most_common()
            ],
        })
    records.sort(key=lambda row: (-row["mention_count"], row["canonical_english"].casefold()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "concept_inventory.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "sample_concept_links.jsonl").open("w", encoding="utf-8") as handle:
        for row in sample_links:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "translation_queue.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            for language in languages:
                handle.write(json.dumps({
                    "translation_id": f"{row['concept_id']}::{language['language']}",
                    "concept_id": row["concept_id"],
                    "canonical_english": row["canonical_english"],
                    "concept_types": row["concept_types"],
                    "language": language["language"],
                    "nllb_code": language["nllb_code"],
                    "status": "pending",
                }, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema": "jailnews_concept_translation_inventory/v1",
        "source_samples": len(samples),
        "unique_concepts": len(records),
        "languages": len(languages),
        "translation_rows": len(records) * len(languages),
        "type_mentions": dict(sorted(type_counts.items())),
        "excluded": ["person names", "time expressions", "full generation instructions"],
        "translation_backend_planned": "facebook/nllb-200-distilled-1.3B",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
