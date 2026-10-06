#!/usr/bin/env python3
"""Merge NLLB concept shards into the 72-language logical table."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue", type=Path,
        default=Path("data/jailnewsbench_person_domain_20260930/concept_translations_v1/translation_queue.jsonl"),
    )
    parser.add_argument(
        "--shards", type=Path,
        default=Path("data/jailnewsbench_person_domain_20260930/concept_translations_v1/nllb_shards"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/jailnewsbench_person_domain_20260930/concept_translations_v1/translations_nllb.jsonl"),
    )
    args = parser.parse_args()

    queue = read_jsonl(args.queue)
    language_codes = []
    seen = set()
    for row in queue:
        pair = (row["language"], row["nllb_code"])
        if pair not in seen:
            seen.add(pair)
            language_codes.append(pair)
    expected_by_language = Counter(row["language"] for row in queue)
    shard_cache: dict[str, dict[str, dict[str, Any]]] = {}
    output_counts: Counter[str] = Counter()
    valid_counts: Counter[str] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for language, code in language_codes:
            if code not in shard_cache:
                path = args.shards / f"{code}.jsonl"
                rows = read_jsonl(path)
                shard_cache[code] = {row["concept_id"]: row for row in rows}
            for queued in (row for row in queue if row["language"] == language):
                source = shard_cache[code][queued["concept_id"]]
                row = {
                    **source,
                    "translation_id": queued["translation_id"],
                    "language": language,
                    "nllb_code": code,
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                output_counts[language] += 1
                valid_counts[language] += int(row["valid"])
    if output_counts != expected_by_language:
        raise RuntimeError(f"logical row mismatch: {output_counts - expected_by_language}")
    summary = {
        "schema": "jailnews_concept_translations/v1",
        "languages": len(output_counts),
        "physical_nllb_codes": len(shard_cache),
        "rows": sum(output_counts.values()),
        "valid_rows": sum(valid_counts.values()),
        "valid_rate": sum(valid_counts.values()) / sum(output_counts.values()),
        "shared_code_expansion": {"srp_Cyrl": ["Montenegrin", "Serbian"]},
        "by_language": {
            language: {
                "rows": output_counts[language],
                "valid": valid_counts[language],
                "valid_rate": valid_counts[language] / output_counts[language],
            }
            for language in sorted(output_counts)
        },
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "by_language"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
