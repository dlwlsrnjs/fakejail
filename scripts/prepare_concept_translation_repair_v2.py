#!/usr/bin/env python3
"""Prepare every unresolved concept translation for local MT + backtranslation QA."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    selected = []
    languages = Counter()
    for row in rows(args.input):
        if row.get("final_status") != "needs_human_review":
            continue
        item = dict(row)
        item["item_id"] = row["concept_id"]
        item["item_type"] = "concept"
        item["repair_route"] = "madlad_400_3b"
        item["original_roundtrip_valid"] = bool(row.get("valid"))
        selected.append(item)
        languages[row["language"]] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(args.output)
    summary = {
        "schema": "concept_translation_repair_inventory/v2", "rows": len(selected),
        "unique_translation_ids": len({row["translation_id"] for row in selected}),
        "empty_input_translations": sum(not str(row.get("translation", "")).strip() for row in selected),
        "languages": dict(sorted(languages.items())),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
