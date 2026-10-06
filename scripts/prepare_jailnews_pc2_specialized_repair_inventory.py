#!/usr/bin/env python3
"""Select Qwen-rejected or numeric-failed translations for specialist MT repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from jailnews_pc2_languages import MADLAD_CODES, MILMMT_LANGUAGE_NAMES


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force-route", choices=["milmmt_46_12b", "madlad_400_3b"])
    args = parser.parse_args()
    source = {row["key"]: row for row in read(args.source)}
    judgments = {row["key"]: row for row in read(args.judgments)}
    translations = {row["translation_id"]: row for row in read(args.translations)}
    if set(source) != set(judgments):
        raise RuntimeError("semantic review is incomplete")
    selected = []
    for key, input_row in source.items():
        judgment = judgments[key]
        if judgment["source_sha256"] != digest(input_row):
            raise RuntimeError(f"semantic review provenance mismatch: {key}")
        parsed = judgment.get("parsed") if judgment.get("valid") else None
        accepted = bool(
            parsed and not parsed["needs_retranslation"] and input_row["numbers_preserved"]
        )
        if accepted:
            continue
        row = dict(translations[key])
        row["semantic_review_before_repair"] = parsed
        row["repair_route"] = args.force_route or (
            "milmmt_46_12b" if row["language"] in MILMMT_LANGUAGE_NAMES
            else "madlad_400_3b"
        )
        if row["language"] not in MADLAD_CODES:
            raise RuntimeError(f"missing MADLAD language mapping: {row['language']}")
        selected.append(row)
    selected.sort(key=lambda row: (row["repair_route"], row["language"], row["item_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, selected)
    print(json.dumps({
        "rows": len(selected),
        "milmmt": sum(row["repair_route"] == "milmmt_46_12b" for row in selected),
        "madlad": sum(row["repair_route"] == "madlad_400_3b" for row in selected),
    }, indent=2))


if __name__ == "__main__":
    main()
