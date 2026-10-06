#!/usr/bin/env python3
"""Materialize only fail-closed Qwen32 semantic rereview acceptances."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


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
    parser.add_argument("--review-model", default="Qwen/Qwen2.5-32B-Instruct")
    args = parser.parse_args()
    source = {row["key"]: row for row in read(args.source)}
    judgments = {row["key"]: row for row in read(args.judgments)}
    translations = {row["translation_id"]: row for row in read(args.translations)}
    if set(source) != set(judgments):
        raise RuntimeError("semantic rereview source/judgment keys differ")
    accepted = []
    rejected = 0
    invalid_judgments = 0
    numeric_blocked = 0
    for key, input_row in source.items():
        judgment = judgments[key]
        if judgment["source_sha256"] != digest(input_row):
            raise RuntimeError(f"rereview provenance mismatch: {key}")
        parsed = judgment.get("parsed") if judgment.get("valid") else None
        if not parsed:
            invalid_judgments += 1
            continue
        if parsed["needs_retranslation"]:
            rejected += 1
            continue
        # A semantic reviewer may approve localized digits, but the experiment's
        # exact numeric contract stays fail-closed until a repaired translation is made.
        if not input_row["numbers_preserved"]:
            numeric_blocked += 1
            continue
        row = dict(translations[key])
        row["strict_roundtrip_valid_original"] = row["valid"]
        row["semantic_rereview"] = parsed
        row["semantic_rereview_model"] = args.review_model
        row["semantic_rereview_raw"] = judgment["raw_judgment"]
        row["valid"] = True
        status = "local_llm_semantic_rereview_valid"
        row["final_translation_status"] = status
        row["translation_backend"] += "+" + args.review_model.replace("/", "_") + "_semantic_rereview"
        accepted.append(row)
    accepted.sort(key=lambda row: (row["item_id"], row["language"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, accepted)
    print(json.dumps({
        "source": len(source), "accepted": len(accepted), "rejected": rejected,
        "invalid_judgments": invalid_judgments, "numeric_blocked": numeric_blocked,
    }, indent=2))


if __name__ == "__main__":
    main()
