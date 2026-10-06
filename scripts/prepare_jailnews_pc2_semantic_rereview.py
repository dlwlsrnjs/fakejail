#!/usr/bin/env python3
"""Convert strict round-trip failures into the frozen Qwen32 semantics-review schema."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


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
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    failures = [row for row in read(args.translations) if not row["valid"]]
    rows = [{
        "key": row["translation_id"],
        "item_id": row["item_id"],
        "item_type": row["item_type"],
        "language": row["language"],
        "english_original": row["canonical_english"],
        "translated": row["translation"],
        "backtranslation": row["backtranslation"],
        "numbers_preserved": row["numbers_preserved"],
        "qa_reason": (
            f"strict round-trip filter failed: cosine={row['backtranslation_similarity']}; "
            f"numbers_preserved={row['numbers_preserved']}"
        ),
    } for row in failures]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, rows)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
