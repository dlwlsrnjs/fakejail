#!/usr/bin/env python3
"""Merge specialist translation-repair shards with exact inventory validation."""

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
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = read(args.inventory)
    rows = [row for path in args.shards for row in read(path)]
    expected = {row["translation_id"] for row in inventory}
    actual = {row["translation_id"] for row in rows}
    if len(rows) != len(actual) or actual != expected:
        raise RuntimeError(
            f"specialized repair grid mismatch: rows={len(rows)} unique={len(actual)} "
            f"expected={len(expected)} missing={len(expected-actual)} extra={len(actual-expected)}"
        )
    rows.sort(key=lambda row: (row["item_id"], row["language"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, rows)
    print(json.dumps({"rows": len(rows), "shards": len(args.shards)}, indent=2))


if __name__ == "__main__":
    main()
