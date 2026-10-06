#!/usr/bin/env python3
"""Create a translation inventory delta against completed translation shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--existing", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-types", nargs="*", default=[])
    args = parser.parse_args()
    inventory = read_jsonl(args.inventory)
    existing = {}
    for path in args.existing:
        for row in read_jsonl(path):
            existing.setdefault(row["item_id"], row)
    delta = [
        row for row in inventory
        if row["item_type"] not in set(args.exclude_types)
        and (
            row["item_id"] not in existing
            or existing[row["item_id"]]["canonical_english"] != row["canonical_english"]
        )
    ]
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in delta:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.output)
    print(json.dumps({
        "inventory": len(inventory),
        "existing_items": len(existing),
        "delta": len(delta),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
