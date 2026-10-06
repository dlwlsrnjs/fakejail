#!/usr/bin/env python3
"""Join rejected Luna QA rows with their original translation inputs for retry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    packet = {row["review_id"]: row for row in read(args.packet)}
    reviews = {row["review_id"]: row for row in read(args.reviews)}
    output = []
    for review_id, review in reviews.items():
        if review.get("strict_accept"):
            continue
        item = dict(packet[review_id])
        item["previous_review"] = review
        output.append(item)

    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"retry_rows": len(output), "output": str(args.output)}))


if __name__ == "__main__":
    main()
