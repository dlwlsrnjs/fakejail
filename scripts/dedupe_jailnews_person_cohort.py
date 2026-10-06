#!/usr/bin/env python3
"""Create a deterministic source-sample view of person-weighted JSONL results."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(args.inputs)):
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())

    # A source is unique only within its public split. Keep the lexicographically
    # first linked person when one source instruction mentions two cohort people.
    selected = {}
    for row in sorted(rows, key=lambda item: str(item.get("victim_person_id", ""))):
        key = (
            str(row.get("source_split", "")),
            str(row.get("raw_jailnews_uid", row.get("source_record_id", ""))),
            str(row.get("attack_type", "")),
            int(row.get("draw_id", 0)),
        )
        selected.setdefault(key, row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(selected):
            handle.write(json.dumps(selected[key], ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"input_rows": len(rows), "unique_rows": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
