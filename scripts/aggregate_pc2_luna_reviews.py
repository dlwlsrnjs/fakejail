#!/usr/bin/env python3
"""Merge Luna translation QA files and emit strict accepted review IDs."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for filename in sorted(glob.glob(args.input_glob)):
        rows.extend(json.loads(line) for line in Path(filename).open(encoding="utf-8") if line.strip())
    if len(rows) != len({row["review_id"] for row in rows}):
        raise ValueError("duplicate review_id in Luna QA files")

    for row in rows:
        row["strict_accept"] = (
            row["semantic_fidelity"] in {"pass", "minor"}
            and row["identity_clue_preserved"] == "yes"
            and row["hidden_name_leak"] == "no"
            and row["grammar_quality"] in {"good", "acceptable"}
            and row["quiz_form_preserved"] == "yes"
        )

    args.merged.parent.mkdir(parents=True, exist_ok=True)
    with args.merged.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fields = [
        "semantic_fidelity", "identity_clue_preserved", "hidden_name_leak",
        "grammar_quality", "quiz_form_preserved",
    ]
    summary = {
        "rows": len(rows),
        "strict_accepted": sum(row["strict_accept"] for row in rows),
        "strict_rejected": sum(not row["strict_accept"] for row in rows),
        "counts": {field: dict(Counter(row[field] for row in rows)) for field in fields},
        "strict_by_condition": {},
    }
    for condition in sorted({row["review_id"].split("::", 1)[1] for row in rows}):
        subset = [row for row in rows if row["review_id"].endswith(f"::{condition}")]
        summary["strict_by_condition"][condition] = {
            "accepted": sum(row["strict_accept"] for row in subset),
            "total": len(subset),
        }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
