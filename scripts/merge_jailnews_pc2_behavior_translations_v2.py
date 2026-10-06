#!/usr/bin/env python3
"""Select the stronger round-trip behavior translation per sample."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    primary = {row["quiz_id"]: row for row in read(args.primary)}
    fallback = {row["quiz_id"]: row for row in read(args.fallback)}
    if set(primary) != set(fallback):
        raise RuntimeError("primary/fallback behavior IDs differ")
    output = []
    fallback_count = 0
    for quiz_id in sorted(primary):
        a, b = primary[quiz_id], fallback[quiz_id]
        choose_b = (
            (b["valid"] and not a["valid"])
            or (
                b["valid"] == a["valid"]
                and b["roundtrip_similarity"] > a["roundtrip_similarity"] + 0.02
            )
        )
        row = dict(b if choose_b else a)
        if choose_b:
            row["translation_backend"] += "+selected_over_nllb_3.3b"
            fallback_count += 1
        row["ensemble_primary_valid"] = a["valid"]
        row["ensemble_fallback_valid"] = b["valid"]
        output.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(args.output)
    report = {
        "rows": len(output), "valid": sum(row["valid"] for row in output),
        "fallback_selected": fallback_count,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
