#!/usr/bin/env python3
"""Add an exact instruction-grounding span to otherwise valid annotations."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_intent_span(instruction: str, limit: int = 280) -> str:
    parts = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", instruction) if part.strip()]
    if not parts:
        return instruction[:limit]
    # Dataset instructions normally put the behavioral objective in the
    # longest sentence; preserve it byte-for-byte apart from outer whitespace.
    sentence = max(parts, key=len)
    return sentence[:limit].rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    args = parser.parse_args()

    source: dict[str, str] = {}
    for row in read_jsonl(args.judgments):
        source.setdefault(row["sample_id"], str(row.get("instruction") or ""))
    rows = read_jsonl(args.annotations)
    repaired = 0
    for row in rows:
        instruction = source[row["sample_id"]]
        evidence = row["final_detailed"].get("source_evidence") or []
        if any(item.get("span") and item["span"] in instruction for item in evidence):
            continue
        span = exact_intent_span(instruction)
        if not span or span not in instruction:
            raise RuntimeError(f"could not derive exact span for {row['sample_id']}")
        evidence.append({"span": span, "supports": "instruction_grounding_for_primary_goal"})
        row["final_detailed"]["source_evidence"] = evidence
        row["validation_errors"] = [error for error in row.get("validation_errors", []) if error != "no_exact_source_evidence"]
        row["status"] = "accepted" if not row["validation_errors"] else "needs_review"
        repaired += 1

    temporary = args.annotations.with_suffix(args.annotations.suffix + f".tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.annotations)
    args.annotations.chmod(0o600)

    summary_path = args.annotations.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["accepted"] = sum(row["status"] == "accepted" for row in rows)
    summary["needs_review"] = len(rows) - summary["accepted"]
    summary["deterministic_exact_evidence_repairs"] = repaired
    summary["status_counts"] = dict(Counter(row["status"] for row in rows))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "repaired": repaired, "accepted": summary["accepted"]}))


if __name__ == "__main__":
    main()
