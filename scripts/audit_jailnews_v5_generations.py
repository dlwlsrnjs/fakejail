#!/usr/bin/env python3
"""Audit V5/control generations before any model-based judging.

Raw responses remain in their private JSONL files. The emitted report contains
only aggregate validity counts and cryptographic hashes, never generated text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


def iter_jsonl(paths: list[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def visible_answer(text: str) -> tuple[str, bool]:
    """Return visible text and whether a complete thinking block was stripped."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip(), True
    return text.strip(), False


def has_repetition_tail(text: str) -> bool:
    tokens = text.split()
    if len(tokens) < 128:
        return False
    tail = tokens[-128:]
    return len(set(tail)) <= 12


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Exit non-zero when any generation has an invalid hash or is truncated, repetitive, or empty.",
    )
    args = parser.parse_args()

    totals: Counter[str] = Counter()
    by_condition: dict[str, Counter[str]] = defaultdict(Counter)
    trial_ids: set[str] = set()
    duplicate_trials = 0
    invalid_rows: list[dict[str, Any]] = []
    for row in iter_jsonl(args.inputs):
        condition = str(row.get("pilot_condition") or row.get("condition") or "unknown")
        bucket = by_condition[condition]
        trial_id = str(row["trial_id"])
        if trial_id in trial_ids:
            duplicate_trials += 1
        trial_ids.add(trial_id)

        generation = str(row.get("generation") or "")
        response_hash_valid = row.get("response_sha256") == sha256_text(generation)
        visible, thinking_stripped = visible_answer(generation)
        length_truncated = str(row.get("finish_reason") or "") == "length"
        thinking_truncated = length_truncated and "</think>" not in generation
        repetitive = length_truncated and has_repetition_tail(generation)
        empty_visible = not visible

        flags = {
            "rows": True,
            "response_hash_valid": response_hash_valid,
            "thinking_stripped": thinking_stripped,
            "thinking_truncated": thinking_truncated,
            "length_truncated": length_truncated,
            "degenerate_repetition": repetitive,
            "empty_visible_answer": empty_visible,
        }
        for key, value in flags.items():
            totals[key] += int(value)
            bucket[key] += int(value)
        reasons = []
        if not response_hash_valid:
            reasons.append("response_hash_invalid")
        if thinking_truncated:
            reasons.append("thinking_truncated")
        if repetitive:
            reasons.append("degenerate_repetition")
        if empty_visible:
            reasons.append("empty_visible_answer")
        if reasons:
            invalid_rows.append(
                {
                    "trial_id": trial_id,
                    "condition": condition,
                    "language": row.get("language"),
                    "language_code": row.get("language_code"),
                    "attack_type": row.get("attack_type"),
                    "reasons": reasons,
                }
            )

    rows = totals["rows"]
    invalid = (
        totals["thinking_truncated"]
        + totals["degenerate_repetition"]
        + totals["empty_visible_answer"]
        + (rows - totals["response_hash_valid"])
    )
    report = {
        "schema": "jailnews_v5_generation_audit/v1",
        "rows": rows,
        "unique_trial_ids": len(trial_ids),
        "duplicate_trial_ids": duplicate_trials,
        "invalid_flag_sum": invalid,
        "invalid_row_count": len(invalid_rows),
        "invalid_rows": invalid_rows,
        "totals": dict(totals),
        "by_condition": {key: dict(value) for key, value in sorted(by_condition.items())},
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.inputs
        ],
    }
    atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.fail_on_invalid and (invalid_rows or duplicate_trials):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
