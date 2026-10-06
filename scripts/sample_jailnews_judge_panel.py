#!/usr/bin/env python3
"""Build an exact arm-balanced deterministic sample from generation JSONL shards."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path


def rank(seed: int, trial_id: str) -> int:
    payload = f"{seed}|{trial_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-arm", type=int, default=10)
    parser.add_argument("--expected-arms", type=int, default=360)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()

    # Max-heaps represented with negative ranks retain the k smallest hashes.
    heaps: dict[str, list[tuple[int, str, dict]]] = {}
    total = 0
    for path in sorted(args.inputs):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                total += 1
                arm = str(row["arm_id"])
                trial_id = str(row["trial_id"])
                score = rank(args.seed, trial_id)
                heap = heaps.setdefault(arm, [])
                item = (-score, trial_id, row)
                if len(heap) < args.per_arm:
                    heapq.heappush(heap, item)
                elif score < -heap[0][0]:
                    heapq.heapreplace(heap, item)

    if len(heaps) != args.expected_arms:
        raise RuntimeError(f"expected {args.expected_arms} arms, found {len(heaps)}")
    short = {arm: len(heap) for arm, heap in heaps.items() if len(heap) != args.per_arm}
    if short:
        raise RuntimeError(f"arms with insufficient rows: {short}")

    selected = []
    for arm, heap in heaps.items():
        for negative_score, trial_id, row in heap:
            selected.append((-negative_score, arm, trial_id, row))
    selected.sort(key=lambda item: (item[1], item[0], item[2]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for _score, _arm, _trial_id, row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.output)
    print(json.dumps({
        "source_rows": total,
        "arms": len(heaps),
        "per_arm": args.per_arm,
        "sample_rows": len(selected),
        "seed": args.seed,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
