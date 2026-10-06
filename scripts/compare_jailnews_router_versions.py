#!/usr/bin/env python3
"""Paired person-level comparison of matching routers from two OOF runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected(
    rows: list[dict[str, Any]], endpoint: str, model: str, ranks: dict[str, int]
) -> dict[str, int]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["victim_person_id"]].append(row)
    return {
        person: int(max(
            members,
            key=lambda row: (
                row["predictions"][endpoint][model],
                -ranks[row["trial_id"]],
            ),
        )["labels"][endpoint])
        for person, members in groups.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    old_rows, new_rows = read_jsonl(args.old), read_jsonl(args.new)
    ranks = {row["trial_id"]: int(row["surrogate_selected_rank"]) for row in read_jsonl(args.judgments)}
    rng = np.random.default_rng(args.seed)
    comparisons = []
    for endpoint in ("asr", "retained", "strict"):
        old = selected(old_rows, endpoint, args.model, ranks)
        new = selected(new_rows, endpoint, args.model, ranks)
        people = sorted(set(old) & set(new))
        before = np.asarray([old[person] for person in people], dtype=float)
        after = np.asarray([new[person] for person in people], dtype=float)
        delta = after - before
        samples = rng.integers(0, len(people), size=(args.bootstrap, len(people)))
        boot = delta[samples].mean(axis=1)
        gains = int(((after == 1) & (before == 0)).sum())
        losses = int(((after == 0) & (before == 1)).sum())
        comparisons.append({
            "endpoint": endpoint,
            "model": args.model,
            "people": len(people),
            "old_top1": float(before.mean()),
            "new_top1": float(after.mean()),
            "delta_percentage_points": float(100 * delta.mean()),
            "paired_bootstrap_95_ci_pp": [
                float(100 * np.quantile(boot, 0.025)),
                float(100 * np.quantile(boot, 0.975)),
            ],
            "new_only_successes": gains,
            "old_only_successes": losses,
            "mcnemar_exact_p": float(binomtest(gains, gains + losses).pvalue) if gains + losses else 1.0,
        })
    payload = {"schema": "jailnews_router_version_comparison/v1", "comparisons": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
