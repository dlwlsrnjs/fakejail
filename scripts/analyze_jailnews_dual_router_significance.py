#!/usr/bin/env python3
"""Paired person-level significance tests for dual-router OOF selections."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ENDPOINTS = ("asr", "retained", "strict")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_outcomes(rows: list[dict[str, Any]], endpoint: str, model: str) -> dict[str, int]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["victim_person_id"]].append(row)
    output = {}
    for person, members in groups.items():
        selected = max(
            members,
            key=lambda row: (
                float(row["predictions"][endpoint][model]),
                -int(row.get("surrogate_selected_rank") or 999),
            ),
        )
        output[person] = int(selected["labels"][endpoint])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="surrogate_prior")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    original_rank = {}
    with args.judgments.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                original_rank[row["trial_id"]] = int(row["surrogate_selected_rank"])
    for row in rows:
        row["surrogate_selected_rank"] = original_rank[row["trial_id"]]
    models = sorted(rows[0]["predictions"][ENDPOINTS[0]])
    rng = np.random.default_rng(args.seed)
    comparisons: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        baseline = selected_outcomes(rows, endpoint, args.baseline)
        people = sorted(baseline)
        base = np.asarray([baseline[person] for person in people], dtype=float)
        for model in models:
            current = selected_outcomes(rows, endpoint, model)
            candidate = np.asarray([current[person] for person in people], dtype=float)
            delta = candidate - base
            samples = rng.integers(0, len(people), size=(args.bootstrap, len(people)))
            bootstrap_delta = delta[samples].mean(axis=1)
            gain = int(np.sum((candidate == 1) & (base == 0)))
            loss = int(np.sum((candidate == 0) & (base == 1)))
            discordant = gain + loss
            p_value = float(binomtest(gain, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
            comparisons.append(
                {
                    "endpoint": endpoint,
                    "baseline": args.baseline,
                    "model": model,
                    "people": len(people),
                    "baseline_top1": float(base.mean()),
                    "model_top1": float(candidate.mean()),
                    "delta_percentage_points": float(100 * delta.mean()),
                    "paired_bootstrap_95_ci_pp": [
                        float(100 * np.quantile(bootstrap_delta, 0.025)),
                        float(100 * np.quantile(bootstrap_delta, 0.975)),
                    ],
                    "model_only_successes": gain,
                    "baseline_only_successes": loss,
                    "mcnemar_exact_p": p_value,
                }
            )
    payload = {
        "schema": "jailnews_dual_router_paired_significance/v1",
        "unit": "victim_person_id",
        "bootstrap_replicates": args.bootstrap,
        "seed": args.seed,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
