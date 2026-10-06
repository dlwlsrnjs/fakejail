#!/usr/bin/env python3
"""Reconstruct v3 held-out replays and compute paired uncertainty tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import evaluate_jailnews_clustered_bai_v3 as v3


def paired_bootstrap(delta: np.ndarray, seed: int, samples: int = 10000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(samples, len(delta)))
    means = delta[draws].mean(axis=1)
    return {
        "delta_pp": 100.0 * float(delta.mean()),
        "ci95_low_pp": 100.0 * float(np.quantile(means, 0.025)),
        "ci95_high_pp": 100.0 * float(np.quantile(means, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--person-embeddings", type=Path, required=True)
    parser.add_argument("--context-embeddings", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    rows = v3.read_jsonl(args.oof)
    people = v3.group_people(rows)
    features = v3.load_context_features(rows, people, args.person_embeddings, args.context_embeddings, args.seed)
    result = json.loads(args.results.read_text(encoding="utf-8"))
    budgets = tuple(map(int, result["budgets"]))
    geometries = {i: v3.arm_geometry(rows, np.asarray(p.rows), "asr") for i, p in enumerate(people)}
    topologies = {
        i: v3.arm_topology(geometries[i], v3.stable_seed(p.person_id, args.seed))
        for i, p in enumerate(people)
    }
    output: dict[str, object] = {"schema": "jailnews_clustered_bai_v3_paired/v1", "comparisons": {}}
    report = ["# Clustered BAI v3 paired comparisons", ""]
    for endpoint in v3.ENDPOINTS:
        observations = {
            policy: {budget: [] for budget in budgets}
            for policy in (
                "nested_best_router_static", "context_cluster_static", "triple_clst_adapted",
                "ccb_adaptive", "budget_aware_clustered_bai",
            )
        }
        for fold_entry in result["fold_audit"][endpoint]:
            fold = int(fold_entry["fold"])
            train = np.asarray([i for i, p in enumerate(people) if p.fold != fold])
            test = np.asarray([i for i, p in enumerate(people) if p.fold == fold])
            prior_context, _, _ = v3.fit_context_cluster_prior(
                rows, people, features, train, endpoint, 16, 12.0, args.seed + fold,
            )
            base_model = fold_entry["base_model"]
            config = fold_entry["chosen"]
            controller = {int(k): value for k, value in fold_entry["controller_by_budget"].items()}
            for pi in test:
                person = people[int(pi)]
                indices = np.asarray(person.rows)
                y = np.asarray([rows[i]["labels"][endpoint] for i in indices], dtype=int)
                base = np.asarray([rows[i]["predictions"][endpoint][base_model] for i in indices])
                prior = np.asarray(v3.sigmoid(
                    (1.0 - config["blend"]) * v3.logit(base)
                    + config["blend"] * v3.logit(prior_context[indices])
                ))
                geometry = geometries[int(pi)]
                replays = {
                    "nested_best_router_static": v3.fixed_replay(np.argsort(-base, kind="stable"), y, budgets),
                    "context_cluster_static": v3.fixed_replay(np.argsort(-prior, kind="stable"), y, budgets),
                    "triple_clst_adapted": v3.fixed_replay(
                        v3.triple_cluster_order(prior, geometry, v3.stable_seed(person.person_id, args.seed)), y, budgets
                    ),
                    "ccb_adaptive": v3.ccb_replay(
                        prior, geometry, y, budgets, config["cluster_weight"], config["kernel_weight"],
                        config["novelty_weight"], v3.stable_seed(person.person_id, args.seed), topologies[int(pi)],
                    ),
                }
                replays["budget_aware_clustered_bai"] = {
                    budget: replays[controller[budget]][budget] for budget in budgets
                }
                for policy, replay in replays.items():
                    for budget in budgets:
                        observations[policy][budget].append(replay[budget])

        endpoint_output = {}
        report.extend([f"## {endpoint}", "", "| policy vs nested | B | delta (pp) | 95% paired CI | McNemar exact p |", "|---|---:|---:|---:|---:|"])
        baseline = observations["nested_best_router_static"]
        for policy in observations:
            if policy == "nested_best_router_static":
                continue
            endpoint_output[policy] = {}
            for budget in budgets:
                left = np.asarray(observations[policy][budget], dtype=int)
                right = np.asarray(baseline[budget], dtype=int)
                delta = left - right
                stats = paired_bootstrap(delta, v3.stable_seed(endpoint + policy + str(budget), args.seed))
                wins = int(np.sum((left == 1) & (right == 0)))
                losses = int(np.sum((left == 0) & (right == 1)))
                discordant = wins + losses
                p = float(binomtest(wins, discordant, 0.5).pvalue) if discordant else 1.0
                stats.update({"wins": wins, "losses": losses, "mcnemar_exact_p": p})
                endpoint_output[policy][str(budget)] = stats
                if budget in (1, 2, 3, 5):
                    report.append(
                        f"| {policy} | {budget} | {stats['delta_pp']:+.2f} | "
                        f"[{stats['ci95_low_pp']:+.2f}, {stats['ci95_high_pp']:+.2f}] | {p:.4f} |"
                    )
        output["comparisons"][endpoint] = endpoint_output
        report.append("")
    v3.write_json(args.output, output)
    args.output.with_suffix(".md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report": str(args.output.with_suffix('.md'))}))


if __name__ == "__main__":
    main()
