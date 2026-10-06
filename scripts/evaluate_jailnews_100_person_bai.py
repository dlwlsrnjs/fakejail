#!/usr/bin/env python3
"""Offline BAI replay over the sanitized 100-person JailNews surrogate matrix.

Each target model is evaluated with the other model's per-person arm rates as
the transferable surrogate prior.  Target-global baselines are leave-one-person
out.  No prompt or generation text is read or retained.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MODELS = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "GraySwanAI/Llama-3-8B-Instruct-RR",
)
POLICIES = (
    "random",
    "target_loo_global",
    "geopolitical_frozen",
    "surrogate_frozen",
    "hybrid_frozen",
    "geopolitical_surrogate_frozen",
    "geopolitical_hybrid_frozen",
    "factorized_ts_surrogate",
    "factorized_ts_hybrid",
    "factorized_ts_geopolitical_surrogate",
    "factorized_ts_geopolitical_hybrid",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped))


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-np.clip(values, -30, 30)))


def smoothed(successes: np.ndarray, trials: np.ndarray) -> np.ndarray:
    return (successes + 0.5) / (trials + 1.0)


def percentile_probability(values: np.ndarray) -> np.ndarray:
    """Turn an unlabeled ranking score into a finite probability-like prior."""
    ranks = np.empty(len(values), dtype=float)
    for value in np.unique(values):
        indices = np.flatnonzero(values == value)
        lower = int(np.sum(values < value))
        upper = lower + len(indices) - 1
        ranks[indices] = (lower + upper) / 2
    # Mid-rank plotting positions avoid zero/one logits.  This is deliberately
    # only an ordering prior, not a claim that the geopolitical score is ASR.
    return (ranks + 0.5) / len(values)


def fixed_order_replay(
    order: np.ndarray,
    outcomes: np.ndarray,
    true_p: np.ndarray,
    budgets: list[int],
) -> dict[int, tuple[float, float, int]]:
    result = {}
    found = False
    recommended = int(order[0])
    for step, arm in enumerate(order[: max(budgets)], 1):
        found = found or bool(outcomes[int(arm)])
        if step in budgets:
            result[step] = (float(found), float(true_p[recommended]), recommended)
    return result


def factorized_replay(
    prior: np.ndarray,
    outcomes: np.ndarray,
    true_p: np.ndarray,
    language_index: np.ndarray,
    method_index: np.ndarray,
    budgets: list[int],
    rng: np.random.Generator,
) -> dict[int, tuple[float, float, int]]:
    n_arms = len(prior)
    base = logit(prior)
    global_mean, global_precision = 0.0, 4.0
    language_mean = np.zeros(int(language_index.max()) + 1)
    language_precision = np.full_like(language_mean, 2.0)
    method_mean = np.zeros(int(method_index.max()) + 1)
    method_precision = np.full_like(method_mean, 2.0)
    pair_mean = np.zeros(n_arms)
    pair_precision = np.ones(n_arms)
    tried = np.zeros(n_arms, dtype=bool)
    found = False
    result = {}
    for step in range(1, max(budgets) + 1):
        means = (
            base + global_mean + language_mean[language_index]
            + method_mean[method_index] + pair_mean
        )
        variances = (
            1 / global_precision + 1 / language_precision[language_index]
            + 1 / method_precision[method_index] + 1 / pair_precision
        )
        sampled = rng.normal(means, np.sqrt(variances))
        sampled[tried] = -np.inf
        arm = int(np.argmax(sampled))
        tried[arm] = True
        outcome = int(outcomes[arm])
        found = found or bool(outcome)
        p_before = float(sigmoid(means[arm]))
        gradient = outcome - p_before
        curvature = max(1e-4, p_before * (1 - p_before))

        global_precision += curvature
        global_mean += gradient / global_precision
        li = int(language_index[arm])
        language_precision[li] += curvature
        language_mean[li] += gradient / language_precision[li]
        mi = int(method_index[arm])
        method_precision[mi] += curvature
        method_mean[mi] += gradient / method_precision[mi]
        pair_precision[arm] += curvature
        pair_mean[arm] += gradient / pair_precision[arm]

        if step in budgets:
            posterior = (
                base + global_mean + language_mean[language_index]
                + method_mean[method_index] + pair_mean
            )
            recommended = int(np.argmax(posterior))
            result[step] = (float(found), float(true_p[recommended]), recommended)
    return result


def mean_ci(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    se = float(array.std(ddof=1) / math.sqrt(len(array)))
    return {"mean": mean, "ci95_low": max(0.0, mean - 1.96 * se), "ci95_high": min(1.0, mean + 1.96 * se)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--geopolitical-rankings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16, 30])
    parser.add_argument("--geo-surrogate-weight", type=float, default=0.15)
    parser.add_argument("--geo-hybrid-weight", type=float, default=0.15)
    args = parser.parse_args()
    if not 0 <= args.geo_surrogate_weight <= 1 or not 0 <= args.geo_hybrid_weight <= 1:
        raise ValueError("geopolitical blend weights must be between zero and one")

    rows = read_jsonl(args.scores)
    geopolitical_rows = read_jsonl(args.geopolitical_rankings)
    samples = sorted({row["sample_id"] for row in rows})
    models = sorted({row["model_id"] for row in rows})
    if models != sorted(MODELS):
        raise ValueError(f"unexpected models: {models}")
    arms = sorted({(row["language"], row["method"]) for row in rows})
    if len(samples) != 100 or len(arms) != 360:
        raise ValueError(f"expected 100 samples and 360 arms, got {len(samples)} and {len(arms)}")
    arm_to_index = {arm: index for index, arm in enumerate(arms)}
    languages = sorted({arm[0] for arm in arms})
    methods = sorted({arm[1] for arm in arms})
    language_to_index = {value: index for index, value in enumerate(languages)}
    method_to_index = {value: index for index, value in enumerate(methods)}
    language_index = np.asarray([language_to_index[arm[0]] for arm in arms])
    method_index = np.asarray([method_to_index[arm[1]] for arm in arms])

    shape = (len(models), len(samples), len(arms))
    successes = np.zeros(shape, dtype=float)
    trials = np.zeros(shape, dtype=float)
    translation_valid = np.zeros((len(samples), len(arms)), dtype=bool)
    sample_to_index = {value: index for index, value in enumerate(samples)}
    model_to_index = {value: index for index, value in enumerate(models)}
    people = {}
    for row in rows:
        mi = model_to_index[row["model_id"]]
        si = sample_to_index[row["sample_id"]]
        ai = arm_to_index[(row["language"], row["method"])]
        successes[mi, si, ai] = row["successes"]
        trials[mi, si, ai] = row["trials"]
        translation_valid[si, ai] = bool(row["translation_valid"])
        people[row["sample_id"]] = row["person"]
    if not np.all(trials == 5):
        raise ValueError("the BAI replay requires exactly five judged trials per model/sample/arm")
    # Priors are smoothed to avoid infinite logits, but the held-out replay
    # endpoint must remain the observed target rate.  Smoothing the endpoint
    # would turn every 0/5 arm into an artificial 8.3% success arm.
    prior_probabilities = smoothed(successes, trials)
    empirical_probabilities = successes / trials

    geopolitical_by_person = {row["person"]: row for row in geopolitical_rows}
    missing_people = sorted({people[sample_id] for sample_id in samples} - set(geopolitical_by_person))
    if missing_people:
        raise ValueError(f"missing geopolitical rankings for {len(missing_people)} people: {missing_people[:5]}")
    geopolitical_probability = np.zeros((len(samples), len(arms)), dtype=float)
    for sample_index, sample_id in enumerate(samples):
        candidates = geopolitical_by_person[people[sample_id]]["candidates"]
        language_score = {
            row["language"]: float(row["score_routed_gap"])
            for row in candidates
        }
        if set(language_score) != set(languages):
            raise ValueError(f"geopolitical language mismatch for {sample_id}")
        language_values = np.asarray([language_score[language] for language in languages])
        language_probability = percentile_probability(language_values)
        geopolitical_probability[sample_index] = language_probability[language_index]

    observations: dict[tuple[str, str, int, str], dict[str, list[float]]] = defaultdict(
        lambda: {"discovery": [], "recommended_p": [], "simple_regret": [], "recommended_translation_valid": []}
    )
    direction_details = {}
    for target_model_index, target_model in enumerate(models):
        surrogate_model_index = 1 - target_model_index
        surrogate_model = models[surrogate_model_index]
        direction = f"{surrogate_model} -> {target_model}"
        for sample_index, sample_id in enumerate(samples):
            true_p = empirical_probabilities[target_model_index, sample_index]
            surrogate_p = prior_probabilities[surrogate_model_index, sample_index]
            geopolitical_p = geopolitical_probability[sample_index]
            mask = np.arange(len(samples)) != sample_index
            loo_success = successes[target_model_index, mask].sum(axis=0)
            loo_trials = trials[target_model_index, mask].sum(axis=0)
            global_p = smoothed(loo_success, loo_trials)
            hybrid_p = sigmoid(0.5 * logit(surrogate_p) + 0.5 * logit(global_p))
            geopolitical_surrogate_p = sigmoid(
                (1 - args.geo_surrogate_weight) * logit(surrogate_p)
                + args.geo_surrogate_weight * logit(geopolitical_p)
            )
            non_geopolitical_weight = (1 - args.geo_hybrid_weight) / 2
            geopolitical_hybrid_p = sigmoid(
                non_geopolitical_weight * logit(surrogate_p)
                + non_geopolitical_weight * logit(global_p)
                + args.geo_hybrid_weight * logit(geopolitical_p)
            )
            oracle_order = np.argsort(true_p)[::-1]
            p_max = float(true_p[oracle_order[0]])
            for simulation in range(args.simulations):
                rng = np.random.default_rng(args.seed + target_model_index * 1_000_000 + sample_index * 10_000 + simulation)
                outcomes = rng.binomial(1, true_p).astype(bool)
                orders = {
                    "random": rng.permutation(len(arms)),
                    "target_loo_global": np.argsort(global_p)[::-1],
                    "geopolitical_frozen": np.argsort(geopolitical_p)[::-1],
                    "surrogate_frozen": np.argsort(surrogate_p)[::-1],
                    "hybrid_frozen": np.argsort(hybrid_p)[::-1],
                    "geopolitical_surrogate_frozen": np.argsort(geopolitical_surrogate_p)[::-1],
                    "geopolitical_hybrid_frozen": np.argsort(geopolitical_hybrid_p)[::-1],
                }
                replays = {
                    policy: fixed_order_replay(order, outcomes, true_p, args.budgets)
                    for policy, order in orders.items()
                }
                replays["factorized_ts_surrogate"] = factorized_replay(
                    surrogate_p, outcomes, true_p, language_index, method_index, args.budgets, rng
                )
                replays["factorized_ts_hybrid"] = factorized_replay(
                    hybrid_p, outcomes, true_p, language_index, method_index, args.budgets, rng
                )
                replays["factorized_ts_geopolitical_surrogate"] = factorized_replay(
                    geopolitical_surrogate_p, outcomes, true_p, language_index, method_index, args.budgets, rng
                )
                replays["factorized_ts_geopolitical_hybrid"] = factorized_replay(
                    geopolitical_hybrid_p, outcomes, true_p, language_index, method_index, args.budgets, rng
                )
                for policy, replay in replays.items():
                    for budget, (found, recommended_p, recommended) in replay.items():
                        bucket = observations[(direction, policy, budget, "all")]
                        bucket["discovery"].append(found)
                        bucket["recommended_p"].append(recommended_p)
                        bucket["simple_regret"].append(p_max - recommended_p)
                        bucket["recommended_translation_valid"].append(float(translation_valid[sample_index, recommended]))
        direction_details[direction] = {
            "surrogate_model": surrogate_model,
            "target_model": target_model,
            "target_overall_asr": float(successes[target_model_index].sum() / trials[target_model_index].sum()),
        }

    metrics = {}
    for direction in direction_details:
        metrics[direction] = {}
        for policy in POLICIES:
            metrics[direction][policy] = {}
            for budget in args.budgets:
                bucket = observations[(direction, policy, budget, "all")]
                metrics[direction][policy][str(budget)] = {
                    "discovery_asr": mean_ci(bucket["discovery"]),
                    "recommended_success_probability": mean_ci(bucket["recommended_p"]),
                    "simple_regret": mean_ci(bucket["simple_regret"]),
                    "recommended_translation_valid_rate": mean_ci(bucket["recommended_translation_valid"]),
                    "replays": len(bucket["discovery"]),
                }

    result = {
        "schema": "jailnews_100_person_bai_replay/v1",
        "scores": str(args.scores),
        "geopolitical_rankings": str(args.geopolitical_rankings),
        "people": len(samples),
        "arms_per_person": len(arms),
        "languages": len(languages),
        "methods": len(methods),
        "trials_per_model_person_arm": 5,
        "simulations_per_person_direction": args.simulations,
        "budgets": args.budgets,
        "directions": direction_details,
        "policies": {
            "random": "uniform random unique-arm order",
            "target_loo_global": "target-model arm ranking aggregated over the other 99 people",
            "geopolitical_frozen": "unlabeled routed-MoE language ranking repeated over all five methods",
            "surrogate_frozen": "same-person arm ranking from the other surrogate model",
            "hybrid_frozen": "equal-weight logit blend of surrogate and target leave-one-person-out priors",
            "geopolitical_surrogate_frozen": "configurable logit blend of surrogate arm prior and routed-MoE language prior",
            "geopolitical_hybrid_frozen": "configurable logit blend of surrogate, target LOO, and routed-MoE priors",
            "factorized_ts_surrogate": "surrogate prior plus online global/language/method/pair residual Thompson sampling",
            "factorized_ts_hybrid": "hybrid prior plus the same online residual Thompson sampling",
            "factorized_ts_geopolitical_surrogate": "geopolitical-surrogate blend plus online residual Thompson sampling",
            "factorized_ts_geopolitical_hybrid": "geopolitical hybrid blend plus online residual Thompson sampling",
        },
        "geopolitical_blend": {
            "score": "PC2-style routed-MoE utility minus domain-routed geopolitical sensitivity risk",
            "calibration": "within-person language percentile plotting position",
            "surrogate_weights": {
                "surrogate": 1 - args.geo_surrogate_weight,
                "geopolitical": args.geo_surrogate_weight,
            },
            "hybrid_weights": {
                "surrogate": (1 - args.geo_hybrid_weight) / 2,
                "target_loo": (1 - args.geo_hybrid_weight) / 2,
                "geopolitical": args.geo_hybrid_weight,
            },
        },
        "metrics": metrics,
        "leakage_controls": [
            "target-global statistics exclude the evaluated person",
            "surrogate prior uses only the other model",
            "raw generations and prompts are not loaded",
        ],
    }
    atomic_json(args.output, result)

    lines = [
        "# JailNews 100-person BAI replay", "",
        f"People: {len(samples)}; arms/person: {len(arms)}; simulations: {args.simulations}", "",
    ]
    for direction in direction_details:
        lines.extend([f"## {direction}", "", "| policy | B | discovery ASR | recommended p | regret | valid-translation selection |", "|---|---:|---:|---:|---:|---:|"])
        for policy in POLICIES:
            for budget in args.budgets:
                row = metrics[direction][policy][str(budget)]
                lines.append(
                    f"| {policy} | {budget} | {row['discovery_asr']['mean']:.4f} | "
                    f"{row['recommended_success_probability']['mean']:.4f} | "
                    f"{row['simple_regret']['mean']:.4f} | "
                    f"{row['recommended_translation_valid_rate']['mean']:.4f} |"
                )
        lines.append("")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "people": len(samples), "arms": len(arms)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
