#!/usr/bin/env python3
"""Leakage-aware offline replay for contextual, clustered JailNews BAI.

The target data contain one judged response for each of ten preselected settings
per person.  Consequently this script evaluates *success discovery* (how many
unique settings are needed to reveal a successful one), not stochastic mean-arm
identification.  All target outcomes used to construct the contextual prior are
restricted to training folds.  Test-fold outcomes are revealed only after a
setting is selected by the replay policy.

The proposed CCB policy has two levels:

1. A soft context cluster prior transfers language/method success statistics
   from embedding-neighbouring training people.
2. A within-person arm topology clusters the ten settings by their OOF router
   response profiles, method, and script.  Observed residuals are propagated to
   nearby untried settings, while a small novelty bonus avoids repeatedly
   querying one failed region.

This is inspired by TRIPLE-CLST's use of prompt embeddings to reduce a large
candidate pool, but it is deliberately contextual and failure-adaptive because
the best language/method setting changes by person and instruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ENDPOINTS = ("asr", "retained", "strict")
BASE_MODELS = (
    "surrogate_prior",
    "formula_method_prior",
    "global_bias",
    "pc2_static_language_router",
    "entity_language_router",
    "intent_metadata_router",
    "intent_technique_router",
    "intent_formula_router",
    "intent_iterative_combined_router",
    "factorized_dual_router",
    "factorized_iterative_dual_router",
    "joint_interaction_router",
    "joint_iterative_interaction_router",
    "joint_plus_surrogate",
)
POLICIES = (
    "random",
    "surrogate_static",
    "nested_best_router_static",
    "context_cluster_static",
    "triple_clst_adapted",
    "ccb_adaptive",
    "budget_aware_clustered_bai",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def logit(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, 1e-4, 1.0 - 1e-4)
    return np.log(x / (1.0 - x))


def unit_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return x / norm


def stable_seed(text: str, seed: int) -> int:
    return seed ^ int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def spherical_kmeans(x: np.ndarray, k: int, seed: int, iterations: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic cosine k-means with k-means++ initialization."""
    x = unit_rows(np.asarray(x, dtype=np.float64))
    k = max(1, min(k, len(x)))
    rng = np.random.default_rng(seed)
    centers = [int(rng.integers(len(x)))]
    closest = 1.0 - x @ x[centers[0]]
    for _ in range(1, k):
        weights = np.maximum(closest, 0.0) ** 2
        if weights.sum() <= 1e-12:
            candidate = int(next(i for i in range(len(x)) if i not in centers))
        else:
            candidate = int(rng.choice(len(x), p=weights / weights.sum()))
        centers.append(candidate)
        closest = np.minimum(closest, 1.0 - x @ x[candidate])
    c = x[centers].copy()
    labels = np.zeros(len(x), dtype=np.int16)
    for _ in range(iterations):
        new_labels = np.argmax(x @ c.T, axis=1).astype(np.int16)
        if np.array_equal(new_labels, labels) and _:
            break
        labels = new_labels
        for cluster in range(k):
            members = x[labels == cluster]
            if len(members):
                c[cluster] = unit_rows(members.mean(axis=0, keepdims=True))[0]
            else:
                farthest = int(np.argmin(np.max(x @ c.T, axis=1)))
                c[cluster] = x[farthest]
    return labels, c


def reduce_embeddings(x: np.ndarray, dimensions: int, seed: int) -> np.ndarray:
    """Fixed random projection; avoids fitting a target-outcome-dependent map."""
    x = unit_rows(np.asarray(x, dtype=np.float32))
    if x.shape[1] <= dimensions:
        return x
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0 / math.sqrt(dimensions), (x.shape[1], dimensions)).astype(np.float32)
    return unit_rows(x @ projection)


@dataclass(frozen=True)
class Person:
    person_id: str
    fold: int
    rows: tuple[int, ...]


def group_people(rows: list[dict[str, Any]]) -> list[Person]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["victim_person_id"]].append(index)
    people = []
    for person_id, indices in sorted(grouped.items()):
        folds = {int(rows[index]["fold"]) for index in indices}
        if len(indices) != 10 or len(folds) != 1:
            raise ValueError(f"expected ten rows and one fold for {person_id}: {len(indices)}, {folds}")
        people.append(Person(person_id, folds.pop(), tuple(indices)))
    return people


def load_context_features(
    rows: list[dict[str, Any]], people: list[Person], person_npz: Path, context_npz: Path, seed: int
) -> np.ndarray:
    person_data = np.load(person_npz, allow_pickle=False)
    person_ids = [str(value) for value in person_data["person_ids"]]
    person_map = {value: index for index, value in enumerate(person_ids)}
    context_data = np.load(context_npz, allow_pickle=False)
    sample_ids = [str(value) for value in context_data["sample_ids"]]
    context_map = {value: index for index, value in enumerate(sample_ids)}
    p_vectors = reduce_embeddings(person_data["vectors"], 96, seed)
    c_vectors = reduce_embeddings(context_data["vectors"], 96, seed + 1)
    features = []
    for person in people:
        p = p_vectors[person_map[person.person_id]]
        sample_id = rows[person.rows[0]]["sample_id"]
        c = c_vectors[context_map[sample_id]]
        features.append(np.concatenate([p, c]))
    return unit_rows(np.asarray(features, dtype=np.float32))


def nested_best_model(
    rows: list[dict[str, Any]], people: list[Person], train_people: np.ndarray, endpoint: str, budgets: Iterable[int]
) -> str:
    """Pick a base router only from outer-training people."""
    best: tuple[float, str] | None = None
    for model in BASE_MODELS:
        scores = []
        for person_index in train_people:
            person = people[int(person_index)]
            indices = np.asarray(person.rows)
            probability = np.asarray([rows[i]["predictions"][endpoint].get(model, -1.0) for i in indices])
            if np.any(probability < 0):
                continue
            outcome = np.asarray([rows[i]["labels"][endpoint] for i in indices], dtype=float)
            order = np.argsort(-probability, kind="stable")
            # Selection must work early, not only at top-1.
            scores.append(np.mean([outcome[order[:b]].max() for b in budgets]))
        candidate = (float(np.mean(scores)), model)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise RuntimeError(f"no usable base router for {endpoint}")
    return best[1]


def fit_context_cluster_prior(
    rows: list[dict[str, Any]], people: list[Person], features: np.ndarray,
    train_people: np.ndarray, endpoint: str, k: int, shrink: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit hard prototypes, but return a soft mixture prior for every row."""
    train_x = features[train_people]
    train_labels, centers = spherical_kmeans(train_x, k, seed)
    k = len(centers)
    language_values = sorted({row["language_code"] for row in rows})
    method_values = sorted({row["attack_type"] for row in rows})
    language_index = {value: index for index, value in enumerate(language_values)}
    method_index = {value: index for index, value in enumerate(method_values)}
    pair_values = sorted({(row["language_code"], row["attack_type"]) for row in rows})
    pair_index = {value: index for index, value in enumerate(pair_values)}

    global_success = 0.0
    global_trials = 0.0
    arm_success = np.zeros(len(pair_values))
    arm_trials = np.zeros(len(pair_values))
    lang_success = np.zeros(len(language_values))
    lang_trials = np.zeros(len(language_values))
    method_success = np.zeros(len(method_values))
    method_trials = np.zeros(len(method_values))
    cluster_success = np.zeros(k)
    cluster_trials = np.zeros(k)
    cluster_arm_success = np.zeros((k, len(pair_values)))
    cluster_arm_trials = np.zeros_like(cluster_arm_success)
    cluster_lang_success = np.zeros((k, len(language_values)))
    cluster_lang_trials = np.zeros_like(cluster_lang_success)
    cluster_method_success = np.zeros((k, len(method_values)))
    cluster_method_trials = np.zeros_like(cluster_method_success)
    for local, person_index in enumerate(train_people):
        cluster = int(train_labels[local])
        for row_index in people[int(person_index)].rows:
            row = rows[row_index]
            y = float(row["labels"][endpoint])
            li = language_index[row["language_code"]]
            mi = method_index[row["attack_type"]]
            ai = pair_index[(row["language_code"], row["attack_type"])]
            global_success += y; global_trials += 1
            arm_success[ai] += y; arm_trials[ai] += 1
            lang_success[li] += y; lang_trials[li] += 1
            method_success[mi] += y; method_trials[mi] += 1
            cluster_success[cluster] += y; cluster_trials[cluster] += 1
            cluster_arm_success[cluster, ai] += y; cluster_arm_trials[cluster, ai] += 1
            cluster_lang_success[cluster, li] += y; cluster_lang_trials[cluster, li] += 1
            cluster_method_success[cluster, mi] += y; cluster_method_trials[cluster, mi] += 1
    global_rate = (global_success + 0.5) / (global_trials + 1.0)
    arm_rate = (arm_success + shrink * global_rate) / (arm_trials + shrink)
    lang_rate = (lang_success + shrink * global_rate) / (lang_trials + shrink)
    method_rate = (method_success + shrink * global_rate) / (method_trials + shrink)
    cluster_rate = (cluster_success + shrink * global_rate) / (cluster_trials + shrink)

    # Cosine-soft assignments prevent brittle hard-cluster boundary effects.
    similarities = features @ centers.T
    temperature = 0.12
    responsibilities = np.exp((similarities - similarities.max(axis=1, keepdims=True)) / temperature)
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    prior = np.zeros(len(rows), dtype=np.float64)
    for person_index, person in enumerate(people):
        q = responsibilities[person_index]
        for row_index in person.rows:
            row = rows[row_index]
            li = language_index[row["language_code"]]
            mi = method_index[row["attack_type"]]
            ai = pair_index[(row["language_code"], row["attack_type"])]
            per_cluster = np.zeros(k)
            for cluster in range(k):
                p_pair = (
                    cluster_arm_success[cluster, ai] + shrink * arm_rate[ai]
                ) / (cluster_arm_trials[cluster, ai] + shrink)
                p_lang = (
                    cluster_lang_success[cluster, li] + shrink * lang_rate[li]
                ) / (cluster_lang_trials[cluster, li] + shrink)
                p_method = (
                    cluster_method_success[cluster, mi] + shrink * method_rate[mi]
                ) / (cluster_method_trials[cluster, mi] + shrink)
                # Sparse exact pairs are stabilized by the two marginals.
                per_cluster[cluster] = float(sigmoid(
                    0.45 * logit(p_pair) + 0.25 * logit(p_lang)
                    + 0.20 * logit(p_method) + 0.10 * logit(cluster_rate[cluster])
                ))
            prior[row_index] = float(q @ per_cluster)
    diagnostics = {
        "clusters": k,
        "train_people": len(train_people),
        "cluster_sizes": Counter(map(int, train_labels)),
        "global_rate": global_rate,
        "soft_assignment_mean_entropy": float(np.mean(-np.sum(responsibilities * np.log(responsibilities + 1e-12), axis=1))),
    }
    diagnostics["cluster_sizes"] = {str(key): value for key, value in sorted(diagnostics["cluster_sizes"].items())}
    return prior, responsibilities, diagnostics


def arm_geometry(rows: list[dict[str, Any]], indices: np.ndarray, endpoint: str) -> np.ndarray:
    numeric = []
    for index in indices:
        predictions = rows[int(index)]["predictions"]
        values = []
        # Reward-topology views across endpoints are more useful than raw prompt
        # text, which mostly clusters by script/language.
        for ep in ENDPOINTS:
            for model in (
                "surrogate_prior", "pc2_static_language_router", "entity_language_router",
                "intent_technique_router", "intent_iterative_combined_router", "joint_plus_surrogate",
            ):
                values.append(float(logit(predictions[ep].get(model, 0.5))))
        numeric.append(values)
    numeric = np.asarray(numeric, dtype=np.float64)
    numeric -= numeric.mean(axis=0, keepdims=True)
    scale = numeric.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    numeric /= scale
    methods = sorted({rows[int(index)]["attack_type"] for index in indices})
    scripts = sorted({rows[int(index)]["language_code"].split("_")[-1] for index in indices})
    categorical = np.zeros((len(indices), len(methods) + len(scripts)), dtype=np.float64)
    for local, index in enumerate(indices):
        row = rows[int(index)]
        categorical[local, methods.index(row["attack_type"])] = 2.0
        categorical[local, len(methods) + scripts.index(row["language_code"].split("_")[-1])] = 0.75
    return unit_rows(np.concatenate([numeric, categorical], axis=1))


def fixed_replay(order: np.ndarray, y: np.ndarray, budgets: tuple[int, ...]) -> dict[int, int]:
    found = 0
    result = {}
    for step, arm in enumerate(order[: max(budgets)], 1):
        found = max(found, int(y[int(arm)]))
        if step in budgets:
            result[step] = found
    return result


def triple_cluster_order(prior: np.ndarray, geometry: np.ndarray, seed: int) -> np.ndarray:
    """A budget-agnostic TRIPLE-CLST analogue for one-observation arms.

    Pull the strongest representative from each reward-topology cluster first,
    ranked by a top-two cluster score, then pull all remaining arms by prior.
    """
    labels, _ = spherical_kmeans(geometry, int(math.ceil(math.sqrt(len(prior)))), seed)
    representatives = []
    for cluster in sorted(set(map(int, labels))):
        members = np.flatnonzero(labels == cluster)
        ordered = members[np.argsort(-prior[members], kind="stable")]
        top = prior[ordered[: min(2, len(ordered))]]
        representatives.append((float(top.mean()), int(ordered[0])))
    first = [arm for _, arm in sorted(representatives, reverse=True)]
    remaining = [int(arm) for arm in np.argsort(-prior, kind="stable") if int(arm) not in set(first)]
    return np.asarray(first + remaining, dtype=int)


def arm_topology(geometry: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels, _ = spherical_kmeans(geometry, int(math.ceil(math.sqrt(len(geometry)))), seed)
    similarity = np.maximum(geometry @ geometry.T, 0.0) ** 2
    return labels, similarity


def ccb_replay(
    prior: np.ndarray, geometry: np.ndarray, y: np.ndarray, budgets: tuple[int, ...],
    cluster_weight: float, kernel_weight: float, novelty_weight: float, seed: int,
    topology: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[int, int]:
    """Clustered contextual BAI with residual propagation after each pull."""
    labels, similarity = topology if topology is not None else arm_topology(geometry, seed)
    tried = np.zeros(len(prior), dtype=bool)
    observations: list[int] = []
    found = 0
    result = {}
    base = np.asarray(logit(prior), dtype=float)
    for step in range(1, max(budgets) + 1):
        score = base.copy()
        if observations:
            obs = np.asarray(observations, dtype=int)
            residual = y[obs] - prior[obs]
            curvature = np.maximum(prior[obs] * (1.0 - prior[obs]), 0.05)
            for arm in range(len(prior)):
                same = (labels[obs] == labels[arm]).astype(float)
                score[arm] += cluster_weight * float(np.sum(same * residual) / (1.5 + np.sum(same * curvature)))
                weights = similarity[arm, obs]
                score[arm] += kernel_weight * float(np.sum(weights * residual) / (1.5 + np.sum(weights * curvature)))
                # When high-prior neighbours fail, explicitly sample another
                # region before spending the entire budget in that basin.
                novelty = 1.0 - float(np.max(similarity[arm, obs]))
                score[arm] += novelty_weight * novelty
        score[tried] = -np.inf
        arm = int(np.argmax(score))
        tried[arm] = True
        observations.append(arm)
        found = max(found, int(y[arm]))
        if step in budgets:
            result[step] = found
    return result


def evaluate_config(
    rows: list[dict[str, Any]], people: list[Person], person_indices: np.ndarray,
    endpoint: str, base_model: str, context_prior: np.ndarray, blend: float,
    cluster_weight: float, kernel_weight: float, novelty_weight: float,
    budgets: tuple[int, ...], seed: int,
    geometries: dict[int, np.ndarray], topologies: dict[int, tuple[np.ndarray, np.ndarray]],
) -> float:
    values = []
    for person_index in person_indices:
        person = people[int(person_index)]
        indices = np.asarray(person.rows)
        base = np.asarray([rows[i]["predictions"][endpoint][base_model] for i in indices])
        prior = np.asarray(sigmoid((1.0 - blend) * logit(base) + blend * logit(context_prior[indices])))
        y = np.asarray([rows[i]["labels"][endpoint] for i in indices], dtype=int)
        geometry = geometries[int(person_index)]
        replay = ccb_replay(prior, geometry, y, budgets, cluster_weight, kernel_weight, novelty_weight,
                            stable_seed(person.person_id, seed), topologies[int(person_index)])
        values.append(np.mean([replay[b] for b in budgets if b <= 5]))
    return float(np.mean(values))


def evaluate_policy_triplet(
    rows: list[dict[str, Any]], people: list[Person], person_indices: np.ndarray,
    endpoint: str, base_model: str, context_prior: np.ndarray, blend: float,
    cluster_weight: float, kernel_weight: float, novelty_weight: float,
    budgets: tuple[int, ...], seed: int,
    geometries: dict[int, np.ndarray], topologies: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[str, dict[int, float]]:
    buckets = {
        policy: {budget: [] for budget in budgets}
        for policy in ("context_cluster_static", "triple_clst_adapted", "ccb_adaptive")
    }
    for person_index in person_indices:
        person = people[int(person_index)]
        indices = np.asarray(person.rows)
        base = np.asarray([rows[i]["predictions"][endpoint][base_model] for i in indices])
        prior = np.asarray(sigmoid((1.0 - blend) * logit(base) + blend * logit(context_prior[indices])))
        y = np.asarray([rows[i]["labels"][endpoint] for i in indices], dtype=int)
        geometry = geometries[int(person_index)]
        replays = {
            "context_cluster_static": fixed_replay(np.argsort(-prior, kind="stable"), y, budgets),
            "triple_clst_adapted": fixed_replay(
                triple_cluster_order(prior, geometry, stable_seed(person.person_id, seed)), y, budgets
            ),
            "ccb_adaptive": ccb_replay(
                prior, geometry, y, budgets, cluster_weight, kernel_weight, novelty_weight,
                stable_seed(person.person_id, seed), topologies[int(person_index)],
            ),
        }
        for policy, replay in replays.items():
            for budget in budgets:
                buckets[policy][budget].append(replay[budget])
    return {
        policy: {budget: float(np.mean(values)) for budget, values in by_budget.items()}
        for policy, by_budget in buckets.items()
    }


def bootstrap_ci(values: np.ndarray, seed: int, samples: int = 4000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--person-embeddings", type=Path, required=True)
    parser.add_argument("--context-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(range(1, 11)))
    args = parser.parse_args()
    budgets = tuple(sorted(set(args.budgets)))
    if budgets[0] < 1 or budgets[-1] > 10:
        raise ValueError("budgets must be within the ten observed settings")

    rows = read_jsonl(args.oof)
    people = group_people(rows)
    features = load_context_features(rows, people, args.person_embeddings, args.context_embeddings, args.seed)
    person_index = {person.person_id: i for i, person in enumerate(people)}
    row_person = np.asarray([person_index[row["victim_person_id"]] for row in rows], dtype=int)
    # Arm geometry is endpoint-independent and outcome-free; cache it once.
    geometries = {
        i: arm_geometry(rows, np.asarray(person.rows), "asr") for i, person in enumerate(people)
    }
    topologies = {
        i: arm_topology(geometries[i], stable_seed(person.person_id, args.seed))
        for i, person in enumerate(people)
    }
    all_results: dict[str, Any] = {}
    audit: dict[str, Any] = {}

    for endpoint in ENDPOINTS:
        policy_person_budget: dict[str, dict[int, list[int]]] = {
            policy: {budget: [] for budget in budgets} for policy in POLICIES
        }
        fold_audit = []
        for outer_fold in sorted({person.fold for person in people}):
            train_people = np.asarray([i for i, p in enumerate(people) if p.fold != outer_fold])
            test_people = np.asarray([i for i, p in enumerate(people) if p.fold == outer_fold])
            base_model = nested_best_model(rows, people, train_people, endpoint, budgets[:5])

            # Tune only on outer-training folds.  Each validation fold receives
            # a context prior fitted without that fold's outcomes.
            configs = [
                (blend, cw, kw, nw)
                for blend in (0.0, 0.5)
                for cw in (0.5, 1.5)
                for kw in (0.0, 0.75)
                for nw in (0.0, 0.15)
            ]
            config_scores: dict[tuple[float, float, float, float], list[float]] = defaultdict(list)
            inner_folds = sorted({people[int(i)].fold for i in train_people})
            for inner_fold in inner_folds:
                inner_train = np.asarray([i for i in train_people if people[int(i)].fold != inner_fold])
                inner_valid = np.asarray([i for i in train_people if people[int(i)].fold == inner_fold])
                inner_prior, _, _ = fit_context_cluster_prior(
                    rows, people, features, inner_train, endpoint, 16, 12.0,
                    args.seed + outer_fold * 100 + inner_fold,
                )
                for config in configs:
                    config_scores[config].append(evaluate_config(
                        rows, people, inner_valid, endpoint, base_model, inner_prior, *config,
                        budgets, args.seed, geometries, topologies,
                    ))
            chosen = max(configs, key=lambda config: (float(np.mean(config_scores[config])), tuple(-x for x in config)))
            # The query budget is known before an API run.  Select the best of
            # the static, cluster-representative, and feedback controllers for
            # each budget using only inner validation folds.
            controller_scores = {
                policy: {budget: [] for budget in budgets}
                for policy in ("context_cluster_static", "triple_clst_adapted", "ccb_adaptive")
            }
            for inner_fold in inner_folds:
                inner_train = np.asarray([i for i in train_people if people[int(i)].fold != inner_fold])
                inner_valid = np.asarray([i for i in train_people if people[int(i)].fold == inner_fold])
                inner_prior, _, _ = fit_context_cluster_prior(
                    rows, people, features, inner_train, endpoint, 16, 12.0,
                    args.seed + outer_fold * 100 + inner_fold,
                )
                triplet = evaluate_policy_triplet(
                    rows, people, inner_valid, endpoint, base_model, inner_prior, *chosen,
                    budgets, args.seed, geometries, topologies,
                )
                for policy, by_budget in triplet.items():
                    for budget, value in by_budget.items():
                        controller_scores[policy][budget].append(value)
            controller_by_budget = {
                budget: max(
                    controller_scores,
                    key=lambda policy: (float(np.mean(controller_scores[policy][budget])), -list(controller_scores).index(policy)),
                )
                for budget in budgets
            }
            context_prior, _, context_diagnostics = fit_context_cluster_prior(
                rows, people, features, train_people, endpoint, 16, 12.0,
                args.seed + outer_fold,
            )
            fold_audit.append({
                "fold": outer_fold,
                "train_people": len(train_people),
                "test_people": len(test_people),
                "base_model": base_model,
                "chosen": {"blend": chosen[0], "cluster_weight": chosen[1], "kernel_weight": chosen[2], "novelty_weight": chosen[3]},
                "inner_score": float(np.mean(config_scores[chosen])),
                "controller_by_budget": {str(key): value for key, value in controller_by_budget.items()},
                "context_clusters": context_diagnostics,
            })
            print(
                json.dumps({"endpoint": endpoint, "outer_fold": outer_fold, "base_model": base_model,
                            "chosen": chosen, "inner_score": float(np.mean(config_scores[chosen]))}),
                flush=True,
            )
            blend, cw, kw, nw = chosen
            for pi in test_people:
                person = people[int(pi)]
                indices = np.asarray(person.rows)
                y = np.asarray([rows[i]["labels"][endpoint] for i in indices], dtype=int)
                base = np.asarray([rows[i]["predictions"][endpoint][base_model] for i in indices])
                surrogate = np.asarray([rows[i]["predictions"][endpoint]["surrogate_prior"] for i in indices])
                prior = np.asarray(sigmoid((1.0 - blend) * logit(base) + blend * logit(context_prior[indices])))
                geometry = geometries[int(pi)]
                rng = np.random.default_rng(stable_seed(person.person_id + endpoint, args.seed))
                replays = {
                    "random": fixed_replay(rng.permutation(10), y, budgets),
                    "surrogate_static": fixed_replay(np.argsort(-surrogate, kind="stable"), y, budgets),
                    "nested_best_router_static": fixed_replay(np.argsort(-base, kind="stable"), y, budgets),
                    "context_cluster_static": fixed_replay(np.argsort(-prior, kind="stable"), y, budgets),
                    "triple_clst_adapted": fixed_replay(
                        triple_cluster_order(prior, geometry, stable_seed(person.person_id, args.seed)), y, budgets
                    ),
                    "ccb_adaptive": ccb_replay(
                        prior, geometry, y, budgets, cw, kw, nw, stable_seed(person.person_id, args.seed),
                        topologies[int(pi)],
                    ),
                }
                replays["budget_aware_clustered_bai"] = {
                    budget: replays[controller_by_budget[budget]][budget] for budget in budgets
                }
                for policy, replay in replays.items():
                    for budget in budgets:
                        policy_person_budget[policy][budget].append(replay[budget])

        endpoint_result: dict[str, Any] = {}
        oracle_values = np.asarray([
            int(any(rows[i]["labels"][endpoint] for i in person.rows)) for person in people
        ], dtype=float)
        oracle = float(oracle_values.mean())
        for policy in POLICIES:
            endpoint_result[policy] = {}
            for budget in budgets:
                values = np.asarray(policy_person_budget[policy][budget], dtype=float)
                low, high = bootstrap_ci(values, stable_seed(endpoint + policy + str(budget), args.seed))
                endpoint_result[policy][str(budget)] = {
                    "success_discovery": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "gap_to_oracle_pp": 100.0 * (oracle - float(values.mean())),
                }
        all_results[endpoint] = {"oracle_at_10": oracle, "policies": endpoint_result}
        audit[endpoint] = fold_audit

    result = {
        "schema": "jailnews_clustered_contextual_bai/v3",
        "objective": "finite-catalog success discovery over ten observed settings per person",
        "not_claimed": "stochastic best-mean-arm identification over all 360 settings",
        "people": len(people),
        "rows": len(rows),
        "budgets": budgets,
        "policies": {
            "random": "one deterministic random permutation per person",
            "surrogate_static": "original transferable surrogate prior",
            "nested_best_router_static": "base router selected using outer-training people only",
            "context_cluster_static": "soft person/context-cluster reward prior blended with the base router",
            "triple_clst_adapted": "reward-topology cluster representatives first, then remaining arms by prior",
            "ccb_adaptive": "context-cluster prior plus online cluster/kernel residual propagation and novelty",
            "budget_aware_clustered_bai": "outer-training validation chooses static, cluster-representative, or adaptive control separately for each known budget",
        },
        "leakage_controls": [
            "target outcomes for a held-out outer fold never enter its context-cluster prior",
            "base-router selection and hyperparameter tuning use outer-training folds only",
            "test outcomes are exposed to the adaptive policy only after the corresponding setting is selected",
            "prompt text and generated response text are not used as clustering features",
        ],
        "results": all_results,
        "fold_audit": audit,
    }
    write_json(args.output, result)

    lines = [
        "# Clustered contextual BAI replay (v3)", "",
        "This evaluates success discovery over the ten observed target settings per person; it is not a 360-arm stochastic BAI claim.", "",
    ]
    for endpoint in ENDPOINTS:
        lines.extend([
            f"## {endpoint}", "",
            f"Observed oracle@10: {all_results[endpoint]['oracle_at_10']:.1%}", "",
            "| policy | B=1 | B=2 | B=3 | B=5 | B=10 | gap@3 (pp) |", "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for policy in POLICIES:
            metrics = all_results[endpoint]["policies"][policy]
            lines.append(
                f"| {policy} | {metrics['1']['success_discovery']:.1%} | {metrics['2']['success_discovery']:.1%} | "
                f"{metrics['3']['success_discovery']:.1%} | {metrics['5']['success_discovery']:.1%} | "
                f"{metrics['10']['success_discovery']:.1%} | {metrics['3']['gap_to_oracle_pp']:.1f} |"
            )
        lines.append("")
    lines.extend(["## Chosen fold configurations", "", "```json", json.dumps(audit, ensure_ascii=False, indent=2), "```", ""])
    args.output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report": str(args.output.with_suffix('.md'))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
