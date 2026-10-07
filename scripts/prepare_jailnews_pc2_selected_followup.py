#!/usr/bin/env python3
"""Select PC2 top-1, observed-oracle, and random-control arms for repeats."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from evaluate_jailnews_pc2_full501 import (
    DEFAULT_CONTEXT_EMBEDDINGS,
    DEFAULT_INTENT_REASONING,
    DEFAULT_LOCALIZATIONS,
    DEFAULT_PERSON_EMBEDDINGS,
    DEFAULT_WIKIPEDIA_CACHE,
    COLD_START_PRIOR_WEIGHTS,
    METHODS,
    arm_graph,
    blend_probabilities,
    combine_probability,
    domain_method_probability,
    effective_candidates,
    iter_jsonl,
    load_external_pc2_prior,
    load_surrogate,
    load_target,
    loo_probability,
    stable_seed,
)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", nargs="+", required=True)
    parser.add_argument("--surrogate", nargs="+", required=True)
    parser.add_argument("--base-arms", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--person-embeddings", type=Path, default=DEFAULT_PERSON_EMBEDDINGS)
    parser.add_argument("--context-embeddings", type=Path, default=DEFAULT_CONTEXT_EMBEDDINGS)
    parser.add_argument("--wikipedia-cache", type=Path, default=DEFAULT_WIKIPEDIA_CACHE)
    parser.add_argument("--localizations", type=Path, default=DEFAULT_LOCALIZATIONS)
    parser.add_argument("--intent-reasoning", type=Path, default=DEFAULT_INTENT_REASONING)
    parser.add_argument("--seed", type=int, default=20261100)
    parser.add_argument("--judge-error-policy", choices=["fail", "zero"], default="fail")
    args = parser.parse_args()

    target = load_target(args.target, "strict", args.judge_error_policy)
    outcome = target["outcome"].astype(np.float64)
    arms = target["arms"]
    people = target["people"]
    languages = sorted({arm[0] for arm in arms})
    methods = [method for method in METHODS if method in {arm[1] for arm in arms}]
    graph, language_index, method_index = arm_graph(arms, languages, methods)
    candidates, equivalence_audit = effective_candidates(target["prompt_hash"], arms)
    external_prior, _, _, external_audit = load_external_pc2_prior(
        target, languages, methods, language_index, method_index,
        args.person_embeddings, args.context_embeddings, args.wikipedia_cache,
        args.localizations, args.intent_reasoning,
    )
    graph_prior = loo_probability(outcome, np.ones_like(outcome), graph)
    domain_prior = domain_method_probability(outcome, target["domains"], method_index, methods)
    structural_prior = combine_probability(graph_prior, domain_prior, 0.70)
    surrogate_prior, _, surrogate_audit = load_surrogate(
        args.surrogate, target, args.judge_error_policy
    )
    fallback = loo_probability(outcome, np.ones_like(outcome), None).mean(axis=0, keepdims=True)
    surrogate_prior = np.where(np.isfinite(surrogate_prior), surrogate_prior, fallback)
    full_prior = blend_probabilities(
        [external_prior, structural_prior, surrogate_prior], list(COLD_START_PRIOR_WEIGHTS)
    )

    selections: dict[tuple[str, tuple[str, str]], set[str]] = defaultdict(set)
    metadata: dict[tuple[str, tuple[str, str]], dict] = {}
    people_with_oracle = 0
    for person_index, person in enumerate(people):
        available = candidates[person_index]
        order = available[np.argsort(-full_prior[person_index, available], kind="stable")]
        top = int(order[0])
        selections[(person, arms[top])].add("pc2_top1")
        metadata[(person, arms[top])] = {
            "pc2_prior": float(full_prior[person_index, top]),
            "external_pc2_prior": float(external_prior[person_index, top]),
            "draw0_strict": int(outcome[person_index, top]),
        }

        successful = available[outcome[person_index, available] == 1]
        oracle = None
        if len(successful):
            people_with_oracle += 1
            oracle = int(successful[np.argmax(full_prior[person_index, successful])])
            selections[(person, arms[oracle])].add("observed_oracle")
            metadata[(person, arms[oracle])] = {
                "pc2_prior": float(full_prior[person_index, oracle]),
                "external_pc2_prior": float(external_prior[person_index, oracle]),
                "draw0_strict": 1,
            }

        excluded = {top}
        if oracle is not None:
            excluded.add(oracle)
        pool = np.asarray([arm for arm in available if int(arm) not in excluded], dtype=int)
        if not len(pool):
            pool = available
        rng = np.random.default_rng(stable_seed(args.seed, person, "random_control"))
        control = int(rng.choice(pool))
        selections[(person, arms[control])].add("random_control")
        metadata[(person, arms[control])] = {
            "pc2_prior": float(full_prior[person_index, control]),
            "external_pc2_prior": float(external_prior[person_index, control]),
            "draw0_strict": int(outcome[person_index, control]),
        }

    output_rows = []
    found = set()
    for row in iter_jsonl([args.base_arms]):
        key = (
            str(row["victim_person_id"]),
            (str(row["language"]), str(row["attack_type"])),
        )
        if key not in selections:
            continue
        found.add(key)
        output_rows.append({
            **row,
            "followup_roles": sorted(selections[key]),
            "followup_selection": metadata[key],
            "followup_selection_contract": "pc2_external_top1__observed_oracle__random_control/v2",
        })
    missing = sorted(set(selections) - found)
    if missing:
        raise RuntimeError(f"selected arms missing from base matrix: {missing[:5]}")
    output_rows.sort(key=lambda row: (row["victim_person_id"], row["arm_id"]))
    atomic_jsonl(args.output, output_rows)
    role_counts = Counter(role for roles in selections.values() for role in roles)
    summary = {
        "schema": "jailnews_pc2_selected_followup/v2",
        "people": len(people),
        "unique_selected_arms": len(output_rows),
        "people_with_observed_oracle": people_with_oracle,
        "role_counts": dict(role_counts),
        "selection_seed": args.seed,
        "selection_uses_target_outcomes": {"pc2_top1": False, "observed_oracle": True, "random_control": False},
        "equivalence_audit": equivalence_audit,
        "surrogate_audit": surrogate_audit,
        "external_pc2_audit": external_audit,
        "prior_weights": {
            "external": COLD_START_PRIOR_WEIGHTS[0],
            "target_population_loo": COLD_START_PRIOR_WEIGHTS[1],
            "surrogate": COLD_START_PRIOR_WEIGHTS[2],
        },
        "output": str(args.output.resolve()),
    }
    atomic_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
