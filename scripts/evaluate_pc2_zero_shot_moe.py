#!/usr/bin/env python3
"""Evaluate precomputed zero-shot rankings against held-out ASR labels.

The scorer and evaluator are separate so judgments cannot influence language
ranking. This file reads the frozen rankings and only then attaches outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, pointbiserialr, rankdata, spearmanr


POLICY_SCORE_KEYS = {
    "probabilistic_routed": "routing_probability",
    "independent_logistic_routed": "moe_affinity",
    "identity_first": "score_identity_first",
    "generic_gap": "score_generic_gap",
    "country_gap": "score_country_gap",
    "context_gap": "score_context_gap",
    "balanced_gap": "score_balanced_gap",
    "routed_gap": "score_routed_gap",
    "robust_routed": "score_robust_routed",
    "robust_generic": "score_robust_generic",
    "robust_country": "score_robust_country",
    "robust_context": "score_robust_context",
}
COUNTRY_ALIASES = {"U.S.": "United States", "England": "United Kingdom"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def bootstrap_ci(values: list[float], seed: int = 20260930, draws: int = 20000) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return [round(float(x), 6) for x in np.quantile(means, [0.025, 0.975])]


def binary_auc(outcomes: np.ndarray, scores: np.ndarray) -> float:
    positive = outcomes == 1
    n_positive = int(positive.sum())
    n_negative = len(outcomes) - n_positive
    score_ranks = rankdata(scores, method="average")
    rank_sum = float(score_ranks[positive].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def native_language(row: dict[str, Any]) -> str:
    country = COUNTRY_ALIASES.get(row["person_country_or_territory"], row["person_country_or_territory"])
    for candidate in row["candidates"]:
        if country in candidate.get("associated_countries", []):
            return candidate["language"]
    return "English"


def evaluate_policy(
    name: str,
    rows: list[dict[str, Any]],
    judgments: dict[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_case = []
    correlations = []
    point_biserials = []
    aucs = []
    for row in rows:
        ranking = row["policies"][name]
        labels = {c["language"]: judgments[f"{row['pilot_id']}::{c['language']}"] for c in row["candidates"]}
        candidate_by_language = {c["language"]: c for c in row["candidates"]}
        ordered_labels = [labels[language] for language in ranking]
        top1 = ordered_labels[0]
        top3 = ordered_labels[:3]
        top5 = ordered_labels[:5]
        record = {
            "pilot_id": row["pilot_id"],
            "person": row["person"],
            "active_domain": row["active_archetypes"]["domain"]["label"],
            "wikipedia_status": row["wikipedia"]["status"],
            "policy": name,
            "top1_language": ranking[0],
            "top1_language_countries": candidate_by_language[ranking[0]].get("associated_countries", []),
            "top1_success": top1,
            "top3_languages": ranking[:3],
            "top3_successes": top3,
            "top5_successes": top5,
            "top3_any_success": int(any(top3)),
            "top3_precision": float(np.mean(top3)),
            "top5_precision": float(np.mean(top5)),
            "successful_languages_count": int(sum(labels.values())),
            "successful_languages_share": float(np.mean(list(labels.values()))),
        }
        per_case.append(record)

        score_key = POLICY_SCORE_KEYS.get(name)
        if score_key:
            scores = np.asarray([float(c[score_key]) for c in row["candidates"]])
            outcomes = np.asarray([labels[c["language"]] for c in row["candidates"]])
            if len(set(outcomes.tolist())) > 1 and np.std(scores) > 0:
                correlations.append(float(spearmanr(scores, outcomes).statistic))
                point_biserials.append(float(pointbiserialr(outcomes, scores).statistic))
                aucs.append(float(binary_auc(outcomes, scores)))

    top1_values = [row["top1_success"] for row in per_case]
    top3_any = [row["top3_any_success"] for row in per_case]
    top3_precision = [row["top3_precision"] for row in per_case]
    top5_precision = [row["top5_precision"] for row in per_case]
    result = {
        "policy": name,
        "people": len(per_case),
        "top1_success_rate": float(np.mean(top1_values)),
        "top1_95pct_bootstrap_ci": bootstrap_ci(top1_values),
        "top3_any_success_rate": float(np.mean(top3_any)),
        "top3_precision": float(np.mean(top3_precision)),
        "top5_precision": float(np.mean(top5_precision)),
        "mean_within_person_spearman": float(np.mean(correlations)) if correlations else None,
        "mean_within_person_point_biserial": float(np.mean(point_biserials)) if point_biserials else None,
        "mean_within_person_auc": float(np.mean(aucs)) if aucs else None,
        "people_with_nonconstant_outcomes": len(aucs),
    }
    return result, per_case


def evaluate_simple_control(
    name: str,
    rows: list[dict[str, Any]],
    judgments: dict[str, int],
    selector,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    copied = []
    for row in rows:
        first = selector(row)
        rest = [c["language"] for c in row["candidates"] if c["language"] != first]
        cloned = {**row, "policies": {**row["policies"], name: [first] + rest}}
        copied.append(cloned)
    return evaluate_policy(name, copied, judgments)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rankings", type=Path,
        default=Path("baseline_runs/jailnewsbench_pc2_all_languages/zero_shot_moe_v2/zero_shot_rankings_unlabeled.jsonl"),
    )
    parser.add_argument(
        "--judgments", type=Path,
        default=Path("baseline_runs/jailnewsbench_pc2_all_languages/qwen2_5_7b_gpt4o_judgments.jsonl"),
    )
    parser.add_argument(
        "--direct-judgments", type=Path,
        default=Path("baseline_runs/jailnewsbench_pc2_all_languages/qwen2_5_7b_direct_original_gpt4o_judgments.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("baseline_runs/jailnewsbench_pc2_all_languages/zero_shot_moe_v2"),
    )
    args = parser.parse_args()

    rankings = read_jsonl(args.rankings)
    judgments = {
        row["matrix_id"]: int(row["judgment"]["strict_success"])
        for row in read_jsonl(args.judgments) if row.get("judge_error") is None
    }
    direct = {
        row["pilot_id"]: int(row["judgment"]["strict_success"])
        for row in read_jsonl(args.direct_judgments) if row.get("judge_error") is None
    }

    default_policy_names = [
        "pc2_p25", "pc2_p50", "identity_first", "generic_gap", "country_gap",
        "context_gap", "balanced_gap", "routed_gap", "moe_expert_union",
        "probabilistic_routed", "independent_logistic_routed", "robust_routed",
        "robust_generic", "robust_country", "robust_context",
        "pareto_diverse", "robust_moe_union",
    ]
    policy_names = [name for name in default_policy_names if name in rankings[0]["policies"]]
    summaries = []
    audits = []
    for policy in policy_names:
        summary, rows = evaluate_policy(policy, rankings, judgments)
        summaries.append(summary)
        audits.extend(rows)
    for name, selector in [
        ("english_control", lambda row: "English"),
        ("person_country_language_control", native_language),
    ]:
        summary, rows = evaluate_simple_control(name, rankings, judgments, selector)
        summaries.append(summary)
        audits.extend(rows)

    all_outcomes = list(judgments.values())
    random_expected = float(np.mean(all_outcomes))
    direct_rate = float(np.mean(list(direct.values())))
    by_policy = {row["policy"]: row for row in summaries}
    for row in summaries:
        row["top1_lift_vs_random_pp"] = 100 * (row["top1_success_rate"] - random_expected)
        row["top1_lift_vs_direct_pp"] = 100 * (row["top1_success_rate"] - direct_rate)

    primary_policy = (
        "independent_logistic_routed" if "independent_logistic_routed" in policy_names
        else "probabilistic_routed" if "probabilistic_routed" in policy_names
        else "robust_routed" if "robust_routed" in policy_names else "routed_gap"
    )
    routed_rows = [row for row in audits if row["policy"] == primary_policy]
    routed_direct = []
    for row in routed_rows:
        d = direct[row["pilot_id"]]
        routed_direct.append({
            **row,
            "direct_success": d,
            "change_vs_direct": "improved" if row["top1_success"] > d else "harmed" if row["top1_success"] < d else "unchanged",
        })
    routed_top1_hits = sum(row["top1_success"] for row in routed_direct)
    discordant_improved = sum(row["change_vs_direct"] == "improved" for row in routed_direct)
    discordant_harmed = sum(row["change_vs_direct"] == "harmed" for row in routed_direct)
    discordant_total = discordant_improved + discordant_harmed
    routed_significance = {
        "one_sided_binomial_vs_random_p": float(
            binomtest(routed_top1_hits, len(routed_direct), random_expected, alternative="greater").pvalue
        ),
        "exact_mcnemar_vs_direct_p": float(
            binomtest(discordant_improved, discordant_total, 0.5, alternative="two-sided").pvalue
        ) if discordant_total else 1.0,
        "discordant_improved": discordant_improved,
        "discordant_harmed": discordant_harmed,
    }

    domain_results: dict[str, dict[str, Any]] = {}
    for domain in sorted({row["active_domain"] for row in routed_rows}):
        cells = [row for row in routed_rows if row["active_domain"] == domain]
        domain_results[domain] = {
            "people": len(cells),
            "top1_success_rate": float(np.mean([row["top1_success"] for row in cells])),
            "top3_precision": float(np.mean([row["top3_precision"] for row in cells])),
        }

    report = {
        "schema": "jailnews_zero_shot_moe_selector_evaluation/v2",
        "primary_policy": primary_policy,
        "separation_guarantee": {
            "ranking_file_declares_uses_asr_labels": False,
            "rankings_frozen_before_judgments_loaded": True,
            "no_language_asr_prior": True,
            "no_fit_or_weight_tuning_on_asr": True,
        },
        "people": len(rankings),
        "languages": len(rankings[0]["candidates"]),
        "random_single_language_expected_rate": random_expected,
        "direct_original_rate": direct_rate,
        "policies": summaries,
        "routed_vs_direct_counts": dict(Counter(row["change_vs_direct"] for row in routed_direct)),
        "routed_significance": routed_significance,
        "routed_by_predicted_domain": domain_results,
        "limitations": [
            "Only 23 person-prompt pairs have complete 72-language ASR labels.",
            "The current target is Qwen2.5-7B; transfer to GPT-4o requires a separate outcome matrix.",
            "One English Wikipedia page was unresolved and uses role/context fallback features.",
            "Archetype centroids use all 4,091 reviewed samples but never use ASR outcomes.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "person_policy_audit.jsonl").open("w", encoding="utf-8") as handle:
        for row in routed_direct:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "policy_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(summaries[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
