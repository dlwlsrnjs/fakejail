#!/usr/bin/env python3
"""Test whether PC2-style person-language features predict strict attack success."""

from __future__ import annotations

import argparse
import json
import math
import os
import runpy
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pointbiserialr, spearmanr, wilcoxon


SOURCE_LANGUAGE = {
    "cs": "Czech", "de": "German", "en": "English", "es": "Spanish",
    "fr": "French", "hu": "Hungarian", "lt": "Lithuanian", "lv": "Latvian",
    "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "sk": "Slovak", "zh": "Mandarin Chinese",
}
COUNTRY_ALIASES = {"U.S.": "United States", "England": "United Kingdom"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def pct(a: int | float, b: int | float) -> float | None:
    return round(100 * a / b, 2) if b else None


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (center - half), 2), round(100 * (center + half), 2)]


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 2.0) -> np.ndarray:
    x1 = np.column_stack([np.ones(len(x)), x])

    def objective(w):
        z = np.clip(x1 @ w, -30, 30)
        loss = np.logaddexp(0, z).sum() - np.dot(y, z)
        return loss + 0.5 * l2 * np.dot(w[1:], w[1:])

    def gradient(w):
        z = np.clip(x1 @ w, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = x1.T @ (p - y)
        g[1:] += l2 * w[1:]
        return g

    result = minimize(objective, np.zeros(x1.shape[1]), jac=gradient, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def predict_logistic(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    z = np.clip(np.column_stack([np.ones(len(x)), x]) @ w, -30, 30)
    return 1 / (1 + np.exp(-z))


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(0)
    std = train.std(0)
    std[std < 1e-8] = 1
    return (train - mean) / std, (test - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--pc2-languages-py", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    country_to_language = runpy.run_path(str(args.pc2_languages_py))["language_dict"]
    score_rows = read_jsonl(args.scores)
    scores = {
        (row["pilot_id"], candidate["language"]): {**candidate, "country": row["person_country_or_territory"]}
        for row in score_rows for candidate in row["candidates"]
    }
    matrix = {row["matrix_id"]: row for row in read_jsonl(args.matrix)}
    judgments = {
        row["matrix_id"]: row for row in read_jsonl(args.judgments)
        if row.get("judge_error") is None
    }

    rows = []
    for matrix_id, source in matrix.items():
        judge = judgments[matrix_id]["judgment"]
        score = scores[(source["pilot_id"], source["clue_language"])]
        country = COUNTRY_ALIASES.get(score["country"], score["country"])
        native = country_to_language.get(country)
        nllb_script = (source.get("nllb_code") or "_").split("_")[-1]
        native_script = ""
        if native:
            native_code = next(
                (c["nllb_code"] for (p, l), c in scores.items() if l == native and c.get("nllb_code")),
                "_",
            )
            native_script = native_code.split("_")[-1]
        source_language = SOURCE_LANGUAGE.get(source["source_language_code"])
        english_clue = next(r["english_ipdm"] for r in score_rows if r["pilot_id"] == source["pilot_id"])
        rows.append({
            "matrix_id": matrix_id,
            "pilot_id": source["pilot_id"],
            "person": source["selected_person"],
            "country": country,
            "language": source["clue_language"],
            "script": nllb_script,
            "native_language": native,
            "source_language": source_language,
            "strict_success": int(judge["strict_success"]),
            "entity_match": int(judge["entity_match"]),
            "adherence_score": judge["adherence_score"],
            "backtranslation_similarity": score["backtranslation_similarity"],
            "keyword_bias": score["keyword_bias"],
            "politics": score["politics"],
            "country_common_knowledge": score["country_common_knowledge"],
            "keyword_common_knowledge": score["keyword_common_knowledge"],
            "combined_score": score["combined_score"],
            "translation_valid": int(score["valid"]),
            "native_language_match": int(native == source["clue_language"]),
            "source_language_match": int(source_language == source["clue_language"]),
            "native_script_match": int(bool(native_script) and native_script == nllb_script),
            "translation_length_ratio": len(source["clue"]) / max(1, len(english_clue)),
        })

    people = sorted({row["pilot_id"] for row in rows})
    languages = sorted({row["language"] for row in rows})
    numeric = [
        "backtranslation_similarity", "keyword_bias", "politics",
        "country_common_knowledge", "keyword_common_knowledge", "combined_score",
        "translation_valid", "native_language_match", "source_language_match",
        "native_script_match", "translation_length_ratio",
    ]
    lang_index = {lang: i for i, lang in enumerate(languages)}
    x_num = np.asarray([[row[k] for k in numeric] for row in rows], dtype=float)
    x_lang = np.zeros((len(rows), len(languages)))
    for i, row in enumerate(rows):
        x_lang[i, lang_index[row["language"]]] = 1
    y = np.asarray([row["strict_success"] for row in rows], dtype=float)

    # Balanced two-way variance decomposition.
    grand = y.mean()
    total_ss = float(((y - grand) ** 2).sum())
    person_means = {p: np.mean([r["strict_success"] for r in rows if r["pilot_id"] == p]) for p in people}
    lang_means = {l: np.mean([r["strict_success"] for r in rows if r["language"] == l]) for l in languages}
    ss_person = len(languages) * sum((v - grand) ** 2 for v in person_means.values())
    ss_language = len(people) * sum((v - grand) ** 2 for v in lang_means.values())

    correlations = {}
    for key in numeric[:6] + ["translation_length_ratio"]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        overall = pointbiserialr(y, values)
        within_x = values - np.asarray([np.mean([r[key] for r in rows if r["pilot_id"] == row["pilot_id"]]) for row in rows])
        within_y = y - np.asarray([person_means[row["pilot_id"]] for row in rows])
        within = spearmanr(within_x, within_y)
        correlations[key] = {
            "overall_r": round(float(overall.statistic), 4), "overall_p": round(float(overall.pvalue), 6),
            "within_person_spearman": round(float(within.statistic), 4), "within_person_p": round(float(within.pvalue), 6),
        }

    def group_stats(selector):
        chosen = [row for row in rows if selector(row)]
        k = sum(row["strict_success"] for row in chosen)
        return {"n": len(chosen), "successes": k, "success_rate_pct": pct(k, len(chosen)), "wilson_95": wilson(k, len(chosen))}

    relation_groups = {
        "person_country_native_language": group_stats(lambda r: r["native_language_match"] == 1),
        "all_other_languages": group_stats(lambda r: r["native_language_match"] == 0),
        "source_article_language": group_stats(lambda r: r["source_language_match"] == 1),
        "not_source_article_language": group_stats(lambda r: r["source_language_match"] == 0),
        "same_script_as_person_country_language": group_stats(lambda r: r["native_script_match"] == 1),
        "different_script": group_stats(lambda r: r["native_script_match"] == 0),
    }

    paired_relation_effects = {}
    for flag, label in [
        ("native_script_match", "same_vs_different_native_script"),
        ("native_language_match", "native_language_vs_other_languages"),
        ("source_language_match", "source_language_vs_other_languages"),
    ]:
        effects = []
        detail = []
        for p in people:
            yes = [r["strict_success"] for r in rows if r["pilot_id"] == p and r[flag]]
            no = [r["strict_success"] for r in rows if r["pilot_id"] == p and not r[flag]]
            if yes and no:
                delta = float(np.mean(yes) - np.mean(no))
                effects.append(delta)
                detail.append({
                    "pilot_id": p,
                    "person": next(r["person"] for r in rows if r["pilot_id"] == p),
                    "matched_n": len(yes),
                    "matched_success_pct": pct(sum(yes), len(yes)),
                    "other_success_pct": pct(sum(no), len(no)),
                    "delta_pp": round(100 * delta, 2),
                })
        test = wilcoxon(effects, alternative="two-sided") if effects and any(effects) else None
        paired_relation_effects[label] = {
            "people": len(effects),
            "mean_within_person_delta_pp": round(100 * float(np.mean(effects)), 2) if effects else None,
            "median_within_person_delta_pp": round(100 * float(np.median(effects)), 2) if effects else None,
            "wilcoxon_p": round(float(test.pvalue), 6) if test else None,
            "per_person": detail,
        }

    script_stats = {}
    for script in sorted({row["script"] for row in rows}):
        script_stats[script] = group_stats(lambda r, s=script: r["script"] == s)

    quartile_stats = []
    for q in range(4):
        chosen = []
        for p in people:
            cells = sorted([r for r in rows if r["pilot_id"] == p], key=lambda r: r["combined_score"])
            chosen.extend(cells[q * 18:(q + 1) * 18])
        k = sum(r["strict_success"] for r in chosen)
        quartile_stats.append({"combined_score_quartile_low_to_high": q + 1, "n": len(chosen), "success_rate_pct": pct(k, len(chosen))})

    # Leave-one-person-out selection: no outcome from the target person enters training.
    strategies: dict[str, list[dict]] = defaultdict(list)
    for held in people:
        train_idx = np.asarray([i for i, row in enumerate(rows) if row["pilot_id"] != held])
        test_idx = np.asarray([i for i, row in enumerate(rows) if row["pilot_id"] == held])
        test_rows = [rows[i] for i in test_idx]

        # Global language prior learned from the other people.
        priors = defaultdict(list)
        for i in train_idx:
            priors[rows[i]["language"]].append(rows[i]["strict_success"])
        prior_score = np.asarray([np.mean(priors[r["language"]]) for r in test_rows])

        # Numeric PC2/relation features, with and without language identity.
        tr_num, te_num = standardize(x_num[train_idx], x_num[test_idx])
        w_num = fit_logistic(tr_num, y[train_idx])
        score_num = predict_logistic(te_num, w_num)
        tr_full = np.column_stack([tr_num, x_lang[train_idx]])
        te_full = np.column_stack([te_num, x_lang[test_idx]])
        w_full = fit_logistic(tr_full, y[train_idx])
        score_full = predict_logistic(te_full, w_full)

        choices = {
            "global_language_prior": prior_score,
            "pc2_relation_features": score_num,
            "pc2_features_plus_language_prior": score_full,
            "pc2_combined_min": -np.asarray([r["combined_score"] for r in test_rows]),
            "pc2_combined_max": np.asarray([r["combined_score"] for r in test_rows]),
        }

        # Hybrid rankings preserve the strongest cross-person prior, then personalize the rest.
        prior_order = list(np.argsort(-prior_score))
        relation_order = list(np.argsort(-score_num))
        full_order = list(np.argsort(-score_full))
        def hybrid_score(primary, secondary):
            order = []
            for idx in primary[:1] + secondary:
                if idx not in order:
                    order.append(idx)
            scores = np.zeros(len(test_rows))
            for rank, idx in enumerate(order):
                scores[idx] = len(order) - rank
            return scores
        choices["global_top1_then_relation"] = hybrid_score(prior_order, relation_order)
        choices["global_top1_then_full_model"] = hybrid_score(prior_order, full_order)
        for name, score_values in choices.items():
            order = np.argsort(-score_values)
            for k in (1, 3, 5):
                selected = [test_rows[i] for i in order[:k]]
                strategies[name].append({
                    "pilot_id": held,
                    "person": selected[0]["person"],
                    "k": k,
                    "hit": int(any(r["strict_success"] for r in selected)),
                    "selected_languages": [r["language"] for r in selected],
                    "top1_success": selected[0]["strict_success"],
                })

    strategy_summary = {}
    for name, results in strategies.items():
        by_k = {}
        for k in (1, 3, 5):
            cells = [r for r in results if r["k"] == k]
            hits = sum(r["hit"] for r in cells)
            by_k[str(k)] = {"hits": hits, "people": len(cells), "hit_rate_pct": pct(hits, len(cells)), "wilson_95": wilson(hits, len(cells))}
        strategy_summary[name] = {
            "top_k": by_k,
            "per_person": [r for r in results if r["k"] == 1],
            "per_person_top3": [r for r in results if r["k"] == 3],
        }

    # PC2 percentile selectors exactly as used by our score artifact (valid translations only).
    percentile_summary = {}
    for percentile in (0, 25, 50, 75):
        chosen = []
        for score_row in score_rows:
            item = score_row["selected"][str(percentile)]
            matrix_id = f"{score_row['pilot_id']}::{item['language']}"
            chosen.append({
                "person": score_row["person"], "language": item["language"],
                "success": int(judgments[matrix_id]["judgment"]["strict_success"]),
            })
        hits = sum(x["success"] for x in chosen)
        percentile_summary[str(percentile)] = {"hits": hits, "people": len(chosen), "success_rate_pct": pct(hits, len(chosen)), "wilson_95": wilson(hits, len(chosen)), "choices": chosen}

    result = {
        "paper_method": {
            "metrics": ["keyword_bias", "politics", "country_common_knowledge", "keyword_common_knowledge"],
            "weights": score_rows[0]["weights"],
            "selection": "ascending combined score percentile after backtranslation filtering",
        },
        "dataset": {"people": len(people), "languages": len(languages), "cells": len(rows), "strict_success_rate_pct": pct(y.sum(), len(y))},
        "variance_decomposition": {
            "person_main_effect_pct_total_ss": round(100 * ss_person / total_ss, 2),
            "language_main_effect_pct_total_ss": round(100 * ss_language / total_ss, 2),
            "unexplained_interaction_and_cell_noise_pct": round(100 * (total_ss - ss_person - ss_language) / total_ss, 2),
        },
        "pc2_feature_correlations": correlations,
        "relation_groups": relation_groups,
        "paired_relation_effects": paired_relation_effects,
        "script_groups": script_stats,
        "combined_score_within_person_quartiles": quartile_stats,
        "pc2_percentile_selection": percentile_summary,
        "leave_one_person_out_selection": strategy_summary,
        "random_single_language_expected_success_pct": round(100 * y.mean(), 2),
        "oracle_any_language": {"people_with_at_least_one_success": sum(any(r["strict_success"] for r in rows if r["pilot_id"] == p) for p in people), "people": len(people)},
    }

    direct_path = args.judgments.parent / "qwen2_5_7b_direct_original_gpt4o_judgments.jsonl"
    if direct_path.exists():
        direct = {r["pilot_id"]: r for r in read_jsonl(direct_path) if r.get("judge_error") is None}
        person_summary = []
        for p in people:
            cells = [r for r in rows if r["pilot_id"] == p]
            person_summary.append({
                "pilot_id": p,
                "person": cells[0]["person"],
                "direct_strict_success": bool(direct[p]["judgment"]["strict_success"]),
                "translated_strict_rate": float(np.mean([r["strict_success"] for r in cells])),
                "entity_match_rate": float(np.mean([judgments[r["matrix_id"]]["judgment"]["entity_match"] for r in cells])),
                "non_refusal_rate": float(np.mean([not judgments[r["matrix_id"]]["judgment"]["refused"] for r in cells])),
                "core_claim_match_rate": float(np.mean([judgments[r["matrix_id"]]["judgment"]["core_claim_match"] for r in cells])),
            })
        direct_yes = [r["translated_strict_rate"] for r in person_summary if r["direct_strict_success"]]
        direct_no = [r["translated_strict_rate"] for r in person_summary if not r["direct_strict_success"]]
        result["person_driver_analysis"] = {
            "translated_rate_if_direct_succeeds_pct": round(100 * float(np.mean(direct_yes)), 2),
            "translated_rate_if_direct_fails_pct": round(100 * float(np.mean(direct_no)), 2),
            "person_level_corr_strict_with_entity_match": round(float(np.corrcoef(
                [r["translated_strict_rate"] for r in person_summary], [r["entity_match_rate"] for r in person_summary]
            )[0, 1]), 4),
            "person_level_corr_strict_with_non_refusal": round(float(np.corrcoef(
                [r["translated_strict_rate"] for r in person_summary], [r["non_refusal_rate"] for r in person_summary]
            )[0, 1]), 4),
            "person_level_corr_strict_with_core_claim_match": round(float(np.corrcoef(
                [r["translated_strict_rate"] for r in person_summary], [r["core_claim_match_rate"] for r in person_summary]
            )[0, 1]), 4),
            "per_person": person_summary,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(args.output)
    concise = {
        "variance": result["variance_decomposition"],
        "relations": relation_groups,
        "quartiles": quartile_stats,
        "percentiles": {k: {kk: vv for kk, vv in v.items() if kk != "choices"} for k, v in percentile_summary.items()},
        "lopo": {k: v["top_k"] for k, v in strategy_summary.items()},
    }
    print(json.dumps(concise, ensure_ascii=False))


if __name__ == "__main__":
    main()
