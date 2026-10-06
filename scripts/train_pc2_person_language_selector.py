#!/usr/bin/env python3
"""Train a small-data, interpretable person-language selector for safety evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import runpy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


FEATURES = [
    "rank_keyword_bias",
    "rank_politics",
    "rank_country_common_knowledge",
    "rank_keyword_common_knowledge",
    "rank_combined_score",
    "backtranslation_similarity",
    "translation_valid",
    "native_language_match",
    "native_script_match",
    "log_translation_length_ratio",
]
METRICS = [
    "keyword_bias", "politics", "country_common_knowledge",
    "keyword_common_knowledge", "combined_score",
]
COUNTRY_ALIASES = {"U.S.": "United States", "England": "United Kingdom"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def secure_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def secure_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def percentile_ranks(values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return (ranks / max(1, len(values) - 1)).tolist()


def build_rows(score_rows, judgments, country_to_language):
    language_code = {}
    for score_row in score_rows:
        for c in score_row["candidates"]:
            language_code.setdefault(c["language"], c["nllb_code"])

    rows = []
    for score_row in score_rows:
        candidates = score_row["candidates"]
        rank_maps = {
            metric: dict(zip(
                [c["language"] for c in candidates],
                percentile_ranks([float(c[metric]) for c in candidates]),
            ))
            for metric in METRICS
        }
        country = COUNTRY_ALIASES.get(score_row["person_country_or_territory"], score_row["person_country_or_territory"])
        native = country_to_language.get(country)
        native_script = (language_code.get(native) or "_").split("_")[-1]
        english_len = max(1, len(score_row["english_ipdm"]))
        for c in candidates:
            matrix_id = f"{score_row['pilot_id']}::{c['language']}"
            judgment = judgments[matrix_id]["judgment"]
            script = (c["nllb_code"] or "_").split("_")[-1]
            feature = {
                **{f"rank_{m}": rank_maps[m][c["language"]] for m in METRICS},
                "backtranslation_similarity": float(c["backtranslation_similarity"]),
                "translation_valid": int(bool(c["valid"])),
                "native_language_match": int(c["language"] == native),
                "native_script_match": int(bool(native_script) and script == native_script),
                "log_translation_length_ratio": math.log(max(1, len(c["translation"])) / english_len),
            }
            rows.append({
                "matrix_id": matrix_id,
                "pilot_id": score_row["pilot_id"],
                "person": score_row["person"],
                "country": country,
                "native_language": native,
                "language": c["language"],
                "strict_success": int(judgment["strict_success"]),
                "entity_match": int(judgment["entity_match"]),
                "feature": feature,
            })
    return rows


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    x1 = np.column_stack([np.ones(len(x)), x])

    def loss(w):
        z = np.clip(x1 @ w, -30, 30)
        return np.logaddexp(0, z).sum() - y @ z + .5 * l2 * (w[1:] @ w[1:])

    def grad(w):
        z = np.clip(x1 @ w, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = x1.T @ (p - y)
        g[1:] += l2 * w[1:]
        return g

    result = minimize(loss, np.zeros(x1.shape[1]), jac=grad, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def standardize_fit(x: np.ndarray):
    mean = x.mean(0)
    std = x.std(0)
    std[std < 1e-8] = 1
    return (x - mean) / std, mean, std


def predict(x, mean, std, weight):
    z = np.column_stack([np.ones(len(x)), (x - mean) / std]) @ weight
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))


def language_priors(rows, alpha: float):
    total_success = sum(r["strict_success"] for r in rows)
    mu = total_success / len(rows)
    languages = sorted({r["language"] for r in rows})
    priors = {}
    for language in languages:
        cells = [r for r in rows if r["language"] == language]
        successes = sum(r["strict_success"] for r in cells)
        priors[language] = {
            "successes": successes,
            "trials": len(cells),
            "posterior_mean": (successes + alpha * mu) / (len(cells) + alpha),
        }
    return priors, mu


def train_model(rows, alpha, l2):
    x = np.asarray([[r["feature"][f] for f in FEATURES] for r in rows], dtype=float)
    y = np.asarray([r["strict_success"] for r in rows], dtype=float)
    xz, mean, std = standardize_fit(x)
    weight = fit_logistic(xz, y, l2)
    priors, mu = language_priors(rows, alpha)
    return {"mean": mean, "std": std, "weight": weight, "priors": priors, "global_mean": mu}


def shortlist(model, cells, k=3):
    x = np.asarray([[r["feature"][f] for f in FEATURES] for r in cells], dtype=float)
    relation = predict(x, model["mean"], model["std"], model["weight"])
    decorated = []
    for r, relation_score in zip(cells, relation):
        prior = model["priors"].get(r["language"], {"posterior_mean": model["global_mean"]})["posterior_mean"]
        decorated.append({**r, "relation_probability": float(relation_score), "language_prior": float(prior)})
    global_first = max(decorated, key=lambda r: (r["language_prior"], r["relation_probability"], r["language"]))
    rest = sorted(
        [r for r in decorated if r["language"] != global_first["language"]],
        key=lambda r: (r["relation_probability"], r["language_prior"], r["language"]),
        reverse=True,
    )
    return [global_first] + rest[:k - 1]


def serializable_model(model, alpha, l2):
    return {
        "schema": "pc2_person_language_selector/v1",
        "formula": {
            "language_prior": "G_l=(success_l + alpha*global_mean)/(trials_l + alpha)",
            "relation_probability": "R_pl=sigmoid(beta_0 + sum_j beta_j*z(feature_plj))",
            "shortlist": "candidate_1=argmax_l G_l; candidates_2..k=top R_pl excluding candidate_1",
            "optional_identity_preflight": "when available, filter or rank identity-recovered languages before applying R_pl",
        },
        "alpha": alpha,
        "l2": l2,
        "feature_names": FEATURES,
        "feature_mean": dict(zip(FEATURES, model["mean"].tolist())),
        "feature_std": dict(zip(FEATURES, model["std"].tolist())),
        "intercept": float(model["weight"][0]),
        "coefficients_standardized": dict(zip(FEATURES, model["weight"][1:].tolist())),
        "global_mean": float(model["global_mean"]),
        "language_priors": model["priors"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--pc2-languages-py", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--l2", type=float, default=5.0)
    args = parser.parse_args()

    score_rows = read_jsonl(args.scores)
    judgments = {r["matrix_id"]: r for r in read_jsonl(args.judgments) if r.get("judge_error") is None}
    country_to_language = runpy.run_path(str(args.pc2_languages_py))["language_dict"]
    rows = build_rows(score_rows, judgments, country_to_language)
    people = sorted({r["pilot_id"] for r in rows})

    lopo = []
    for held in people:
        train = [r for r in rows if r["pilot_id"] != held]
        test = [r for r in rows if r["pilot_id"] == held]
        selected = shortlist(train_model(train, args.alpha, args.l2), test, 3)
        lopo.append({
            "pilot_id": held,
            "person": test[0]["person"],
            "country": test[0]["country"],
            "selected_languages": [r["language"] for r in selected],
            "top1_success": bool(selected[0]["strict_success"]),
            "top3_hit": any(r["strict_success"] for r in selected),
            "details": [{
                "language": r["language"], "language_prior": r["language_prior"],
                "relation_probability": r["relation_probability"],
                "retrospective_success": bool(r["strict_success"]),
            } for r in selected],
        })

    full_model = train_model(rows, args.alpha, args.l2)
    recommendations = []
    for score_row in score_rows:
        cells = [r for r in rows if r["pilot_id"] == score_row["pilot_id"]]
        selected = shortlist(full_model, cells, 3)
        recommendations.append({
            "pilot_id": score_row["pilot_id"], "person": score_row["person"],
            "country": score_row["person_country_or_territory"],
            "warning": "in-sample recommendation; use LOPO rows for unbiased audit",
            "selected": [{
                "rank": i + 1, "language": r["language"],
                "language_prior": round(r["language_prior"], 6),
                "relation_probability": round(r["relation_probability"], 6),
            } for i, r in enumerate(selected)],
        })

    top1 = sum(r["top1_success"] for r in lopo)
    top3 = sum(r["top3_hit"] for r in lopo)
    summary = {
        "schema": "pc2_person_language_selector_evaluation/v1",
        "people": len(people), "languages": len({r["language"] for r in rows}),
        "cells": len(rows), "alpha": args.alpha, "l2": args.l2,
        "evaluation": "leave-one-person-out; all 72 outcomes for held-out person excluded",
        "top1": {"hits": top1, "people": len(people), "rate_pct": round(100 * top1 / len(people), 2)},
        "top3": {"hits": top3, "people": len(people), "rate_pct": round(100 * top3 / len(people), 2)},
        "limitations": [
            "23 people is a pilot-sized training set",
            "the relation model uses proxy PC2 features rather than the paper-exact Wikipedia metrics",
            "the recommended identity-reconstruction preflight is not yet included",
            "one deterministic target generation per person-language cell",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    secure_json(args.output_dir / "selector_model.json", serializable_model(full_model, args.alpha, args.l2))
    secure_json(args.output_dir / "summary.json", summary)
    secure_jsonl(args.output_dir / "lopo_audit.jsonl", lopo)
    secure_jsonl(args.output_dir / "recommendations_in_sample.jsonl", recommendations)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
