#!/usr/bin/env python3
"""Apply a trained PC2 person-language selector to new language-score rows."""

from __future__ import annotations

import argparse
import json
import math
import os
import runpy
from pathlib import Path
from typing import Any

import numpy as np

from train_pc2_person_language_selector import (
    COUNTRY_ALIASES, FEATURES, METRICS, percentile_ranks,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def secure_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def features_for(score_row, country_to_language):
    candidates = score_row["candidates"]
    rank_maps = {
        metric: dict(zip(
            [c["language"] for c in candidates],
            percentile_ranks([float(c[metric]) for c in candidates]),
        ))
        for metric in METRICS
    }
    codes = {c["language"]: c["nllb_code"] for c in candidates}
    country = COUNTRY_ALIASES.get(score_row["person_country_or_territory"], score_row["person_country_or_territory"])
    native = country_to_language.get(country)
    native_script = (codes.get(native) or "_").split("_")[-1]
    english_len = max(1, len(score_row["english_ipdm"]))
    rows = []
    for c in candidates:
        script = (c["nllb_code"] or "_").split("_")[-1]
        f = {
            **{f"rank_{m}": rank_maps[m][c["language"]] for m in METRICS},
            "backtranslation_similarity": float(c["backtranslation_similarity"]),
            "translation_valid": int(bool(c["valid"])),
            "native_language_match": int(c["language"] == native),
            "native_script_match": int(bool(native_script) and script == native_script),
            "log_translation_length_ratio": math.log(max(1, len(c["translation"])) / english_len),
        }
        rows.append({"language": c["language"], "feature": f, "translation_valid": bool(c["valid"])})
    return rows, country, native


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--pc2-languages-py", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--identity-preflight", type=Path,
                        help="Optional JSONL with pilot_id, language, identity_probability")
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    model = json.load(args.model.open(encoding="utf-8"))
    score_rows = read_jsonl(args.scores)
    country_to_language = runpy.run_path(str(args.pc2_languages_py))["language_dict"]
    means = np.asarray([model["feature_mean"][f] for f in FEATURES], dtype=float)
    stds = np.asarray([model["feature_std"][f] for f in FEATURES], dtype=float)
    beta = np.asarray([model["coefficients_standardized"][f] for f in FEATURES], dtype=float)
    intercept = float(model["intercept"])
    priors = model["language_priors"]
    global_mean = float(model["global_mean"])

    identity = {}
    if args.identity_preflight:
        for row in read_jsonl(args.identity_preflight):
            identity[(row["pilot_id"], row["language"])] = float(row["identity_probability"])

    output = []
    for score_row in score_rows:
        rows, country, native = features_for(score_row, country_to_language)
        if args.valid_only:
            rows = [r for r in rows if r["translation_valid"]]
        if not rows:
            raise ValueError(f"no candidates for {score_row['pilot_id']}")
        x = np.asarray([[r["feature"][f] for f in FEATURES] for r in rows], dtype=float)
        z = intercept + ((x - means) / stds) @ beta
        relation = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        candidates = []
        for row, relation_probability in zip(rows, relation):
            prior = float(priors.get(row["language"], {"posterior_mean": global_mean})["posterior_mean"])
            candidates.append({
                **row,
                "language_prior": prior,
                "relation_probability": float(relation_probability),
                "identity_probability": identity.get((score_row["pilot_id"], row["language"])),
            })

        if identity:
            identified = [r for r in candidates if (r["identity_probability"] or 0) >= .5]
            pool = identified if len(identified) >= args.top_k else candidates
        else:
            pool = candidates
        first = max(pool, key=lambda r: (r["language_prior"], r["relation_probability"], r["language"]))
        rest = sorted(
            [r for r in pool if r["language"] != first["language"]],
            key=lambda r: (
                -1 if r["identity_probability"] is None else r["identity_probability"],
                r["relation_probability"], r["language_prior"], r["language"],
            ),
            reverse=True,
        )
        selected = [first] + rest[:args.top_k - 1]
        output.append({
            "pilot_id": score_row["pilot_id"],
            "person": score_row["person"],
            "country": country,
            "native_language": native,
            "selector_schema": model["schema"],
            "identity_preflight_used": bool(identity),
            "selected": [{
                "rank": i + 1,
                "language": r["language"],
                "language_prior": round(r["language_prior"], 6),
                "relation_probability": round(r["relation_probability"], 6),
                "identity_probability": r["identity_probability"],
                "translation_valid": r["translation_valid"],
                "selection_role": "global_anchor" if i == 0 else "person_conditioned",
                "feature_values": r["feature"],
            } for i, r in enumerate(selected)],
        })

    secure_jsonl(args.output, output)
    print(json.dumps({"people": len(output), "top_k": args.top_k, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
