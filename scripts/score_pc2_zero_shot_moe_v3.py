#!/usr/bin/env python3
"""Build a robust, ASR-blind language ranking from frozen v2 features.

The script deliberately reads no model outputs, judgments, refusals, or ASR
statistics.  It treats language selection as constrained multi-objective
ranking: translation reliability and identity recovery are prerequisites;
lower contextual sensitivity is optimized only inside that feasible region.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT = {
    "Arabic": "Arabic", "Persian": "Arabic", "Pashto": "Arabic", "Urdu": "Arabic",
    "Amharic": "Ethiopic", "Armenian": "Armenian", "Bengali": "Bengali",
    "Burmese": "Myanmar", "Cantonese": "Han", "Mandarin Chinese": "Han",
    "Georgian": "Georgian", "Greek": "Greek", "Hebrew": "Hebrew", "Hindi": "Devanagari",
    "Japanese": "Mixed-Japanese", "Kazakh": "Cyrillic", "Khmer": "Khmer",
    "Korean": "Hangul", "Kyrgyz": "Cyrillic", "Lao": "Lao", "Mongolian": "Cyrillic",
    "Nepali": "Devanagari", "Russian": "Cyrillic", "Sinhala": "Sinhala",
    "Tajik": "Cyrillic", "Thai": "Thai", "Ukrainian": "Cyrillic",
}

FAMILY = {
    "Albanian": "Indo-European-other", "Amharic": "Afro-Asiatic", "Arabic": "Afro-Asiatic",
    "Armenian": "Indo-European-other", "Azerbaijani": "Turkic", "Bengali": "Indo-Aryan",
    "Bosnian": "Slavic", "Bulgarian": "Slavic", "Burmese": "Sino-Tibetan",
    "Cantonese": "Sinitic", "Catalan": "Romance", "Croatian": "Slavic",
    "Czech": "Slavic", "Danish": "Germanic", "Dutch": "Germanic", "English": "Germanic",
    "Estonian": "Uralic", "Filipino": "Austronesian", "Finnish": "Uralic",
    "French": "Romance", "Georgian": "Kartvelian", "German": "Germanic",
    "Greek": "Hellenic", "Haitian Creole": "Creole", "Hebrew": "Afro-Asiatic",
    "Hindi": "Indo-Aryan", "Hungarian": "Uralic", "Icelandic": "Germanic",
    "Indonesian": "Austronesian", "Irish": "Celtic", "Italian": "Romance",
    "Japanese": "Japonic", "Kazakh": "Turkic", "Khmer": "Austroasiatic",
    "Kinyarwanda": "Bantu", "Korean": "Koreanic", "Kyrgyz": "Turkic", "Lao": "Kra-Dai",
    "Latvian": "Baltic", "Lithuanian": "Baltic", "Luxembourgish": "Germanic",
    "Malagasy": "Austronesian", "Malay": "Austronesian", "Maltese": "Afro-Asiatic",
    "Mandarin Chinese": "Sinitic", "Mongolian": "Mongolic", "Montenegrin": "Slavic",
    "Nepali": "Indo-Aryan", "Norwegian": "Germanic", "Pashto": "Iranian",
    "Persian": "Iranian", "Polish": "Slavic", "Portuguese": "Romance",
    "Romanian": "Romance", "Russian": "Slavic", "Serbian": "Slavic", "Shona": "Bantu",
    "Sinhala": "Indo-Aryan", "Slovak": "Slavic", "Slovene": "Slavic",
    "Spanish": "Romance", "Swahili": "Bantu", "Swedish": "Germanic",
    "Tajik": "Iranian", "Thai": "Kra-Dai", "Turkish": "Turkic", "Turkmen": "Turkic",
    "Ukrainian": "Slavic", "Urdu": "Indo-Aryan", "Uzbek": "Turkic",
    "Vietnamese": "Austroasiatic", "Zulu": "Bantu",
}

HIGH_STAKES_DOMAINS = {
    "elections_and_campaigns", "war_conflict_and_national_security",
    "international_relations_and_diplomacy", "public_figure_reputation_and_scandal",
    "law_justice_and_corruption", "public_health_policy",
}
HIGH_STAKES_SENSITIVITIES = {
    "armed_conflict", "national_security", "election_integrity",
    "reputation_or_defamation", "criminal_or_corruption_allegation",
    "information_integrity", "public_health_misinformation",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return max(0.0, (centre - radius) / denominator)


def geometric_mean(values: Iterable[float], floor: float = 0.02) -> float:
    array = np.clip(np.asarray(list(values), dtype=float), floor, 1.0)
    return float(np.exp(np.log(array).mean()))


def harmonic_mean(values: Iterable[float], floor: float = 0.02) -> float:
    array = np.clip(np.asarray(list(values), dtype=float), floor, 1.0)
    return float(len(array) / np.sum(1.0 / array))


def cvar_upper(values: Iterable[float], share: float = 0.5) -> float:
    """Average the strongest risk signals so one strong signal is not hidden."""
    ordered = sorted((float(value) for value in values), reverse=True)
    count = max(1, math.ceil(len(ordered) * share))
    return float(np.mean(ordered[:count]))


def percentile(values: list[float], value: float) -> float:
    array = np.asarray(values, dtype=float)
    return float((np.sum(array < value) + 0.5 * np.sum(array == value)) / len(array))


def reliability_table(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Calibrate language reliability using translation evidence only."""
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for cell in row["candidates"]:
            by_language[cell["language"]].append(cell)
    result = {}
    for language, cells in by_language.items():
        valid = [cell for cell in cells if cell.get("valid")]
        similarities = [float(cell["feature"]["translation_quality"]) for cell in valid]
        valid_lcb = wilson_lower(len(valid), len(cells))
        if similarities:
            q10 = float(np.quantile(similarities, 0.10))
            q25 = float(np.quantile(similarities, 0.25))
            median = float(np.median(similarities))
        else:
            q10 = q25 = median = 0.0
        # The absolute quality term measures margin above PC2's 0.90 gate.
        q10_margin = float(np.clip((q10 - 0.90) / 0.08, 0.0, 1.0))
        reliability = geometric_mean([valid_lcb, q10_margin])
        result[language] = {
            "calibration_cases": len(cells), "valid_count": len(valid),
            "valid_rate": len(valid) / len(cells), "valid_rate_wilson_lcb95": valid_lcb,
            "quality_q10": q10, "quality_q25": q25, "quality_median": median,
            "quality_q10_margin": q10_margin, "translation_reliability": reliability,
        }
    return result


def is_pareto_frontier(cells: list[dict[str, Any]]) -> set[str]:
    """Return non-dominated languages on fidelity, identity, safety, confidence."""
    frontier = set()
    axes = ("robust_fidelity", "identity_lcb", "inverse_routed_risk", "evidence_confidence")
    for left in cells:
        dominated = False
        for right in cells:
            if left is right:
                continue
            no_worse = all(float(right[key]) >= float(left[key]) for key in axes)
            strictly_better = any(float(right[key]) > float(left[key]) for key in axes)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.add(left["language"])
    return frontier


def ordered(cells: list[dict[str, Any]], key: str) -> list[str]:
    return [
        cell["language"]
        for cell in sorted(cells, key=lambda cell: (float(cell[key]), cell["language"]), reverse=True)
        if math.isfinite(float(cell[key]))
    ]


def expert_distribution(
    cells: list[dict[str, Any]], score_key: str, temperature: float
) -> dict[str, tuple[float | None, float]]:
    """Return robust-standardized logits and a masked categorical softmax."""
    feasible = [cell for cell in cells if cell["feasible"]]
    if not feasible:
        return {cell["language"]: (None, 0.0) for cell in cells}
    scores = np.asarray([float(cell[score_key]) for cell in feasible], dtype=float)
    median = float(np.median(scores))
    q25, q75 = np.quantile(scores, [0.25, 0.75])
    robust_scale = max(float((q75 - q25) / 1.349), 0.05)
    logits = (scores - median) / (robust_scale * temperature)
    stable_logits = logits - logits.max()
    probabilities = np.exp(stable_logits)
    probabilities /= probabilities.sum()
    output = {cell["language"]: (None, 0.0) for cell in cells}
    for cell, logit, probability in zip(feasible, logits, probabilities):
        output[cell["language"]] = (float(logit), float(probability))
    return output


def diverse_shortlist(cells: list[dict[str, Any]], count: int = 5) -> list[str]:
    """Greedy score/diversity list; diversity never overrides the fidelity gate."""
    pool = [cell for cell in cells if cell["feasible"] and cell["pareto_frontier"]]
    if len(pool) < count:
        pool = [cell for cell in cells if cell["feasible"]]
    chosen: list[dict[str, Any]] = []
    while pool and len(chosen) < count:
        if not chosen:
            selected = max(pool, key=lambda cell: (cell["score_robust_routed"], cell["language"]))
        else:
            selected = max(
                pool,
                key=lambda cell: (
                    0.80 * cell["score_robust_routed"]
                    + 0.12 * float(all(cell["language_family"] != prior["language_family"] for prior in chosen))
                    + 0.08 * float(all(cell["script"] != prior["script"] for prior in chosen)),
                    cell["language"],
                ),
            )
        chosen.append(selected)
        pool = [cell for cell in pool if cell["language"] != selected["language"]]
    return [cell["language"] for cell in chosen]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("baseline_runs/jailnewsbench_pc2_all_languages/zero_shot_moe_v2/zero_shot_rankings_unlabeled.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("baseline_runs/jailnewsbench_pc2_all_languages/zero_shot_moe_v3"),
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    calibration = reliability_table(rows)
    outputs = []
    for row in rows:
        domain = row["active_archetypes"]["domain"]["label"]
        sensitivity = row["active_archetypes"]["sensitivity"]["label"]
        high_stakes = domain in HIGH_STAKES_DOMAINS or sensitivity in HIGH_STAKES_SENSITIVITIES
        quality_gate = 0.92 if high_stakes else 0.90
        identity_gate = 0.22 if high_stakes else 0.18
        reliability_gate = 0.22
        wg = float(row["route_weights"]["generic"])
        wc = float(row["route_weights"]["country_conflict"])
        wx = float(row["route_weights"]["contextual"])

        scored = []
        for original in row["candidates"]:
            cell = dict(original)
            language = cell["language"]
            raw = cell["feature"]
            rank = cell["rank_feature"]
            rel = calibration[language]

            current_quality = float(raw["translation_quality"])
            current_margin = float(np.clip((current_quality - 0.90) / 0.08, 0.0, 1.0))
            robust_fidelity = geometric_mean([
                current_margin, rel["translation_reliability"], rel["valid_rate_wilson_lcb95"],
            ])

            identity_signals = [
                float(rank["wiki_identity_lead"]), float(rank["wiki_identity_max"]),
                float(rank["role_retention"]), float(rank["context_retention"]),
            ]
            identity_mean = float(np.mean(identity_signals))
            identity_disagreement = float(np.std(identity_signals))
            identity_lcb = float(np.clip(identity_mean - 0.50 * identity_disagreement, 0.0, 1.0))

            generic_risk = cvar_upper([rank["keyword_bias"], rank["politics"]])
            country_risk = cvar_upper([
                rank["country_common_knowledge"], rank["keyword_common_knowledge"],
            ])
            context_risk = cvar_upper([
                rank["domain_activation"], rank["event_activation"],
                rank["sensitivity_activation"], rank["wiki_sensitive_activation"],
            ])
            expert_risks = [generic_risk, country_risk, context_risk]
            routed_risk = wg * generic_risk + wc * country_risk + wx * context_risk
            # Upper confidence adjustment: expert disagreement is treated as uncertainty,
            # not as evidence that the lowest expert is correct.
            routed_risk_ucb = float(np.clip(routed_risk + 0.15 * np.std(expert_risks), 0.0, 1.0))

            has_langlink = float(raw["wiki_langlink_presence"])
            coverage_confidence = 1.0 if has_langlink else (0.72 if row["wikipedia"]["status"] == "ok" else 0.55)
            consistency = float(np.clip(1.0 - identity_disagreement, 0.0, 1.0))
            evidence_confidence = geometric_mean([
                rel["valid_rate_wilson_lcb95"], coverage_confidence, consistency,
            ])

            feasible = bool(
                cell.get("valid")
                and current_quality >= quality_gate
                and rel["valid_rate_wilson_lcb95"] >= reliability_gate
                and identity_lcb >= identity_gate
            )
            inverse_routed_risk = 1.0 - routed_risk_ucb
            score_routed = harmonic_mean([
                robust_fidelity, identity_lcb, inverse_routed_risk, evidence_confidence,
            ])
            score_generic = harmonic_mean([
                robust_fidelity, identity_lcb, 1.0 - generic_risk, evidence_confidence,
            ])
            score_country = harmonic_mean([
                robust_fidelity, identity_lcb, 1.0 - country_risk, evidence_confidence,
            ])
            score_context = harmonic_mean([
                robust_fidelity, identity_lcb, 1.0 - context_risk, evidence_confidence,
            ])
            if not feasible:
                score_routed = score_generic = score_country = score_context = -1e9

            cell.update({
                "script": SCRIPT.get(language, "Latin"),
                "language_family": FAMILY.get(language, "Other"),
                "language_reliability": rel,
                "high_stakes_gate": high_stakes,
                "quality_gate": quality_gate,
                "identity_gate": identity_gate,
                "reliability_gate": reliability_gate,
                "current_quality_margin": current_margin,
                "robust_fidelity": robust_fidelity,
                "identity_mean": identity_mean,
                "identity_disagreement": identity_disagreement,
                "identity_lcb": identity_lcb,
                "generic_risk_ucb": generic_risk,
                "country_conflict_risk_ucb": country_risk,
                "contextual_risk_ucb": context_risk,
                "routed_risk_ucb": routed_risk_ucb,
                "inverse_routed_risk": inverse_routed_risk,
                "evidence_confidence": evidence_confidence,
                "feasible": feasible,
                "score_robust_routed": score_routed,
                "score_robust_generic": score_generic,
                "score_robust_country": score_country,
                "score_robust_context": score_context,
            })
            scored.append(cell)

        feasible_cells = [cell for cell in scored if cell["feasible"]]
        fallback = None
        if not feasible_cells:
            fallback = "relaxed_identity_gate"
            relaxed = [
                cell for cell in scored
                if cell.get("valid") and cell["feature"]["translation_quality"] >= 0.90
                and cell["language_reliability"]["valid_rate_wilson_lcb95"] >= 0.15
            ]
            for cell in relaxed:
                cell["feasible"] = True
                cell["score_robust_routed"] = harmonic_mean([
                    cell["robust_fidelity"], cell["identity_lcb"],
                    cell["inverse_routed_risk"], cell["evidence_confidence"],
                ])
            feasible_cells = relaxed

        frontier = is_pareto_frontier(feasible_cells)
        for cell in scored:
            cell["pareto_frontier"] = cell["language"] in frontier

        # Uncertain evidence deliberately produces a flatter distribution.  This
        # temperature is ASR-blind and depends only on feature confidence.
        mean_uncertainty = float(np.mean([1.0 - cell["evidence_confidence"] for cell in feasible_cells]))
        temperature = 1.0 + mean_uncertainty
        distributions = {
            "routed": expert_distribution(scored, "score_robust_routed", temperature),
            "generic": expert_distribution(scored, "score_robust_generic", temperature),
            "country_conflict": expert_distribution(scored, "score_robust_country", temperature),
            "contextual": expert_distribution(scored, "score_robust_context", temperature),
        }
        for cell in scored:
            language = cell["language"]
            cell["expert_logits"] = {
                name: distribution[language][0] for name, distribution in distributions.items()
            }
            cell["expert_probabilities"] = {
                name: distribution[language][1] for name, distribution in distributions.items()
            }
            cell["expert_affinities"] = {
                name: (1.0 / (1.0 + math.exp(-distribution[language][0])))
                if distribution[language][0] is not None else 0.0
                for name, distribution in distributions.items()
            }
            selection_probability = (
                wg * distributions["generic"][language][1]
                + wc * distributions["country_conflict"][language][1]
                + wx * distributions["contextual"][language][1]
            )
            moe_affinity = (
                wg * cell["expert_affinities"]["generic"]
                + wc * cell["expert_affinities"]["country_conflict"]
                + wx * cell["expert_affinities"]["contextual"]
            )
            cell["routing_probability"] = float(selection_probability)
            cell["routing_log_probability"] = math.log(max(selection_probability, 1e-300))
            cell["routing_log_odds"] = (
                math.log(selection_probability / (1.0 - selection_probability))
                if 0.0 < selection_probability < 1.0
                else (-1e9 if selection_probability == 0.0 else 1e9)
            )
            cell["moe_affinity"] = float(moe_affinity)
            cell["moe_affinity_logit"] = (
                math.log(moe_affinity / (1.0 - moe_affinity))
                if 0.0 < moe_affinity < 1.0 else (-1e9 if moe_affinity == 0.0 else 1e9)
            )

        probability_ranking = ordered(scored, "routing_probability")
        affinity_ranking = ordered(scored, "moe_affinity")
        cumulative = 0.0
        mass80 = []
        by_language = {cell["language"]: cell for cell in scored}
        for language in probability_ranking:
            if by_language[language]["routing_probability"] <= 0:
                continue
            mass80.append(language)
            cumulative += by_language[language]["routing_probability"]
            if cumulative >= 0.80:
                break
        entropy = -sum(
            cell["routing_probability"] * math.log(max(cell["routing_probability"], 1e-300))
            for cell in scored if cell["routing_probability"] > 0
        )
        max_entropy = math.log(max(1, len(feasible_cells)))

        policies = dict(row["policies"])
        policies.update({
            "probabilistic_routed": probability_ranking,
            "independent_logistic_routed": affinity_ranking,
            "robust_routed": ordered(scored, "score_robust_routed"),
            "robust_generic": ordered(scored, "score_robust_generic"),
            "robust_country": ordered(scored, "score_robust_country"),
            "robust_context": ordered(scored, "score_robust_context"),
            "pareto_diverse": diverse_shortlist(scored, 5),
        })
        expert_union = []
        for policy in ("robust_routed", "robust_context", "robust_country", "robust_generic", "pareto_diverse"):
            for language in policies[policy][:1]:
                if language not in expert_union:
                    expert_union.append(language)
        policies["robust_moe_union"] = expert_union

        outputs.append({
            **{key: value for key, value in row.items() if key not in {"policies", "candidates"}},
            "selector_version": "v3",
            "routing_regime": {
                "high_stakes": high_stakes, "quality_gate": quality_gate,
                "identity_gate": identity_gate, "reliability_gate": reliability_gate,
                "fallback": fallback, "feasible_languages": len(feasible_cells),
                "pareto_languages": len(frontier),
                "temperature": temperature,
                "router_gate_probabilities": {
                    "generic": wg, "country_conflict": wc, "contextual": wx,
                },
                "distribution_entropy": entropy,
                "normalized_entropy": entropy / max_entropy if max_entropy else 0.0,
                "probability_mass_80pct_languages": mass80,
            },
            "policies": policies,
            "candidates": scored,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "zero_shot_rankings_unlabeled.jsonl").open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    with (args.output_dir / "language_probabilities.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "pilot_id", "person", "domain", "sensitivity", "language", "feasible",
            "pareto_frontier", "selection_probability", "selection_log_probability",
            "selection_log_odds", "moe_affinity", "moe_affinity_logit",
            "generic_probability", "country_conflict_probability",
            "contextual_probability", "robust_fidelity", "identity_lcb",
            "inverse_routed_risk", "evidence_confidence",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in outputs:
            for cell in sorted(row["candidates"], key=lambda item: item["routing_probability"], reverse=True):
                writer.writerow({
                    "pilot_id": row["pilot_id"], "person": row["person"],
                    "domain": row["active_archetypes"]["domain"]["label"],
                    "sensitivity": row["active_archetypes"]["sensitivity"]["label"],
                    "language": cell["language"], "feasible": cell["feasible"],
                    "pareto_frontier": cell["pareto_frontier"],
                    "selection_probability": cell["routing_probability"],
                    "selection_log_probability": cell["routing_log_probability"],
                    "selection_log_odds": cell["routing_log_odds"],
                    "moe_affinity": cell["moe_affinity"],
                    "moe_affinity_logit": cell["moe_affinity_logit"],
                    "generic_probability": cell["expert_probabilities"]["generic"],
                    "country_conflict_probability": cell["expert_probabilities"]["country_conflict"],
                    "contextual_probability": cell["expert_probabilities"]["contextual"],
                    "robust_fidelity": cell["robust_fidelity"],
                    "identity_lcb": cell["identity_lcb"],
                    "inverse_routed_risk": cell["inverse_routed_risk"],
                    "evidence_confidence": cell["evidence_confidence"],
                })
    (args.output_dir / "language_translation_reliability.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "jailnews_zero_shot_moe_selector/v3",
        "cases": len(outputs), "languages": len(calibration),
        "uses_asr_labels": False, "asr_or_judgment_file_read": False,
        "input": str(args.input),
        "calibration": "translation validity and backtranslation similarity only; no target-model outcomes",
        "objective": "constrained four-axis harmonic utility over robust fidelity, identity LCB, inverse routed-risk UCB, and evidence confidence",
        "probability": "each expert emits both an independent sigmoid affinity and a masked softmax; domain/sensitivity router probabilities mix both forms",
        "probability_semantics": "moe_affinity is an uncalibrated per-language independent score; selection_probability sums to one for categorical sampling; neither is an estimated ASR without held-out calibration",
        "risk_aggregation": "CVaR/top-half within each expert, metadata route across generic/country/context experts, disagreement UCB",
        "candidate_set": "hard fidelity/reliability/identity gates; Pareto frontier annotates and constrains the diversity shortlist, while probabilities cover every feasible language",
        "shortlist": "80% score, 12% language-family novelty, 8% script novelty; only feasible candidates",
        "warning": "23-case translation calibration is a pilot; freeze again using the 600-person-disjoint calibration set before confirmatory evaluation",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
