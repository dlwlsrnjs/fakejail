#!/usr/bin/env python3
"""Score languages without ASR priors using PC2-style and Wikipedia features.

This script intentionally has no judgments/ASR argument. It builds language
rankings from translation fidelity, PC2 knowledge scores, Wikipedia identity
coverage, and archetypes learned from the 4,091 reviewed English samples.
"""

from __future__ import annotations

import argparse
import json
import math
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from transformers import AutoModel, AutoTokenizer


WIKI_CODES = {
    "Albanian": "sq", "Amharic": "am", "Arabic": "ar", "Armenian": "hy",
    "Azerbaijani": "az", "Bengali": "bn", "Bosnian": "bs", "Bulgarian": "bg",
    "Burmese": "my", "Cantonese": "zh-yue", "Catalan": "ca", "Croatian": "hr",
    "Czech": "cs", "Danish": "da", "Dutch": "nl", "English": "en",
    "Estonian": "et", "Filipino": "tl", "Finnish": "fi", "French": "fr",
    "Georgian": "ka", "German": "de", "Greek": "el", "Haitian Creole": "ht",
    "Hebrew": "he", "Hindi": "hi", "Hungarian": "hu", "Icelandic": "is",
    "Indonesian": "id", "Irish": "ga", "Italian": "it", "Japanese": "ja",
    "Kazakh": "kk", "Khmer": "km", "Kinyarwanda": "rw", "Korean": "ko",
    "Kyrgyz": "ky", "Lao": "lo", "Latvian": "lv", "Lithuanian": "lt",
    "Luxembourgish": "lb", "Malagasy": "mg", "Malay": "ms", "Maltese": "mt",
    "Mandarin Chinese": "zh", "Mongolian": "mn", "Montenegrin": "sr",
    "Nepali": "ne", "Norwegian": "no", "Pashto": "ps", "Persian": "fa",
    "Polish": "pl", "Portuguese": "pt", "Romanian": "ro", "Russian": "ru",
    "Serbian": "sr", "Shona": "sn", "Sinhala": "si", "Slovak": "sk",
    "Slovene": "sl", "Spanish": "es", "Swahili": "sw", "Swedish": "sv",
    "Tajik": "tg", "Thai": "th", "Turkish": "tr", "Turkmen": "tk",
    "Ukrainian": "uk", "Urdu": "ur", "Uzbek": "uz", "Vietnamese": "vi",
    "Zulu": "zu",
}

ROUTE_WEIGHTS = {
    "elections_and_campaigns": (0.20, 0.40, 0.40),
    "war_conflict_and_national_security": (0.15, 0.45, 0.40),
    "international_relations_and_diplomacy": (0.15, 0.45, 0.40),
    "public_figure_reputation_and_scandal": (0.40, 0.10, 0.50),
    "law_justice_and_corruption": (0.35, 0.15, 0.50),
    "public_health_policy": (0.25, 0.10, 0.65),
    "technology_platforms_and_information": (0.40, 0.10, 0.50),
    "civil_rights_identity_and_migration": (0.30, 0.15, 0.55),
    "governance_and_leadership": (0.40, 0.20, 0.40),
    "economy_trade_and_labor": (0.35, 0.25, 0.40),
    "royalty_and_constitutional_affairs": (0.40, 0.10, 0.50),
    "climate_energy_and_environment": (0.25, 0.20, 0.55),
    "science_space_and_infrastructure": (0.25, 0.10, 0.65),
    "culture_sports_and_other": (0.30, 0.10, 0.60),
    "unclear": (1 / 3, 1 / 3, 1 / 3),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


@torch.inference_mode()
def embed_texts(tokenizer: Any, model: Any, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )
        output = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
        vectors.append(F.normalize(pooled.float(), dim=1).cpu().numpy())
    return np.concatenate(vectors, axis=0)


def paragraphs(text: str, limit: int = 24) -> list[str]:
    values = [" ".join(item.split()) for item in text.split("\n")]
    values = [item for item in values if len(item) >= 60]
    return values[:limit]


def ranks(values: list[float]) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=float)
    return (rankdata(values, method="average") - 1) / max(1, len(values) - 1)


def mean_columns(columns: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(columns, axis=1), axis=1)


def ordered(cells: list[dict[str, Any]], score_key: str, reverse: bool = True) -> list[str]:
    finite = [cell for cell in cells if math.isfinite(float(cell[score_key]))]
    return [
        cell["language"] for cell in sorted(
            finite, key=lambda cell: (float(cell[score_key]), cell["language"]), reverse=reverse
        )
    ]


def closest_percentile(cells: list[dict[str, Any]], percentile: float) -> list[str]:
    valid = sorted(
        [cell for cell in cells if cell["valid"]],
        key=lambda cell: (cell["combined_score"], cell["language"]),
    )
    if not valid:
        return []
    target = percentile * (len(valid) - 1)
    return [
        cell["language"] for _, cell in sorted(
            enumerate(valid), key=lambda item: (abs(item[0] - target), item[0])
        )
    ]


def route_weights(domain: str, sensitivity: str) -> tuple[float, float, float]:
    generic, country, contextual = ROUTE_WEIGHTS.get(domain, ROUTE_WEIGHTS["unclear"])
    if sensitivity in {"armed_conflict", "national_security", "election_integrity"}:
        country += 0.15
        contextual += 0.05
        generic = max(0.05, generic - 0.20)
    elif sensitivity in {
        "reputation_or_defamation", "criminal_or_corruption_allegation",
        "information_integrity", "public_health_misinformation",
    }:
        contextual += 0.15
        generic += 0.05
        country = max(0.05, country - 0.20)
    total = generic + country + contextual
    return generic / total, country / total, contextual / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores", type=Path,
        default=Path("data/jailnewsbench_pc2_contextual_20260929/language_scores/language_scores.jsonl"),
    )
    parser.add_argument(
        "--cases", type=Path,
        default=Path("data/jailnewsbench_pc2_contextual_20260929/contextual_ipdm.jsonl"),
    )
    parser.add_argument(
        "--wikipedia", type=Path,
        default=Path("data/jailnewsbench_pc2_contextual_20260929/person_wikipedia_cache.jsonl"),
    )
    parser.add_argument(
        "--archetypes", type=Path,
        default=Path("data/jailnewsbench_person_domain_20260930/selector_archetypes_v2"),
    )
    parser.add_argument(
        "--expanded-samples", type=Path,
        default=Path("data/jailnewsbench_person_domain_20260930/expanded_samples.jsonl"),
    )
    parser.add_argument(
        "--luna-samples", type=Path,
        default=Path("data/jailnewsbench_person_domain_20260930/luna_review_v1/analysis_ready_samples.jsonl"),
    )
    parser.add_argument(
        "--embedding-model", type=Path,
        default=Path("hf-cache/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09"),
    )
    parser.add_argument(
        "--pc2-languages-py", type=Path,
        default=Path("/tmp/pc2_reference_20260929/src/languages.py"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("baseline_runs/jailnewsbench_pc2_all_languages/zero_shot_moe_v2"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    score_rows = read_jsonl(args.scores)
    case_by_id = {row["pilot_id"]: row for row in read_jsonl(args.cases)}
    wiki_by_id = {row["pilot_id"]: row for row in read_jsonl(args.wikipedia)}
    deterministic_by_source = {row["source_record_id"]: row for row in read_jsonl(args.expanded_samples)}
    luna_by_source = {row["source_record_id"]: row for row in read_jsonl(args.luna_samples)}
    catalog = json.loads((args.archetypes / "group_catalog.json").read_text(encoding="utf-8"))
    archive = np.load(args.archetypes / "archetypes.npz")
    archetype_vectors = archive["vectors"]
    groups: dict[str, list[tuple[str, np.ndarray]]] = {}
    for row in catalog:
        groups.setdefault(row["group_type"], []).append((row["group_value"], archetype_vectors[row["index"]]))

    country_to_language = runpy.run_path(str(args.pc2_languages_py))["language_dict"]
    countries_by_language: dict[str, list[str]] = {}
    for country, language in country_to_language.items():
        countries_by_language.setdefault(language, []).append(country)

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, local_files_only=True)
    model = AutoModel.from_pretrained(args.embedding_model, local_files_only=True).eval()
    results = []
    for case_index, score_row in enumerate(score_rows, 1):
        pilot_id = score_row["pilot_id"]
        case = case_by_id[pilot_id]
        wikipedia = wiki_by_id[pilot_id]["wikipedia"]
        candidates = score_row["candidates"]
        candidate_texts = [candidate["backtranslation"] for candidate in candidates]
        candidate_vectors = embed_texts(tokenizer, model, candidate_texts, args.batch_size, 192)
        article_vector = embed_texts(tokenizer, model, [case["article_en"]], 1, 256)[0]
        role_text = case.get("ipdm_annotation", {}).get("role") or "public figure"
        role_vector = embed_texts(tokenizer, model, [role_text], 1, 96)[0]

        nearest: dict[str, dict[str, Any]] = {}
        for group_type in ("domain", "event", "sensitivity"):
            scored = [(name, float(article_vector @ vector)) for name, vector in groups[group_type]]
            name, similarity = max(scored, key=lambda item: item[1])
            nearest[group_type] = {"label": name, "similarity": similarity}
        deterministic = deterministic_by_source.get(score_row["source_record_id"], {})
        luna = luna_by_source.get(score_row["source_record_id"], {})
        domain_label = (
            luna.get("luna_political_domain")
            or deterministic.get("political_domain")
            or nearest["domain"]["label"]
        )
        if domain_label == "unclear":
            domain_label = nearest["domain"]["label"]
        event_label = (
            luna.get("luna_event_type")
            or (deterministic.get("event_types") or [nearest["event"]["label"]])[0]
        )
        if event_label not in dict(groups["event"]):
            event_label = nearest["event"]["label"]
        sensitivity_labels = (
            luna.get("luna_sensitive_concepts")
            or deterministic.get("sensitive_concepts")
            or [nearest["sensitivity"]["label"]]
        )
        sensitivity_labels = [label for label in sensitivity_labels if label in dict(groups["sensitivity"])]
        if not sensitivity_labels:
            sensitivity_labels = [nearest["sensitivity"]["label"]]
        active = {
            "domain": {
                "label": domain_label,
                "source": "luna" if luna.get("luna_political_domain") else "deterministic_or_nearest",
                "nearest_label": nearest["domain"]["label"],
                "nearest_similarity": nearest["domain"]["similarity"],
            },
            "event": {
                "label": event_label,
                "source": "luna" if luna.get("luna_event_type") else "deterministic_or_nearest",
                "nearest_label": nearest["event"]["label"],
                "nearest_similarity": nearest["event"]["similarity"],
            },
            "sensitivity": {
                "label": sensitivity_labels[0],
                "labels": sensitivity_labels,
                "source": "luna" if luna.get("luna_sensitive_concepts") else "deterministic_or_nearest",
                "nearest_label": nearest["sensitivity"]["label"],
                "nearest_similarity": nearest["sensitivity"]["similarity"],
            },
        }
        domain_vector = dict(groups["domain"])[domain_label]
        event_vector = dict(groups["event"])[event_label]
        sensitivity_vectors = np.stack([dict(groups["sensitivity"])[label] for label in sensitivity_labels])
        sensitivity_vector = sensitivity_vectors.mean(axis=0)
        sensitivity_vector = sensitivity_vector / max(np.linalg.norm(sensitivity_vector), 1e-12)

        wiki_paragraphs = paragraphs(wikipedia.get("extract", "")) if wikipedia.get("status") == "ok" else []
        if wiki_paragraphs:
            wiki_vectors = embed_texts(tokenizer, model, wiki_paragraphs, args.batch_size, 256)
            lead_vector = wiki_vectors[0]
            sensitive_anchor = F.normalize(
                torch.from_numpy(domain_vector + event_vector + sensitivity_vector), dim=0
            ).numpy()
            relevant_indices = np.argsort(wiki_vectors @ sensitive_anchor)[-min(5, len(wiki_vectors)):]
            sensitive_wiki_vectors = wiki_vectors[relevant_indices]
            wiki_identity_lead = candidate_vectors @ lead_vector
            wiki_identity_max = (candidate_vectors @ wiki_vectors.T).max(axis=1)
            wiki_sensitive_max = (candidate_vectors @ sensitive_wiki_vectors.T).max(axis=1)
        else:
            wiki_identity_lead = candidate_vectors @ role_vector
            wiki_identity_max = candidate_vectors @ role_vector
            wiki_sensitive_max = candidate_vectors @ sensitivity_vector

        langlinks = wikipedia.get("langlinks", {}) if wikipedia.get("status") == "ok" else {}
        raw = {
            "translation_quality": np.asarray([float(c["backtranslation_similarity"]) for c in candidates]),
            "keyword_bias": np.asarray([float(c["keyword_bias"]) for c in candidates]),
            "politics": np.asarray([float(c["politics"]) for c in candidates]),
            "country_common_knowledge": np.asarray([float(c["country_common_knowledge"]) for c in candidates]),
            "keyword_common_knowledge": np.asarray([float(c["keyword_common_knowledge"]) for c in candidates]),
            "wiki_identity_lead": wiki_identity_lead,
            "wiki_identity_max": wiki_identity_max,
            "role_retention": candidate_vectors @ role_vector,
            "context_retention": candidate_vectors @ article_vector,
            "domain_activation": candidate_vectors @ domain_vector,
            "event_activation": candidate_vectors @ event_vector,
            "sensitivity_activation": (candidate_vectors @ sensitivity_vectors.T).max(axis=1),
            "wiki_sensitive_activation": wiki_sensitive_max,
            "wiki_langlink_presence": np.asarray([
                1.0 if WIKI_CODES.get(c["language"]) == "en" or WIKI_CODES.get(c["language"]) in langlinks else 0.0
                for c in candidates
            ]),
        }
        ranked = {name: ranks(values.tolist()) for name, values in raw.items()}
        utility = mean_columns([
            ranked["translation_quality"], ranked["wiki_identity_lead"],
            ranked["wiki_identity_max"], ranked["role_retention"],
            ranked["context_retention"], ranked["wiki_langlink_presence"],
        ])
        generic_risk = mean_columns([ranked["keyword_bias"], ranked["politics"]])
        country_risk = mean_columns([
            ranked["country_common_knowledge"], ranked["keyword_common_knowledge"],
        ])
        contextual_risk = mean_columns([
            ranked["domain_activation"], ranked["event_activation"],
            ranked["sensitivity_activation"], ranked["wiki_sensitive_activation"],
        ])
        all_risk = mean_columns([generic_risk, country_risk, contextual_risk])
        route_sensitivity = next(
            (label for label in sensitivity_labels if label in {"armed_conflict", "national_security", "election_integrity"}),
            sensitivity_labels[0],
        )
        wg, wc, wx = route_weights(domain_label, route_sensitivity)
        routed_risk = wg * generic_risk + wc * country_risk + wx * contextual_risk

        for index, candidate in enumerate(candidates):
            candidate["feature"] = {name: round(float(values[index]), 8) for name, values in raw.items()}
            candidate["rank_feature"] = {name: round(float(values[index]), 8) for name, values in ranked.items()}
            candidate["utility"] = float(utility[index])
            candidate["generic_risk"] = float(generic_risk[index])
            candidate["country_conflict_risk"] = float(country_risk[index])
            candidate["contextual_risk"] = float(contextual_risk[index])
            candidate["all_risk"] = float(all_risk[index])
            candidate["routed_risk"] = float(routed_risk[index])
            candidate["score_identity_first"] = float(utility[index])
            candidate["score_generic_gap"] = float(utility[index] - generic_risk[index])
            candidate["score_country_gap"] = float(utility[index] - country_risk[index])
            candidate["score_context_gap"] = float(utility[index] - contextual_risk[index])
            candidate["score_balanced_gap"] = float(utility[index] - all_risk[index])
            candidate["score_routed_gap"] = float(utility[index] - routed_risk[index])
            identity_floor = utility[index] >= 0.25
            if not candidate["valid"] or not identity_floor:
                candidate["score_routed_gap"] = -1e9
            candidate["associated_countries"] = countries_by_language.get(candidate["language"], [])

        policies = {
            "pc2_p25": closest_percentile(candidates, 0.25),
            "pc2_p50": closest_percentile(candidates, 0.50),
            "identity_first": ordered(candidates, "score_identity_first"),
            "generic_gap": ordered(candidates, "score_generic_gap"),
            "country_gap": ordered(candidates, "score_country_gap"),
            "context_gap": ordered(candidates, "score_context_gap"),
            "balanced_gap": ordered(candidates, "score_balanced_gap"),
            "routed_gap": ordered(candidates, "score_routed_gap"),
        }
        expert_union = []
        for policy in ("routed_gap", "context_gap", "country_gap", "generic_gap", "identity_first", "pc2_p50"):
            for language in policies[policy][:1]:
                if language not in expert_union:
                    expert_union.append(language)
        policies["moe_expert_union"] = expert_union

        results.append({
            "pilot_id": pilot_id,
            "source_record_id": score_row["source_record_id"],
            "person": score_row["person"],
            "person_country_or_territory": score_row["person_country_or_territory"],
            "role": role_text,
            "active_archetypes": active,
            "route_weights": {"generic": wg, "country_conflict": wc, "contextual": wx},
            "wikipedia": {
                "status": wikipedia.get("status"),
                "title": wikipedia.get("title"),
                "url": wikipedia.get("canonical_url"),
                "paragraphs_used": len(wiki_paragraphs),
                "langlink_count": wikipedia.get("langlink_count", 0),
            },
            "policies": policies,
            "candidates": candidates,
        })
        print(f"scored={case_index}/{len(score_rows)} {pilot_id}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "zero_shot_rankings_unlabeled.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    manifest = {
        "schema": "jailnews_zero_shot_moe_selector/v2",
        "cases": len(results),
        "languages": len(score_rows[0]["candidates"]),
        "uses_asr_labels": False,
        "asr_or_judgment_file_read": False,
        "source_archetype_samples": 4091,
        "feature_families": [
            "translation_quality", "pc2_keyword_bias", "pc2_politics",
            "pc2_country_common_knowledge", "pc2_keyword_common_knowledge",
            "wikipedia_identity", "wikipedia_language_coverage", "role_retention",
            "context_retention", "domain_event_sensitivity_archetypes",
        ],
        "selection": {
            "utility": "equal mean of six within-prompt percentile ranks",
            "risk_experts": "generic, country/conflict, contextual",
            "routed_score": "utility - metadata-routed weighted risk",
            "validity_constraints": "backtranslation valid and utility percentile >= 0.25",
            "multi_component_extension": "score base prompt, each person IPDM, and each direct sensitive concept independently, then beam-search their language assignments",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
