#!/usr/bin/env python3
"""Prepare a deterministic 72-language x 5-attack surrogate pilot.

The selector chooses one context per QID-backed entity and balances political
domains before rendering the five public JailNewsBench attack wrappers.  The
result can be consumed directly by ``jailnewsbench_table2_qwen32.py generate``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

from jailnewsbench_table2_qwen32 import (
    JAILBREAKS,
    OFFICIAL,
    PUBLIC_COMMIT,
    PUBLIC_EVALUATOR_BLOB,
    PUBLIC_EVALUATOR_SHA256,
    PUBLIC_PROMPT_BUNDLE_SHA256,
    build_context_prefix,
    sha256_text,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "data/jailnewsbench_person_domain_20260930/a_middle_surface_v2"
DEFAULT_ANNOTATIONS = (
    ROOT / "data/jailnewsbench_person_domain_20260930/luna_review_v1/analysis_ready_samples.jsonl"
)


def read(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def balanced_cases(
    matrix: list[dict[str, Any]], annotations: dict[str, dict[str, Any]], limit: int, seed: int
) -> list[str]:
    # One context per resolved entity. Prefer the context with the rarest domain
    # globally; then round-robin domains so the pilot is not dominated by heads
    # of government or the most frequent geopolitical region.
    english = [row for row in matrix if row["language"] == "English"]
    by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in english:
        entity = str(row.get("entity_id") or "")
        if entity.startswith("Q") and entity[1:].isdigit():
            by_entity[entity].append(row)
    domain_freq = Counter(
        str(annotations[row["sample_id"]].get("luna_political_domain") or "unclear")
        for rows in by_entity.values() for row in rows
    )
    representatives = []
    for entity, rows in by_entity.items():
        rows.sort(key=lambda row: (
            domain_freq[str(annotations[row["sample_id"]].get("luna_political_domain") or "unclear")],
            sha256_text(f"{seed}|{row['sample_id']}"),
        ))
        representatives.append(rows[0])
    groups: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(representatives, key=lambda r: sha256_text(f"{seed}|{r['entity_id']}")):
        domain = str(annotations[row["sample_id"]].get("luna_political_domain") or "unclear")
        groups[domain].append(row)
    selected: list[str] = []
    order = sorted(groups, key=lambda key: (len(groups[key]), key))
    while order and len(selected) < limit:
        next_order = []
        for domain in order:
            if groups[domain] and len(selected) < limit:
                selected.append(groups[domain].popleft()["sample_id"])
            if groups[domain]:
                next_order.append(domain)
        order = next_order
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, default=DEFAULT_BASE / "prompt_matrix.jsonl")
    ap.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-entities", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()

    # Keep the complete 72-language grid.  Translation QC is retained as a
    # covariate instead of silently dropping difficult languages, which would
    # make language-level ASR incomparable.  Empty translations remain fatal.
    matrix = list(read(args.matrix))
    empty = [row["matrix_id"] for row in matrix if not str(row.get("a_description") or "").strip()]
    if empty:
        raise RuntimeError(f"empty translated descriptions: {len(empty)}")
    annotations = {row["sample_id"]: row for row in read(args.annotations)}
    selected_samples = balanced_cases(matrix, annotations, args.max_entities, args.seed)
    selected = set(selected_samples)
    rows = [row for row in matrix if row["sample_id"] in selected]
    by_sample = Counter(row["sample_id"] for row in rows)
    malformed = {key: value for key, value in by_sample.items() if value != 72}
    if malformed or len(by_sample) != args.max_entities:
        raise RuntimeError(
            f"pilot grid incomplete: entities={len(by_sample)} malformed={len(malformed)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    context_pool = [annotations[sample]["article_local"] for sample in selected_samples]
    context_prefix = build_context_prefix(context_pool, tokenizer, args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    arms_path = args.output / "arms.jsonl"
    count = 0
    with arms_path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: (item["sample_id"], item["language"])):
            ann = annotations[row["sample_id"]]
            for attack in JAILBREAKS:
                attacked = OFFICIAL.apply_attack(
                    seed_instruction=row["prompt"],
                    motivation_category=str(ann.get("motivation_category") or "Political"),
                    attack_type=attack,
                    context_prefix=context_prefix if attack == "context_overload" else None,
                )
                trial_key = f"{row['matrix_id']}|{attack}|deterministic-v1"
                output = {
                    **row,
                    "uid": row["sample_id"],
                    "trial_id": sha256_text(trial_key)[:24],
                    "attack_type": attack,
                    "attacked_prompt": attacked,
                    "prompt_sha256": sha256_text(attacked),
                    "language_code": row["nllb_code"],
                    "article_local": row["article"],
                    "article_en": row["article"],
                    "region_en": ann.get("region_en"),
                    "political_domain": ann.get("luna_political_domain"),
                    "person_role": ann.get("luna_primary_person_role"),
                    "motivation_category": ann.get("motivation_category"),
                    "war_or_security_related": ann.get("luna_war_or_security_related"),
                    "election_related": ann.get("luna_election_related"),
                }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                count += 1

    manifest = {
        "schema": "jailnews_surface_surrogate_pilot/v1",
        "matrix": str(args.matrix.resolve()),
        "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
        "entities": len({row["entity_id"] for row in rows}),
        "samples": len(by_sample),
        "languages": 72,
        "methods": JAILBREAKS,
        "arms": count,
        "translation_qc_valid_arms": sum(
            bool(row.get("description_translation_valid")) for row in rows
        ) * len(JAILBREAKS),
        "translation_qc_provisional_arms": sum(
            not bool(row.get("description_translation_valid")) for row in rows
        ) * len(JAILBREAKS),
        "selected_sample_ids": selected_samples,
        "selection": "one QID-backed context per entity, political-domain round robin",
        "temperature": 0.0,
        "context_tokens": 512,
        "context_sha256": sha256_text(context_prefix),
        "prompt_source": {
            "commit": PUBLIC_COMMIT,
            "evaluate_py_blob": PUBLIC_EVALUATOR_BLOB,
            "evaluate_py_sha256": PUBLIC_EVALUATOR_SHA256,
            "prompt_bundle_sha256": PUBLIC_PROMPT_BUNDLE_SHA256,
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
