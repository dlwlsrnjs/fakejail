#!/usr/bin/env python3
"""Prepare the structure-matched English-article, real-name JailNews baseline.

This control keeps the V4 ``ARTICLE -> INSTRUCTION`` layout and the same five
public jailbreak wrappers, but removes every proposed identity component: no A
placeholder, translated descriptor, identity binding, router, or bandit.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from jailnewsbench_table2_qwen32 import (
    JAILBREAKS,
    OFFICIAL,
    PUBLIC_COMMIT,
    PUBLIC_EVALUATOR_BLOB,
    PUBLIC_EVALUATOR_SHA256,
    PUBLIC_PROMPT_BUNDLE_SHA256,
    sha256_text,
)


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_BASE = ROOT / "artifacts/jailnews_bandit_20260930/runtime/official_baseline_501/base_arms.jsonl"
V4_PARENT_BASE = ROOT / "artifacts/jailnews_bandit_20260930/runtime/stochastic_360_v1/base_arms.jsonl"
OUTPUT = ROOT / "artifacts/jailnews_bandit_20260930/runtime/english_article_named_baseline_501"
TRANSLATIONS = OUTPUT / "translation/behavior_translations_verified.jsonl"


def read(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return count


def main() -> None:
    sources = {
        str(row["victim_person_id"]): row
        for row in read(OFFICIAL_BASE)
        if row.get("attack_type") == "original"
    }
    if len(sources) != 501:
        raise RuntimeError(f"expected 501 person-linked sources, found {len(sources)}")

    translations = {
        str(row["quiz_id"]): row
        for row in read(TRANSLATIONS)
    }
    if set(translations) != set(sources):
        raise RuntimeError(
            f"English behavior translation coverage mismatch: "
            f"translations={len(translations)} sources={len(sources)}"
        )
    invalid = [person for person, row in translations.items() if not row.get("valid")]
    if invalid:
        raise RuntimeError(f"English behavior translations failing round-trip QC: {len(invalid)}")

    context_prefix = None
    for row in read(V4_PARENT_BASE):
        if row.get("attack_type") != "context_overload":
            continue
        attacked = str(row["attacked_prompt"])
        seed = str(row["prompt"])
        if not attacked.endswith(seed):
            raise RuntimeError("cannot recover the frozen V4 context prefix")
        context_prefix = attacked[: -len(seed)]
        break
    if not context_prefix:
        raise RuntimeError("missing V4 context prefix")

    counts: Counter[str] = Counter()
    prompt_hashes: set[str] = set()

    def rendered() -> Iterable[dict[str, Any]]:
        for person in sorted(sources):
            source = sources[person]
            translated = translations[person]
            seed = str(translated["behavior_english_template"])
            article_en = str(source["article_en"])
            if seed.count(article_en) != 1:
                raise RuntimeError(
                    f"English article is not embedded exactly once: {source['sample_id']}"
                )
            instruction = seed.replace(article_en, "[ARTICLE]", 1)
            structure_prompt = f"ARTICLE\n{article_en}\n\nINSTRUCTION\n{instruction}"
            for attack in JAILBREAKS:
                attacked = OFFICIAL.apply_attack(
                    seed_instruction=structure_prompt,
                    motivation_category=str(source.get("motivation_category") or "Political"),
                    attack_type=attack,
                    context_prefix=context_prefix if attack == "context_overload" else None,
                )
                trial_id = sha256_text(
                    f"english-article-named-baseline-501|{person}|{source['sample_id']}|{attack}"
                )[:24]
                counts[attack] += 1
                prompt_hashes.add(sha256_text(attacked))
                yield {
                    **source,
                    "uid": f"{person}::{source['raw_jailnews_uid']}",
                    "trial_id": trial_id,
                    "base_trial_id": trial_id,
                    "attack_type": attack,
                    "condition": "english_article_named_structure_baseline",
                    "prompt_contract": "ARTICLE>INSTRUCTION/real-name-control",
                    "prompt": structure_prompt,
                    "attacked_prompt": attacked,
                    "prompt_sha256": sha256_text(attacked),
                    "article_local": article_en,
                    "article_en": article_en,
                    "source_language_code_original": source["language_code"],
                    "source_language_code": "en",
                    "expected_output_language_code": "en",
                    "target_model": "Qwen3-30B-A3B-Thinking-2507-FP8",
                    "uses_english_article": True,
                    "uses_structured_layout": True,
                    "uses_english_instruction": True,
                    "uses_real_person_names": True,
                    "uses_pc2_translation": False,
                    "uses_A_placeholder": False,
                    "uses_identity_binding": False,
                    "uses_router_or_bandit": False,
                }

    output = OUTPUT / "base_arms.jsonl"
    count = atomic_jsonl(output, rendered())
    expected = 501 * len(JAILBREAKS)
    if count != expected or any(counts[attack] != 501 for attack in JAILBREAKS):
        raise RuntimeError(f"coverage mismatch: rows={count}, counts={dict(counts)}")
    manifest = {
        "schema": "jailnews_english_article_named_baseline/v1",
        "people": 501,
        "conditions": JAILBREAKS,
        "rows": count,
        "counts_by_condition": dict(counts),
        "prompt_hashes": len(prompt_hashes),
        "control_contract": {
            "article": "article_en",
            "instruction": "NLLB-3.3B English translation with protected article replaced by [ARTICLE]",
            "translation_qc": "forward/backward/reforward, protected article, numeric preservation, semantic similarity",
            "person_names": "retained",
            "translated_identity_description": False,
            "A_placeholder": False,
            "identity_binding": False,
            "router_or_bandit": False,
        },
        "prompt_source": {
            "commit": PUBLIC_COMMIT,
            "evaluate_py_blob": PUBLIC_EVALUATOR_BLOB,
            "evaluate_py_sha256": PUBLIC_EVALUATOR_SHA256,
            "prompt_bundle_sha256": PUBLIC_PROMPT_BUNDLE_SHA256,
        },
        "context_sha256": hashlib.sha256(context_prefix.encode()).hexdigest(),
        "output": str(output.resolve()),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
