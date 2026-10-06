#!/usr/bin/env python3
"""Select a diverse, auditable JailNewsBench Political-person pilot.

This script only selects existing benchmark records.  It does not execute the
embedded manipulation request or create new claims about the named people.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/ljk98/POLY")
DEFAULT_INPUT = ROOT / "data/jailnewsbench_person_views"
DEFAULT_OUTPUT = ROOT / "data/jailnewsbench_pc2_person_pilot_20260929"
ROLE_RE = re.compile(
    r"\b(president|prime minister|minister|governor|senator|representative|"
    r"lawmaker|legislator|member of parliament|mayor|secretary|chancellor|"
    r"party leader|political leader|candidate|ambassador|attorney general|"
    r"deputy prime minister|congressman|congresswoman)\b",
    re.I,
)
NON_POLITICAL_ROLE_RE = re.compile(
    r"\b(university|company|corporation|institute|association|safety representative|"
    r"public health director|research director|vice-president of the university)\b",
    re.I,
)
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žА-Яа-я'-]+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def candidate(row: dict[str, Any]) -> tuple[bool, float]:
    entities = row.get("person_entities") or []
    if len(entities) != 1:
        return False, 0.0
    name = entities[0]["text"].strip()
    if name.endswith("'s") or name.endswith("’s"):
        return False, 0.0
    words = TOKEN_RE.findall(name)
    if not 2 <= len(words) <= 5:
        return False, 0.0
    article = str(row.get("article_en") or "")
    if name.casefold() not in article.casefold() or not ROLE_RE.search(article):
        return False, 0.0
    if not 30 <= len(article) <= 900:
        return False, 0.0
    match = re.search(re.escape(name), article, re.I)
    local = article[max(0, match.start() - 140): min(len(article), match.end() + 140)] if match else article
    suffix = article[match.end(): match.end() + 12] if match else ""
    if re.match(r"-[A-Za-zÀ-ÖØ-öø-ÿĀ-ž]{1,8}\b", suffix):
        return False, 0.0
    if NON_POLITICAL_ROLE_RE.search(local):
        return False, 0.0
    near_role = bool(ROLE_RE.search(local))
    score = 3.0 if near_role else 1.0
    score += 1.0 / (1.0 + abs(len(article) - 260) / 260)
    score += 0.25 if len(words) >= 2 else 0.0
    return True, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pool: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        path = args.input_root / split / "person_records_political.jsonl"
        for row in read_jsonl(path):
            ok, score = candidate(row)
            if ok:
                pool.append({**row, "selection_split": split, "selection_score": score})

    rng.shuffle(pool)
    pool.sort(key=lambda row: row["selection_score"], reverse=True)

    selected: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    region_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    quotas = {"train": args.n // 2, "validation": args.n // 4, "test": args.n - 3 * (args.n // 4)}

    while len(selected) < args.n:
        best = None
        best_score = float("-inf")
        for row in pool:
            name = row["person_entities"][0]["text"].casefold()
            split = row["selection_split"]
            if name in seen_names or split_counts[split] >= quotas[split]:
                continue
            diversity = (
                2.5 / (1 + region_counts[row["region_en"]])
                + 2.0 / (1 + language_counts[row["language_code"]])
            )
            value = row["selection_score"] + diversity
            if value > best_score:
                best, best_score = row, value
        if best is None:
            raise RuntimeError(f"could select only {len(selected)} of {args.n} records")
        name = best["person_entities"][0]["text"]
        selected.append(
            {
                **best,
                "pilot_id": f"JNBPC2_{len(selected):03d}",
                "selected_person": name,
                "article_sha256": hashlib.sha256(best["article_en"].encode("utf-8")).hexdigest(),
                "seed_instruction_sha256": hashlib.sha256(
                    best["seed_instruction_local"].encode("utf-8")
                ).hexdigest(),
            }
        )
        seen_names.add(name.casefold())
        region_counts[best["region_en"]] += 1
        language_counts[best["language_code"]] += 1
        split_counts[best["selection_split"]] += 1

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "selected_records.jsonl", selected)
    manifest = {
        "schema": "jailnewsbench_pc2_person_pilot/v1",
        "source_population": 24040,
        "pilot_size": len(selected),
        "seed": args.seed,
        "split_counts": dict(split_counts),
        "region_counts": dict(region_counts),
        "language_counts": dict(language_counts),
        "selection_requirements": [
            "Political motivation row",
            "exactly one automatic PERSON candidate",
            "two to five name tokens",
            "political/public-office role appears in article context",
            "unique selected name",
        ],
        "required_review": (
            "Automatic NER and public-role heuristics are not entity linking; each selected person "
            "and generated clue must pass manual review before any target-model study."
        ),
        "planned_conditions": [
            "direct_name",
            "english_ipdm",
            "random_valid_language_ipdm",
            "pc2_ranked_language_ipdm",
        ],
        "safe_endpoint": (
            "entity/country/manipulation-intent reconstruction and refusal response only; "
            "the pilot does not request a fabricated article about a real person"
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
