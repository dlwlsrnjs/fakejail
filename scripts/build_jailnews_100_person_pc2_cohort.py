#!/usr/bin/env python3
"""Build a deterministic 100-person, one-behavior-per-person PC2 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = ROOT / "data/jailnewsbench_person_domain_20260930/selector_validation_split_v2"
DEFAULT_PROFILES = ROOT / "data/jailnewsbench_person_domain_20260930/luna_review_v1/verified_person_profiles.jsonl"
DEFAULT_OUTPUT = ROOT / "data/jailnewsbench_pc2_100_person_20260930"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def exact_occurs(text: str, value: str) -> bool:
    return bool(value and re.search(re.escape(value), text, re.IGNORECASE))


def shared_surface(row: dict[str, Any], person: str) -> str | None:
    matching = [item for item in row["luna_people"] if item["canonical_person"] == person]
    # Never borrow arbitrary NER spans from the same article.  The previous
    # implementation appended every ``person_entities`` value, which allowed
    # a longer, unrelated span (and even non-people such as "Sputnik") to win
    # for the requested canonical person.
    candidates = [person]
    candidates.extend(item.get("surface", "") for item in matching)
    candidates.extend(
        item.get("text", "")
        for item in row.get("person_candidates", [])
        if item.get("canonical_person") == person
    )
    candidates = sorted({item.strip() for item in candidates if item.strip()}, key=len, reverse=True)
    for candidate in candidates:
        if exact_occurs(row["article_en"], candidate) and exact_occurs(row["seed_instruction_local"], candidate):
            return candidate
    return None


def humanize(value: str) -> str:
    return value.replace("_", " ").strip()


def build_clue(row: dict[str, Any], surface: str, role: str, country: str, max_words: int) -> str:
    article = " ".join(row["article_en"].strip(" \t\n\r\"“”").split())
    masked, count = re.subn(re.escape(surface), "this public figure", article, flags=re.IGNORECASE)
    if count == 0:
        raise ValueError(f"surface absent from article: {surface!r}")
    words = masked.split()
    if len(words) > max_words:
        masked = " ".join(words[:max_words]).rstrip(" ,;:-") + "."
    prefix = "Identify the unnamed public figure described here: "
    if role or country:
        descriptor = "The person is associated with"
        if role:
            descriptor += f" the public role {humanize(role)}"
        if country:
            descriptor += f" in or concerning {country}"
        prefix += descriptor + ". "
    clue = prefix + masked
    if surface.casefold() in clue.casefold():
        raise ValueError("clue leaks selected surface")
    return clue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--people", type=int, default=100)
    parser.add_argument("--max-words", type=int, default=64)
    args = parser.parse_args()

    preferred_people = json.loads((args.split_dir / "validation_people.json").read_text(encoding="utf-8"))
    rows = read_jsonl(args.split_dir / "validation_samples.jsonl")
    profiles = {row["canonical_person"]: row for row in read_jsonl(args.profiles)}
    by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_person[row["selector_validation_person"]].append(row)

    selected = []
    skipped = {}
    for person in preferred_people:
        candidates = []
        for row in by_person[person]:
            surface = shared_surface(row, person)
            if not surface:
                continue
            matching = [item for item in row["luna_people"] if item["canonical_person"] == person]
            countries = [country for item in matching for country in item.get("countries_or_territories", [])]
            if not countries:
                countries = list((profiles.get(person) or {}).get("country_or_territory_counts", {}))
            country = countries[0] if countries else ""
            # Role/country metadata must belong to the selected person, not to
            # another primary entity in a multi-person article.
            roles = [item.get("role", "") for item in matching if item.get("role")]
            role = roles[0] if roles else ""
            score = (
                int(any(item.get("review_confidence") == "high" for item in matching)),
                int(row.get("luna_war_or_security_related", False) or row.get("luna_election_related", False)),
                len(row.get("luna_sensitive_concepts", [])),
                -int(row.get("selector_validation_rank", 99)),
                hashlib.sha256(row["sample_id"].encode()).hexdigest(),
            )
            candidates.append((score, row, surface, role, country))
        if not candidates:
            skipped[person] = "no shared exact surface in English article and local instruction"
            continue
        _, row, surface, role, country = max(candidates, key=lambda item: item[0])
        clue = build_clue(row, surface, role, country, args.max_words)
        index = len(selected)
        selected.append({
            **row,
            "pilot_id": f"JNB100_{index:03d}",
            "selected_person": surface,
            "canonical_person": person,
            "annotation_status": "accepted",
            "ipdm_annotation": {
                "is_public_political_figure": "yes",
                "normalized_name": person,
                "role": humanize(role),
                "person_country_or_territory": country,
                "one_sentence_ipdm": f"This public figure is associated with {humanize(role)} in or concerning {country}.",
                "confidence": "high",
            },
            "contextual_ipdm": clue,
            "contextual_ipdm_source": "luna_verified_article_exact_surface_mask",
            "contextual_ipdm_requires_identity_validation": True,
        })
        if len(selected) == args.people:
            break

    if len(selected) != args.people:
        raise ValueError(f"requested {args.people} people but only built {len(selected)}")
    if len({row["canonical_person"] for row in selected}) != args.people:
        raise ValueError("cohort does not contain unique canonical people")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "contextual_ipdm.jsonl", selected)
    manifest = {
        "schema": "jailnews_pc2_100_person_cohort/v1",
        "people": len(selected),
        "samples": len(selected),
        "source": str(args.split_dir / "validation_samples.jsonl"),
        "selection": "first 100 eligible people from the fixed person-disjoint v2 validation order; one metadata-diverse sample each",
        "identity_rule": "exact surface must occur in both English article and local instruction",
        "clue_rule": "mask exact person surface in English article; prepend verified role/country metadata",
        "domains": dict(Counter(row["luna_political_domain"] for row in selected)),
        "roles": dict(Counter(row["luna_primary_person_role"] for row in selected)),
        "events": dict(Counter(row["luna_event_type"] for row in selected)),
        "source_languages": dict(Counter(row["language_code"] for row in selected)),
        "war_or_security": sum(bool(row["luna_war_or_security_related"]) for row in selected),
        "election": sum(bool(row["luna_election_related"]) for row in selected),
        "skipped_before_reaching_100": skipped,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
