#!/usr/bin/env python3
"""Prepare all JailNews A-middle cases using only reviewed Luna surface spans.

No canonical-name search or fuzzy surname replacement is used.  Every replaced
span is traceable to ``analysis_ready_samples.jsonl:luna_people[].surface``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    tmp.replace(path)
    return count


def redact(text: str, people: list[dict[str, str]]) -> tuple[str, list[dict[str, Any]]]:
    audit = []
    output = text
    unique = {(row["surface"], row.get("canonical_person", "")): row for row in people}
    for surface, canonical in sorted(unique, key=lambda value: (-len(value[0]), value[0].casefold())):
        # Unicode word boundaries avoid changing a surface embedded in an
        # unrelated word, but punctuation and inflection are left untouched.
        # A reviewed span ending in punctuation can be a tokenizer-truncated
        # prefix (the dataset contains ``Lee Jun-`` for ``Lee Jun-seok``).
        # Extend only the observed attached word, never from a canonical name.
        anchored_extension = ""
        if canonical.casefold().startswith(surface.casefold()) and len(canonical) > len(surface):
            anchored_extension = canonical[len(surface):]
        elif canonical:
            surface_tokens = surface.split()
            canonical_tokens = canonical.split()
            last = surface_tokens[-1].casefold() if surface_tokens else ""
            positions = [i for i, token in enumerate(canonical_tokens) if token.casefold() == last]
            if positions and positions[-1] + 1 < len(canonical_tokens):
                anchored_extension = " " + " ".join(canonical_tokens[positions[-1] + 1:])
            else:
                # Honorific-only surfaces such as "Datuk Seri" are followed
                # by the complete reviewed person's name in the article.
                anchored_extension = " " + canonical
        if anchored_extension:
            # The canonical label is never scanned independently.  It is used
            # only to extend a reviewed Luna surface at the exact same start
            # position when the article literally contains the continuation.
            suffix = re.escape(anchored_extension) + r"(?!\w)"
            rule = "luna_surface_anchor_plus_observed_contiguous_extension"
        elif surface and not surface[-1].isalnum():
            suffix = r"\w+"
            rule = "surface_plus_attached_word"
        else:
            suffix = r"(?!\w)"
            rule = "exact_luna_surface"
        pattern = re.compile(r"(?<!\w)" + re.escape(surface) + suffix, re.IGNORECASE)
        output, count = pattern.subn("A", output)
        # If the strict contiguous extension is absent (for example a local
        # instruction uses a shorter observed spelling), fall back to the
        # reviewed surface itself, not to a global canonical-name search.
        if not count and anchored_extension:
            fallback = re.compile(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", re.IGNORECASE)
            output, count = fallback.subn("A", output)
            rule = "exact_luna_surface_fallback"
        audit.append({
            "surface": surface, "replacements": count,
            "match_rule": rule,
        })
    output = re.sub(r"(?<!\w)A(?:\s+A)+(?!\w)", "A", output)
    return output, audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/jailnewsbench_person_domain_20260930/luna_review_v1/analysis_ready_samples.jsonl")
    ap.add_argument("--entity-map", type=Path, required=True)
    ap.add_argument("--entities", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    mapping = {row["dataset_label"]: row for row in read(args.entity_map)}
    entities = {row["entity_id"]: row for row in read(args.entities)}
    cases, failures = [], []
    description_inventory: dict[str, dict[str, Any]] = {}
    stats = Counter()
    for source in read(args.contexts):
        sample_id = source["sample_id"]
        # Luna surface offsets were reviewed against article_en.  Using that
        # same source is what makes replacement traceable and prevents a
        # canonical name from being substituted for the observed span.
        article = str(source.get("article_en") or "")
        embedded_article = str(source.get("article_local") or "")
        behavior = str(source.get("seed_instruction_local") or "")
        surface_people = [{
            "surface": str(person.get("surface") or "").strip(),
            "canonical_person": str(person.get("canonical_person") or "").strip(),
        } for person in source.get("luna_people", []) if str(person.get("surface") or "").strip()]
        surfaces = [row["surface"] for row in surface_people]
        if not article or not embedded_article or behavior.count(embedded_article) != 1:
            failures.append({"sample_id": sample_id, "reason": "embedded_article_not_unique", "occurrences": behavior.count(embedded_article)})
            continue
        if not surfaces:
            failures.append({"sample_id": sample_id, "reason": "no_luna_person_surface"})
            continue
        instruction = behavior.replace(embedded_article, "[ARTICLE]", 1)
        clean_article, article_audit = redact(article, surface_people)
        clean_instruction, instruction_audit = redact(instruction, surface_people)
        if not any(row["replacements"] for row in article_audit + instruction_audit):
            failures.append({"sample_id": sample_id, "reason": "surface_not_found", "surfaces": surfaces})
            continue
        primary = source.get("luna_primary_person")
        if not primary and source.get("luna_people"):
            primary = source["luna_people"][0].get("canonical_person")
        mapped = mapping.get(primary or "")
        if mapped is None:
            # The one historical missing-primary Bolsonaro row is recoverable
            # from the reviewed surface, without scanning arbitrary names.
            surface_matches = [mapping[value] for value in surfaces if value in mapping]
            mapped = surface_matches[0] if len(surface_matches) == 1 else None
        entity_id = mapped["entity_id"] if mapped else f"unresolved-sample:{sample_id}"
        entity = entities.get(entity_id, {})
        description = str(entity.get("description") or "").strip()
        if not description:
            people = source.get("luna_people", [])
            role = str(source.get("luna_primary_person_role") or (people[0].get("role") if people else "public figure"))
            countries = [country for person in people for country in person.get("countries_or_territories", [])]
            place = countries[0] if countries else str(source.get("region_en") or "the reported region")
            description = f"a public figure associated with {place}; role category: {role.replace('_', ' ')}"
            stats["fallback_descriptions"] += 1
        description_inventory.setdefault(entity_id, {
            "item_id": f"entity_description:{entity_id}", "entity_id": entity_id,
            "wikidata_qid": mapped.get("wikidata_qid") if mapped else None,
            "canonical_english": description, "item_type": "person_description",
        })
        cases.append({
            "sample_id": sample_id, "source_record_id": source["source_record_id"],
            "source_language_code": source.get("language_code"), "primary_dataset_label": primary,
            "entity_id": entity_id, "wikidata_qid": mapped.get("wikidata_qid") if mapped else None,
            "luna_people_surfaces": surfaces, "article": clean_article,
            "instruction": clean_instruction, "description_item_id": f"entity_description:{entity_id}",
            "prompt_template": "ARTICLE\n{article}\n\nA: {translated_person_description}\n\nINSTRUCTION\n{instruction}",
            "article_reference_marker": "[ARTICLE]", "replacement_symbol": "A",
            "surface_replacement_audit": {"article": article_audit, "instruction": instruction_audit},
            "article_sha256": hashlib.sha256(clean_article.encode()).hexdigest(),
            "instruction_sha256": hashlib.sha256(clean_instruction.encode()).hexdigest(),
        })
        stats["prepared"] += 1
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "person_cases.jsonl", sorted(cases, key=lambda row: row["sample_id"]))
    write(args.output / "description_translation_inventory.jsonl", sorted(description_inventory.values(), key=lambda row: row["item_id"]))
    write(args.output / "prepare_failures.jsonl", failures)
    manifest = {
        "schema": "jailnews_a_middle_surface/v2", "contexts_seen": len(cases) + len(failures),
        "prepared_cases": len(cases), "failures": len(failures),
        "description_items": len(description_inventory), "fallback_descriptions": stats["fallback_descriptions"],
        "replacement_source": "luna_people[].surface only", "prompt_order": ["ARTICLE", "A", "INSTRUCTION"],
        "article_source": "article_en (the text against which Luna surface spans were reviewed)",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
