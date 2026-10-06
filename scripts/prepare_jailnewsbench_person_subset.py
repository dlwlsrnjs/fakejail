#!/usr/bin/env python3
"""Extract auditable person-centered subsets from JailNewsBench.

The public dataset has region and language fields, but no structured person
field.  This script runs English NER over ``article_en`` and preserves the
original multilingual fields without modifying the benchmark instructions.
It writes both the broad person subset and the narrower Political subset so
that entity-IPDM experiments do not conflate political and non-political news.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import spacy


SOURCE_URL = "https://huggingface.co/datasets/MasahiroKaneko/JailNewsBench"
SOURCE_LICENSE = "CC-BY-4.0"


def jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized_entity(text: str) -> str:
    return " ".join(text.split()).strip(" \t\n,.;:()[]{}\"'“”‘’")


def accepted_person(text: str) -> bool:
    """Drop obvious NER fragments while retaining single-token public names."""
    value = normalized_entity(text)
    if not value or len(value) < 3 or value.isdigit():
        return False
    if value.casefold() in {
        "government", "president", "minister", "parliament", "congress",
        "senate", "police", "court", "party", "administration",
        "covid", "covid-19", "vaccine", "pfizer", "moderna", "twitter",
        "instagram", "spacex", "tesla", "jaxa", "seimas", "anvisa",
        "perseverance", "juno", "zhurong", "bennu", "brexit",
    }:
        return False
    if any(token in value.casefold().split() for token in {
        "telescope", "vaccine", "virus", "mission", "spacecraft",
    }):
        return False
    return any(ch.isalpha() for ch in value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(args.input).to_pylist()
    nlp = spacy.load(args.model, disable=["tagger", "parser", "lemmatizer"])
    texts = [str(row.get("article_en") or "") for row in rows]

    person_rows: list[dict[str, Any]] = []
    entity_counts: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    motivation_counts: Counter[str] = Counter()

    for row, doc in zip(
        rows,
        nlp.pipe(texts, batch_size=args.batch_size),
        strict=True,
    ):
        seen: set[str] = set()
        entities: list[dict[str, Any]] = []
        for ent in doc.ents:
            if ent.label_ != "PERSON" or not accepted_person(ent.text):
                continue
            name = normalized_entity(ent.text)
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            entities.append({
                "text": name,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "extractor": args.model,
                "label": "PERSON",
            })
            entity_counts[name] += 1
        if not entities:
            continue

        uid = str(row["uid"])
        record = {
            "collection_id": f"jailnewsbench:{args.split}:{uid}",
            "source_dataset": "JailNewsBench",
            "source_record_id": uid,
            "source_split": args.split,
            "source_url": SOURCE_URL,
            "source_license": SOURCE_LICENSE,
            "region_en": row["region_en"],
            "language_code": row["language_code"],
            "motivation_category": row["motivation_category"],
            "article_local": row["article_local"],
            "article_en": row["article_en"],
            "seed_instruction_local": row["seed_instruction_local"],
            "person_entities": entities,
            "person_extraction_status": "automatic_ner_requires_review",
            "prompt_sha256": hashlib.sha256(
                str(row["seed_instruction_local"]).encode("utf-8")
            ).hexdigest(),
        }
        person_rows.append(record)
        region_counts[str(row["region_en"])] += 1
        language_counts[str(row["language_code"])] += 1
        motivation_counts[str(row["motivation_category"])] += 1

    political_rows = [
        row for row in person_rows if row["motivation_category"] == "Political"
    ]
    jsonl_write(args.output / "person_records_all_motivations.jsonl", person_rows)
    jsonl_write(args.output / "person_records_political.jsonl", political_rows)

    political_regions = Counter(row["region_en"] for row in political_rows)
    political_languages = Counter(row["language_code"] for row in political_rows)
    political_entities = Counter(
        entity["text"]
        for row in political_rows
        for entity in row["person_entities"]
    )
    summary = {
        "source_rows": len(rows),
        "person_rows_all_motivations": len(person_rows),
        "person_row_rate_all_motivations": len(person_rows) / len(rows),
        "person_rows_political": len(political_rows),
        "political_source_rows": sum(
            row["motivation_category"] == "Political" for row in rows
        ),
        "person_rate_within_political": (
            len(political_rows)
            / max(1, sum(row["motivation_category"] == "Political" for row in rows))
        ),
        "all_person_motivation_counts": dict(motivation_counts.most_common()),
        "all_person_region_counts": dict(region_counts.most_common()),
        "all_person_language_counts": dict(language_counts.most_common()),
        "political_person_region_counts": dict(political_regions.most_common()),
        "political_person_language_counts": dict(political_languages.most_common()),
        "top_political_person_mentions": dict(political_entities.most_common(100)),
        "distinct_ner_strings_all": len(entity_counts),
        "distinct_ner_strings_political": len(political_entities),
        "caveat": (
            "spaCy NER is an automatic candidate extractor. Human or stronger "
            "entity-linking review is required before reporting unique-person counts."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    readme = f"""# JailNewsBench person-centered subset ({args.split})

This directory extracts explicit person mentions from the English article field
of the official JailNewsBench `{args.split}` split. The original multilingual
article and seed instruction are preserved verbatim. No new attack prompt is
generated here.

- Source rows: {len(rows):,}
- Rows with at least one PERSON candidate: {len(person_rows):,}
- Political rows with at least one PERSON candidate: {len(political_rows):,}
- Source: {SOURCE_URL}
- License: {SOURCE_LICENSE}

`person_records_all_motivations.jsonl` is the primary person-centered view.
The motivation field is retained for stratified analysis rather than used as
an exclusion criterion. `person_records_political.jsonl` is only an optional
derived view for comparisons that explicitly isolate political motivation.

All person annotations are automatic NER candidates and require review or
entity linking before unique-person statistics are treated as ground truth.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
