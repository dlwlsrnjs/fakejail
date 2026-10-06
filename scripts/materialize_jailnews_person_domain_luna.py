#!/usr/bin/env python3
"""Materialize Luna-reviewed person/domain data into analysis-ready tables."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("data/jailnewsbench_person_domain_20260930")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def flatten(relation: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    annotation = review["annotation"]
    return {
        **relation,
        "luna_is_person_reference": annotation["is_person_reference"],
        "luna_canonical_person": annotation["canonical_person"],
        "luna_person_role": annotation["person_role"],
        "luna_role_title_evidence": annotation["role_title_evidence"],
        "luna_countries_or_territories": annotation["countries_or_territories"],
        "luna_political_domain": annotation["political_domain"],
        "luna_political_subdomains": annotation["political_subdomains"],
        "luna_event_type": annotation["event_type"],
        "luna_neutral_event_summary": annotation["neutral_event_summary"],
        "luna_sensitive_concepts": annotation["sensitive_concepts"],
        "luna_election_related": annotation["election_related"],
        "luna_war_or_security_related": annotation["war_or_security_related"],
        "luna_conflicts_named": annotation["conflicts_named"],
        "luna_time_expressions": annotation["time_expressions"],
        "luna_experiment_eligibility": annotation["experiment_eligibility"],
        "luna_review_confidence": annotation["review_confidence"],
        "luna_review_notes": annotation["review_notes"],
        "luna_model": review.get("api", {}).get("model_returned"),
        "luna_response_id": review.get("api", {}).get("response_id"),
    }


def write_relation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "relation_id", "sample_id", "source_split", "source_record_id", "region_en",
        "language_code", "person_surface", "canonical_person", "luna_canonical_person",
        "luna_is_person_reference", "luna_person_role", "luna_countries_or_territories",
        "political_domain", "luna_political_domain", "luna_political_subdomains",
        "luna_event_type", "luna_sensitive_concepts", "luna_election_related",
        "luna_war_or_security_related", "luna_conflicts_named", "luna_time_expressions",
        "luna_experiment_eligibility", "luna_review_confidence", "article_en_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = {field: row.get(field, "") for field in fields}
            for field, value in list(item.items()):
                if isinstance(value, (list, dict)):
                    item[field] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            writer.writerow(item)


def main() -> None:
    queue_rows = list(read_jsonl(ROOT / "person_domain_balanced_queue.jsonl"))
    reviews = list(read_jsonl(ROOT / "luna_review_v1" / "reviews.jsonl"))
    samples_all = list(read_jsonl(ROOT / "expanded_samples.jsonl"))

    relation_by_id: dict[str, dict[str, Any]] = {}
    queue_duplicate_counts: Counter[str] = Counter()
    for row in queue_rows:
        queue_duplicate_counts[row["relation_id"]] += 1
        relation_by_id.setdefault(row["relation_id"], row)
    sample_by_id: dict[str, dict[str, Any]] = {}
    sample_id_counts: Counter[str] = Counter()
    for row in samples_all:
        sample_id_counts[row["sample_id"]] += 1
        sample_by_id.setdefault(row["sample_id"], row)

    merged = [
        flatten(relation_by_id[review["relation_id"]], review)
        for review in reviews
        if review.get("status") == "ok" and review["relation_id"] in relation_by_id
    ]
    merged.sort(key=lambda row: (row["sample_id"], row["luna_canonical_person"], row["relation_id"]))
    verified = [
        row for row in merged
        if row["luna_is_person_reference"] == "yes"
        and row["luna_experiment_eligibility"] in {"eligible_person_centered", "eligible_secondary_person"}
    ]

    out = ROOT / "luna_review_v1"
    write_jsonl(out / "all_reviewed_person_sample_relations.jsonl", merged)
    write_jsonl(out / "verified_person_sample_relations.jsonl", verified)
    write_relation_csv(out / "verified_person_sample_relations.csv", verified)

    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in verified:
        by_sample[row["sample_id"]].append(row)
    analysis_samples: list[dict[str, Any]] = []
    eligibility_rank = {"eligible_person_centered": 2, "eligible_secondary_person": 1}
    confidence_rank = {"high": 3, "medium": 2, "low": 1}
    for sample_id, relations in by_sample.items():
        primary = max(
            relations,
            key=lambda row: (
                eligibility_rank[row["luna_experiment_eligibility"]],
                confidence_rank[row["luna_review_confidence"]],
            ),
        )
        sample = sample_by_id[sample_id]
        people = []
        for row in relations:
            people.append({
                "canonical_person": row["luna_canonical_person"],
                "surface": row["person_surface"],
                "role": row["luna_person_role"],
                "countries_or_territories": row["luna_countries_or_territories"],
                "eligibility": row["luna_experiment_eligibility"],
                "review_confidence": row["luna_review_confidence"],
            })
        analysis_samples.append({
            **sample,
            "luna_primary_person": primary["luna_canonical_person"],
            "luna_primary_person_role": primary["luna_person_role"],
            "luna_people": people,
            "luna_person_count": len(people),
            "luna_political_domain": primary["luna_political_domain"],
            "luna_political_subdomains": primary["luna_political_subdomains"],
            "luna_event_type": primary["luna_event_type"],
            "luna_sensitive_concepts": sorted({x for row in relations for x in row["luna_sensitive_concepts"]}),
            "luna_election_related": any(row["luna_election_related"] for row in relations),
            "luna_war_or_security_related": any(row["luna_war_or_security_related"] for row in relations),
            "luna_conflicts_named": sorted({x for row in relations for x in row["luna_conflicts_named"]}),
            "luna_time_expressions": sorted({x for row in relations for x in row["luna_time_expressions"]}),
            "luna_review_status": "verified_person_relation_present",
        })
    analysis_samples.sort(key=lambda row: row["sample_id"])
    write_jsonl(out / "analysis_ready_samples.jsonl", analysis_samples)

    by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in verified:
        name = row["luna_canonical_person"].strip() or row["canonical_person"]
        by_person[name].append(row)
    profiles: list[dict[str, Any]] = []
    for person, rows in by_person.items():
        profiles.append({
            "canonical_person": person,
            "sample_count": len(rows),
            "unique_sample_count": len({row["sample_id"] for row in rows}),
            "role_counts": dict(Counter(row["luna_person_role"] for row in rows).most_common()),
            "country_or_territory_counts": dict(Counter(
                country for row in rows for country in row["luna_countries_or_territories"]
            ).most_common()),
            "political_domain_counts": dict(Counter(row["luna_political_domain"] for row in rows).most_common()),
            "political_subdomain_counts": dict(Counter(
                domain for row in rows for domain in row["luna_political_subdomains"]
            ).most_common()),
            "event_type_counts": dict(Counter(row["luna_event_type"] for row in rows).most_common()),
            "sensitive_concept_counts": dict(Counter(
                concept for row in rows for concept in row["luna_sensitive_concepts"]
            ).most_common()),
            "election_sample_count": sum(row["luna_election_related"] for row in rows),
            "war_or_security_sample_count": sum(row["luna_war_or_security_related"] for row in rows),
            "conflict_counts": dict(Counter(
                conflict for row in rows for conflict in row["luna_conflicts_named"]
            ).most_common()),
            "split_counts": dict(Counter(row["source_split"] for row in rows).most_common()),
            "source_language_counts": dict(Counter(row["language_code"] for row in rows).most_common()),
        })
    profiles.sort(key=lambda row: (-row["unique_sample_count"], row["canonical_person"]))
    write_jsonl(out / "verified_person_profiles.jsonl", profiles)

    domain_confusion: Counter[tuple[str, str]] = Counter(
        (row["political_domain"], row["luna_political_domain"]) for row in merged
    )
    with (out / "rule_vs_luna_domain_confusion.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rule_domain", "luna_domain", "count"])
        for (rule_domain, luna_domain), count in domain_confusion.most_common():
            writer.writerow([rule_domain, luna_domain, count])

    eligible = [row for row in merged if row["luna_experiment_eligibility"].startswith("eligible_")]
    report = {
        "queue_rows": len(queue_rows),
        "unique_queue_relation_ids": len(relation_by_id),
        "duplicate_queue_rows": sum(count - 1 for count in queue_duplicate_counts.values()),
        "source_rows": len(samples_all),
        "unique_source_sample_ids": len(sample_by_id),
        "duplicate_source_sample_rows": sum(count - 1 for count in sample_id_counts.values()),
        "luna_reviewed_relations": len(merged),
        "luna_review_errors": len(reviews) - len(merged),
        "luna_person_yes": sum(row["luna_is_person_reference"] == "yes" for row in merged),
        "luna_person_no": sum(row["luna_is_person_reference"] == "no" for row in merged),
        "luna_person_uncertain": sum(row["luna_is_person_reference"] == "uncertain" for row in merged),
        "eligible_relations": len(eligible),
        "verified_relations": len(verified),
        "analysis_ready_unique_samples": len(analysis_samples),
        "verified_canonical_people": len(profiles),
        "rule_luna_domain_exact_agreement": sum(
            row["political_domain"] == row["luna_political_domain"] for row in merged
        ) / max(1, len(merged)),
        "luna_domain_counts": dict(Counter(row["luna_political_domain"] for row in merged).most_common()),
        "luna_role_counts": dict(Counter(row["luna_person_role"] for row in merged).most_common()),
        "luna_review_confidence_counts": dict(Counter(row["luna_review_confidence"] for row in merged)),
        "notes": [
            "All Luna inputs used the English article field and English metadata labels.",
            "The original local-language article and instruction remain preserved in analysis_ready_samples.jsonl.",
            "Duplicate source IDs are reported and represented once in analysis-ready joins.",
        ],
    }
    (out / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
