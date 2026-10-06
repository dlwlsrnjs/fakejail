#!/usr/bin/env python3
"""Validate structural completeness and quality gates for JailNews bandit assets."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


RUNTIME = Path("artifacts/jailnews_bandit_20260930/runtime")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=RUNTIME / "catalog.sqlite")
    ap.add_argument("--person-localizations", type=Path, default=RUNTIME / "person_localizations_resolved.jsonl")
    ap.add_argument("--entity-map", type=Path, default=RUNTIME / "entity_links_v2/label_entity_map.jsonl")
    ap.add_argument("--prior", type=Path)
    ap.add_argument("--output", type=Path, default=RUNTIME / "validation_report.json")
    args = ap.parse_args()
    con = sqlite3.connect(args.catalog); con.row_factory = sqlite3.Row
    counts = {
        "contexts": con.execute("SELECT COUNT(*) FROM contexts").fetchone()[0],
        "people": con.execute("SELECT COUNT(*) FROM persons").fetchone()[0],
        "arms": con.execute("SELECT COUNT(*) FROM arms").fetchone()[0],
        "languages": con.execute("SELECT COUNT(DISTINCT language) FROM arms").fetchone()[0],
        "methods": con.execute("SELECT COUNT(DISTINCT method) FROM arms").fetchone()[0],
        "concept_translations": con.execute("SELECT COUNT(*) FROM concept_translation").fetchone()[0],
        "concepts": con.execute("SELECT COUNT(DISTINCT concept_id) FROM concept_translation").fetchone()[0],
        "translation_pairs": con.execute("SELECT COUNT(*) FROM (SELECT concept_id,language FROM concept_translation GROUP BY concept_id,language)").fetchone()[0],
    }
    status_counts = dict(con.execute("SELECT review_status,COUNT(*) FROM concept_translation GROUP BY review_status").fetchall())
    errors = []
    entity_map = list(read_jsonl(args.entity_map)) if args.entity_map.exists() else []
    expected_people = len({row["entity_id"] for row in entity_map}) if entity_map else counts["people"]
    expected = {"contexts": 4091, "people": expected_people, "arms": 360, "languages": 72, "methods": 5,
                "concept_translations": 303480, "concepts": 4215, "translation_pairs": 303480}
    for key, value in expected.items():
        if counts[key] != value:
            errors.append(f"{key}: expected {value}, got {counts[key]}")
    if entity_map and len(entity_map) != 501:
        errors.append(f"entity label map expected 501 labels, got {len(entity_map)}")
    unusable_conflicts = con.execute(
        "SELECT COUNT(*) FROM concept_translation WHERE valid != usable_for_rendering"
    ).fetchone()[0]
    if unusable_conflicts:
        errors.append(f"translation final-valid/usability conflicts: {unusable_conflicts}")
    missing_context_entity = con.execute(
        "SELECT COUNT(*) FROM contexts WHERE primary_entity_id IS NULL OR primary_entity_id=''"
    ).fetchone()[0]
    localizations: dict[str, Any] = {"present": args.person_localizations.exists()}
    if args.person_localizations.exists():
        rows = list(read_jsonl(args.person_localizations))
        keys = {(row["person_id"], row["target_language"]) for row in rows}
        localizations.update({
            "rows": len(rows), "unique_pairs": len(keys),
            "status_counts": dict(Counter(row["status"] for row in rows)),
        })
        if len(rows) != 36072 or len(keys) != 36072:
            errors.append(f"person localizations expected 36,072 unique pairs, got rows={len(rows)} unique={len(keys)}")
    prior: dict[str, Any] = {"present": bool(args.prior and args.prior.exists())}
    if prior["present"]:
        pcon = sqlite3.connect(args.prior)
        prior.update({
            "context_arm_priors": pcon.execute("SELECT COUNT(*) FROM arm_prior").fetchone()[0],
            "global_arm_priors": pcon.execute("SELECT COUNT(*) FROM global_prior").fetchone()[0],
        })
    report = {
        "schema": "jailnews_bandit_asset_validation/v1", "passed": not errors,
        "catalog_counts": counts, "translation_status_counts": status_counts,
        "person_localizations": localizations, "prior": prior, "errors": errors,
        "entity_labels": len(entity_map), "expected_deduplicated_entities": expected_people,
        "translation_usability_conflicts": unusable_conflicts,
        "contexts_without_entity_id": missing_context_entity,
        "quality_gate_note": "Unresolved entity links and rejected translations remain explicitly unusable.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
