#!/usr/bin/env python3
"""Build a deterministic, paired V5 control pilot from frozen arm matrices.

The script never regenerates prompts. It selects rows from the archived V5 and
English real-name control matrices, records exact source/output hashes, and
fails closed when the expected paired grid is incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/v5_pilot_preregister.json"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


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


def prompt_set_digest(rows: Iterable[dict[str, Any]]) -> str:
    hashes = sorted(str(row["prompt_sha256"]) for row in rows)
    return sha256_text("\n".join(hashes) + "\n")


def summarize_people(path: Path) -> dict[str, dict[str, Any]]:
    people: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        person = str(row["victim_person_id"])
        current = people.setdefault(
            person,
            {
                "victim_person_id": person,
                "wikidata_qid": row.get("wikidata_qid"),
                "canonical_person": row.get("expected_canonical_english_name"),
                "political_domain": row.get("political_domain"),
                "region_en": row.get("region_en"),
                "election_related": bool(row.get("election_related")),
                "war_or_security_related": bool(row.get("war_or_security_related")),
                "surrogate_prior_sum": 0.0,
                "rows": 0,
            },
        )
        invariant = (
            current["political_domain"] == row.get("political_domain")
            and current["region_en"] == row.get("region_en")
            and current["wikidata_qid"] == row.get("wikidata_qid")
        )
        if not invariant:
            raise RuntimeError(f"person metadata changed across arms: {person}")
        current["surrogate_prior_sum"] += float(row["surrogate_prior_mean"])
        current["rows"] += 1

    if len(people) != 501:
        raise RuntimeError(f"expected 501 people, found {len(people)}")
    for person, value in people.items():
        if value["rows"] != 360:
            raise RuntimeError(f"expected 360 arms for {person}, found {value['rows']}")
        value["mean_surrogate_prior"] = value.pop("surrogate_prior_sum") / value["rows"]
    return people


def select_people(
    people: dict[str, dict[str, Any]], people_per_domain: int
) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in people.values():
        by_domain[str(value["political_domain"])].append(value)
    selected: list[dict[str, Any]] = []
    for domain in sorted(by_domain):
        candidates = sorted(
            by_domain[domain],
            key=lambda value: (
                value["mean_surrogate_prior"],
                sha256_text(value["victim_person_id"]),
            ),
        )
        if len(candidates) < people_per_domain:
            raise RuntimeError(
                f"domain {domain} has {len(candidates)} people, fewer than {people_per_domain}"
            )
        used: set[int] = set()
        for index in range(people_per_domain):
            quantile = (index + 0.5) / people_per_domain
            target = round(quantile * (len(candidates) - 1))
            while target in used:
                target += 1
            used.add(target)
            selected.append(
                {
                    **candidates[target],
                    "domain_quantile_target": quantile,
                    "domain_rank": target,
                    "domain_size": len(candidates),
                }
            )
    return selected


def select_v5_rows(
    path: Path,
    *,
    selected_people: set[str],
    languages: list[str],
    methods: list[str],
    mode: str,
) -> list[dict[str, Any]]:
    language_set = set(languages)
    method_set = set(methods)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in iter_jsonl(path):
        person = str(row["victim_person_id"])
        language = str(row["language"])
        method = str(row["attack_type"])
        if person not in selected_people or language not in language_set or method not in method_set:
            continue
        key = (person, language, method)
        if key in seen:
            raise RuntimeError(f"duplicate {mode} pilot cell: {key}")
        seen.add(key)
        rows.append(
            {
                **row,
                "pilot_condition": f"v5_{mode}",
                "pilot_pair_id": sha256_text("|".join(key))[:24],
            }
        )
    expected = len(selected_people) * len(languages) * len(methods)
    if len(rows) != expected or len(seen) != expected:
        raise RuntimeError(f"{mode} grid mismatch: rows={len(rows)} expected={expected}")
    return sorted(
        rows,
        key=lambda row: (
            row["victim_person_id"],
            languages.index(row["language"]),
            methods.index(row["attack_type"]),
        ),
    )


def select_control_rows(
    path: Path, *, selected_people: set[str], methods: list[str]
) -> list[dict[str, Any]]:
    method_set = set(methods)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in iter_jsonl(path):
        person = str(row["victim_person_id"])
        method = str(row["attack_type"])
        if person not in selected_people or method not in method_set:
            continue
        key = (person, method)
        if key in seen:
            raise RuntimeError(f"duplicate control pilot cell: {key}")
        seen.add(key)
        rows.append(
            {
                **row,
                "pilot_condition": "english_real_name_control",
                "pilot_pair_id": sha256_text("|".join(key))[:24],
                "control_shared_across_languages": True,
            }
        )
    expected = len(selected_people) * len(methods)
    if len(rows) != expected or len(seen) != expected:
        raise RuntimeError(f"control grid mismatch: rows={len(rows)} expected={expected}")
    return sorted(
        rows,
        key=lambda row: (row["victim_person_id"], methods.index(row["attack_type"])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-root", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    cohort = config["cohort"]
    languages = list(config["arms"]["languages"])
    methods = list(config["arms"]["methods"])
    people_per_domain = int(cohort["people_per_domain"])

    literal_source = args.v5_root / "a_literal/base_arms.jsonl"
    canonical_source = args.v5_root / "canonical_en/base_arms.jsonl"
    for source in (literal_source, canonical_source, args.control):
        if not source.is_file():
            raise FileNotFoundError(source)

    people = summarize_people(literal_source)
    selected_people = select_people(people, people_per_domain)
    expected_domains = int(cohort["expected_domains"])
    expected_people = int(cohort["expected_people"])
    domains = {str(row["political_domain"]) for row in selected_people}
    if len(domains) != expected_domains or len(selected_people) != expected_people:
        raise RuntimeError(
            f"cohort mismatch: domains={len(domains)} people={len(selected_people)}"
        )
    selected_ids = {str(row["victim_person_id"]) for row in selected_people}

    control_rows = select_control_rows(
        args.control, selected_people=selected_ids, methods=methods
    )
    literal_rows = select_v5_rows(
        literal_source,
        selected_people=selected_ids,
        languages=languages,
        methods=methods,
        mode="a_literal",
    )
    canonical_rows = select_v5_rows(
        canonical_source,
        selected_people=selected_ids,
        languages=languages,
        methods=methods,
        mode="canonical_en",
    )

    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "control": args.output / "arms_control.jsonl",
        "a_literal": args.output / "arms_v5_a_literal.jsonl",
        "canonical_en": args.output / "arms_v5_canonical_en.jsonl",
        "combined": args.output / "arms_all.jsonl",
    }
    counts = {
        "control": atomic_jsonl(paths["control"], control_rows),
        "a_literal": atomic_jsonl(paths["a_literal"], literal_rows),
        "canonical_en": atomic_jsonl(paths["canonical_en"], canonical_rows),
        "combined": atomic_jsonl(
            paths["combined"], [*control_rows, *literal_rows, *canonical_rows]
        ),
    }
    atomic_json(args.output / "selected_people.json", selected_people)

    smoke_person = selected_people[0]["victim_person_id"]
    smoke_method = "role_play"
    smoke = [
        next(
            row
            for row in rows
            if row["victim_person_id"] == smoke_person
            and row["attack_type"] == smoke_method
            and (row.get("language") == "English" if row.get("language") else True)
        )
        for rows in (control_rows, literal_rows, canonical_rows)
    ]
    smoke_path = args.output / "smoke_3.jsonl"
    atomic_jsonl(smoke_path, smoke)

    manifest = {
        "schema": "jailnews_v5_controlled_pilot_plan/v1",
        "preregister": config,
        "source": {
            "v5_root": str(args.v5_root.resolve()),
            "control": str(args.control.resolve()),
            "v5_manifest_sha256": sha256_file(args.v5_root / "manifest.json"),
            "control_manifest_sha256": sha256_file(args.control.parent / "manifest.json"),
        },
        "selection": {
            "domains": len(domains),
            "people": len(selected_people),
            "languages": languages,
            "methods": methods,
            "selected_people_sha256": sha256_file(args.output / "selected_people.json"),
        },
        "counts": counts,
        "prompt_set_sha256": {
            "control": prompt_set_digest(control_rows),
            "a_literal": prompt_set_digest(literal_rows),
            "canonical_en": prompt_set_digest(canonical_rows),
        },
        "output_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "smoke": {
            "rows": len(smoke),
            "person": smoke_person,
            "method": smoke_method,
            "v5_language": "English",
            "path": str(smoke_path.resolve()),
            "sha256": sha256_file(smoke_path),
        },
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
