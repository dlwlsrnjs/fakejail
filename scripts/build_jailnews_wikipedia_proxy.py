#!/usr/bin/env python3
"""Build a provenance-explicit Wikipedia langlink proxy for selected cases.

The localization catalog was populated from Wikidata/Wikipedia titles.  This
adapter exposes only those verified titles as langlinks; it deliberately does
not invent Wikipedia article text.  Downstream scorers therefore use their
role/context embedding fallback while retaining real identity-coverage data.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--localizations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = read_jsonl(args.cases)
    wanted = {row["canonical_person"] for row in cases}
    by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.localizations):
        if row.get("canonical_name") in wanted:
            by_person[row["canonical_name"]].append(row)

    output = []
    for case in cases:
        person = case["canonical_person"]
        rows = by_person.get(person, [])
        verified = [row for row in rows if row.get("status") == "verified_localized"]
        langlinks = {
            row["target_language_code"]: row.get("source_title") or row.get("localized_name")
            for row in verified
            if row.get("target_language_code")
            and (row.get("source_title") or row.get("localized_name"))
        }
        english = next(
            (row for row in verified if row.get("target_language_code") == "en"),
            None,
        )
        output.append({
            "pilot_id": case["pilot_id"],
            "person": person,
            "wikipedia": {
                "status": "ok" if langlinks else "missing",
                "title": (english or {}).get("source_title") or person,
                "canonical_url": (english or {}).get("source_url") or "",
                "extract": "",
                "langlinks": langlinks,
                "langlink_count": len(langlinks),
                "proxy_source": "person_localizations_resolved/verified_localized",
                "proxy_limit": "langlinks_only_no_article_extract",
            },
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "cases": len(output),
        "people_with_verified_langlinks": sum(bool(row["wikipedia"]["langlinks"]) for row in output),
        "verified_langlinks": sum(row["wikipedia"]["langlink_count"] for row in output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
