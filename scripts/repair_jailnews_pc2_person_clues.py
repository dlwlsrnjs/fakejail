#!/usr/bin/env python3
"""Repair person clues using only validated role/country annotation fields."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


MANUAL_EXCLUSIONS = {
    "JNBPC2_009": "organization president, not a political officeholder",
    "JNBPC2_021": "sports representative, not a political officeholder",
    "JNBPC2_022": "automatic NER omitted the first component of the person's name",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    repaired = []
    for row in rows:
        ann = dict(row.get("ipdm_annotation") or {})
        reason = MANUAL_EXCLUSIONS.get(row["pilot_id"])
        if reason:
            ann.update({
                "is_public_political_figure": "no",
                "role": "",
                "person_country_or_territory": "",
                "one_sentence_ipdm": "",
            })
            status = "excluded"
            errors = [reason]
        elif ann.get("is_public_political_figure") == "yes" and ann.get("role") and ann.get("person_country_or_territory"):
            role = " ".join(ann["role"].split()).strip(" .")
            country = " ".join(ann["person_country_or_territory"].split()).strip(" .")
            clue = f"This public figure holds or held the role of {role} in {country}."
            name_tokens = [
                token.casefold() for token in re.findall(
                    r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž]{4,}", row["selected_person"]
                )
            ]
            leaks = [token for token in name_tokens if re.search(rf"\b{re.escape(token)}\b", clue, re.I)]
            words = re.findall(r"\b\w+(?:[-']\w+)*\b", clue)
            errors = []
            if leaks:
                errors.append("template_leaks_name:" + ",".join(leaks))
            if not 10 <= len(words) <= 24:
                errors.append("template_length")
            ann["one_sentence_ipdm"] = clue
            status = "accepted" if not errors else "needs_review"
        else:
            status = "excluded"
            errors = ["annotator_did_not_confirm_public_political_figure"]
        repaired.append({
            **row,
            "ipdm_annotation_original": row.get("ipdm_annotation"),
            "ipdm_annotation": ann,
            "annotation_errors": errors,
            "annotation_status": status,
            "clue_repair": "deterministic_role_country_template" if status == "accepted" else None,
        })
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in repaired),
        encoding="utf-8",
    )
    summary = {
        "rows": len(repaired),
        "accepted": sum(row["annotation_status"] == "accepted" for row in repaired),
        "excluded": sum(row["annotation_status"] == "excluded" for row in repaired),
        "needs_review": sum(row["annotation_status"] == "needs_review" for row in repaired),
        "manual_exclusions": MANUAL_EXCLUSIONS,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
