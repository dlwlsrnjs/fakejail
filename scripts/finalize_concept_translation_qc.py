#!/usr/bin/env python3
"""Merge NLLB, Luna adjudication, and corrected round-trip QA into a gated cache."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


BASE = Path("data/jailnewsbench_person_domain_20260930/concept_translations_v1")


def read(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=BASE / "translations_nllb.jsonl")
    ap.add_argument("--adjudications", type=Path, required=True)
    ap.add_argument("--correction-qc", type=Path)
    ap.add_argument("--output", type=Path, default=BASE / "translations_verified.jsonl")
    args = ap.parse_args()
    reviews = {row["translation_id"]: row for row in read(args.adjudications)}
    corrections = {row["translation_id"]: row for row in read(args.correction_qc)}
    counts: Counter[str] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in read(args.base):
            review = reviews.get(row["translation_id"])
            final = dict(row)
            # Keep the automatic round-trip signal separate from the final
            # rendering gate.  Older files overloaded `valid` with both.
            final["roundtrip_valid"] = bool(row.get("valid"))
            status = "provisional_auto" if row["valid"] else "needs_review"
            if review:
                direct_pass = (
                    review["decision"] == "accept" and review["semantic_fidelity"] == "pass"
                    and review["coverage"] == "complete" and review["entity_preservation"] == "pass"
                    and review["terminology_quality"] == "appropriate"
                    and review["grammar_quality"] in {"good", "acceptable"}
                    and review["adjudication_confidence"] in {"high", "medium"}
                )
                correction = corrections.get(row["translation_id"])
                corrected_pass = (
                    review["decision"] == "retranslate" and correction is not None
                    and correction["corrected_roundtrip_valid"]
                    and review["adjudication_confidence"] in {"high", "medium"}
                )
                if direct_pass:
                    final["valid"] = True
                    status = "verified_luna"
                elif corrected_pass:
                    final["translation"] = correction["corrected_translation"]
                    final["backtranslation"] = correction["corrected_backtranslation"]
                    final["backtranslation_similarity"] = correction["corrected_backtranslation_similarity"]
                    final["translation_backend"] = "luna_correction+nllb_backtranslation"
                    final["valid"] = True
                    status = "verified_luna"
                else:
                    status = "needs_human_review"
                final["luna_adjudication"] = review
            final["final_status"] = status
            final["usable_for_rendering"] = bool(
                final.get("translation", "").strip()
                and status in {"verified_luna", "verified_human"}
            )
            # `valid` is retained for compatibility, but in v2 it is exactly
            # the authoritative final gate.  Use `roundtrip_valid` for the
            # original automatic QA result.
            final["valid"] = final["usable_for_rendering"]
            counts[status] += 1
            handle.write(json.dumps(final, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema": "concept_translation_qc/v2", "rows": sum(counts.values()),
        "status_counts": dict(counts),
        "valid_semantics": "authoritative final rendering gate",
        "roundtrip_valid_semantics": "original automatic round-trip QA signal",
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
