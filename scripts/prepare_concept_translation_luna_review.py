#!/usr/bin/env python3
"""Prepare a resumable Luna adjudication queue for neutral concept translations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DIR = Path("data/jailnewsbench_person_domain_20260930/concept_translations_v1")


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def audit_pick(translation_id: str, rate: float) -> bool:
    bucket = int(hashlib.sha256(translation_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < rate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_DIR / "translations_nllb.jsonl")
    ap.add_argument("--output", type=Path, default=DEFAULT_DIR / "luna_review_queue.jsonl")
    ap.add_argument("--valid-audit-rate", type=float, default=0.02)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    selected = []
    for row in rows(args.input):
        reason = None
        if not row["valid"]:
            reason = "automatic_invalid"
        elif audit_pick(row["translation_id"], args.valid_audit_rate):
            reason = "deterministic_valid_audit"
        if reason is None:
            continue
        priority = 1.0 - float(row["backtranslation_similarity"])
        selected.append({
            "schema": "concept_translation_adjudication_input/v1",
            "translation_id": row["translation_id"], "concept_id": row["concept_id"],
            "language": row["language"], "canonical_english": row["canonical_english"],
            "concept_types": row["concept_types"], "nllb_translation": row["translation"],
            "nllb_backtranslation": row["backtranslation"],
            "backtranslation_similarity": row["backtranslation_similarity"],
            "automatic_valid": row["valid"], "selection_reason": reason,
            "priority": round(priority, 8),
        })
        counts[reason] += 1
    selected.sort(key=lambda row: (-row["priority"], row["language"], row["translation_id"]))
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema": "concept_translation_luna_queue/v1", "input": str(args.input),
        "rows": len(selected), "selection_counts": dict(counts),
        "valid_audit_rate": args.valid_audit_rate,
        "note": "Similarity is a queue priority signal, not the final decision.",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
