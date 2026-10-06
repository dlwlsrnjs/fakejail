#!/usr/bin/env python3
"""Merge local MT candidates and fail-closed Qwen32 review into translation v2."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path


def read(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def paths(pattern: str) -> list[Path]:
    matched = [Path(value) for value in sorted(glob.glob(pattern))]
    if not matched:
        raise FileNotFoundError(pattern)
    return matched


def parse_judgment(value: dict) -> dict | None:
    parsed = value.get("parsed") if value.get("valid") else None
    if parsed is not None:
        return parsed
    text = str(value.get("raw_judgment") or "")
    required = {"equivalent", "target_language_valid", "needs_retranslation", "error_types", "reason"}
    decoder = json.JSONDecoder()
    for candidate_text in (text, text.replace("\\'", "'")):
        for match in re.finditer(r"\{", candidate_text):
            try:
                candidate, _ = decoder.raw_decode(candidate_text[match.start():])
            except ValueError:
                continue
            if not isinstance(candidate, dict) or set(candidate) != required:
                continue
            if not all(isinstance(candidate[k], bool) for k in ("equivalent", "target_language_valid", "needs_retranslation")):
                continue
            if not isinstance(candidate["error_types"], list) or not all(isinstance(x, str) for x in candidate["error_types"]):
                continue
            if not isinstance(candidate["reason"], str):
                continue
            expected = not (candidate["equivalent"] and candidate["target_language_valid"])
            if candidate["needs_retranslation"] == expected:
                return candidate
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--candidates", required=True, help="Glob for MT candidate shards")
    ap.add_argument("--judgments", required=True, help="Glob for Qwen32 judgment shards")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    candidate_rows = [row for path in paths(args.candidates) for row in read(path)]
    judgment_rows = [row for path in paths(args.judgments) for row in read(path)]
    candidate = {row["translation_id"]: row for row in candidate_rows}
    judgment = {row["key"]: row for row in judgment_rows}
    if len(candidate) != len(candidate_rows):
        raise ValueError("Duplicate translation_id in MT candidates")
    if len(judgment) != len(judgment_rows):
        raise ValueError("Duplicate key in Qwen32 judgments")
    source_rows = list(read(args.base))
    pending_ids = {
        row["translation_id"] for row in source_rows
        if row.get("final_status", "needs_human_review") == "needs_human_review"
    }
    missing_candidates = pending_ids - candidate.keys()
    missing_judgments = pending_ids - judgment.keys()
    extra_candidates = candidate.keys() - pending_ids
    extra_judgments = judgment.keys() - pending_ids
    if missing_candidates or missing_judgments or extra_candidates or extra_judgments:
        raise ValueError(
            "Review coverage mismatch: "
            f"pending={len(pending_ids)} candidates={len(candidate)} judgments={len(judgment)} "
            f"missing_candidates={len(missing_candidates)} missing_judgments={len(missing_judgments)} "
            f"extra_candidates={len(extra_candidates)} extra_judgments={len(extra_judgments)}"
        )
    counts = Counter()
    empty_translation_rows = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for source in source_rows:
            row = dict(source)
            row["roundtrip_valid"] = bool(source.get("roundtrip_valid", source.get("valid")))
            status = source.get("final_status", "needs_human_review")
            usable = status in {"verified_luna", "verified_human", "verified_local_qwen32"}
            if status == "needs_human_review":
                repaired = candidate.get(source["translation_id"])
                judged = judgment.get(source["translation_id"])
                parsed = parse_judgment(judged) if judged else None
                # Preserve a non-empty targeted repair for the three formerly
                # blank rows even when the semantic judge rejects it.  The
                # final usability gate still remains false, so this text can
                # be audited but cannot be rendered into an experiment.
                if (
                    repaired and not str(source.get("translation", "")).strip()
                    and str(repaired.get("translation", "")).strip()
                ):
                    row["pre_v2_translation"] = source.get("translation")
                    row["pre_v2_backtranslation"] = source.get("backtranslation")
                    row["translation"] = repaired["translation"]
                    row["backtranslation"] = repaired.get("backtranslation", "")
                    row["translation_backend"] = repaired["translation_backend"] + "+Qwen2.5-32B-review"
                passed = bool(
                    repaired and parsed and parsed.get("equivalent")
                    and parsed.get("target_language_valid") and not parsed.get("needs_retranslation")
                    and str(repaired.get("translation", "")).strip()
                    and str(repaired.get("backtranslation", "")).strip()
                )
                if passed:
                    row["pre_v2_translation"] = source.get("translation")
                    row["pre_v2_backtranslation"] = source.get("backtranslation")
                    row["translation"] = repaired["translation"]
                    row["backtranslation"] = repaired["backtranslation"]
                    row["translation_backend"] = repaired["translation_backend"] + "+Qwen2.5-32B-review"
                    row["local_qwen32_review"] = parsed
                    status = "verified_local_qwen32"
                    usable = True
                else:
                    row["local_qwen32_review"] = parsed
                    # A valid negative judgment is a completed adjudication, not
                    # another pending review.  Invalid model output is tracked
                    # separately and remains unusable (fail closed).
                    status = "rejected_local_qwen32" if parsed else "review_invalid_local_qwen32"
                    usable = False
            row["final_status"] = status
            row["usable_for_rendering"] = bool(usable and str(row.get("translation", "")).strip())
            row["valid"] = row["usable_for_rendering"]
            empty_translation_rows += not bool(str(row.get("translation", "")).strip())
            counts[status] += 1
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if counts.get("review_invalid_local_qwen32", 0):
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"Unparseable Qwen32 judgments: {counts['review_invalid_local_qwen32']}; "
            "refusing to finalize the catalog"
        )
    if counts.get("needs_human_review", 0) or empty_translation_rows:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"Final QC incomplete: needs_human_review={counts.get('needs_human_review', 0)} "
            f"empty_translation_rows={empty_translation_rows}"
        )
    tmp.replace(args.output)
    summary = {
        "schema": "concept_translation_qc/v2", "rows": sum(counts.values()),
        "status_counts": dict(counts), "candidate_rows": len(candidate),
        "judgment_rows": len(judgment),
        "pending_input_rows": len(pending_ids),
        "remaining_needs_human_review": counts.get("needs_human_review", 0),
        "empty_translation_rows": empty_translation_rows,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
