#!/usr/bin/env python3
"""Prepare low-roundtrip-similarity PC2 clue translations for Luna review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for score in read_jsonl(args.scores):
        for candidate in score["candidates"]:
            if candidate["valid"]:
                continue
            review_id = hashlib.sha256(
                f"{score['pilot_id']}\0{candidate['language']}\0{candidate['translation']}".encode()
            ).hexdigest()[:24]
            rows.append({
                "review_id": f"pc2clue:{review_id}",
                "pilot_id": score["pilot_id"],
                "source_record_id": score["source_record_id"],
                "person": score["person"],
                "language": candidate["language"],
                "nllb_code": candidate["nllb_code"],
                "canonical_english": score["english_ipdm"],
                "translation": candidate["translation"],
                "backtranslation": candidate["backtranslation"],
                "backtranslation_similarity": candidate["backtranslation_similarity"],
                "automatic_valid": False,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(args.output)
    summary = {"rows": len(rows), "source": str(args.scores), "output": str(args.output)}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
