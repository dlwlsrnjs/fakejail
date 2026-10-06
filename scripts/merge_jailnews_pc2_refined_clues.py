#!/usr/bin/env python3
"""Promote accepted Qwen-32B clues into the language-scoring schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    merged = []
    for row in rows:
        if row["refinement_status"] != "accepted":
            continue
        annotation = dict(row["ipdm_annotation"])
        annotation["one_sentence_ipdm"] = row["refined_ipdm"]["one_sentence_ipdm"]
        merged.append({
            **row,
            "ipdm_annotation": annotation,
            "annotation_status": "accepted",
            "clue_repair": "qwen2.5_32b_identity_preserving_refinement",
        })
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged),
        encoding="utf-8",
    )
    print(json.dumps({"accepted": len(merged)}, indent=2))


if __name__ == "__main__":
    main()
