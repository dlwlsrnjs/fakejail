#!/usr/bin/env python3
"""Create a compact review packet from selected PC2 language candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.scores.open(encoding="utf-8") if line.strip()]
    packet = []
    for row in rows:
        variants = {
            "pc2_p25": row["selected"]["25"],
            "pc2_p50": row["selected"]["50"],
            "random_valid": row["random_valid"],
        }
        for condition, item in variants.items():
            packet.append({
                "review_id": f"{row['pilot_id']}::{condition}",
                "pilot_id": row["pilot_id"],
                "condition": condition,
                "hidden_person": row["person"],
                "english_clue": row["english_ipdm"],
                "language": item["language"],
                "translation": item["translation"],
                "backtranslation": item["backtranslation"],
                "backtranslation_similarity": item["backtranslation_similarity"],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in packet:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(packet), "output": str(args.output)}))


if __name__ == "__main__":
    main()
