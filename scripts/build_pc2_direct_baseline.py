#!/usr/bin/env python3
"""Build one direct, untranslated baseline prompt per PC2 pilot case."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    first_by_pilot = {}
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                first_by_pilot.setdefault(row["pilot_id"], row)

    output = []
    for pilot_id, source in sorted(first_by_pilot.items()):
        row = dict(source)
        row.update({
            "matrix_id": f"{pilot_id}::direct_original",
            "clue_language": "DirectOriginal",
            "nllb_code": None,
            "clue": None,
            "backtranslation": None,
            "backtranslation_similarity": None,
            "nllb_similarity_valid": True,
            "translation_source": "none-direct-original",
            "attacked_prompt": source["seed_instruction_local"],
        })
        output.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"rows": len(output), "output": str(args.output)}))


if __name__ == "__main__":
    main()
