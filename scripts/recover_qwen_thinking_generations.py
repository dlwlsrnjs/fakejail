#!/usr/bin/env python3
"""Extract and merge length-truncated Qwen Thinking generations."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def read(patterns: Iterable[str]) -> Iterable[dict[str, Any]]:
    for pattern in patterns:
        for path in sorted(Path(value) for value in glob.glob(pattern)):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)


def truncated(row: dict[str, Any]) -> bool:
    text = str(row.get("generation") or "")
    return str(row.get("finish_reason")) == "length" and "</think>" not in text


def extract(args: argparse.Namespace) -> None:
    source = list(read(args.inputs))
    selected = [row for row in source if truncated(row)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            # The generation command only needs the original arm fields and
            # overwrites generation metadata in its output.
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "source_rows": len(source),
        "truncated_without_visible_answer": len(selected),
        "recovery_fraction": len(selected) / len(source) if source else None,
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def merge(args: argparse.Namespace) -> None:
    recovery = {row["trial_id"]: row for row in read(args.recovery)}
    source = list(read(args.original))
    if len(recovery) != sum(truncated(row) for row in source):
        raise RuntimeError(
            f"recovery coverage mismatch: recovery={len(recovery)} "
            f"expected={sum(truncated(row) for row in source)}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    handles = [
        (args.output / f"shard_{index}.jsonl").open("w", encoding="utf-8")
        for index in range(args.shards)
    ]
    recovered_visible = 0
    still_truncated = 0
    try:
        for row in source:
            final = recovery.get(row["trial_id"], row)
            if row["trial_id"] in recovery:
                visible = str(final.get("generation") or "").rsplit("</think>", 1)
                if len(visible) == 2 and visible[1].strip():
                    recovered_visible += 1
                elif truncated(final):
                    still_truncated += 1
                final = {**final, "recovered_from_1024_token_truncation": True}
            shard = int(hashlib.sha256(final["trial_id"].encode()).hexdigest()[:16], 16) % args.shards
            handles[shard].write(json.dumps(final, ensure_ascii=False) + "\n")
    finally:
        for handle in handles:
            handle.close()
    report = {
        "source_rows": len(source),
        "recovery_rows": len(recovery),
        "recovered_visible_answers": recovered_visible,
        "still_truncated_without_visible_answer": still_truncated,
        "final_rows": len(source),
        "shards": args.shards,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(function=extract)
    p = sub.add_parser("merge")
    p.add_argument("--original", nargs="+", required=True)
    p.add_argument("--recovery", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shards", type=int, default=4)
    p.set_defaults(function=merge)
    args = ap.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
