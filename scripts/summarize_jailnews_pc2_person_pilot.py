#!/usr/bin/env python3
"""Write a compact audit report for the JailNewsBench PC2 person pilot."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="Refined identity-preserving clues")
    args = parser.parse_args()

    scores = read_jsonl(args.score_dir / "language_scores.jsonl")
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf-8"))
    selected = {
        p: Counter(row["selected"][p]["language"] for row in scores if row["selected"].get(p))
        for p in ("25", "50")
    }
    valid_counts = [sum(candidate["valid"] for candidate in row["candidates"]) for row in scores]

    lines = [
        "# JailNewsBench person-IPDM × PC2 local pilot",
        "",
        f"**Variant:** {args.label}",
        "",
        f"- Source population: 24,040 Political rows with an automatic PERSON candidate",
        f"- Validated pilot people: {len(scores)}",
        f"- Candidate languages: 72 PC2 labels",
        f"- Languages passing cosine QC per person: min {min(valid_counts)}, mean {sum(valid_counts)/len(valid_counts):.1f}, max {max(valid_counts)}",
        "- Endpoint: safe identity/country reconstruction, not fake-news generation or ASR",
        "",
        "## Qwen2.5-7B reconstruction",
        "",
        "| Condition | n | Person exact | 95% CI | Country |",
        "|---|---:|---:|---:|---:|",
    ]
    order = [
        "direct_name", "english_ipdm", "random_valid_language_ipdm",
        "pc2_p25_language_ipdm", "pc2_p50_language_ipdm",
    ]
    for condition in order:
        row = summary["conditions"][condition]
        lo, hi = row["person_exact_95ci"]
        lines.append(
            f"| {condition} | {row['n']} | {pct(row['person_exact'])} | "
            f"[{pct(lo)}, {pct(hi)}] | {pct(row['country'])} |"
        )
    lines += [
        "",
        "## Selected-language frequencies",
        "",
        "- p25: " + ", ".join(f"{lang} ({count})" for lang, count in selected["25"].most_common()),
        "- p50: " + ", ".join(f"{lang} ({count})" for lang, count in selected["50"].most_common()),
        "",
        "## Interpretation guardrail",
        "",
        "A language condition is viable only if it retains identity substantially better than a random valid language. "
        "This report cannot support an attack-success claim because the target is never asked to execute the embedded manipulation request.",
        "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
