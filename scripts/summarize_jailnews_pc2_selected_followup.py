#!/usr/bin/env python3
"""Summarize five-draw stability for PC2/oracle/random selected arms."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from evaluate_jailnews_pc2_full501 import iter_jsonl, labels


ROLES = ("pc2_top1", "observed_oracle", "random_control")


def interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean()) if len(array) else 0.0
    se = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return {
        "mean": mean,
        "ci95_low": max(0.0, mean - 1.96 * se),
        "ci95_high": min(1.0, mean + 1.96 * se),
        "n": int(len(array)),
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--judgments", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-followup-draws", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--judge-error-policy", choices=["fail", "zero"], default="fail")
    args = parser.parse_args()

    selected = {}
    for row in iter_jsonl([args.manifest]):
        key = (str(row["victim_person_id"]), str(row["language"]), str(row["attack_type"]))
        selected[key] = {
            "roles": tuple(row["followup_roles"]),
            "draw0": int(row["followup_selection"]["draw0_strict"]),
        }
    observed: dict[tuple[str, str, str], dict[int, int]] = defaultdict(dict)
    judge_errors = 0
    for row in iter_jsonl(args.judgments):
        key = (str(row["victim_person_id"]), str(row["language"]), str(row["attack_type"]))
        if key not in selected:
            raise RuntimeError(f"judgment outside selected manifest: {key}")
        draw = int(row["draw"])
        if draw in observed[key]:
            raise RuntimeError(f"duplicate selected judgment: {key}, draw={draw}")
        if (row.get("paper_qwen32") or {}).get("judge_error"):
            judge_errors += 1
        _, _, strict, _ = labels(row, args.judge_error_policy)
        observed[key][draw] = strict
    expected = set(args.expected_followup_draws)
    incomplete = [key for key in selected if set(observed[key]) != expected]
    if incomplete:
        raise RuntimeError(f"incomplete selected followup for {len(incomplete)} arms: {incomplete[:5]}")

    by_role_draw: dict[str, dict[int, list[float]]] = {
        role: defaultdict(list) for role in ROLES
    }
    arm_frequency: dict[str, list[float]] = {role: [] for role in ROLES}
    person_draw: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for key, state in selected.items():
        outcomes = {0: state["draw0"], **observed[key]}
        for role in state["roles"]:
            for draw, outcome in outcomes.items():
                by_role_draw[role][draw].append(float(outcome))
                person_draw[key[0]][draw][role] = float(outcome)
            arm_frequency[role].append(float(np.mean(list(outcomes.values()))))

    role_metrics = {}
    for role in ROLES:
        role_metrics[role] = {
            "by_draw": {str(draw): interval(values) for draw, values in sorted(by_role_draw[role].items())},
            "per_arm_five_draw_success_frequency": interval(arm_frequency[role]),
            "always_successful_arms": int(sum(value == 1.0 for value in arm_frequency[role])),
            "never_successful_arms": int(sum(value == 0.0 for value in arm_frequency[role])),
        }

    paired = {}
    for left, right in (("pc2_top1", "random_control"), ("observed_oracle", "random_control")):
        paired[f"{left}_minus_{right}"] = {}
        for draw in [0, *args.expected_followup_draws]:
            deltas = [
                values[left] - values[right]
                for values in (person_draw[person][draw] for person in person_draw)
                if left in values and right in values
            ]
            paired[f"{left}_minus_{right}"][str(draw)] = interval(deltas)

    result = {
        "schema": "jailnews_pc2_selected_followup_summary/v1",
        "selected_unique_arms": len(selected),
        "followup_rows": sum(len(value) for value in observed.values()),
        "draws": [0, *args.expected_followup_draws],
        "judge_errors_fail_closed": judge_errors,
        "role_counts": dict(Counter(role for state in selected.values() for role in state["roles"])),
        "role_metrics": role_metrics,
        "paired_by_draw": paired,
        "interpretation": "draw 0 selected the observed-oracle role; draws 1-4 estimate its persistence without redefining the arm",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "results.json", result)
    lines = [
        "# PC2 selected-arm five-draw stability", "",
        "| role | " + " | ".join(f"draw {draw}" for draw in result["draws"]) + " | five-draw arm mean |",
        "|---|" + "---:|" * (len(result["draws"]) + 1),
    ]
    for role in ROLES:
        rates = [role_metrics[role]["by_draw"][str(draw)]["mean"] for draw in result["draws"]]
        mean = role_metrics[role]["per_arm_five_draw_success_frequency"]["mean"]
        lines.append("| " + role + " | " + " | ".join(f"{value:.2%}" for value in rates) + f" | {mean:.2%} |")
    lines.extend(["", "Observed-oracle draw 0 is selected by construction; persistence is judged on draws 1-4.", ""])
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
