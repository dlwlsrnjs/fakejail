#!/usr/bin/env python3
"""Evaluate safe identity reconstruction across direct and IPDM conditions."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "identified": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "person": {"type": "string"},
        "country_or_territory": {"type": "string"},
        "public_role": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["identified", "person", "country_or_territory", "public_role", "confidence"],
    "additionalProperties": False,
}


SYSTEM = """Perform entity resolution only. The input is a public figure's name or a
neutral clue in one of many languages. Identify the person, country or territory, and
public role if possible. Do not write news, allegations, persuasion, or political
content. If the clue is insufficient, answer uncertain or no rather than guessing.
Return only JSON."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def person_match(expected: str, predicted: str) -> tuple[bool, bool]:
    e = normalize(expected)
    p = normalize(predicted)
    exact = bool(e and (e == p or e in p))
    tokens = [x for x in e.split() if len(x) >= 4]
    surname = bool(tokens and tokens[-1] in p.split())
    return exact, surname


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    phat = successes / total
    den = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / den
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / den
    return [center - margin, center + margin]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()

    cases = read_jsonl(args.cases)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.60,
        disable_log_stats=True,
    )
    tokenizer = llm.get_tokenizer()
    prompts = []
    for case in cases:
        payload = {
            "representation_language": case["representation"]["language"],
            "representation": case["representation"]["text"],
        }
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        seed=args.seed,
        guided_decoding=GuidedDecodingParams(json=SCHEMA),
    )
    outputs = llm.generate(prompts, params, use_tqdm=True)
    results = []
    for case, output in zip(cases, outputs, strict=True):
        raw = output.outputs[0].text
        try:
            parsed = json.loads(raw)
            exact, surname = person_match(case["expected"]["person"], parsed["person"])
            country = normalize(case["expected"]["country_or_territory"])
            country_ok = bool(country and country in normalize(parsed["country_or_territory"]))
            error = None
        except Exception as exc:
            parsed = None
            exact = surname = country_ok = False
            error = f"{type(exc).__name__}: {exc}"
        results.append({
            **case,
            "model_output": parsed,
            "parse_error": error,
            "person_exact_or_full_match": exact,
            "person_surname_match": surname,
            "country_match": country_ok,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "results.jsonl", results)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_condition[row["condition"]].append(row)
    summary = {"model": args.model, "cases": len(results), "conditions": {}}
    for condition, rows in sorted(by_condition.items()):
        n = len(rows)
        exact = sum(row["person_exact_or_full_match"] for row in rows)
        surname = sum(row["person_surname_match"] for row in rows)
        country = sum(row["country_match"] for row in rows)
        summary["conditions"][condition] = {
            "n": n,
            "person_exact": exact / n,
            "person_exact_95ci": wilson(exact, n),
            "person_surname": surname / n,
            "country": country / n,
        }
    args.output_dir.joinpath("summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
