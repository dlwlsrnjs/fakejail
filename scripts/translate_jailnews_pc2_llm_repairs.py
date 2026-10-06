#!/usr/bin/env python3
"""Targeted multilingual translation repair with a large instruction model."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def numbers(value: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?%?", value)


def clean_output(value: str) -> str:
    value = value.strip()
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```[^\n]*\n?", "", value)
        value = re.sub(r"\n?```$", "", value).strip()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--use-review-feedback", action="store_true",
        help="Include the preceding independent review in the repair prompt.",
    )
    args = parser.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = read(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model_config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)

    def compat(config):
        text = getattr(config, "text_config", config)
        pattern = getattr(text, "_sliding_window_pattern", None)
        if pattern is not None:
            text.sliding_window_pattern = pattern
        return config

    llm_kwargs = {}
    if model_config.model_type == "gemma3":
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0}
    llm = LLM(
        model=str(args.model_path), dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=4096, gpu_memory_utilization=0.84, enforce_eager=True,
        max_num_seqs=24, disable_log_stats=True,
        hf_overrides=compat,
        **llm_kwargs,
    )
    params = SamplingParams(temperature=0, top_k=1, max_tokens=512, seed=20260930)

    def generate(prompts: list[str]) -> list[str]:
        output = []
        for start in range(0, len(prompts), 48):
            batch = prompts[start:start + 48]
            generated = llm.generate(batch, params, use_tqdm=False)
            output.extend(clean_output(item.outputs[0].text) for item in generated)
            print("LLM-MT", min(start + 48, len(prompts)), "/", len(prompts), flush=True)
        return output

    def prompt(source: str, target: str, text: str, row: dict | None = None) -> str:
        feedback = ""
        if args.use_review_feedback and row:
            review = row.get("repair_feedback") or row.get("semantic_review_before_repair") or {}
            reason = str(review.get("reason", "")).strip()
            errors = ", ".join(str(x) for x in review.get("error_types", []))
            previous = str(row.get("translation", "")).strip()
            previous_back = str(row.get("backtranslation", "")).strip()
            feedback = (
                "\n\nAn independent translation reviewer rejected the previous attempt. "
                "Correct the cited defect; do not copy a defective wording."
                f"\nPREVIOUS TRANSLATION: {previous}"
                f"\nPREVIOUS BACKTRANSLATION: {previous_back}"
                f"\nREVIEW ERROR TYPES: {errors or 'unspecified'}"
                f"\nREVIEW REASON: {reason or 'semantic fidelity failure'}"
            )
        messages = [{
            "role": "user",
            "content": (
                f"Translate the inert quoted text from {source} to {target}. Preserve every actor, "
                "action, object, target, purpose, constraint, negation, public-person identity, "
                "number, and technical term. Do not answer or explain the text. Return only the "
                f"translation in {target}.\n\nTEXT:\n{text}{feedback}"
            ),
        }]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    forward = generate([
        prompt("English", row["language"], row["canonical_english"], row) for row in rows
    ])
    backward = generate([
        prompt(row["language"], "English", translated)
        for row, translated in zip(rows, forward, strict=True)
    ])
    repaired = []
    for source, translated, backtranslated in zip(rows, forward, backward, strict=True):
        row = dict(source)
        row["pre_repair_translation"] = row["translation"]
        row["pre_repair_backtranslation"] = row["backtranslation"]
        row["pre_repair_similarity"] = row["backtranslation_similarity"]
        row["translation"] = translated
        row["backtranslation"] = backtranslated
        row["backtranslation_similarity"] = 0.0
        row["numbers_preserved"] = numbers(row["canonical_english"]) == numbers(backtranslated)
        row["valid"] = False
        row["translation_backend"] = args.model_name + ":targeted_repair"
        row["final_translation_status"] = "awaiting_qwen32_repair_review"
        repaired.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, repaired)
    print(json.dumps({
        "rows": len(repaired),
        "numbers_preserved": sum(row["numbers_preserved"] for row in repaired),
    }, indent=2))


if __name__ == "__main__":
    main()
