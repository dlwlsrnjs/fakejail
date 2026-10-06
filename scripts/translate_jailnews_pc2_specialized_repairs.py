#!/usr/bin/env python3
"""Retranslate strict failures with MiLMMT-46 or MADLAD-400 and backtranslate."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from jailnews_pc2_languages import MADLAD_CODES, MILMMT_LANGUAGE_NAMES


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


def translate_milmmt(rows: list[dict], model_path: Path) -> list[dict]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    def compat(config):
        text = config.text_config
        pattern = text._sliding_window_pattern
        text.sliding_window_pattern = pattern
        return config

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    llm = LLM(
        model=str(model_path), dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=4096, gpu_memory_utilization=0.82, enforce_eager=True,
        max_num_seqs=32, disable_log_stats=True, limit_mm_per_prompt={"image": 0},
        hf_overrides=compat,
    )
    params = SamplingParams(temperature=0, top_k=1, max_tokens=512, seed=20260930)

    def generate(prompts: list[str]) -> list[str]:
        output = []
        for start in range(0, len(prompts), 64):
            batch = prompts[start:start + 64]
            token_ids = [tokenizer.encode(value, add_special_tokens=False) for value in batch]
            generated = llm.generate(
                [{"prompt_token_ids": value} for value in token_ids], params, use_tqdm=False
            )
            output.extend(item.outputs[0].text.strip() for item in generated)
            print("MiLMMT", min(start + 64, len(prompts)), "/", len(prompts), flush=True)
        return output

    forward_prompts = []
    for row in rows:
        language = MILMMT_LANGUAGE_NAMES[row["language"]]
        forward_prompts.append(
            f"Translate this from English to {language}:\n"
            f"English: {row['canonical_english']}\n{language}:"
        )
    forward = generate(forward_prompts)
    back_prompts = []
    for row, translated in zip(rows, forward, strict=True):
        language = MILMMT_LANGUAGE_NAMES[row["language"]]
        back_prompts.append(
            f"Translate this from {language} to English:\n"
            f"{language}: {translated}\nEnglish:"
        )
    backward = generate(back_prompts)
    return materialize(rows, forward, backward, "xiaomi-research/MiLMMT-46-12B-v1.0")


def translate_madlad(rows: list[dict], model_path: Path, batch_size: int) -> list[dict]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()

    def generate(texts: list[str]) -> list[str]:
        output = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to("cuda")
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, max_new_tokens=512, num_beams=4, do_sample=False,
                    early_stopping=True,
                )
            output.extend(
                value.strip() for value in tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
            print("MADLAD", min(start + batch_size, len(texts)), "/", len(texts), flush=True)
        return output

    forward = generate([
        f"<2{MADLAD_CODES[row['language']]}> {row['canonical_english']}" for row in rows
    ])
    backward = generate([f"<2en> {value}" for value in forward])
    backend = (
        "google/madlad400-10b-mt" if "10b" in str(model_path).lower()
        else "google/madlad400-3b-mt"
    )
    return materialize(rows, forward, backward, backend)


def materialize(
    rows: list[dict], forward: list[str], backward: list[str], backend: str
) -> list[dict]:
    output = []
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
        row["translation_backend"] = backend + ":targeted_repair"
        row["final_translation_status"] = "awaiting_qwen32_repair_review"
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-family", choices=["milmmt", "madlad"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    route = "milmmt_46_12b" if args.model_family == "milmmt" else "madlad_400_3b"
    candidates = [row for row in read(args.input) if row["repair_route"] == route]
    rows = [row for index, row in enumerate(candidates) if index % args.shard_count == args.shard_index]
    if not rows:
        raise RuntimeError("empty repair shard")
    output = (
        translate_milmmt(rows, args.model_path)
        if args.model_family == "milmmt"
        else translate_madlad(rows, args.model_path, args.batch_size)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, output)
    print(json.dumps({
        "route": route, "rows": len(output),
        "numbers_preserved": sum(row["numbers_preserved"] for row in output),
    }, indent=2))


if __name__ == "__main__":
    main()
