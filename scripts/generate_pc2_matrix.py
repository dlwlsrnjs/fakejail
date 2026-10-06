#!/usr/bin/env python3
"""Generate target-model responses for a prepared PC2 attack matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
from pathlib import Path
from typing import Any

from tqdm import tqdm


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        row["matrix_id"] for row in read_jsonl(path)
        if row.get("generation_error") is None
    }


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_vllm(rows: list[dict], model: str, output: Path, batch_size: int,
             max_tokens: int, tensor_parallel_size: int) -> None:
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size)
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    for start in tqdm(range(0, len(rows), batch_size), desc="vLLM batches"):
        batch = rows[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["attacked_prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
            for row in batch
        ]
        generated = llm.generate(prompts, params)
        append_rows(output, [
            {**row, "target_model": model, "generation": result.outputs[0].text,
             "generation_error": None}
            for row, result in zip(batch, generated)
        ])


def run_openai(rows: list[dict], model: str, output: Path, max_tokens: int,
               concurrency: int) -> None:
    from openai import OpenAI
    local = threading.local()

    def generate(row: dict) -> dict:
        if not hasattr(local, "client"):
            local.client = OpenAI()
        error = None
        generation = ""
        try:
            response = local.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": row["attacked_prompt"]}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            generation = response.choices[0].message.content or ""
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return {**row, "target_model": model, "generation": generation,
                "generation_error": error}

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(generate, row) for row in rows]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=model):
            append_rows(output, [future.result()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["vllm", "openai"], required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Concurrent OpenAI requests")
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.valid_only:
        rows = [row for row in rows if row["nllb_similarity_valid"]]
    done = existing_ids(args.output)
    rows = [row for row in rows if row["matrix_id"] not in done]
    if args.limit is not None:
        rows = rows[:args.limit]
    print(json.dumps({"pending": len(rows), "already_done": len(done), "backend": args.backend}))
    if args.backend == "vllm":
        run_vllm(rows, args.model, args.output, args.batch_size, args.max_tokens,
                 args.tensor_parallel_size)
    else:
        run_openai(rows, args.model, args.output, args.max_tokens, args.concurrency)


if __name__ == "__main__":
    main()
