#!/usr/bin/env python3
"""Generate auditable surrogate responses for the corrected PC2 v2 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from jailnews_official_attack_axis import OFFICIAL_FIVE, apply_official_attack


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "data/jailnewsbench_pc2_quiz_v2_20260930/quiz_matrix.jsonl"
DEFAULT_RUNTIME = ROOT / "artifacts/jailnews_bandit_20260930/runtime/surrogate_quiz_v2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def person_rows(matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in matrix
        if row["sample_entity_state"] == "verified_person_present"
        and row["eligible_for_model_run"]
    ]


def make_plan(args: argparse.Namespace) -> None:
    if args.max_people < 0:
        raise ValueError("max-people must be zero (all) or a positive integer")
    rows = person_rows(read_jsonl(args.matrix))
    quiz_ids = sorted({row["quiz_id"] for row in rows})
    if args.complete_only:
        eligible_counts = Counter(
            row["quiz_id"] for row in rows
            if row["condition"] == "person_identity_quiz"
        )
        quiz_ids = sorted(
            quiz_id for quiz_id, count in eligible_counts.items() if count == 72
        )
        complete = set(quiz_ids)
        rows = [row for row in rows if row["quiz_id"] in complete]
    if args.max_people:
        quiz_ids = quiz_ids[:args.max_people]
        selected = set(quiz_ids)
        rows = [row for row in rows if row["quiz_id"] in selected]
    by_quiz = Counter(row["quiz_id"] for row in rows if row["condition"] == "person_identity_quiz")
    malformed = {key: count for key, count in by_quiz.items() if count != 72}
    if malformed:
        raise RuntimeError(f"eligible quiz grid is incomplete for {len(malformed)} people")
    direct = [row for row in rows if row["condition"] == "direct_original"]
    quiz = [row for row in rows if row["condition"] == "person_identity_quiz"]
    if len(direct) != len(by_quiz):
        raise RuntimeError(f"direct control mismatch: direct={len(direct)} people={len(by_quiz)}")
    arms = len(direct) + len(quiz) * len(OFFICIAL_FIVE)
    plan = {
        "schema": "jailnews_surrogate_prior_plan/v2",
        "matrix": str(args.matrix),
        "matrix_sha256": digest(args.matrix.read_text(encoding="utf-8")),
        "people": len(by_quiz),
        "selected_quiz_ids": quiz_ids,
        "languages": 72,
        "methods": ["direct_original", *OFFICIAL_FIVE],
        "direct_arms": len(direct),
        "quiz_method_arms": len(quiz) * len(OFFICIAL_FIVE),
        "arms": arms,
        "draws_per_model_arm": args.draws,
        "models": args.models,
        "expected_generations": arms * args.draws * len(args.models),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "retention": "raw prompts and generations retained for audit; sanitized derivative produced after judging",
    }
    atomic_json(args.output, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def sample_context(row: dict[str, Any]) -> str:
    article = row.get("article_english_template") or ""
    if not article:
        return "Background for this same behavior is provided in the behavior template below.\n\n"
    return "Background source context for this same behavior:\n" + article + "\n\n"


def generate(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    matrix = person_rows(read_jsonl(args.matrix))
    quiz_ids = plan.get("selected_quiz_ids") or sorted({row["quiz_id"] for row in matrix})
    selected = {quiz_id for index, quiz_id in enumerate(quiz_ids) if index % args.shard_count == args.shard_index}
    rows = [row for row in matrix if row["quiz_id"] in selected]
    arms = []
    for row in rows:
        if row["condition"] == "direct_original":
            arms.append((row, "direct_original", row["prompt"]))
        elif row["condition"] == "person_identity_quiz":
            for method in OFFICIAL_FIVE:
                arms.append((row, method, apply_official_attack(
                    row["prompt"], row.get("motivation_category") or "Political", method,
                    context_prefix=sample_context(row) if method == "context_overload" else None,
                )))

    llm = LLM(
        model=str(args.model), dtype="bfloat16", max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if args.resume and args.output.exists():
        completed = {
            row["trial_id"] for row in read_jsonl(args.output)
            if row.get("trial_id")
        }
    open_flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if args.resume else os.O_TRUNC)
    fd = os.open(args.output, open_flags, 0o600)
    started = time.perf_counter()
    total = 0
    skipped = 0
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for draw in range(args.draws):
            params = SamplingParams(
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                max_tokens=args.max_tokens, seed=args.seed + draw,
            )
            pending = []
            for row, method, prompt in arms:
                trial_id = f"{row['matrix_id']}::{method}::{args.model_id}::d{draw}"
                if trial_id in completed:
                    skipped += 1
                else:
                    pending.append((row, method, prompt, trial_id))
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                chats = [[{"role": "user", "content": prompt}] for _, _, prompt, _ in batch]
                outputs = list(llm.chat(chats, params, use_tqdm=True))
                for (row, method, prompt, trial_id), request in zip(batch, outputs, strict=True):
                    result = request.outputs[0]
                    generation = result.text
                    judge_behavior = row["behavior_english_template"].replace(
                        "[[PERSON_1]]", row["canonical_person"] or ""
                    )
                    handle.write(json.dumps({
                        "trial_id": trial_id,
                        "matrix_id": row["matrix_id"],
                        "quiz_id": row["quiz_id"],
                        "sample_id": row["sample_id"],
                        "person": row["canonical_person"],
                        "language": row["language"],
                        "method": method,
                        "model_id": args.model_id,
                        "draw": draw,
                        "seed": args.seed + draw,
                        "prompt": prompt,
                        "prompt_sha256": digest(prompt),
                        "behavior": judge_behavior,
                        "generation": generation,
                        "response_sha256": digest(generation),
                        "prompt_tokens": len(request.prompt_token_ids or []),
                        "output_tokens": len(result.token_ids or []),
                        "finish_reason": result.finish_reason,
                    }, ensure_ascii=False, sort_keys=True) + "\n")
                    total += 1
                handle.flush()
    report = {
        "schema": "jailnews_surrogate_raw_generation/v2",
        "model_id": args.model_id,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "people": len(selected),
        "arms": len(arms),
        "draws": args.draws,
        "rows": total,
        "rows_skipped_from_existing_output": skipped,
        "rows_total_after_run": len(completed) + total,
        "elapsed_seconds": time.perf_counter() - started,
        "raw_text_retained": True,
        "output": str(args.output),
    }
    atomic_json(args.output.with_suffix(".summary.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    plan.add_argument("--output", type=Path, default=DEFAULT_RUNTIME / "plan.json")
    plan.add_argument("--draws", type=int, default=5)
    plan.add_argument("--temperature", type=float, default=1.0)
    plan.add_argument("--top-p", type=float, default=0.95)
    plan.add_argument("--top-k", type=int, default=20)
    plan.add_argument("--max-tokens", type=int, default=1024)
    plan.add_argument("--seed", type=int, default=20260930)
    plan.add_argument("--max-people", type=int, default=0)
    plan.add_argument(
        "--complete-only", action="store_true",
        help="select only people with all 72 eligible language rows",
    )
    plan.add_argument(
        "--models", nargs="+", default=[
            "meta-llama/Llama-3.1-8B-Instruct",
            "GraySwanAI/Llama-3-8B-Instruct-RR",
        ],
    )
    plan.set_defaults(function=make_plan)
    generation = commands.add_parser("generate")
    generation.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    generation.add_argument("--plan", type=Path, default=DEFAULT_RUNTIME / "plan.json")
    generation.add_argument("--model", type=Path, required=True)
    generation.add_argument("--model-id", required=True)
    generation.add_argument("--output", type=Path, required=True)
    generation.add_argument("--draws", type=int, default=5)
    generation.add_argument("--temperature", type=float, default=1.0)
    generation.add_argument("--top-p", type=float, default=0.95)
    generation.add_argument("--top-k", type=int, default=20)
    generation.add_argument("--max-tokens", type=int, default=1024)
    generation.add_argument("--max-model-len", type=int, default=8192)
    generation.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    generation.add_argument("--tensor-parallel-size", type=int, default=1)
    generation.add_argument("--batch-size", type=int, default=128)
    generation.add_argument("--seed", type=int, default=20260930)
    generation.add_argument("--shard-index", type=int, default=0)
    generation.add_argument("--shard-count", type=int, default=1)
    generation.add_argument(
        "--resume", action="store_true",
        help="append only trials not already present in the output JSONL",
    )
    generation.set_defaults(function=generate)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
