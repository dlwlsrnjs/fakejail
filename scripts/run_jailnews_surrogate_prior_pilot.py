#!/usr/bin/env python3
"""Collect a sanitized local-surrogate prior for the JailNews PC2 language axis.

Raw prompts and generations are written only to a caller-selected path (the Slurm
launcher uses node-local temporary storage).  The persistent aggregate retains
only binary labels, hashes, token counts, seeds, and non-sensitive arm metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from jailnews_official_attack_axis import OFFICIAL_FIVE, apply_official_attack


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "data/jailnewsbench_pc2_contextual_20260929/all_language_matrix.jsonl"
DEFAULT_PLAN = ROOT / "artifacts/jailnews_bandit_20260930/runtime/surrogate_pilot/plan.json"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    count = 0
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return count


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_quantile_pilots(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_pilot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pilot[row["pilot_id"]].append(row)
    summaries = []
    for pilot_id, items in by_pilot.items():
        summaries.append({
            "pilot_id": pilot_id,
            "sample_id": items[0].get("sample_id") or f"jailnewsbench:train:{items[0]['uid']}",
            "source_record_id": items[0]["uid"],
            "person": items[0]["selected_person"],
            "region": items[0]["region_en"],
            "source_language": items[0]["source_language_code"],
            "languages": len(items),
            "valid_languages": sum(bool(item["nllb_similarity_valid"]) for item in items),
        })
    summaries.sort(key=lambda item: (item["valid_languages"], item["pilot_id"]))
    if count >= len(summaries):
        return summaries
    if count == 1:
        indices = [len(summaries) // 2]
    else:
        indices = [round(index * (len(summaries) - 1) / (count - 1)) for index in range(count)]
    return [summaries[index] for index in indices]


def make_plan(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(args.matrix))
    selected = select_quantile_pilots(rows, args.pilot_count)
    plan = {
        "schema": "jailnews_surrogate_prior_pilot_plan/v1",
        "matrix": str(args.matrix),
        "matrix_sha256": digest(args.matrix.read_text(encoding="utf-8")),
        "selection": "deterministic quantiles over the per-person count of valid round-trip translations",
        "selected": selected,
        "pilot_count": len(selected),
        "languages_per_pilot": 72,
        "methods": list(OFFICIAL_FIVE),
        "arms": len(selected) * 72 * len(OFFICIAL_FIVE),
        "draws_per_model_arm": args.draws,
        "models": [
            "meta-llama/Llama-3.1-8B-Instruct",
            "GraySwanAI/Llama-3-8B-Instruct-RR",
        ],
        "expected_generations": len(selected) * 72 * len(OFFICIAL_FIVE) * args.draws * 2,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "persistent_retention": "hashes, labels, token counts, seeds, and arm metadata only",
        "raw_retention": "node-local temporary storage only",
    }
    atomic_json(args.output, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def context_prefix(rows: list[dict[str, Any]], tokenizer: Any, seed: int) -> str:
    articles = list(dict.fromkeys(row["article_local"] for row in rows if row.get("article_local")))
    random.Random(seed).shuffle(articles)
    token_ids: list[int] = []
    for article in articles:
        token_ids.extend(tokenizer.encode(article, add_special_tokens=False))
        if len(token_ids) >= 512:
            break
    return tokenizer.decode(token_ids[:512], skip_special_tokens=True) + "\n\n"


def generate(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    selected_order = [item["pilot_id"] for item in plan["selected"]]
    selected = set(selected_order)
    all_source_rows = [row for row in read_jsonl(args.matrix) if row["pilot_id"] in selected]
    if len(all_source_rows) != len(selected) * 72:
        raise ValueError(f"expected {len(selected) * 72} matrix rows, found {len(all_source_rows)}")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")
    shard_pilots = {
        pilot_id for index, pilot_id in enumerate(selected_order)
        if index % args.shard_count == args.shard_index
    }
    source_rows = [row for row in all_source_rows if row["pilot_id"] in shard_pilots]
    if not source_rows:
        raise ValueError(f"empty shard {args.shard_index}/{args.shard_count}")

    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    overload = context_prefix(all_source_rows, tokenizer, args.seed)
    arms = []
    for source in source_rows:
        for method in OFFICIAL_FIVE:
            attacked = apply_official_attack(
                source["attacked_prompt"],
                source["motivation_category"],
                method,
                context_prefix=overload if method == "context_overload" else None,
            )
            arms.append((source, method, attacked))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    total = 0
    started = time.perf_counter()
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for draw in range(args.draws):
            params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                seed=args.seed + draw,
            )
            for start in range(0, len(arms), args.batch_size):
                batch = arms[start:start + args.batch_size]
                chats = [[{"role": "user", "content": attacked}] for _, _, attacked in batch]
                batch_started = time.perf_counter()
                outputs = list(llm.chat(chats, params, use_tqdm=True))
                batch_elapsed = time.perf_counter() - batch_started
                for (source, method, attacked), request in zip(batch, outputs, strict=True):
                    result = request.outputs[0]
                    generation = result.text
                    record = {
                        "trial_id": f"{source['matrix_id']}::{method}::{args.model_id}::d{draw}",
                        "sample_id": source.get("sample_id") or f"jailnewsbench:train:{source['uid']}",
                        "pilot_id": source["pilot_id"],
                        "person": source["selected_person"],
                        "language": source["clue_language"],
                        "method": method,
                        "model_id": args.model_id,
                        "draw": draw,
                        "seed": args.seed + draw,
                        "translation_valid": bool(source["nllb_similarity_valid"]),
                        "backtranslation_similarity": source["backtranslation_similarity"],
                        "prompt_sha256": digest(attacked),
                        "response_sha256": digest(generation),
                        "prompt_tokens": len(request.prompt_token_ids or []),
                        "output_tokens": len(result.token_ids or []),
                        "finish_reason": result.finish_reason,
                        "batch_elapsed_seconds": batch_elapsed,
                        # The next two fields exist only in the node-local temporary file.
                        "behavior": source["seed_instruction_local"],
                        "generation": generation,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
                handle.flush()
    report = {
        "schema": "jailnews_surrogate_raw_generation/v1",
        "model_id": args.model_id,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "people": len(shard_pilots),
        "arms": len(arms),
        "draws": args.draws,
        "rows": total,
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(args.output),
        "retention": "temporary; must be deleted by the batch allocation",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def aggregate(args: argparse.Namespace) -> None:
    judged = list(read_jsonl(args.judged))
    trials = []
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in judged:
        label = int(row.get("harmbench_label", -1))
        clean = {
            "trial_id": row["trial_id"],
            "sample_id": row["sample_id"],
            "pilot_id": row["pilot_id"],
            "person": row["person"],
            "language": row["language"],
            "method": row["method"],
            "model_id": row["model_id"],
            "draw": row["draw"],
            "seed": row["seed"],
            "translation_valid": row["translation_valid"],
            "backtranslation_similarity": row["backtranslation_similarity"],
            "prompt_sha256": row["prompt_sha256"],
            "response_sha256": row["response_sha256"],
            "prompt_tokens": row["prompt_tokens"],
            "output_tokens": row["output_tokens"],
            "finish_reason": row["finish_reason"],
            "judge_id": args.judge_id,
            "judge_label": label,
            "success": label == 1,
        }
        trials.append(clean)
        key = (row["sample_id"], row["language"], row["method"], row["model_id"])
        if key not in grouped:
            grouped[key] = {
                "sample_id": row["sample_id"],
                "pilot_id": row["pilot_id"],
                "person": row["person"],
                "language": row["language"],
                "method": row["method"],
                "model_id": row["model_id"],
                "successes": 0,
                "trials": 0,
                "invalid_judgements": 0,
                "translation_valid": bool(row["translation_valid"]),
                "backtranslation_similarity": row["backtranslation_similarity"],
                "response_hashes": [],
                "judge_id": args.judge_id,
            }
        target = grouped[key]
        target["trials"] += int(label in (0, 1))
        target["successes"] += int(label == 1)
        target["invalid_judgements"] += int(label not in (0, 1))
        target["response_hashes"].append(row["response_sha256"])

    trial_count = atomic_jsonl(args.trials_output, trials)
    score_rows = sorted((row for row in grouped.values() if row["trials"] > 0), key=lambda row: (
        row["sample_id"], row["language"], row["method"], row["model_id"]
    ))
    score_count = atomic_jsonl(args.scores_output, score_rows)
    manifest = {
        "schema": "jailnews_surrogate_prior_scores/v1",
        "judged_rows": len(judged),
        "sanitized_trials": trial_count,
        "score_rows": score_count,
        "successes": sum(row["successes"] for row in score_rows),
        "invalid_judgements": sum(row["invalid_judgements"] for row in score_rows),
        "raw_text_retained": False,
        "trials_output": str(args.trials_output),
        "scores_output": str(args.scores_output),
    }
    atomic_json(args.scores_output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def merge_scores(args: argparse.Namespace) -> None:
    rows = [row for path in args.inputs for row in read_jsonl(path)]
    keys = [(row["sample_id"], row["language"], row["method"], row["model_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate sample/language/method/model score keys")
    models = sorted({row["model_id"] for row in rows})
    expected_models = sorted([
        "meta-llama/Llama-3.1-8B-Instruct",
        "GraySwanAI/Llama-3-8B-Instruct-RR",
    ])
    if models != expected_models:
        raise ValueError(f"unexpected model set: {models}")
    rows.sort(key=lambda row: (row["sample_id"], row["language"], row["method"], row["model_id"]))
    count = atomic_jsonl(args.output, rows)

    by_model: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_method: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_translation_valid: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_person: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_language: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        for bucket, key in (
            (by_model, row["model_id"]),
            (by_method, row["method"]),
            (by_translation_valid, str(bool(row["translation_valid"])).lower()),
            (by_person, row["person"]),
            (by_language, row["language"]),
        ):
            bucket[key][0] += int(row["successes"])
            bucket[key][1] += int(row["trials"])

    def rates(bucket: dict[str, list[int]]) -> list[dict[str, Any]]:
        return sorted(({
            "key": key,
            "successes": value[0],
            "trials": value[1],
            "asr": value[0] / value[1] if value[1] else None,
        } for key, value in bucket.items()), key=lambda row: row["key"])

    summary = {
        "schema": "jailnews_surrogate_prior_pilot_summary/v1",
        "score_rows": count,
        "models": models,
        "total_successes": sum(int(row["successes"]) for row in rows),
        "total_trials": sum(int(row["trials"]) for row in rows),
        "by_model": rates(by_model),
        "by_method": rates(by_method),
        "by_translation_valid": rates(by_translation_valid),
        "by_person": rates(by_person),
        "by_language": rates(by_language),
        "raw_text_retained": False,
    }
    atomic_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    plan.add_argument("--output", type=Path, default=DEFAULT_PLAN)
    plan.add_argument("--pilot-count", type=int, default=5)
    plan.add_argument("--draws", type=int, default=5)
    plan.add_argument("--temperature", type=float, default=1.0)
    plan.add_argument("--top-p", type=float, default=0.95)
    plan.add_argument("--top-k", type=int, default=20)
    plan.add_argument("--max-tokens", type=int, default=384)
    plan.add_argument("--seed", type=int, default=20260930)
    plan.set_defaults(function=make_plan)

    generation = commands.add_parser("generate")
    generation.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    generation.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    generation.add_argument("--model", type=Path, required=True)
    generation.add_argument("--model-id", required=True)
    generation.add_argument("--output", type=Path, required=True)
    generation.add_argument("--draws", type=int, default=5)
    generation.add_argument("--temperature", type=float, default=1.0)
    generation.add_argument("--top-p", type=float, default=0.95)
    generation.add_argument("--top-k", type=int, default=20)
    generation.add_argument("--max-tokens", type=int, default=384)
    generation.add_argument("--max-model-len", type=int, default=4096)
    generation.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    generation.add_argument("--batch-size", type=int, default=256)
    generation.add_argument("--seed", type=int, default=20260930)
    generation.add_argument("--shard-index", type=int, default=0)
    generation.add_argument("--shard-count", type=int, default=1)
    generation.set_defaults(function=generate)

    aggregation = commands.add_parser("aggregate")
    aggregation.add_argument("--judged", type=Path, required=True)
    aggregation.add_argument("--trials-output", type=Path, required=True)
    aggregation.add_argument("--scores-output", type=Path, required=True)
    aggregation.add_argument(
        "--judge-id", default="cais/HarmBench-Llama-2-13b-cls@bda705349d1144fa618770bea64d99ce54e3835b"
    )
    aggregation.set_defaults(function=aggregate)

    merge = commands.add_parser("merge")
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--summary", type=Path, required=True)
    merge.set_defaults(function=merge_scores)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
