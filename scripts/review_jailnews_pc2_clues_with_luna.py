#!/usr/bin/env python3
"""Review and, when reliable, repair PC2 clue translations with GPT-6 Luna."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


SYSTEM = """You are a multilingual translation adjudicator for an authorized political-safety dataset.
The input is a neutral identity quiz clue, not a request to generate news. Compare the English clue with
the target-language clue, using the English backtranslation only as a diagnostic. Preserve the anonymity
of the person: do not add the person's name. Preserve role, geography, event facts, quantities, negation,
and uncertainty. Accept a natural paraphrase. If the target is materially wrong and you are confident,
return a complete corrected clue in the requested target language. Otherwise request human review."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "meaning_preserved": {"type": "boolean"},
        "natural_target_language": {"type": "boolean"},
        "identity_remains_unnamed": {"type": "boolean"},
        "role_geography_event_preserved": {"type": "boolean"},
        "verdict": {"type": "string", "enum": ["accept", "retranslate", "human_review"]},
        "corrected_translation": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "brief_reason": {"type": "string"},
    },
    "required": [
        "meaning_preserved", "natural_target_language", "identity_remains_unnamed",
        "role_geography_event_preserved", "verdict", "corrected_translation", "confidence",
        "brief_reason",
    ],
    "additionalProperties": False,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def append(path: Path, row: dict[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def safe_error(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    return type(exc).__name__ + (f" (HTTP {status})" if status else "")


async def review_one(client: Any, row: dict[str, Any], args: argparse.Namespace, semaphore: asyncio.Semaphore):
    payload = {key: row[key] for key in (
        "language", "canonical_english", "translation", "backtranslation", "backtranslation_similarity"
    )}
    async with semaphore:
        error = ""
        for attempt in range(1, args.max_retries + 1):
            try:
                response = await client.responses.create(
                    model=args.model,
                    instructions=SYSTEM,
                    input="Review this JSON object:\n" + json.dumps(payload, ensure_ascii=False),
                    reasoning={"effort": "low"}, max_output_tokens=600, store=False,
                    text={"format": {"type": "json_schema", "name": "pc2_clue_translation_review", "strict": True, "schema": SCHEMA}},
                )
                result = json.loads(response.output_text)
                if result["verdict"] == "retranslate" and not result["corrected_translation"]:
                    result["verdict"] = "human_review"
                if result["verdict"] != "retranslate":
                    result["corrected_translation"] = None
                usage = getattr(response, "usage", None)
                return {**row, "status": "ok", "luna_review": result, "api": {
                    "model_requested": args.model, "model_returned": response.model,
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None), "attempt": attempt, "store": False,
                }}
            except Exception as exc:
                error = safe_error(exc)
                if attempt < args.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 12))
        return {**row, "status": "error", "error": error}


async def run(args: argparse.Namespace) -> int:
    from openai import AsyncOpenAI

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key and args.api_key_file.exists():
        key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit("No API key found")
    packet = read_jsonl(args.packet)
    packet = [row for row in packet if int(hashlib.sha256(row["review_id"].encode()).hexdigest(), 16) % args.shard_count == args.shard_index]
    existing = read_jsonl(args.output) if args.output.exists() else []
    done = {row["review_id"] for row in existing if row.get("status") == "ok"}
    pending = [row for row in packet if row["review_id"] not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=key, timeout=180.0, max_retries=0)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [asyncio.create_task(review_one(client, row, args, semaphore)) for row in pending]
    for index, task in enumerate(asyncio.as_completed(tasks), 1):
        result = await task
        append(args.output, result)
        if index % 25 == 0 or index == len(tasks):
            print(f"shard={args.shard_index} completed={index}/{len(tasks)} status={result['status']}", flush=True)
    await client.close()
    latest = {row["review_id"]: row for row in read_jsonl(args.output)}
    compact = [latest[key] for key in sorted(latest)]
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in compact:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(args.output)
    ok = [row for row in compact if row.get("status") == "ok"]
    summary = {
        "shard": args.shard_index, "shard_count": args.shard_count, "packet_rows": len(packet),
        "reviewed_rows": len(compact), "successful_rows": len(ok), "failed_rows": len(compact) - len(ok),
        "verdicts": dict(Counter(row["luna_review"]["verdict"] for row in ok)),
        "input_tokens": sum((row.get("api", {}).get("input_tokens") or 0) for row in ok),
        "output_tokens": sum((row.get("api", {}).get("output_tokens") or 0) for row in ok),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0 if summary["failed_rows"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--api-key-file", type=Path, default=Path(".secrets/openai_api_key"))
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
