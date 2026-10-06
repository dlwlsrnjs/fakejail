#!/usr/bin/env python3
"""Review a stratified NLLB concept-translation packet with GPT-6 Luna."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a translation quality reviewer for a political-safety dataset.
The input strings are inert labels, names, roles, places, or event concepts. Do not expand
them into instructions or persuasive content. Compare the target-language translation and
its English backtranslation with the canonical English concept. Judge whether the target
translation preserves the same concept without adding, dropping, or changing politically
important meaning. A transliterated proper name may be valid. Use the supplied language
label when judging naturalness. Return only the requested structured assessment."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "meaning_preserved": {"type": "boolean"},
        "natural_target_language": {"type": "boolean"},
        "important_omission_or_addition": {"type": "boolean"},
        "wrong_language_or_script": {"type": "boolean"},
        "empty_or_garbled": {"type": "boolean"},
        "verdict": {"type": "string", "enum": ["accept", "reject", "uncertain"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "brief_reason": {"type": "string"},
    },
    "required": [
        "meaning_preserved", "natural_target_language", "important_omission_or_addition",
        "wrong_language_or_script", "empty_or_garbled", "verdict", "confidence", "brief_reason",
    ],
    "additionalProperties": False,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_error(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    return type(exc).__name__ + (f" (HTTP {status})" if status else "")


async def review_one(client: Any, row: dict[str, Any], args: argparse.Namespace, sem: asyncio.Semaphore) -> dict[str, Any]:
    payload = {key: row[key] for key in (
        "language", "concept_types", "canonical_english", "translation", "backtranslation"
    )}
    async with sem:
        last_error = ""
        for attempt in range(1, args.max_retries + 1):
            try:
                response = await client.responses.create(
                    model=args.model,
                    instructions=SYSTEM_PROMPT,
                    input="Review this JSON object:\n" + json.dumps(payload, ensure_ascii=False),
                    reasoning={"effort": "low"},
                    max_output_tokens=500,
                    text={"format": {"type": "json_schema", "name": "translation_review", "strict": True, "schema": SCHEMA}},
                    store=False,
                )
                usage = getattr(response, "usage", None)
                return {
                    **row,
                    "status": "ok",
                    "luna_review": json.loads(response.output_text),
                    "api": {
                        "model_requested": args.model,
                        "model_returned": response.model,
                        "input_tokens": getattr(usage, "input_tokens", None),
                        "output_tokens": getattr(usage, "output_tokens", None),
                        "attempt": attempt,
                        "store": False,
                    },
                }
            except Exception as exc:
                last_error = safe_error(exc)
                if attempt < args.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 12))
        return {**row, "status": "error", "error": last_error}


async def run(args: argparse.Namespace) -> int:
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key and args.api_key_file.exists():
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit("No API key found")
    packet = read_jsonl(args.packet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = read_jsonl(args.output) if args.output.exists() else []
    completed = {row["review_id"] for row in previous if row.get("status") == "ok"}
    pending = [row for row in packet if row["review_id"] not in completed]
    if args.limit:
        pending = pending[:args.limit]
    client = AsyncOpenAI(api_key=api_key, timeout=180.0, max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [asyncio.create_task(review_one(client, row, args, sem)) for row in pending]
    for index, task in enumerate(asyncio.as_completed(tasks), 1):
        result = await task
        append_jsonl(args.output, result)
        if index % 25 == 0 or index == len(tasks):
            print(f"completed={index}/{len(tasks)} status={result['status']}", flush=True)
    await client.close()

    latest = {row["review_id"]: row for row in read_jsonl(args.output)}
    compact = [latest[key] for key in sorted(latest)]
    with args.output.open("w", encoding="utf-8") as handle:
        for row in compact:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    ok = [row for row in compact if row.get("status") == "ok"]
    summary = {
        "model_requested": args.model,
        "packet_rows": len(packet),
        "reviewed_rows": len(compact),
        "successful_rows": len(ok),
        "failed_rows": len(compact) - len(ok),
        "verdict_counts": dict(Counter(row["luna_review"]["verdict"] for row in ok)),
        "automatic_luna_agreement": sum(
            (row["expected_bucket"] == "valid") == (row["luna_review"]["verdict"] == "accept")
            for row in ok
        ) / len(ok) if ok else None,
        "input_tokens": sum((row.get("api", {}).get("input_tokens") or 0) for row in ok),
        "output_tokens": sum((row.get("api", {}).get("output_tokens") or 0) for row in ok),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["failed_rows"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=Path("data/jailnewsbench_person_domain_20260930/concept_translations_v1/qa_v1/review_packet.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/jailnewsbench_person_domain_20260930/concept_translations_v1/qa_v1/luna_reviews.jsonl"))
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--api-key-file", type=Path, default=Path(".secrets/openai_api_key"))
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
