#!/usr/bin/env python3
"""Adjudicate neutral concept translations with an OpenAI model.

The runner is resumable, never accepts an API key on the command line, stores no API
response object, and writes only the structured translation judgement.  It is an
optional batch worker; the local Codex Luna review establishes the rubric, while this
worker can scale adjudication using a model available to the caller's API project.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["accept", "retranslate", "human_review"]},
        "semantic_fidelity": {"type": "string", "enum": ["pass", "minor", "fail"]},
        "coverage": {"type": "string", "enum": ["complete", "partial", "missing"]},
        "grammar_quality": {"type": "string", "enum": ["good", "acceptable", "poor"]},
        "terminology_quality": {"type": "string", "enum": ["appropriate", "questionable", "wrong"]},
        "entity_preservation": {"type": "string", "enum": ["pass", "minor", "fail"]},
        "corrected_translation": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "concise_rationale": {"type": "string"},
        "adjudication_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": [
        "decision", "semantic_fidelity", "coverage", "grammar_quality", "terminology_quality",
        "entity_preservation", "corrected_translation", "concise_rationale", "adjudication_confidence",
    ],
    "additionalProperties": False,
}

SYSTEM = """You are a multilingual translation adjudicator. Review only the neutral concept term supplied.
Compare the English source directly with the target-language translation; use the English backtranslation
only as a diagnostic. Preserve meaning scope, negation, entities, roles, events, and geography. A cosine
score is not a pass/fail rule. Choose accept only if the target term is semantically faithful and usable.
Choose retranslate only when you can provide a reliable full corrected target-language term. Otherwise
choose human_review. corrected_translation must be null unless decision is retranslate. Return the schema."""


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", required=True, help="Exact API model ID available to the project")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--delay", type=float, default=0.0)
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; do not pass secrets on the command line")
    completed = {}
    if args.output.exists():
        completed = {row["translation_id"]: row for row in read_jsonl(args.output)}
    queue = [row for row in read_jsonl(args.input) if row["translation_id"] not in completed]
    if args.limit is not None:
        queue = queue[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    with args.output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(queue, 1):
            prompt = json.dumps({
                "canonical_english": row["canonical_english"], "target_language": row["language"],
                "target_translation": row["nllb_translation"],
                "english_backtranslation": row["nllb_backtranslation"],
                "backtranslation_similarity": row["backtranslation_similarity"],
                "concept_types": row["concept_types"],
            }, ensure_ascii=False)
            response = client.responses.create(
                model=args.model, store=False,
                input=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                text={"format": {"type": "json_schema", "name": "translation_adjudication", "strict": True, "schema": SCHEMA}},
            )
            judgement = json.loads(response.output_text)
            if judgement["decision"] == "retranslate" and not judgement["corrected_translation"]:
                judgement["decision"] = "human_review"
                judgement["concise_rationale"] += " [automatic gate: missing corrected_translation]"
            if judgement["decision"] != "retranslate":
                judgement["corrected_translation"] = None
            result = {
                "schema": "concept_translation_adjudication/v1", "translation_id": row["translation_id"],
                "concept_id": row["concept_id"], "language": row["language"],
                "canonical_english": row["canonical_english"], "nllb_translation": row["nllb_translation"],
                "nllb_backtranslation": row["nllb_backtranslation"],
                "backtranslation_similarity": row["backtranslation_similarity"],
                "review_model": args.model, **judgement,
            }
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(f"{index}/{len(queue)} {row['translation_id']} {judgement['decision']}", flush=True)
            if args.delay:
                time.sleep(args.delay)


if __name__ == "__main__":
    main()
