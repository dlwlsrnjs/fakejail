#!/usr/bin/env python3
"""Repair strict NLLB English-instruction failures with MADLAD-400-10B."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer


NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?%?")


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def clean(value: str) -> str:
    return " ".join(value.split())


def numbers(value: str) -> list[str]:
    return NUMBER_RE.findall(value)


def join(prefix: str, article: str, suffix: str) -> str:
    left, right = prefix.rstrip(), suffix.lstrip()
    return (left + (" " if left else "") + article + (" " if right else "") + right).strip()


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


@torch.inference_mode()
def embed(tokenizer, model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size], return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to("cuda")
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        vectors.append(F.normalize(pooled.float(), dim=1).cpu().numpy())
    return np.concatenate(vectors, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.92)
    args = parser.parse_args()

    original_rows = read(args.translations)
    cohort = {str(row["quiz_id"]): row for row in read(args.cohort)}
    bad = [row for row in original_rows if not row.get("valid")]
    if not bad:
        write(args.output, original_rows)
        print(json.dumps({"repaired": 0, "rows": len(original_rows)}, indent=2))
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()

    # Translate numeric-free fragments independently and splice every observed
    # number back byte-for-byte. This prevents dates, vote counts, and
    # percentages from being reformatted during either round-trip direction.
    @torch.inference_mode()
    def translate_preserving_numbers(texts: list[str], target: str) -> list[str]:
        segment_sets = [NUMBER_RE.split(text) for text in texts]
        observed = [NUMBER_RE.findall(text) for text in texts]
        flat = [segment for segments in segment_sets for segment in segments]
        translated: list[str] = []
        for start in range(0, len(flat), 16):
            prompts = [f"<2{target}> {value}" for value in flat[start:start + 16]]
            encoded = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).to("cuda")
            generated = model.generate(
                **encoded, max_new_tokens=1024, num_beams=4, do_sample=False,
                early_stopping=True,
            )
            translated.extend(
                clean(value) for value in tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        output, offset = [], 0
        for segments, values in zip(segment_sets, observed, strict=True):
            rendered = translated[offset]
            offset += 1
            for number in values:
                rendered = rendered.rstrip() + " " + number + " " + translated[offset].lstrip()
                offset += 1
            output.append(clean(rendered))
        return output

    shells = []
    for row in bad:
        source = cohort[str(row["quiz_id"])]
        local_behavior = str(source["behavior_local_template"])
        local_article = str(source["article_local_template"])
        if local_behavior.count(local_article) != 1:
            raise RuntimeError(f"unprotected source article: {row['quiz_id']}")
        shells.append(local_behavior.split(local_article, 1))

    flat_local = [part for pair in shells for part in pair]
    flat_english = translate_preserving_numbers(flat_local, "en")
    english_shells = [
        clean(flat_english[index * 2] + " " + flat_english[index * 2 + 1])
        for index in range(len(bad))
    ]

    back_flat = []
    for index, row in enumerate(bad):
        language = str(row["source_language_code"])
        pair = flat_english[index * 2:index * 2 + 2]
        back_flat.extend(translate_preserving_numbers(pair, language))
    backward_shells = [
        clean(back_flat[index * 2] + " " + back_flat[index * 2 + 1])
        for index in range(len(bad))
    ]

    reforward_flat = translate_preserving_numbers(back_flat, "en")
    reforward_shells = [
        clean(reforward_flat[index * 2] + " " + reforward_flat[index * 2 + 1])
        for index in range(len(bad))
    ]

    del model
    torch.cuda.empty_cache()
    embed_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, local_files_only=True)
    embed_model = AutoModel.from_pretrained(
        args.embedding_model, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    forward_vectors = embed(embed_tokenizer, embed_model, english_shells)
    reforward_vectors = embed(embed_tokenizer, embed_model, reforward_shells)
    similarities = np.sum(forward_vectors * reforward_vectors, axis=1)

    repairs = {}
    for index, row in enumerate(bad):
        source = cohort[str(row["quiz_id"])]
        article_en = str(source["article_english_template"])
        article_local = str(source["article_local_template"])
        forward = join(flat_english[index * 2], article_en, flat_english[index * 2 + 1])
        backward = join(back_flat[index * 2], article_local, back_flat[index * 2 + 1])
        reforward = join(
            reforward_flat[index * 2], article_en, reforward_flat[index * 2 + 1]
        )
        similarity = float(similarities[index])
        numeric_ok = numbers(str(source["behavior_local_template"])) == numbers(backward)
        repaired = {
            **row,
            "behavior_english_template": forward,
            "behavior_backtranslation": backward,
            "behavior_reforward_english": reforward,
            "roundtrip_similarity": round(similarity, 8),
            "numbers_preserved": numeric_ok,
            "article_segment_protected": (
                forward.count(article_en) == 1
                and backward.count(article_local) == 1
                and reforward.count(article_en) == 1
            ),
            "translation_backend": "google/madlad400-10b-mt:targeted-repair:numbers-protected",
        }
        repaired["valid"] = bool(
            similarity >= args.threshold
            and repaired["numbers_preserved"]
            and repaired["article_segment_protected"]
        )
        repairs[str(row["quiz_id"])] = repaired

    merged = [repairs.get(str(row["quiz_id"]), row) for row in original_rows]
    remaining = [row for row in merged if not row.get("valid")]
    write(args.output, merged)
    summary = {
        "schema": "jailnews_english_behavior_translation_repair/v1",
        "rows": len(merged),
        "repair_candidates": len(bad),
        "repaired_valid": len(bad) - len(remaining),
        "remaining_invalid": len(remaining),
        "remaining": [
            {
                "quiz_id": row["quiz_id"],
                "roundtrip_similarity": row["roundtrip_similarity"],
                "numbers_preserved": row["numbers_preserved"],
            }
            for row in remaining
        ],
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if remaining:
        raise RuntimeError(f"MADLAD repair left {len(remaining)} invalid translations")


if __name__ == "__main__":
    main()
