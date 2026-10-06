#!/usr/bin/env python3
"""Translate the deduplicated concept inventory with NLLB and round-trip QA."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


@torch.inference_mode()
def translate(
    tokenizer: Any,
    model: Any,
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    max_length: int,
) -> list[str]:
    tokenizer.src_lang = src_lang
    target_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(model.device)
        generated = model.generate(
            **encoded,
            forced_bos_token_id=target_id,
            max_new_tokens=max_length,
            num_beams=1,
            do_sample=False,
        )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return [" ".join(text.split()) for text in outputs]


@torch.inference_mode()
def embed(
    tokenizer: Any,
    model: Any,
    texts: list[str],
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(model.device)
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        vectors.append(F.normalize(pooled.float(), dim=1).cpu().numpy())
    return np.concatenate(vectors, axis=0)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory", type=Path,
        default=Path("data/jailnewsbench_person_domain_20260930/concept_translations_v1/concept_inventory.jsonl"),
    )
    parser.add_argument(
        "--queue", type=Path,
        default=Path("data/jailnewsbench_person_domain_20260930/concept_translations_v1/translation_queue.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/jailnewsbench_person_domain_20260930/concept_translations_v1/nllb_shards"),
    )
    parser.add_argument("--nllb-model", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--translation-batch-size", type=int, default=128)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--similarity-threshold", type=float, default=0.90)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    inventory = read_jsonl(args.inventory)
    rows = read_jsonl(args.queue)
    languages = []
    seen = set()
    for row in rows:
        key = (row["language"], row["nllb_code"])
        if key not in seen:
            seen.add(key)
            languages.append(key)
    assigned = [item for index, item in enumerate(languages) if index % args.num_shards == args.shard_index]
    texts = [row["canonical_english"] for row in inventory]
    concept_ids = [row["concept_id"] for row in inventory]
    concept_types = [row["concept_types"] for row in inventory]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    nllb_tokenizer = AutoTokenizer.from_pretrained(args.nllb_model, local_files_only=True)
    nllb = AutoModelForSeq2SeqLM.from_pretrained(
        args.nllb_model, local_files_only=True, torch_dtype=dtype
    ).to(device).eval()
    embed_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, local_files_only=True)
    embed_model = AutoModel.from_pretrained(
        args.embedding_model, local_files_only=True, torch_dtype=dtype
    ).to(device).eval()
    source_vectors = embed(
        embed_tokenizer, embed_model, texts, args.embedding_batch_size, args.max_length
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for language, nllb_code in assigned:
        output_path = args.output_dir / f"{safe_name(nllb_code)}.jsonl"
        if output_path.exists():
            existing_count = sum(1 for line in output_path.open(encoding="utf-8") if line.strip())
            if existing_count == len(inventory):
                print(f"skip complete {language} rows={existing_count}", flush=True)
                continue
            raise RuntimeError(f"partial shard requires inspection: {output_path} rows={existing_count}")
        forward = translate(
            nllb_tokenizer, nllb, texts, "eng_Latn", nllb_code,
            args.translation_batch_size, args.max_length,
        )
        backward = translate(
            nllb_tokenizer, nllb, forward, nllb_code, "eng_Latn",
            args.translation_batch_size, args.max_length,
        )
        back_vectors = embed(
            embed_tokenizer, embed_model, backward, args.embedding_batch_size, args.max_length
        )
        similarities = np.sum(source_vectors * back_vectors, axis=1)
        with output_path.open("w", encoding="utf-8") as handle:
            for index, (translated, backtranslated, similarity) in enumerate(
                zip(forward, backward, similarities)
            ):
                handle.write(json.dumps({
                    "translation_id": f"{concept_ids[index]}::{language}",
                    "concept_id": concept_ids[index],
                    "concept_types": concept_types[index],
                    "canonical_english": texts[index],
                    "language": language,
                    "nllb_code": nllb_code,
                    "translation": translated,
                    "backtranslation": backtranslated,
                    "backtranslation_similarity": round(float(similarity), 8),
                    "valid": bool(float(similarity) >= args.similarity_threshold),
                    "translation_backend": "facebook/nllb-200-distilled-1.3B",
                }, ensure_ascii=False, sort_keys=True) + "\n")
        record = {
            "language": language, "nllb_code": nllb_code, "rows": len(inventory),
            "valid": int(np.sum(similarities >= args.similarity_threshold)),
            "mean_similarity": float(np.mean(similarities)),
            "output": str(output_path),
        }
        summary.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    (args.output_dir / f"worker_{args.shard_index:02d}_summary.json").write_text(
        json.dumps({
            "shard_index": args.shard_index, "num_shards": args.num_shards,
            "assigned_languages": len(assigned), "completed": summary,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
