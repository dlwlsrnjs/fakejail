#!/usr/bin/env python3
"""Translate PC2 v2 identity clues, concepts, names, and behavior templates.

Forward translations use NLLB-200 3.3B.  Every item is translated back to
English and scored with an English semantic encoder.  Behavior placeholders
are protected by translating the surrounding segments independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from jailnews_pc2_languages import LANGUAGES, SOURCE_TO_NLLB


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/jailnewsbench_pc2_quiz_v2_20260930"
PLACEHOLDER = "[[PERSON_1]]"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def clean(value: str) -> str:
    return " ".join(value.split())


def numbers(value: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?%?", value)


@torch.inference_mode()
def translate(
    tokenizer: Any,
    model: Any,
    texts: list[str],
    source: str,
    target: str,
    batch_size: int,
    max_length: int,
) -> list[str]:
    if not texts:
        return []
    if source == target:
        return [clean(text) for text in texts]
    tokenizer.src_lang = source
    output = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(model.device)
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target),
            max_new_tokens=max_length,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
            do_sample=False,
        )
        output.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return [clean(text) for text in output]


@torch.inference_mode()
def embed(
    tokenizer: Any, model: Any, texts: list[str], batch_size: int, max_length: int
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


def load_models(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    translation_tokenizer = AutoTokenizer.from_pretrained(args.nllb_model, local_files_only=True)
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.nllb_model, local_files_only=True, torch_dtype=dtype
    ).to(device).eval()
    supported = set(translation_tokenizer.additional_special_tokens)
    missing = sorted((set(LANGUAGES.values()) | set(SOURCE_TO_NLLB.values())) - supported)
    if missing:
        raise ValueError(f"NLLB checkpoint lacks language codes: {missing}")
    embed_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, local_files_only=True)
    embed_model = AutoModel.from_pretrained(
        args.embedding_model, local_files_only=True, torch_dtype=dtype
    ).to(device).eval()
    return translation_tokenizer, translation_model, embed_tokenizer, embed_model


def build_inventory(args: argparse.Namespace) -> None:
    people = read_jsonl(args.person_cohort)
    controls = read_jsonl(args.no_person_controls)
    rows = []
    for row in people:
        rows.append({
            "item_id": f"identity:{row['quiz_id']}",
            "item_type": "identity_clue",
            "owner_quiz_ids": [row["quiz_id"]],
            "canonical_english": row["identity_clue_english"],
            "canonical_person": row["canonical_person"],
        })
        rows.append({
            "item_id": "person:" + digest(row["canonical_person"].casefold()),
            "item_type": "person_name",
            "owner_quiz_ids": [row["quiz_id"]],
            "canonical_english": row["canonical_person"],
            "canonical_person": row["canonical_person"],
        })
    concept_items: dict[str, dict[str, Any]] = {}
    for owner in people + controls:
        for concept in owner["concepts"]:
            item = concept_items.setdefault(concept["concept_id"], {
                "item_id": concept["concept_id"],
                "item_type": "concept",
                "owner_quiz_ids": [],
                "canonical_english": concept["canonical_english"],
                "concept_type": concept["concept_type"],
            })
            if item["canonical_english"].casefold() != concept["canonical_english"].casefold():
                raise ValueError(f"concept collision: {concept['concept_id']}")
            item["owner_quiz_ids"].append(owner["quiz_id"])
    rows.extend(concept_items.values())
    # Merge duplicate person-name inventory items while preserving owners.
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["item_id"] not in merged:
            merged[row["item_id"]] = row
        else:
            merged[row["item_id"]]["owner_quiz_ids"].extend(row["owner_quiz_ids"])
    output = sorted(merged.values(), key=lambda row: (row["item_type"], row["item_id"]))
    if args.item_types:
        allowed = set(args.item_types)
        output = [row for row in output if row["item_type"] in allowed]
    for row in output:
        row["owner_quiz_ids"] = sorted(set(row["owner_quiz_ids"]))
    count = atomic_jsonl(args.output, output)
    report = {
        "schema": "jailnews_pc2_translation_inventory/v2",
        "rows": count,
        "by_type": {
            item_type: sum(row["item_type"] == item_type for row in output)
            for item_type in sorted({row["item_type"] for row in output})
        },
        "languages": len(LANGUAGES),
        "expected_translations": count * len(LANGUAGES),
    }
    atomic_json(args.output.with_suffix(".summary.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def translate_language_shard(args: argparse.Namespace) -> None:
    inventory = read_jsonl(args.inventory)
    languages = [
        (name, code) for index, (name, code) in enumerate(LANGUAGES.items())
        if index % args.shard_count == args.shard_index
    ]
    if not languages:
        raise ValueError("language shard is empty")
    translation_tokenizer, translation_model, embed_tokenizer, embed_model = load_models(args)
    originals = [row["canonical_english"] for row in inventory]
    original_vectors = embed(
        embed_tokenizer, embed_model, originals, args.embedding_batch_size, args.embedding_max_length
    )
    output = []
    for language, nllb_code in languages:
        forward = translate(
            translation_tokenizer, translation_model, originals, "eng_Latn", nllb_code,
            args.translation_batch_size, args.translation_max_length,
        )
        backward = translate(
            translation_tokenizer, translation_model, forward, nllb_code, "eng_Latn",
            args.translation_batch_size, args.translation_max_length,
        )
        back_vectors = embed(
            embed_tokenizer, embed_model, backward,
            args.embedding_batch_size, args.embedding_max_length,
        )
        similarities = np.sum(original_vectors * back_vectors, axis=1)
        for index, item in enumerate(inventory):
            source = originals[index]
            back = backward[index]
            similarity = float(similarities[index])
            numeric_ok = numbers(source) == numbers(back)
            back_length_ratio = len(back) / max(1, len(source))
            threshold = args.name_similarity_threshold if item["item_type"] == "person_name" else args.similarity_threshold
            valid = bool(
                forward[index].strip()
                and back.strip()
                and similarity >= threshold
                and 0.45 <= back_length_ratio <= 2.20
                and numeric_ok
            )
            output.append({
                **item,
                "translation_id": f"{item['item_id']}::{language}",
                "language": language,
                "nllb_code": nllb_code,
                "translation": forward[index],
                "backtranslation": back,
                "backtranslation_similarity": round(similarity, 8),
                "backtranslation_length_ratio": round(back_length_ratio, 8),
                "numbers_preserved": numeric_ok,
                "valid": valid,
                "translation_backend": translation_backend(args.nllb_model),
            })
        print(json.dumps({
            "language": language,
            "rows": len(inventory),
            "valid": sum(row["valid"] for row in output if row["language"] == language),
        }), flush=True)
    count = atomic_jsonl(args.output, output)
    report = {
        "schema": "jailnews_pc2_translation_shard/v2",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "languages": [name for name, _ in languages],
        "rows": count,
        "valid": sum(row["valid"] for row in output),
        "invalid": sum(not row["valid"] for row in output),
    }
    atomic_json(args.output.with_suffix(".summary.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def translate_segmented(
    tokenizer: Any,
    model: Any,
    texts: list[str],
    source: str,
    target: str,
    batch_size: int,
    max_length: int,
) -> list[str]:
    split_texts = [text.split(PLACEHOLDER) for text in texts]
    flat = [segment for segments in split_texts for segment in segments]
    translated = translate(tokenizer, model, flat, source, target, batch_size, max_length)
    output = []
    offset = 0
    for segments in split_texts:
        count = len(segments)
        output.append(PLACEHOLDER.join(translated[offset:offset + count]))
        offset += count
    return output


def join_protected_article(prefix: str, article: str, suffix: str) -> str:
    """Join translated shell text while keeping the official article byte-exact."""
    left = prefix.rstrip()
    right = suffix.lstrip()
    return (left + (" " if left else "") + article + (" " if right else "") + right).strip()


def translate_with_protected_articles(
    tokenizer: Any,
    model: Any,
    texts: list[str],
    source_articles: list[str],
    target_articles: list[str],
    source: str,
    target: str,
    batch_size: int,
    max_length: int,
) -> tuple[list[str], list[str], list[bool]]:
    """Translate only text around an article and insert the official target article.

    The behavior contains a quoted source article. Translating that article again
    can silently drop dates, vote counts, or other facts, so the dataset's paired
    article is treated as a protected span in both translation directions.
    """
    shells: list[tuple[str, str]] = []
    protected: list[bool] = []
    for text, article in zip(texts, source_articles, strict=True):
        if article and text.count(article) == 1:
            before, after = text.split(article, 1)
            shells.append((before, after))
            protected.append(True)
        else:
            # Retain an auditable result, but the caller must reject this row.
            shells.append((text, ""))
            protected.append(False)
    flat_shells = [part for pair in shells for part in pair]
    translated = translate_segmented(
        tokenizer, model, flat_shells, source, target, batch_size, max_length
    )
    output: list[str] = []
    translated_shells: list[str] = []
    for index, target_article in enumerate(target_articles):
        before = translated[index * 2]
        after = translated[index * 2 + 1]
        output.append(join_protected_article(before, target_article, after))
        translated_shells.append(clean(before + " " + after))
    return output, translated_shells, protected


def translation_backend(model_path: Path, article_protected: bool = False) -> str:
    value = str(model_path).lower()
    if "distilled-1.3b" in value:
        model = "facebook/nllb-200-distilled-1.3B"
    elif "3.3b" in value:
        model = "facebook/nllb-200-3.3B"
    else:
        model = str(model_path)
    backend = model + ":num_beams=4"
    return backend + (":official_article_protected" if article_protected else "")


def translate_behaviors(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.person_cohort) + read_jsonl(args.no_person_controls)
    translation_tokenizer, translation_model, embed_tokenizer, embed_model = load_models(args)
    output = []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["source_language_code"] not in SOURCE_TO_NLLB:
            raise ValueError(f"unsupported source language: {row['source_language_code']}")
        by_source[row["source_language_code"]].append(row)
    for source_language, items in sorted(by_source.items()):
        source_code = SOURCE_TO_NLLB[source_language]
        originals = [row["behavior_local_template"] for row in items]
        local_articles = [row["article_local_template"] for row in items]
        english_articles = [row["article_english_template"] for row in items]
        forward, forward_shells, forward_protected = translate_with_protected_articles(
            translation_tokenizer, translation_model, originals, local_articles, english_articles,
            source_code, "eng_Latn",
            args.translation_batch_size, args.translation_max_length,
        )
        backward, backward_shells, backward_protected = translate_with_protected_articles(
            translation_tokenizer, translation_model, forward, english_articles, local_articles,
            "eng_Latn", source_code,
            args.translation_batch_size, args.translation_max_length,
        )
        reforward, reforward_shells, reforward_protected = translate_with_protected_articles(
            translation_tokenizer, translation_model, backward, local_articles, english_articles,
            source_code, "eng_Latn",
            args.translation_batch_size, args.translation_max_length,
        )
        forward_vectors = embed(
            embed_tokenizer, embed_model, forward_shells,
            args.embedding_batch_size, args.embedding_max_length,
        )
        reforward_vectors = embed(
            embed_tokenizer, embed_model, reforward_shells,
            args.embedding_batch_size, args.embedding_max_length,
        )
        similarities = np.sum(forward_vectors * reforward_vectors, axis=1)
        for index, row in enumerate(items):
            expected_placeholders = row["behavior_local_template"].count(PLACEHOLDER)
            placeholder_ok = (
                forward[index].count(PLACEHOLDER) == expected_placeholders
                and backward[index].count(PLACEHOLDER) == expected_placeholders
                and reforward[index].count(PLACEHOLDER) == expected_placeholders
            )
            numeric_ok = numbers(originals[index]) == numbers(backward[index])
            similarity = float(similarities[index])
            article_protected = bool(
                forward_protected[index]
                and backward_protected[index]
                and reforward_protected[index]
                and english_articles[index] in forward[index]
                and local_articles[index] in backward[index]
                and english_articles[index] in reforward[index]
            )
            output.append({
                "quiz_id": row["quiz_id"],
                "sample_id": row["sample_id"],
                "source_language_code": source_language,
                "source_nllb_code": source_code,
                "behavior_local_template": originals[index],
                "behavior_english_template": forward[index],
                "behavior_backtranslation": backward[index],
                "behavior_reforward_english": reforward[index],
                "roundtrip_similarity": round(similarity, 8),
                "roundtrip_scope": "instruction_shell_excluding_official_article",
                "article_segment_protected": article_protected,
                "placeholders_preserved": placeholder_ok,
                "numbers_preserved": numeric_ok,
                "valid": bool(
                    similarity >= args.behavior_similarity_threshold
                    and article_protected and placeholder_ok and numeric_ok
                ),
                "translation_backend": translation_backend(args.nllb_model, article_protected=True),
            })
    output.sort(key=lambda row: row["quiz_id"])
    count = atomic_jsonl(args.output, output)
    report = {
        "schema": "jailnews_pc2_behavior_translation/v2",
        "rows": count,
        "valid": sum(row["valid"] for row in output),
        "invalid": sum(not row["valid"] for row in output),
        "placeholder_failures": sum(not row["placeholders_preserved"] for row in output),
        "numeric_failures": sum(not row["numbers_preserved"] for row in output),
        "article_protection_failures": sum(not row["article_segment_protected"] for row in output),
        "source_languages": dict(sorted((
            language, sum(row["source_language_code"] == language for row in output)
        ) for language in by_source)),
    }
    atomic_json(args.output.with_suffix(".summary.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory")
    inventory.add_argument("--person-cohort", type=Path, default=BASE / "person_cohort.jsonl")
    inventory.add_argument("--no-person-controls", type=Path, default=BASE / "no_person_controls.jsonl")
    inventory.add_argument("--output", type=Path, default=BASE / "translation_inventory.jsonl")
    inventory.add_argument(
        "--item-types", nargs="*", choices=["identity_clue", "person_name", "concept"]
    )
    inventory.set_defaults(function=build_inventory)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--nllb-model", type=Path, required=True)
    common.add_argument("--embedding-model", type=Path, required=True)
    common.add_argument("--translation-batch-size", type=int, default=32)
    common.add_argument("--embedding-batch-size", type=int, default=128)
    common.add_argument("--translation-max-length", type=int, default=768)
    common.add_argument("--embedding-max-length", type=int, default=512)

    shard = commands.add_parser("translate-shard", parents=[common])
    shard.add_argument("--inventory", type=Path, default=BASE / "translation_inventory.jsonl")
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--shard-count", type=int, default=8)
    shard.add_argument("--similarity-threshold", type=float, default=0.90)
    shard.add_argument("--name-similarity-threshold", type=float, default=0.82)
    shard.set_defaults(function=translate_language_shard)

    behavior = commands.add_parser("behavior", parents=[common])
    behavior.add_argument("--person-cohort", type=Path, default=BASE / "person_cohort.jsonl")
    behavior.add_argument("--no-person-controls", type=Path, default=BASE / "no_person_controls.jsonl")
    behavior.add_argument("--output", type=Path, default=BASE / "behavior_translations.jsonl")
    behavior.add_argument("--behavior-similarity-threshold", type=float, default=0.92)
    behavior.set_defaults(function=translate_behaviors)
    return root


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "shard_index") and not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must satisfy 0 <= index < shard count")
    args.function(args)


if __name__ == "__main__":
    main()
