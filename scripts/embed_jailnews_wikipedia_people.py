#!/usr/bin/env python3
"""Embed cached Wikipedia person profiles with the frozen BGE encoder."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--wikipedia-cache", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    required = sorted({row["victim_person_id"] for row in read_jsonl(args.judgments)})
    cache = {row["person_id"]: row for row in read_jsonl(args.wikipedia_cache)}
    missing = [person_id for person_id in required if person_id not in cache]
    if missing:
        raise RuntimeError(f"Wikipedia profiles missing for {len(missing)} people")

    texts = []
    metadata = []
    for person_id in required:
        row = cache[person_id]
        wiki = row.get("wikipedia") or {}
        extract = str(wiki.get("extract") or "")
        text = "\n".join(
            part for part in [
                str(row.get("canonical_name") or ""),
                str(row.get("country_hint") or ""),
                extract[:12000],
            ] if part
        )
        texts.append(text)
        metadata.append(
            {
                "person_id": person_id,
                "wikidata_qid": wiki.get("wikidata_qid"),
                "wiki_status": wiki.get("status"),
                "langlink_count": int(wiki.get("langlink_count") or 0),
                "extract_characters": len(extract),
            }
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch = tokenizer(
                texts[start:start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (output * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype(np.float32))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        person_ids=np.asarray(required),
        vectors=np.concatenate(vectors),
        metadata=np.asarray([json.dumps(row, ensure_ascii=False) for row in metadata]),
    )
    manifest = {
        "schema": "jailnews_wikipedia_person_embeddings/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "people": len(required),
        "dimensions": int(vectors[0].shape[1]),
        "model": args.model,
        "pooling": "attention_mask_mean_then_l2",
        "max_length": args.max_length,
        "device": str(device),
        "uses_target_outcomes": False,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
