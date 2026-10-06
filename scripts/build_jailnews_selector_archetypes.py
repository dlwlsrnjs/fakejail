#!/usr/bin/env python3
"""Build label-conditioned embedding archetypes from Luna-reviewed samples.

No ASR labels or model outcomes are read. The resulting centroids are used to
route a new prompt among domain/role/event/sensitivity experts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


@torch.inference_mode()
def embed_batch(tokenizer: Any, model: Any, texts: list[str], max_length: int) -> torch.Tensor:
    encoded = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    output = model(**encoded).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1)
    pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
    return F.normalize(pooled.float(), dim=1).cpu()


def labels_for(row: dict[str, Any]) -> list[tuple[str, str]]:
    labels = [
        ("domain", row["luna_political_domain"]),
        ("role", row["luna_primary_person_role"]),
        ("event", row["luna_event_type"]),
    ]
    labels.extend(("sensitivity", value) for value in row["luna_sensitive_concepts"])
    labels.extend(("conflict", value) for value in row["luna_conflicts_named"])
    labels.append(("flag", "election_related" if row["luna_election_related"] else "not_election_related"))
    labels.append(("flag", "war_or_security_related" if row["luna_war_or_security_related"] else "not_war_or_security_related"))
    labels.append((
        "composite_domain_role",
        f"{row['luna_political_domain']}||{row['luna_primary_person_role']}",
    ))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/jailnewsbench_person_domain_20260930/luna_review_v1/analysis_ready_samples.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/jailnewsbench_person_domain_20260930/selector_archetypes_v2"),
    )
    parser.add_argument(
        "--embedding-model", type=Path,
        default=Path("hf-cache/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--min-composite-count", type=int, default=8)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, local_files_only=True)
    model = AutoModel.from_pretrained(args.embedding_model, local_files_only=True).eval()

    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        counts.update(labels_for(row))
    sums: dict[tuple[str, str], np.ndarray] = {}
    global_sum: np.ndarray | None = None

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        vectors = embed_batch(tokenizer, model, [str(row["article_en"]) for row in batch], args.max_length).numpy()
        for row, vector in zip(batch, vectors, strict=True):
            global_sum = vector.copy() if global_sum is None else global_sum + vector
            for key in labels_for(row):
                if key[0] == "composite_domain_role" and counts[key] < args.min_composite_count:
                    continue
                sums[key] = vector.copy() if key not in sums else sums[key] + vector
        done = min(len(rows), start + len(batch))
        if done % 320 == 0 or done == len(rows):
            print(f"embedded={done}/{len(rows)}", flush=True)

    keys = sorted(sums)
    vectors = []
    catalog = []
    for index, key in enumerate(keys):
        vector = sums[key] / counts[key]
        vector = vector / max(np.linalg.norm(vector), 1e-12)
        vectors.append(vector.astype(np.float32))
        catalog.append({
            "index": index,
            "group_type": key[0],
            "group_value": key[1],
            "sample_count": counts[key],
        })
    global_vector = global_sum / len(rows)
    global_vector = global_vector / max(np.linalg.norm(global_vector), 1e-12)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "archetypes.npz",
        vectors=np.stack(vectors),
        global_vector=global_vector.astype(np.float32),
    )
    (args.output_dir / "group_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema": "jailnews_zero_shot_selector_archetypes/v2",
        "source_rows": len(rows),
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "embedding_pooling": "attention_mask_mean_pooling_then_l2_normalize",
        "uses_asr_labels": False,
        "group_count": len(catalog),
        "group_type_counts": dict(Counter(row["group_type"] for row in catalog)),
        "min_composite_count": args.min_composite_count,
        "purpose": "route new prompts among domain, role, event, sensitivity, conflict, and composite experts",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
