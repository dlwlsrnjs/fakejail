#!/usr/bin/env python3
"""Seal the complete 501-person transfer run into a key-free archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    sources = [
        run_dir / "plan_manifest.json",
        run_dir / "requests.jsonl",
        run_dir / "responses.jsonl",
        run_dir / "generation_manifest.json",
        run_dir / "qwen32_judgments.jsonl",
        run_dir / "per_person_results.jsonl",
        run_dir / "final_summary.json",
        ROOT / "logs/jnb_gpt501_29841.out",
        ROOT / "logs/jnb_gpt501_29841.err",
        ROOT / "logs/jnb_gpt501_judge_29842.out",
        ROOT / "logs/jnb_gpt501_judge_29842.err",
        ROOT / "scripts/run_gpt4omini_501_transfer.py",
        ROOT / "scripts/archive_gpt4omini_501_transfer.py",
        ROOT / "scripts/jailnewsbench_table2_qwen32.py",
        ROOT / "scripts/prepare_jailnews_surface_prior.py",
        ROOT / "scripts/render_jailnews_surface_matrix_v2.py",
        ROOT / "slurm/jailnews_gpt4omini_target_501.sbatch",
        ROOT / "slurm/jailnews_gpt4omini_target_501_judge.sbatch",
        ROOT / "external/jail_news_bench/evaluate.py",
        ROOT / "artifacts/jailnews_bandit_20260930/runtime/surface_prior_ensemble_100/ensemble_arm_prior.jsonl",
        ROOT / "artifacts/jailnews_bandit_20260930/runtime/surface_prior_ensemble_100/manifest.json",
        ROOT / "artifacts/jailnews_bandit_20260930/runtime/entity_links_v2_final/label_entity_map.jsonl",
    ]
    inventory = []
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_relative_to(run_dir):
            relative = Path("artifacts") / source.relative_to(run_dir)
        else:
            relative = source.relative_to(ROOT)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        inventory.append(
            {"path": relative.as_posix(), "bytes": destination.stat().st_size, "sha256": digest(destination)}
        )

    summary = json.loads((run_dir / "final_summary.json").read_text(encoding="utf-8"))
    generation = json.loads((run_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    readme = output / "README.md"
    readme.write_text(
        "# GPT-4o-mini 501-person transfer evaluation\n\n"
        "This archive contains all selected requests, all 501 raw model responses, "
        "Qwen2.5-32B judgments, compact per-person results, prior provenance, exact code, "
        "Slurm logs, and SHA-256 checksums. The API key is intentionally excluded.\n",
        encoding="utf-8",
    )
    inventory.append({"path": "README.md", "bytes": readme.stat().st_size, "sha256": digest(readme)})
    manifest = {
        "schema": "jailnews_gpt4omini_501_archive/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contains_api_key": False,
        "target_model": generation["target_model"],
        "jobs": {"generation": "29841", "qwen32_judge": "29842"},
        "persons": 501,
        "completed": generation["completed"],
        "input_tokens": generation["input_tokens"],
        "output_tokens": generation["output_tokens"],
        "final_result": summary["overall"],
        "files": sorted(inventory, key=lambda row: row["path"]),
    }
    temporary = output / f"MANIFEST.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "MANIFEST.json")
    print(json.dumps({"output": str(output), "files": len(inventory), "contains_api_key": False}))


if __name__ == "__main__":
    main()
