#!/usr/bin/env python3
"""Create a self-contained, key-free archive of the one-shot transfer run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    sources = [
        run_dir / "selection.json",
        run_dir / "gpt_generation.jsonl",
        run_dir / "qwen32_judgment.jsonl",
        run_dir / "result.json",
        run_dir / "qwen32_judgment_corrected.jsonl",
        run_dir / "result_corrected.json",
        ROOT / "logs/jnb_gpt_api1_29838.out",
        ROOT / "logs/jnb_gpt_api1_29838.err",
        ROOT / "logs/jnb_gpt_judge1_29839.out",
        ROOT / "logs/jnb_gpt_judge1_29839.err",
        ROOT / "logs/jnb_gpt_judge1_29840.out",
        ROOT / "logs/jnb_gpt_judge1_29840.err",
        ROOT / "scripts/run_gpt4omini_one_shot_transfer.py",
        ROOT / "scripts/archive_gpt4omini_one_shot.py",
        ROOT / "scripts/jailnewsbench_table2_qwen32.py",
        ROOT / "scripts/render_jailnews_surface_matrix_v2.py",
        ROOT / "slurm/jailnews_gpt4omini_api_once.sbatch",
        ROOT / "slurm/jailnews_gpt4omini_judge_once.sbatch",
        ROOT / "external/jail_news_bench/evaluate.py",
    ]

    copied: list[dict[str, object]] = []
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
        copied.append(
            {
                "path": relative.as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )

    corrected = json.loads((run_dir / "result_corrected.json").read_text(encoding="utf-8"))
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    manifest = {
        "schema": "jailnews_gpt4omini_one_shot_archive/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contains_api_key": False,
        "api_calls": corrected["calls"],
        "target_model": corrected["model"],
        "response_id": corrected["response_id"],
        "sample_id": corrected["sample_id"],
        "selected_arm": selection["selected_arm"],
        "final_result": corrected,
        "jobs": {
            "api": "29838",
            "initial_judge": "29839",
            "corrected_language_judge": "29840",
        },
        "notes": [
            "The API key and key-file contents are intentionally excluded.",
            "The initial judgment used the descriptor language as the fluency language.",
            "The corrected judgment uses the source instruction/output language (pt).",
            "No second OpenAI generation call was made for the corrected judgment.",
        ],
        "files": sorted(copied, key=lambda item: str(item["path"])),
    }
    readme = output / "README.md"
    readme.write_text(
        "# GPT-4o-mini one-shot transfer archive\n\n"
        "This directory preserves the selected prompt/arm, the complete model generation, "
        "the initial and corrected Qwen2.5-32B judgments, Slurm logs, and exact execution "
        "scripts. `MANIFEST.json` records SHA-256 hashes for every copied file.\n\n"
        "The OpenAI API key is not included. The corrected result is "
        "`artifacts/result_corrected.json`.\n",
        encoding="utf-8",
    )
    copied.append(
        {
            "path": "README.md",
            "bytes": readme.stat().st_size,
            "sha256": sha256(readme),
        }
    )
    manifest["files"] = sorted(copied, key=lambda item: str(item["path"]))
    manifest_path = output / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "files": len(copied), "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
