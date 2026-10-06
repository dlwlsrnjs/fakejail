#!/usr/bin/env python3
"""Build the clean ARTICLE -> A description -> INSTRUCTION PC2 condition.

Only A's neutral description is translated.  The source article and original
instruction remain byte-identical across the 72 language conditions after all
person names, surnames, aliases, and the person placeholder are replaced by A.
No quiz-solving or response-compliance instruction is added.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from jailnews_pc2_languages import LANGUAGES


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/jailnewsbench_pc2_quiz_v2_20260930/person_cohort.jsonl"
LOCALIZATIONS = (
    ROOT / "artifacts/jailnews_bandit_20260930/runtime/person_localizations_resolved.jsonl"
)
OUTPUT = ROOT / "data/jailnewsbench_pc2_a_middle_v3_20261001"
PLACEHOLDER = "[[PERSON_1]]"
CLUE_PREFIX = "Identify the unnamed public figure."


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


def compact(value: str) -> str:
    return " ".join(value.split())


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def comparison_form(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def localized_aliases(path: Path) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(path):
        canonical = compact(str(row.get("canonical_name") or ""))
        if not canonical:
            continue
        for value in [row.get("localized_name"), *(row.get("aliases") or [])]:
            alias = compact(str(value or ""))
            if alias:
                output[canonical].add(alias)
    return output


def name_tokens(values: Iterable[str]) -> set[str]:
    output: set[str] = set()
    for value in values:
        for token in re.findall(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", value, flags=re.UNICODE):
            # Two-letter tokens generate many false replacements (for example,
            # Li in ordinary words). Full names remain covered independently.
            if len(token) >= 3:
                output.add(token)
    return output


def aliases_for(source: dict[str, Any], localized: dict[str, set[str]]) -> list[str]:
    canonical = compact(source["canonical_person"])
    core = {
        canonical,
        compact(source.get("person_surface_local") or ""),
        compact(source.get("person_surface_english") or ""),
    }
    core.discard("")
    aliases = set(core) | localized.get(canonical, set()) | name_tokens(core)
    aliases.discard("A")
    return sorted(aliases, key=lambda value: (-len(value), value.casefold()))


def replace_person_expressions(
    text: str,
    aliases: Iterable[str],
    replacement: str = "A",
    fuzzy_tokens: Iterable[str] = (),
) -> str:
    # Some source languages attach case/genitive suffixes directly to the
    # placeholder (for example [[PERSON_1]]nak or [[PERSON_1]]s).  Keeping the
    # suffix would render "Anak"/"As", which no longer exposes A as a distinct
    # symbol. Consume a directly attached alphabetic suffix with the name span.
    result = re.sub(re.escape(PLACEHOLDER) + r"[^\W\d_]+", replacement, text)
    result = result.replace(PLACEHOLDER, replacement)
    for alias in aliases:
        if not alias or alias.casefold() == replacement.casefold():
            continue
        pattern = re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.IGNORECASE)
        result = pattern.sub(replacement, result)
    # Catch case endings attached to a name token that survived exact matching
    # (Janezovi, Orbánnak, Trumps, etc.). Keep this pass case-sensitive to avoid
    # treating an ordinary lower-case word that merely shares a name prefix as
    # a person expression.
    for alias in aliases:
        if len(alias) < 4 or re.search(r"\s", alias):
            continue
        inflected = re.compile(r"(?<!\w)" + re.escape(alias) + r"[^\W\d_]+")
        result = inflected.sub(replacement, result)
    # Handle stem-changing case endings (Janša -> Janši, Babiš -> Babiše).
    # Candidate words must be capitalized, share the first three characters,
    # and be strongly similar to a token from the target person's own name.
    roots = [token for token in fuzzy_tokens if len(token) >= 4]
    words = list(re.finditer(r"[^\W\d_]+", result, flags=re.UNICODE))
    replacements: list[tuple[int, int]] = []
    for match in words:
        candidate = match.group(0)
        if not candidate[:1].isupper() or candidate == replacement:
            continue
        folded = comparison_form(candidate)
        for root in roots:
            root_folded = comparison_form(root)
            if folded[:3] != root_folded[:3]:
                continue
            similarity = difflib.SequenceMatcher(None, folded, root_folded).ratio()
            prefix_extension = folded.startswith(root_folded) and len(folded) <= len(root_folded) + 6
            left = result[max(0, match.start() - 4):match.start()]
            right = result[match.end():min(len(result), match.end() + 4)]
            adjacent_replaced_name_token = bool(
                re.search(r"(?:^|\s)A\s+$", left) or re.match(r"^\s+A(?:\s|$)", right)
            )
            if similarity >= 0.78 or prefix_extension or (
                adjacent_replaced_name_token and similarity >= 0.68
            ):
                replacements.append(match.span())
                break
    for start, end in reversed(replacements):
        result = result[:start] + replacement + result[end:]
    result = re.sub(r"(?<!\w)A(?:\s+A)+(?!\w)", "A", result)
    return result


def contains_alias(text: str, aliases: Iterable[str]) -> list[str]:
    hits = []
    for alias in aliases:
        if not alias:
            continue
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.IGNORECASE):
            hits.append(alias)
    return hits


def prepare(args: argparse.Namespace) -> None:
    people = read_jsonl(args.source)
    localized = localized_aliases(args.localizations)
    cases = []
    inventory = []
    failures = []
    for source in people:
        quiz_id = source["quiz_id"]
        article = source["article_local_template"]
        behavior = source["behavior_local_template"]
        if behavior.count(article) != 1:
            failures.append({
                "quiz_id": quiz_id,
                "reason": "article_not_unique_in_behavior",
                "occurrences": behavior.count(article),
            })
            continue
        aliases = aliases_for(source, localized)
        core_tokens = name_tokens({
            compact(source["canonical_person"]),
            compact(source.get("person_surface_local") or ""),
            compact(source.get("person_surface_english") or ""),
        })
        instruction = behavior.replace(article, "[ARTICLE]", 1)
        clean_article = replace_person_expressions(article, aliases, fuzzy_tokens=core_tokens)
        clean_instruction = replace_person_expressions(instruction, aliases, fuzzy_tokens=core_tokens)
        description = source["identity_clue_english"].strip()
        if description.startswith(CLUE_PREFIX):
            description = description[len(CLUE_PREFIX):].lstrip()
        description = replace_person_expressions(
            description, aliases, "this public figure", fuzzy_tokens=core_tokens
        )
        description = compact(description)
        leaks = {
            "article": contains_alias(clean_article, aliases),
            "instruction": contains_alias(clean_instruction, aliases),
            "description": contains_alias(description, aliases),
        }
        if any(leaks.values()) or PLACEHOLDER in clean_article or PLACEHOLDER in clean_instruction:
            failures.append({"quiz_id": quiz_id, "reason": "person_expression_leak", "leaks": leaks})
            continue
        case = {
            "quiz_id": quiz_id,
            "sample_id": source["sample_id"],
            "source_record_id": source["source_record_id"],
            "source_language_code": source["source_language_code"],
            "motivation_category": source.get("motivation_category"),
            "canonical_person": source["canonical_person"],
            "person_role": source.get("person_role"),
            "person_countries_or_territories": source.get("person_countries_or_territories", []),
            "article": clean_article,
            "instruction": clean_instruction,
            "description_english": description,
            "article_sha256": digest(clean_article),
            "instruction_sha256": digest(clean_instruction),
            "replacement_symbol": "A",
            "article_reference_marker": "[ARTICLE]",
            "alias_count_used_for_redaction": len(aliases),
        }
        cases.append(case)
        inventory.append({
            "item_id": f"a_description:{quiz_id}",
            "item_type": "identity_clue",
            "owner_quiz_ids": [quiz_id],
            "canonical_english": description,
            "canonical_person": source["canonical_person"],
        })

    if failures:
        atomic_jsonl(args.output / "prepare_failures.jsonl", failures)
        raise RuntimeError(f"failed to prepare {len(failures)} person cases")
    cases.sort(key=lambda row: row["quiz_id"])
    inventory.sort(key=lambda row: row["item_id"])
    atomic_jsonl(args.output / "person_cases.jsonl", cases)
    atomic_jsonl(args.output / "description_translation_inventory.jsonl", inventory)
    report = {
        "schema": "jailnews_pc2_a_middle_prepare/v3",
        "person_cases": len(cases),
        "inventory_rows": len(inventory),
        "languages": len(LANGUAGES),
        "expected_translation_rows": len(inventory) * len(LANGUAGES),
        "prompt_order": ["ARTICLE", "A", "INSTRUCTION"],
        "translated_span": "A description only",
        "added_compliance_instructions": False,
    }
    atomic_json(args.output / "prepare_manifest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def load_grid(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = [row for path in paths for row in read_jsonl(path)]
    output = {}
    for row in rows:
        key = (row["item_id"], row["language"])
        if key in output:
            raise RuntimeError(f"duplicate translation row: {key}")
        output[key] = row
    return output


def merge_render(args: argparse.Namespace) -> None:
    cases = read_jsonl(args.output / "person_cases.jsonl")
    inventory = read_jsonl(args.output / "description_translation_inventory.jsonl")
    primary_paths = sorted(args.primary.glob("shard_*.jsonl"))
    fallback_paths = sorted(args.fallback.glob("shard_*.jsonl")) if args.fallback else []
    primary = load_grid(primary_paths)
    fallback = load_grid(fallback_paths) if fallback_paths else {}
    expected_keys = {
        (item["item_id"], language) for item in inventory for language in LANGUAGES
    }
    missing_primary = expected_keys - set(primary)
    if missing_primary:
        raise RuntimeError(f"primary translation grid is missing {len(missing_primary)} rows")
    if fallback and expected_keys - set(fallback):
        raise RuntimeError(f"fallback translation grid is missing {len(expected_keys - set(fallback))} rows")
    source_by_item = {item["item_id"]: item["canonical_english"] for item in inventory}
    stale_primary = [
        key for key, row in primary.items()
        if row.get("canonical_english") != source_by_item.get(key[0])
    ]
    stale_fallback = [
        key for key, row in fallback.items()
        if row.get("canonical_english") != source_by_item.get(key[0])
    ]
    if stale_primary or stale_fallback:
        raise RuntimeError(
            f"stale translation source: primary={len(stale_primary)} fallback={len(stale_fallback)}"
        )

    selected = []
    selection_counts: Counter[str] = Counter()
    for item_id, language in sorted(expected_keys):
        candidates = [primary[(item_id, language)]]
        if fallback:
            candidates.append(fallback[(item_id, language)])
        candidates.sort(
            key=lambda row: (
                bool(row.get("valid")),
                float(row.get("backtranslation_similarity") or -1.0),
            ),
            reverse=True,
        )
        winner = dict(candidates[0])
        winner["candidate_count"] = len(candidates)
        winner["selected_backend"] = winner.get("translation_backend")
        winner["alternate_candidates"] = [
            {
                "translation_backend": row.get("translation_backend"),
                "translation": row.get("translation"),
                "backtranslation": row.get("backtranslation"),
                "backtranslation_similarity": row.get("backtranslation_similarity"),
                "valid": row.get("valid"),
            }
            for row in candidates[1:]
        ]
        selection_counts[str(winner["selected_backend"])] += 1
        selected.append(winner)
    atomic_jsonl(args.output / "description_translations_merged.jsonl", selected)

    translation_index = {(row["item_id"], row["language"]): row for row in selected}
    prompt_rows = []
    for case in cases:
        item_id = f"a_description:{case['quiz_id']}"
        for language, nllb_code in LANGUAGES.items():
            translated = translation_index[(item_id, language)]
            description = translated["translation"].strip()
            prompt = (
                "ARTICLE\n"
                + case["article"]
                + "\n\nA: "
                + description
                + "\n\nINSTRUCTION\n"
                + case["instruction"]
            )
            prompt_rows.append({
                "matrix_id": f"{case['quiz_id']}::{language}",
                "quiz_id": case["quiz_id"],
                "sample_id": case["sample_id"],
                "canonical_person": case["canonical_person"],
                "person_role": case.get("person_role"),
                "person_countries_or_territories": case.get("person_countries_or_territories", []),
                "language": language,
                "nllb_code": nllb_code,
                "condition": "pc2_a_middle_description_language",
                "article": case["article"],
                "a_description": description,
                "instruction": case["instruction"],
                "prompt": prompt,
                "description_backtranslation": translated.get("backtranslation"),
                "description_backtranslation_similarity": translated.get("backtranslation_similarity"),
                "description_translation_valid": bool(translated.get("valid")),
                "description_translation_backend": translated.get("selected_backend"),
                "eligible_for_model_run": bool(translated.get("valid")),
                "article_sha256": case["article_sha256"],
                "instruction_sha256": case["instruction_sha256"],
            })
    prompt_rows.sort(key=lambda row: (row["quiz_id"], list(LANGUAGES).index(row["language"])))
    atomic_jsonl(args.output / "prompt_matrix.jsonl", prompt_rows)
    manifest = audit_rows(cases, selected, prompt_rows)
    manifest["translation_backend_selection"] = dict(selection_counts)
    manifest["translation_qc_by_language"] = {
        language: {
            "rows": sum(row["language"] == language for row in prompt_rows),
            "valid": sum(
                row["language"] == language and row["description_translation_valid"]
                for row in prompt_rows
            ),
            "invalid": sum(
                row["language"] == language and not row["description_translation_valid"]
                for row in prompt_rows
            ),
        }
        for language in LANGUAGES
    }
    atomic_json(args.output / "prompt_matrix.manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def audit_rows(
    cases: list[dict[str, Any]],
    translations: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
) -> dict[str, Any]:
    by_quiz: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prompts:
        by_quiz[row["quiz_id"]].append(row)
    failures = []
    for case in cases:
        rows = by_quiz[case["quiz_id"]]
        if len(rows) != len(LANGUAGES):
            failures.append(f"{case['quiz_id']}: language_count={len(rows)}")
            continue
        if {row["language"] for row in rows} != set(LANGUAGES):
            failures.append(f"{case['quiz_id']}: language_set_mismatch")
        if len({row["article"] for row in rows}) != 1:
            failures.append(f"{case['quiz_id']}: article_varies_by_language")
        if len({row["instruction"] for row in rows}) != 1:
            failures.append(f"{case['quiz_id']}: instruction_varies_by_language")
        for row in rows:
            expected = (
                f"ARTICLE\n{row['article']}\n\nA: {row['a_description']}"
                f"\n\nINSTRUCTION\n{row['instruction']}"
            )
            if row["prompt"] != expected:
                failures.append(f"{row['matrix_id']}: prompt_contract_mismatch")
            positions = [
                row["prompt"].find("ARTICLE\n"),
                row["prompt"].find("\n\nA: "),
                row["prompt"].find("\n\nINSTRUCTION\n"),
            ]
            if not positions[0] == 0 or not positions[0] < positions[1] < positions[2]:
                failures.append(f"{row['matrix_id']}: section_order_failure")
            if PLACEHOLDER in row["prompt"]:
                failures.append(f"{row['matrix_id']}: placeholder_leak")
    unique_ids = {row["matrix_id"] for row in prompts}
    if len(unique_ids) != len(prompts):
        failures.append("duplicate_matrix_ids")
    expected_rows = len(cases) * len(LANGUAGES)
    if len(prompts) != expected_rows:
        failures.append(f"matrix_size={len(prompts)} expected={expected_rows}")
    if failures:
        raise RuntimeError("audit failed: " + "; ".join(failures[:20]))
    return {
        "schema": "jailnews_pc2_a_middle_matrix/v3",
        "person_cases": len(cases),
        "languages": len(LANGUAGES),
        "total_rows": len(prompts),
        "expected_rows": expected_rows,
        "eligible_rows": sum(row["eligible_for_model_run"] for row in prompts),
        "ineligible_rows": sum(not row["eligible_for_model_run"] for row in prompts),
        "translation_valid": sum(bool(row.get("valid")) for row in translations),
        "translation_invalid": sum(not bool(row.get("valid")) for row in translations),
        "article_language_invariance_failures": 0,
        "instruction_language_invariance_failures": 0,
        "placeholder_leaks": 0,
        "prompt_order": ["ARTICLE", "A", "INSTRUCTION"],
        "translated_span": "A description only",
        "person_replacement": "all detected names, surnames, aliases, and placeholders -> A",
        "instruction_article_substitution": "original inline article -> [ARTICLE]",
        "added_quiz_or_compliance_instructions": False,
        "primary_comparison": "each language versus English under the same A-middle structure",
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", type=Path, default=SOURCE)
    prep.add_argument("--localizations", type=Path, default=LOCALIZATIONS)
    prep.add_argument("--output", type=Path, default=OUTPUT)
    prep.set_defaults(function=prepare)

    render = commands.add_parser("merge-render")
    render.add_argument("--output", type=Path, default=OUTPUT)
    render.add_argument("--primary", type=Path, required=True)
    render.add_argument("--fallback", type=Path)
    render.set_defaults(function=merge_render)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
