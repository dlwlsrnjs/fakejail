#!/usr/bin/env python3
"""Build and operate a replayable multilingual safety-evaluation bandit.

This program deliberately does not call a closed model and does not generate attack
prompts.  It prepares normalized context/arm assets, ingests authorized surrogate
judgements, recommends language/method settings probabilistically, and updates a
target-specific residual posterior from externally supplied binary outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/jailnewsbench_person_domain_20260930"
TRANSLATIONS = DATA / "concept_translations_v1"
DEFAULT_RUNTIME = ROOT / "artifacts/jailnews_bandit_20260930/runtime"
LANGUAGE_CONFIG = ROOT / "data/political_safety_collection_20260929/multilingual_pc2/pc2_languages.json"
METHODS = ["role_play", "system_override", "research_front", "neg_prompting", "context_overload"]
SURROGATES = ["meta-llama/Llama-3.1-8B-Instruct", "GraySwanAI/Llama-3-8B-Instruct-RR"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def languages() -> list[dict[str, Any]]:
    rows = json.loads(LANGUAGE_CONFIG.read_text(encoding="utf-8"))["languages"]
    if len(rows) != 72 or len({row["name"] for row in rows}) != 72:
        raise ValueError("expected exactly 72 unique PC2 languages")
    return rows


def arm_id(language: str, method: str) -> str:
    return f"{language}::{method}"


def all_arms() -> list[dict[str, Any]]:
    result = []
    for language in languages():
        for method in METHODS:
            result.append({
                "arm_id": arm_id(language["name"], method),
                "language": language["name"],
                "language_code": language["code"],
                "language_index": language["index"],
                "method": method,
            })
    return result


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def build_catalog(args: argparse.Namespace) -> None:
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {out}; pass --force")
    if out.exists():
        out.unlink()
    Path(str(out) + "-wal").unlink(missing_ok=True)
    Path(str(out) + "-shm").unlink(missing_ok=True)
    links = {row["source_record_id"]: row for row in read_jsonl(args.links)}
    person_links: dict[str, dict[str, Any]] = {}
    if args.person_links and args.person_links.exists():
        person_links = {row["dataset_label"]: row for row in read_jsonl(args.person_links)}
    con = connect(out)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=MEMORY")
    con.executescript(
        """
        CREATE TABLE contexts(
          sample_id TEXT PRIMARY KEY, source_record_id TEXT, primary_person TEXT,
          primary_entity_id TEXT, person_surfaces_json TEXT,
          article_en TEXT, source_language TEXT, region TEXT, domain TEXT, event_type TEXT,
          person_role TEXT, sensitive_json TEXT, conflicts_json TEXT, concepts_json TEXT,
          behavior_sha256 TEXT NOT NULL, metadata_json TEXT NOT NULL
        );
        CREATE TABLE arms(
          arm_id TEXT PRIMARY KEY, language TEXT, language_code TEXT,
          language_index INTEGER, method TEXT
        );
        CREATE TABLE concept_translation(
          concept_id TEXT, language TEXT, translation TEXT, canonical_english TEXT,
          valid INTEGER, roundtrip_valid INTEGER, usable_for_rendering INTEGER,
          similarity REAL, nllb_code TEXT, translation_id TEXT, review_status TEXT,
          CHECK(valid IN (0,1)), CHECK(roundtrip_valid IN (0,1)),
          CHECK(usable_for_rendering IN (0,1))
        );
        CREATE TABLE persons(
          entity_id TEXT PRIMARY KEY, wikidata_qid TEXT, canonical_person TEXT,
          dataset_labels_json TEXT NOT NULL, profile_json TEXT NOT NULL
        );
        CREATE INDEX context_source_record_idx ON contexts(source_record_id);
        """
    )
    con.executemany(
        "INSERT INTO arms VALUES(:arm_id,:language,:language_code,:language_index,:method)", all_arms()
    )
    context_count = 0
    for row in read_jsonl(args.contexts):
        record_id = row["source_record_id"]
        link = links.get(record_id, {})
        concept_refs = link.get("concepts", [])
        safe_meta = {
            "election_related": row.get("luna_election_related"),
            "war_or_security_related": row.get("luna_war_or_security_related"),
            "countries_or_territories": [
                country
                for person in row.get("luna_people", [])
                for country in person.get("countries_or_territories", [])
            ],
            "political_subdomains": row.get("luna_political_subdomains", []),
        }
        people = row.get("luna_people", [])
        primary_person = row.get("luna_primary_person")
        if not primary_person and people:
            primary_person = people[0].get("canonical_person")
        mapped = person_links.get(primary_person or "", {})
        if not mapped:
            surface_matches = [person_links.get(person.get("surface", "")) for person in people]
            surface_matches = [item for item in surface_matches if item]
            if len(surface_matches) == 1:
                mapped = surface_matches[0]
        con.execute(
            "INSERT INTO contexts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                row["sample_id"], record_id, primary_person, mapped.get("entity_id"),
                json.dumps([person.get("surface") for person in people if person.get("surface")], ensure_ascii=False),
                row.get("article_en"),
                row.get("language_code"), row.get("region_en"), row.get("luna_political_domain"),
                row.get("luna_event_type"), row.get("luna_primary_person_role"),
                json.dumps(row.get("luna_sensitive_concepts", []), ensure_ascii=False),
                json.dumps(row.get("luna_conflicts_named", []), ensure_ascii=False),
                json.dumps(concept_refs, ensure_ascii=False), row["prompt_sha256"],
                json.dumps(safe_meta, ensure_ascii=False),
            ),
        )
        context_count += 1
    profiles_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.people):
        label = row["canonical_person"]
        mapped = person_links.get(label, {})
        entity_id = mapped.get("entity_id") or f"legacy-label:{label}"
        profiles_by_entity[entity_id].append(row)
    entity_catalog = (
        {row["entity_id"]: row for row in read_jsonl(args.entities)}
        if args.entities and args.entities.exists() else {}
    )
    person_count = 0
    for entity_id, profiles in sorted(profiles_by_entity.items()):
        entity = entity_catalog.get(entity_id, {})
        labels = sorted({row["canonical_person"] for row in profiles})
        canonical = entity.get("canonical_label") or labels[0]
        country_counts: dict[str, int] = defaultdict(int)
        role_counts: dict[str, int] = defaultdict(int)
        for profile in profiles:
            for key, value in profile.get("country_or_territory_counts", {}).items():
                country_counts[key] += int(value)
            for key, value in profile.get("role_counts", {}).items():
                role_counts[key] += int(value)
        merged_profile = {
            "entity": entity, "label_profiles": profiles,
            "country_or_territory_counts": dict(country_counts),
            "role_counts": dict(role_counts),
        }
        con.execute(
            "INSERT INTO persons VALUES(?,?,?,?,?)",
            (
                entity_id, entity.get("wikidata_qid"), canonical,
                json.dumps(labels, ensure_ascii=False),
                json.dumps(merged_profile, ensure_ascii=False),
            ),
        )
        person_count += 1
    translation_count = 0
    batch = []
    for row in read_jsonl(args.translations):
        batch.append((
            row["concept_id"], row["language"], row["translation"], row["canonical_english"],
            int(bool(row.get("usable_for_rendering", row.get("valid", False)))),
            int(bool(row.get("roundtrip_valid", row.get("valid", False)))),
            int(bool(row.get("usable_for_rendering", False))),
            float(row["backtranslation_similarity"]), row["nllb_code"],
            row["translation_id"], row.get("final_status", "provisional_auto" if row["valid"] else "needs_review"),
        ))
        if len(batch) >= 5000:
            con.executemany("INSERT INTO concept_translation VALUES(?,?,?,?,?,?,?,?,?,?,?)", batch)
            translation_count += len(batch)
            batch.clear()
    if batch:
        con.executemany("INSERT INTO concept_translation VALUES(?,?,?,?,?,?,?,?,?,?,?)", batch)
        translation_count += len(batch)
    con.execute("CREATE UNIQUE INDEX concept_key_idx ON concept_translation(concept_id, language)")
    con.execute("CREATE INDEX concept_language_idx ON concept_translation(language, valid)")
    con.commit()
    manifest = {
        "schema": "jailnews_bandit_catalog/v2", "created_at": now(),
        "catalog": str(out), "contexts": context_count, "people": person_count,
        "languages": 72, "methods": METHODS, "arms": 360,
        "concept_translations": translation_count,
        "execution_boundary": "No attack text or closed-model caller is stored in this catalog.",
    }
    dump_json(out.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_person_queue(args: argparse.Namespace) -> None:
    people = list(read_jsonl(args.people))
    langs = languages()

    def rows() -> Iterable[dict[str, Any]]:
        for person in people:
            name = person["canonical_person"]
            pid = "person:" + hashlib.sha256(name.casefold().encode()).hexdigest()[:16]
            for lang in langs:
                yield {
                    "person_id": pid, "canonical_name": name, "wikidata_qid": None,
                    "target_language": lang["name"], "target_language_code": lang["code"],
                    "localized_name": name if lang["name"] == "English" else None,
                    "aliases": [],
                    "rendering_method": "original_fallback" if lang["name"] == "English" else None,
                    "source_title": None, "source_url": None, "source_accessed_at": None,
                    "identity_match": "verified" if lang["name"] == "English" else "unchecked",
                    "status": "fallback_original" if lang["name"] == "English" else "unresolved",
                    "confidence": "high" if lang["name"] == "English" else "low",
                    "profile_evidence": {
                        "countries": person.get("country_or_territory_counts", {}),
                        "roles": person.get("role_counts", {}),
                    },
                    "reviewer": None, "reviewed_at": None, "notes": "",
                }

    count = write_jsonl(args.output, rows())
    summary = {
        "schema": "person_localization_queue/v1", "people": len(people), "languages": len(langs),
        "rows": count, "verified_english_rows": len(people),
        "unresolved_rows": count - len(people),
        "rule": "Use verified Wikipedia/Wikidata langlinks or official sources; never machine-translate names blindly.",
    }
    dump_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_offline_plan(args: argparse.Namespace) -> None:
    con = connect(args.catalog)
    sample_ids = [row[0] for row in con.execute("SELECT sample_id FROM contexts ORDER BY sample_id")]
    shards = []
    for start in range(0, len(sample_ids), args.contexts_per_shard):
        subset = sample_ids[start:start + args.contexts_per_shard]
        shards.append({
            "shard_id": f"context_{start:05d}_{start + len(subset) - 1:05d}",
            "context_start": start, "context_stop_exclusive": start + len(subset),
            "sample_ids": subset, "arms_per_context": 360,
            "responses_per_arm_per_model": 5, "surrogate_models": SURROGATES,
            "expected_arm_rows": len(subset) * 360,
            "expected_generations": len(subset) * 360 * 10,
        })
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "shards.jsonl", shards)
    manifest = {
        "schema": "jailnews_offline_surrogate_plan/v1", "created_at": now(),
        "contexts": len(sample_ids), "languages": 72, "methods": 5,
        "arms_per_context": 360, "surrogate_models": SURROGATES,
        "responses_per_arm_per_model": 5,
        "arm_rows": len(sample_ids) * 360,
        "surrogate_generations": len(sample_ids) * 360 * 10,
        "shards": len(shards), "shard_catalog": "shards.jsonl",
        "required_score_fields": [
            "sample_id", "language", "method", "model_id", "successes", "trials"
        ],
        "forbidden_retention": ["raw harmful response text"],
        "allowed_retention": ["response hash", "binary judgement", "judge version", "latency", "token counts", "seed"],
    }
    dump_json(args.output / "manifest.json", manifest)
    dump_json(args.output / "score_row.example.json", {
        "sample_id": "jailnewsbench:test:...", "language": "Korean",
        "method": "role_play", "model_id": SURROGATES[0], "successes": 0, "trials": 5,
        "judge_id": "authorized-safety-judge/version", "response_hashes": [],
    })
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def initialize_prior_db(path: Path, force: bool) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    if path.exists():
        path.unlink()
    Path(str(path) + "-wal").unlink(missing_ok=True)
    Path(str(path) + "-shm").unlink(missing_ok=True)
    con = connect(path)
    con.executescript(
        """
        CREATE TABLE model_score(
          sample_id TEXT, arm_id TEXT, model_id TEXT, successes INTEGER, trials INTEGER,
          PRIMARY KEY(sample_id, arm_id, model_id)
        );
        CREATE TABLE arm_prior(
          sample_id TEXT, arm_id TEXT, p_mean REAL, alpha REAL, beta REAL,
          strength REAL, disagreement REAL, models INTEGER, trials INTEGER,
          PRIMARY KEY(sample_id, arm_id)
        );
        CREATE TABLE global_prior(
          arm_id TEXT PRIMARY KEY, p_mean REAL, alpha REAL, beta REAL, trials INTEGER
        );
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value_json TEXT);
        """
    )
    return con


def ingest_surrogate(args: argparse.Namespace) -> None:
    con = initialize_prior_db(args.output, args.force)
    rows_seen = 0
    for row in read_jsonl(args.scores):
        language, method = row["language"], row["method"]
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}")
        successes = int(row.get("successes", int(bool(row.get("success", False)))))
        trials = int(row.get("trials", 1))
        if trials <= 0 or successes < 0 or successes > trials:
            raise ValueError(f"invalid counts: {row}")
        con.execute(
            """INSERT INTO model_score VALUES(?,?,?,?,?)
               ON CONFLICT(sample_id,arm_id,model_id) DO UPDATE SET
               successes=successes+excluded.successes, trials=trials+excluded.trials""",
            (row["sample_id"], arm_id(language, method), row["model_id"], successes, trials),
        )
        rows_seen += 1
    con.commit()
    cursor = con.execute(
        "SELECT sample_id,arm_id,model_id,successes,trials FROM model_score ORDER BY sample_id,arm_id,model_id"
    )
    current = None
    group: list[sqlite3.Row] = []

    def emit(items: list[sqlite3.Row]) -> None:
        if not items:
            return
        total_s = sum(item["successes"] for item in items)
        total_n = sum(item["trials"] for item in items)
        rates = [(item["successes"] + 0.5) / (item["trials"] + 1.0) for item in items]
        disagreement = max(rates) - min(rates) if len(rates) > 1 else 1.0
        strength = total_n * max(0.0, 1.0 - disagreement)
        raw_p = (total_s + 0.5) / (total_n + 1.0)
        alpha = 0.5 + raw_p * strength
        beta = 0.5 + (1.0 - raw_p) * strength
        con.execute(
            "INSERT INTO arm_prior VALUES(?,?,?,?,?,?,?,?,?)",
            (items[0]["sample_id"], items[0]["arm_id"], raw_p, alpha, beta,
             strength, disagreement, len(items), total_n),
        )

    for row in cursor:
        key = (row["sample_id"], row["arm_id"])
        if current is not None and key != current:
            emit(group)
            group = []
        current = key
        group.append(row)
    emit(group)
    for row in con.execute(
        "SELECT arm_id,SUM(successes) s,SUM(trials) n FROM model_score GROUP BY arm_id"
    ):
        p = (row["s"] + 0.5) / (row["n"] + 1.0)
        con.execute("INSERT INTO global_prior VALUES(?,?,?,?,?)", (row["arm_id"], p, row["s"] + 0.5, row["n"] - row["s"] + 0.5, row["n"]))
    metadata = {
        "schema": "jailnews_surrogate_prior/v1", "created_at": now(),
        "source": str(args.scores), "input_rows": rows_seen,
        "arm_priors": con.execute("SELECT COUNT(*) FROM arm_prior").fetchone()[0],
        "global_priors": con.execute("SELECT COUNT(*) FROM global_prior").fetchone()[0],
        "prior": "Jeffreys beta-binomial; strength reduced by model disagreement",
    }
    for key, value in metadata.items():
        con.execute("INSERT INTO metadata VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False)))
    con.commit()
    dump_json(args.output.with_suffix(".manifest.json"), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def neutral_prior() -> dict[str, float]:
    return {"p_mean": 0.5, "alpha": 0.5, "beta": 0.5, "strength": 0.0, "disagreement": 1.0}


def priors_for_sample(prior_db: Path | None, sample_id: str, arms: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {arm["arm_id"]: neutral_prior() for arm in arms}
    if prior_db is None or not prior_db.exists():
        return result
    con = connect(prior_db)
    global_rows = {row["arm_id"]: dict(row) for row in con.execute("SELECT * FROM global_prior")}
    sample_rows = {row["arm_id"]: dict(row) for row in con.execute("SELECT * FROM arm_prior WHERE sample_id=?", (sample_id,))}
    for arm in arms:
        row = sample_rows.get(arm["arm_id"]) or global_rows.get(arm["arm_id"])
        if row:
            result[arm["arm_id"]] = row
    return result


def priors_for_state(state: dict[str, Any], prior_db: Path | None, arms: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    inline = state.get("inline_priors")
    if inline:
        result = {arm["arm_id"]: neutral_prior() for arm in arms}
        result.update(inline)
        return result
    return priors_for_sample(prior_db, state["sample_id"], arms)


def fresh_residual() -> dict[str, Any]:
    return {"global": {"mean": 0.0, "precision": 4.0}, "language": {}, "method": {}, "pair": {}}


def init_state(args: argparse.Namespace) -> None:
    con = connect(args.catalog)
    if not con.execute("SELECT 1 FROM contexts WHERE sample_id=?", (args.sample_id,)).fetchone():
        raise KeyError(f"unknown sample_id: {args.sample_id}")
    state = {
        "schema": "jailnews_target_bandit_state/v1", "sample_id": args.sample_id,
        "created_at": now(), "updated_at": now(), "prior_db": str(args.prior) if args.prior else None,
        "residual": fresh_residual(), "observations": [], "success_found": False,
        "note": "Target residual posterior over a frozen surrogate prior.",
    }
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --force")
    dump_json(args.output, state)
    print(json.dumps({"state": str(args.output), "sample_id": args.sample_id}, ensure_ascii=False))


def get_term(state: dict[str, Any], family: str, key: str | None = None) -> tuple[float, float]:
    if family == "global":
        item = state["residual"]["global"]
        return float(item["mean"]), 1.0 / float(item["precision"])
    defaults = {"language": 2.0, "method": 2.0, "pair": 1.0}
    item = state["residual"][family].get(key)
    if item is None:
        return 0.0, 1.0 / defaults[family]
    return float(item["mean"]), 1.0 / float(item["precision"])


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def score_arms(state: dict[str, Any], priors: dict[str, dict[str, float]], arms: list[dict[str, Any]], seed: int, draws: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    means, variances = [], []
    for arm in arms:
        p = float(priors[arm["arm_id"]]["p_mean"])
        components = [get_term(state, "global"), get_term(state, "language", arm["language"]),
                      get_term(state, "method", arm["method"]), get_term(state, "pair", arm["arm_id"])]
        means.append(logit(p) + sum(item[0] for item in components))
        variances.append(sum(item[1] for item in components))
    means_a, vars_a = np.asarray(means), np.asarray(variances)
    posterior_p = sigmoid(means_a / np.sqrt(1.0 + np.pi * vars_a / 8.0))
    winners = np.zeros(len(arms), dtype=int)
    for _ in range(draws):
        sampled = rng.normal(means_a, np.sqrt(vars_a))
        winners[int(np.argmax(sampled))] += 1
    tried = {item["arm_id"] for item in state.get("observations", [])}
    rows = []
    for index, arm in enumerate(arms):
        prior = priors[arm["arm_id"]]
        rows.append({
            **arm, "posterior_success_probability": round(float(posterior_p[index]), 8),
            "posterior_logit_mean": round(float(means_a[index]), 8),
            "posterior_logit_sd": round(float(np.sqrt(vars_a[index])), 8),
            "thompson_selection_probability": round(float(winners[index] / draws), 8),
            "surrogate_prior_probability": round(float(prior["p_mean"]), 8),
            "surrogate_prior_strength": round(float(prior.get("strength", prior.get("trials", 0))), 8),
            "surrogate_disagreement": round(float(prior.get("disagreement", 0)), 8),
            "already_tried": arm["arm_id"] in tried,
        })
    return rows


def recommend(args: argparse.Namespace) -> None:
    state = json.loads(args.state.read_text(encoding="utf-8"))
    con = connect(args.catalog)
    arms = [dict(row) for row in con.execute("SELECT * FROM arms ORDER BY language_index,method")]
    prior_path = args.prior or (Path(state["prior_db"]) if state.get("prior_db") else None)
    priors = priors_for_state(state, prior_path, arms)
    rows = score_arms(state, priors, arms, args.seed, args.draws)
    candidates = rows if args.allow_repeat else [row for row in rows if not row["already_tried"]]
    candidates.sort(key=lambda row: (row["thompson_selection_probability"], row["posterior_success_probability"]), reverse=True)
    result = {
        "schema": "jailnews_bandit_recommendation/v1", "sample_id": state["sample_id"],
        "observations": len(state.get("observations", [])), "seed": args.seed, "draws": args.draws,
        "top_k": candidates[:args.top_k],
        "probability_semantics": {
            "posterior_success_probability": "calibrated Bernoulli affinity estimate per arm",
            "thompson_selection_probability": "probability that the arm is best under posterior draws",
        },
    }
    if args.output:
        dump_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def posterior_mean_for_arm(state: dict[str, Any], prior: dict[str, float], language: str, method: str) -> float:
    aid = arm_id(language, method)
    value = logit(float(prior["p_mean"]))
    value += get_term(state, "global")[0]
    value += get_term(state, "language", language)[0]
    value += get_term(state, "method", method)[0]
    value += get_term(state, "pair", aid)[0]
    return float(sigmoid(value))


def update_term(state: dict[str, Any], family: str, key: str | None, gradient: float, curvature: float) -> None:
    defaults = {"global": 4.0, "language": 2.0, "method": 2.0, "pair": 1.0}
    if family == "global":
        item = state["residual"]["global"]
    else:
        item = state["residual"][family].setdefault(key, {"mean": 0.0, "precision": defaults[family]})
    new_precision = float(item["precision"]) + curvature
    item["mean"] = float(item["mean"]) + gradient / new_precision
    item["precision"] = new_precision


def update(args: argparse.Namespace) -> None:
    state = json.loads(args.state.read_text(encoding="utf-8"))
    arms = all_arms()
    known = {row["arm_id"] for row in arms}
    aid = arm_id(args.language, args.method)
    if aid not in known:
        raise ValueError(f"unknown arm: {aid}")
    prior_path = args.prior or (Path(state["prior_db"]) if state.get("prior_db") else None)
    prior = priors_for_state(state, prior_path, [{"arm_id": aid}])[aid]
    p_before = posterior_mean_for_arm(state, prior, args.language, args.method)
    outcome = int(args.outcome)
    gradient = outcome - p_before
    curvature = max(1e-4, p_before * (1.0 - p_before))
    update_term(state, "global", None, gradient, curvature)
    update_term(state, "language", args.language, gradient, curvature)
    update_term(state, "method", args.method, gradient, curvature)
    update_term(state, "pair", aid, gradient, curvature)
    observation = {
        "iteration": len(state["observations"]) + 1, "observed_at": now(), "arm_id": aid,
        "language": args.language, "method": args.method, "outcome": outcome,
        "p_before": round(p_before, 8), "source": args.source,
        "external_record_id": args.external_record_id,
    }
    state["observations"].append(observation)
    state["success_found"] = bool(state.get("success_found") or outcome == 1)
    state["updated_at"] = now()
    destination = args.output or args.state
    temp = destination.with_suffix(destination.suffix + ".tmp")
    dump_json(temp, state)
    os.replace(temp, destination)
    print(json.dumps(observation, ensure_ascii=False, indent=2))


def inspect_context(args: argparse.Namespace) -> None:
    con = connect(args.catalog)
    context = con.execute("SELECT * FROM contexts WHERE sample_id=?", (args.sample_id,)).fetchone()
    if not context:
        raise KeyError(args.sample_id)
    row = dict(context)
    concepts = json.loads(row.pop("concepts_json"))
    translated = []
    for ref in concepts:
        item = con.execute(
            "SELECT * FROM concept_translation WHERE concept_id=? AND language=?",
            (ref["concept_id"], args.language),
        ).fetchone()
        if item:
            item = dict(item)
            item["usable"] = bool(
                item["usable_for_rendering"]
                or (args.allow_provisional and item["roundtrip_valid"])
            )
            translated.append(item)
    result = {
        "schema": "jailnews_setting_materialization/v1", "sample_id": args.sample_id,
        "language": args.language, "method": args.method,
        "person": row["primary_person"], "concepts": translated,
        "context_metadata": {
            "domain": row["domain"], "event_type": row["event_type"], "person_role": row["person_role"],
            "region": row["region"], "source_language": row["source_language"],
        },
        "behavior_reference": {"sha256": row["behavior_sha256"]},
        "ready_for_render": bool(translated) and all(item["usable"] for item in translated),
        "rendering_boundary": "This spec contains neutral metadata only; an authorized harness owns prompt rendering and target calls.",
    }
    if args.output:
        dump_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def local_snapshot(cache_name: str) -> Path:
    candidates = sorted((ROOT / "hf-cache/hub" / cache_name / "snapshots").glob("*"))
    if not candidates:
        raise FileNotFoundError(f"missing local model snapshot: {cache_name}")
    return candidates[-1]


def embed_texts(model_path: Path, texts: list[str], batch_size: int = 64) -> np.ndarray:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype).to(device).eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start:start + batch_size], return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors.append(functional.normalize(pooled.float(), dim=1).cpu().numpy())
    return np.concatenate(vectors, axis=0)


def build_embedding_index(args: argparse.Namespace) -> None:
    con = connect(args.catalog)
    rows = list(con.execute("SELECT sample_id,article_en FROM contexts ORDER BY sample_id"))
    vectors = embed_texts(args.model, [row["article_en"] for row in rows], args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, sample_ids=np.asarray([row["sample_id"] for row in rows]), vectors=vectors)
    manifest = {
        "schema": "jailnews_context_embedding_index/v1", "rows": len(rows),
        "dimensions": int(vectors.shape[1]), "embedding_model": str(args.model),
        "pooling": "attention_mask_mean_then_l2", "source": str(args.catalog),
    }
    dump_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def transfer_neighbor_priors(prior_db: Path | None, neighbors: list[tuple[str, float]], arms: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if prior_db is None or not prior_db.exists():
        return {arm["arm_id"]: neutral_prior() for arm in arms}
    con = connect(prior_db)
    result: dict[str, dict[str, float]] = {}
    similarities = np.asarray([score for _, score in neighbors], dtype=float)
    weights = np.exp((similarities - similarities.max()) / 0.07)
    weights /= weights.sum()
    for arm in arms:
        values, used = [], []
        for (sample_id, _), weight in zip(neighbors, weights):
            row = con.execute("SELECT * FROM arm_prior WHERE sample_id=? AND arm_id=?", (sample_id, arm["arm_id"])).fetchone()
            if row:
                values.append(dict(row)); used.append(float(weight))
        if values:
            used_a = np.asarray(used); used_a /= used_a.sum()
            p = float(sum(w * float(row["p_mean"]) for w, row in zip(used_a, values)))
            strength = float(sum(w * float(row["strength"]) for w, row in zip(used_a, values)))
            disagreement = float(sum(w * float(row["disagreement"]) for w, row in zip(used_a, values)))
            result[arm["arm_id"]] = {
                "p_mean": p, "alpha": 0.5 + p * strength, "beta": 0.5 + (1 - p) * strength,
                "strength": strength, "disagreement": disagreement,
            }
        else:
            global_row = con.execute("SELECT * FROM global_prior WHERE arm_id=?", (arm["arm_id"],)).fetchone()
            result[arm["arm_id"]] = dict(global_row) if global_row else neutral_prior()
    return result


def route_text(args: argparse.Namespace) -> None:
    text_value = args.text if args.text is not None else args.input.read_text(encoding="utf-8")
    digest = hashlib.sha256(text_value.encode()).hexdigest()
    index = np.load(args.index)
    query = embed_texts(args.model, [text_value], 1)[0]
    similarities = index["vectors"] @ query
    k = min(args.neighbors, len(similarities))
    selected = np.argpartition(-similarities, k - 1)[:k]
    selected = selected[np.argsort(-similarities[selected])]
    neighbors = [(str(index["sample_ids"][i]), float(similarities[i])) for i in selected]
    con = connect(args.catalog)
    lowered = text_value.casefold()
    people = []
    for row in con.execute("SELECT entity_id,canonical_person,dataset_labels_json,profile_json FROM persons"):
        labels = json.loads(row["dataset_labels_json"])
        hits = [label for label in labels if len(label) >= 4 and label.casefold() in lowered]
        if hits:
            profile = json.loads(row["profile_json"])
            people.append({
                "entity_id": row["entity_id"], "canonical_person": row["canonical_person"],
                "matched_dataset_labels": hits,
                "countries": profile.get("country_or_territory_counts", {}),
                "roles": profile.get("role_counts", {}),
            })
    concepts = []
    seen = set()
    for row in con.execute("SELECT DISTINCT concept_id,canonical_english FROM concept_translation"):
        phrase = row["canonical_english"].casefold()
        if len(phrase) >= 4 and phrase in lowered and row["concept_id"] not in seen:
            seen.add(row["concept_id"]); concepts.append(dict(row))
    arms = [dict(row) for row in con.execute("SELECT * FROM arms ORDER BY language_index,method")]
    inline = transfer_neighbor_priors(args.prior, neighbors, arms)
    state = {
        "schema": "jailnews_target_bandit_state/v1", "sample_id": f"online:{digest[:20]}",
        "created_at": now(), "updated_at": now(), "prior_db": str(args.prior) if args.prior else None,
        "inline_priors": inline, "residual": fresh_residual(), "observations": [],
        "success_found": False,
        "routing": {
            "input_sha256": digest, "people": people, "concepts": concepts,
            "nearest_contexts": [{"sample_id": sid, "cosine_similarity": round(score, 8)} for sid, score in neighbors],
            "web_feature_policy": "Known person profiles are metadata only; web evidence must be calibrated on surrogate data before changing arm logits.",
        },
    }
    dump_json(args.state_output, state)
    scored = score_arms(state, inline, arms, args.seed, args.draws)
    scored.sort(key=lambda row: (row["thompson_selection_probability"], row["posterior_success_probability"]), reverse=True)
    result = {
        "schema": "jailnews_online_route/v1", "state": str(args.state_output),
        "sample_id": state["sample_id"], "people": people, "concepts": concepts,
        "nearest_contexts": state["routing"]["nearest_contexts"], "top_k": scored[:args.top_k],
    }
    if args.output:
        dump_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build-catalog")
    p.add_argument("--contexts", type=Path, default=DATA / "luna_review_v1/analysis_ready_samples.jsonl")
    p.add_argument("--links", type=Path, default=TRANSLATIONS / "sample_concept_links.jsonl")
    p.add_argument("--translations", type=Path, default=TRANSLATIONS / "translations_nllb.jsonl")
    p.add_argument("--people", type=Path, default=DATA / "luna_review_v1/verified_person_profiles.jsonl")
    p.add_argument("--person-links", type=Path, default=DEFAULT_RUNTIME / "entity_links_v2/label_entity_map.jsonl")
    p.add_argument("--entities", type=Path, default=DEFAULT_RUNTIME / "entity_links_v2/entities_deduplicated_by_qid.jsonl")
    p.add_argument("--output", type=Path, default=DEFAULT_RUNTIME / "catalog.sqlite")
    p.add_argument("--force", action="store_true"); p.set_defaults(func=build_catalog)
    p = sub.add_parser("build-person-queue")
    p.add_argument("--people", type=Path, default=DATA / "luna_review_v1/verified_person_profiles.jsonl")
    p.add_argument("--output", type=Path, default=DEFAULT_RUNTIME / "person_localization_queue.jsonl")
    p.set_defaults(func=build_person_queue)
    p = sub.add_parser("build-offline-plan")
    p.add_argument("--catalog", type=Path, default=DEFAULT_RUNTIME / "catalog.sqlite")
    p.add_argument("--output", type=Path, default=DEFAULT_RUNTIME / "offline_plan")
    p.add_argument("--contexts-per-shard", type=int, default=25); p.set_defaults(func=build_offline_plan)
    p = sub.add_parser("ingest-surrogate")
    p.add_argument("--scores", type=Path, required=True); p.add_argument("--output", type=Path, default=DEFAULT_RUNTIME / "prior.sqlite")
    p.add_argument("--force", action="store_true"); p.set_defaults(func=ingest_surrogate)
    p = sub.add_parser("init-state")
    p.add_argument("--catalog", type=Path, default=DEFAULT_RUNTIME / "catalog.sqlite")
    p.add_argument("--sample-id", required=True); p.add_argument("--prior", type=Path)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--force", action="store_true"); p.set_defaults(func=init_state)
    p = sub.add_parser("recommend")
    p.add_argument("--catalog", type=Path, default=DEFAULT_RUNTIME / "catalog.sqlite")
    p.add_argument("--state", type=Path, required=True); p.add_argument("--prior", type=Path)
    p.add_argument("--top-k", type=int, default=10); p.add_argument("--draws", type=int, default=4000)
    p.add_argument("--seed", type=int, default=20260930); p.add_argument("--allow-repeat", action="store_true")
    p.add_argument("--output", type=Path); p.set_defaults(func=recommend)
    p = sub.add_parser("update")
    p.add_argument("--state", type=Path, required=True); p.add_argument("--prior", type=Path)
    p.add_argument("--language", required=True); p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--outcome", type=int, choices=[0, 1], required=True)
    p.add_argument("--source", default="authorized_external_evaluation")
    p.add_argument("--external-record-id"); p.add_argument("--output", type=Path); p.set_defaults(func=update)
    p = sub.add_parser("materialize-setting")
    p.add_argument("--catalog", type=Path, default=DEFAULT_RUNTIME / "catalog.sqlite")
    p.add_argument("--sample-id", required=True); p.add_argument("--language", required=True)
    p.add_argument("--method", choices=METHODS, required=True); p.add_argument("--output", type=Path)
    p.add_argument("--allow-provisional", action="store_true", help="Allow NLLB-only rows that have not passed Luna/human review")
    p.set_defaults(func=inspect_context)
    p = sub.add_parser("build-embedding-index")
    p.add_argument("--catalog", type=Path, default=DEFAULT_RUNTIME / "catalog.sqlite")
    p.add_argument("--model", type=Path, default=local_snapshot("models--BAAI--bge-large-en-v1.5"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=DEFAULT_RUNTIME / "context_embeddings.npz")
    p.set_defaults(func=build_embedding_index)
    p = sub.add_parser("route-text")
    source = p.add_mutually_exclusive_group(required=True); source.add_argument("--text"); source.add_argument("--input", type=Path)
    p.add_argument("--catalog", type=Path, default=DEFAULT_RUNTIME / "catalog.sqlite")
    p.add_argument("--index", type=Path, default=DEFAULT_RUNTIME / "context_embeddings.npz")
    p.add_argument("--prior", type=Path); p.add_argument("--model", type=Path, default=local_snapshot("models--BAAI--bge-large-en-v1.5"))
    p.add_argument("--neighbors", type=int, default=20); p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--draws", type=int, default=4000); p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--state-output", type=Path, required=True); p.add_argument("--output", type=Path)
    p.set_defaults(func=route_text)
    return ap


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
