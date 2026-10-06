#!/usr/bin/env python3
"""Repair JailNews person links, deduplicate by Wikidata QID, and localize names.

The script never merges records by display name.  Dataset labels are retained in
``label_entity_map.jsonl`` and point to a QID-backed entity when one can be
resolved with high confidence.  Uncertain matches remain explicit and are not
silently attached to the best search hit.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "artifacts/jailnews_bandit_20260930/runtime"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "POLY-safety-evaluation/2.0 (JailNews entity-link audit)"
WIKI_CODE_OVERRIDES = {"zh-CN": "zh", "fil": "tl"}

# Audited aliases whose old cache entry was a spouse, parent, event, or a
# similarly named person.  The API response is still fetched and retained.
QID_OVERRIDES = {
    "Prince Andrew (Duke of York)": "Q153330",
    "Prince Andrew, Duke of York": "Q153330",
    "Prince Andrew": "Q153330",
    "Kei Komuro": "Q63382863",
    "Harry (son of Diana)": "Q152316",
    "António Costa e Silva": "Q97170467",
    "Carl Philip of Sweden": "Q214704",
    "Kate Middleton": "Q34787",
    "Kate Middleton (Catherine)": "Q34787",
    "Michael Collins": "Q104859",
    "Andrew Pollard": "Q64861148",
    # Exact identities recovered during the 429 retry pass or independently
    # checked against unambiguous context (office, profession, or alias).
    "Melanie Brinkmann": "Q83979239",
    "Raman Pratasevich": "Q106949139",
    "Lee Nak-yeon": "Q12611134",
    "Ilias Mosialos": "Q14337846",
    "Attila Vidnyánszky": "Q732953",
    "René Fasel": "Q688764",
    "Andrew Bailey": "Q51561618",
    "Meng Wanzhou": "Q59485866",
    "Svetlana Tikhanovskaya": "Q97010473",
    "Jack Dorsey": "Q335552",
    "Oh Se-hoon": "Q494239",
    "Tan Sri Muhyiddin": "Q1060949",
    "Nicolás Maduro": "Q58132",
    "Mārtiņš Staķis": "Q58371519",
    "Peter Pellegrini": "Q17330556",
    "Sywert van Lienden": "Q2764748",
    "Bojana Beović": "Q89561883",
    "George Soros": "Q12908",
    "Marcelo Queiroga": "Q105969706",
    "Jason Kenney": "Q3162959",
    "Aung San Suu Kyi": "Q36740",
    "Bernie Sanders": "Q359442",
    "Raluca Turcan": "Q7288407",
    "Dimitris Lignadis": "Q40682521",
    "Martin Bashir": "Q951095",
    "Nicușor Dan": "Q11163124",
    "Jan Blatný": "Q56974631",
    "Monica Anisie": "Q73515246",
    "Sylvi Listhaug": "Q7660866",
    "Monika Beňová": "Q441032",
    "Pope Francis": "Q450675",
    "Recep Tayyip Erdogan": "Q39259",
    "Olaf Scholz": "Q61053",
    "Pedro Sánchez": "Q6070218",
    "Włodzimierz Gut": "Q88096567",
    "Jarosław Jakimowicz": "Q11720859",
    "Tamayo Marukawa": "Q7680979",
    "López Obrador": "Q318508",
    "Stephen Donnelly": "Q76130565",
    "Charles Darwin": "Q1035",
    "Margaret Thatcher": "Q7416",
    "Tom Cruise": "Q37079",
    "Narendra Modi": "Q1058",
    "Neil Armstrong": "Q1615",
    "Jeffrey Epstein": "Q994932",
    "Ron DeSantis": "Q543522",
    "James Webb": "Q537520",
    "Michael Freilich": "Q18225134",
    "Navalny (first name not stated in the article)": "Q155979",
    "Bolsonaro (given name not established in the text)": "Q10304982",
    "Aleksandr Lukashenko": "Q2866",
    "Aliaksandr Lukashenko": "Q2866",
    "Netanyahu": "Q43723",
    "Willem-Alexander": "Q154952",
    "Queen Elizabeth": "Q9682",
    "Máxima": "Q460960",
    "De Jonge": "Q2649226",
    "Suga": "Q122465",
    # Exact-name retry results audited against the local JailNews article
    # context.  Pinning these makes a rerun independent of Wikidata search
    # ranking changes and prevents another 429 retry from changing identity.
    "Ugur Sahin": "Q2505089",
    "Jen Psaki": "Q12066523",
    "Rochelle Walensky": "Q34445539",
    "Albert Bourla": "Q63400080",
    "Milan Lučanský": "Q60234617",
    "Marianna Schreiber": "Q112122178",
    "Richard Bergström": "Q96333304",
    "Hunter Biden": "Q5944264",
    "Gladys Berejiklian": "Q5566385",
    "Borut Štrukelj": "Q79330998",
    "Jarosław Gowin": "Q222905",
    # The automatic exact-name hit was a different researcher (Q89952338).
    "Scott Kelly": "Q362190",
    "Jimmy Lai": "Q1447095",
    "Marek Krajčí": "Q26198001",
    "Ilze Viņķele": "Q2607120",
    "Simona Halep": "Q230156",
    "Hirofumi Yoshimura": "Q21529384",
    "Milan Krek": "Q65491176",
    "Ramūnas Karbauskis": "Q145315",
    "Paul Pogba": "Q129027",
    # The automatic exact-name hit was Edward VIII (Q590227).
    "Prince Edward": "Q154920",
    "Lefteris Avgenakis": "Q12880163",
    "Mauricio Macri": "Q561837",
    "Valeriu Gheorghiță": "Q92448330",
    "Linas Linkevičius": "Q345625",
    "Roman Prymula": "Q17277123",
    "Ismail Sabri Yaakob": "Q6084858",
    "Ferd Grapperhaus": "Q2195581",
    "Nikol Pashinyan": "Q7035479",
    "Heiko Maas": "Q108447",
    "Mateja Logar": "Q98093926",
    "Jill Biden": "Q235349",
    "Vlad Voiculescu": "Q24174632",
    "Robert Fico": "Q57606",
    "Fumiko Kinoshita": "Q60230151",
    "Sarah Everard": "Q105908485",
    "Margaret Keenan": "Q104089209",
    "Jonathan Van-Tam": "Q41448199",
    "Sandra Ciesek": "Q87922222",
    "Michał Dworczyk": "Q11778705",
    "Joan Laporta": "Q311611",
}

# Exact-name search is unsafe for these dataset labels.  One is a Nikkei
# byline and the other search result is a different Czech namesake.  They stay
# explicit unresolved entities unless an audited QID is supplied later.
NO_SAFE_AUTOMATIC_NAMES = {"Ryo Nakamura", "Ladislav Dušek"}

FORCE_REFRESH_NAMES = {"Marianna Schreiber"}

QUERY_OVERRIDES = {
    "Carl Philip of Sweden": "Prince Carl Philip Duke of Värmland",
    "Tan Sri Muhyiddin": "Muhyiddin Yassin",
    "Navalny (first name not stated in the article)": "Alexei Navalny",
    "Bolsonaro (given name not established in the text)": "Jair Bolsonaro",
    "Kate Middleton (Catherine)": "Catherine Princess of Wales",
}

BAD_DESCRIPTION_TERMS = {
    "album", "film", "episode", "list of", "surname", "given name", "wedding",
    "election", "government", "university", "country", "valley", "franchise",
    "novel", "video game", "television series", "basketball club",
}
PERSON_DESCRIPTION_TERMS = {
    "politician", "minister", "president", "king", "queen", "prince", "princess",
    "actor", "actress", "journalist", "scientist", "physician", "lawyer", "activist",
    "business", "footballer", "royal", "pope", "diplomat", "economist", "director",
    "professor", "researcher", "born", "died", "executive", "official", "person",
}
CONTEXT_CUES = {
    "astronaut", "apollo", "nasa", "vaccine", "vaccinologist", "immunologist",
    "virologist", "epidemiologist", "physician", "doctor", "professor", "scientist",
    "politician", "minister", "president", "prime minister", "governor", "mayor",
    "footballer", "football", "actor", "actress", "journalist", "royal", "prince",
    "princess", "pope", "businessman", "businesswoman", "activist", "general",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def norm(value: str) -> str:
    value = "".join(
        char for char in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(char)
    )
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def request(params: dict[str, Any], retries: int = 8) -> dict[str, Any]:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    req = urllib.request.Request(f"{WIKIDATA_API}?{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            code = getattr(exc, "code", None)
            if code not in {429, 500, 502, 503, 504} or attempt + 1 == retries:
                raise
            retry_after = getattr(exc, "headers", {}).get("Retry-After") if getattr(exc, "headers", None) else None
            wait = float(retry_after) if retry_after and retry_after.isdigit() else min(90.0, 2.0 ** (attempt + 1))
            time.sleep(wait)
    raise RuntimeError("unreachable retry loop")


def fetch_entity(qid: str) -> dict[str, Any] | None:
    data = request({
        "action": "wbgetentities", "ids": qid,
        "props": "labels|aliases|descriptions|sitelinks",
        "languages": "en", "sitefilter": "",
    })
    entity = data.get("entities", {}).get(qid)
    return None if not entity or entity.get("missing") else entity


def fetch_entities(qids: list[str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for start in range(0, len(qids), 50):
        batch = qids[start:start + 50]
        data = request({
            "action": "wbgetentities", "ids": "|".join(batch),
            "props": "labels|aliases|descriptions|sitelinks", "languages": "en",
        })
        for qid, entity in data.get("entities", {}).items():
            if not entity.get("missing"):
                output[qid] = entity
        time.sleep(0.8)
    return output


def search(name: str, evidence: str) -> list[dict[str, Any]]:
    query = QUERY_OVERRIDES.get(name, name)
    cues = [cue for cue in sorted(CONTEXT_CUES, key=len, reverse=True) if cue in evidence.casefold()]
    if cues:
        query += " " + " ".join(cues[:3])
    return request({
        "action": "wbsearchentities", "search": query, "language": "en",
        "uselang": "en", "type": "item", "limit": 10,
    }).get("search", [])


def candidate_score(name: str, country: str, evidence: str, candidate: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    query = norm(QUERY_OVERRIDES.get(name, name))
    names = [candidate.get("label", ""), *(candidate.get("aliases") or [])]
    normalized = [norm(value) for value in names if value]
    similarity = max((difflib.SequenceMatcher(None, query, value).ratio() for value in normalized), default=0.0)
    exact = any(query == value for value in normalized)
    query_tokens = set(query.split())
    coverage = max((len(query_tokens & set(value.split())) / max(1, len(query_tokens)) for value in normalized), default=0.0)
    description = str(candidate.get("description") or "").casefold()
    person_like = any(term in description for term in PERSON_DESCRIPTION_TERMS)
    bad_kind = any(term in description for term in BAD_DESCRIPTION_TERMS)
    country_hit = bool(country and norm(country) in norm(description))
    evidence_cues = [cue for cue in CONTEXT_CUES if cue in evidence.casefold()]
    context_hits = sorted({cue for cue in evidence_cues if cue in description})
    score = similarity + 0.65 * coverage + (0.65 if exact else 0.0)
    score += 0.15 if person_like else 0.0
    score += 0.15 if country_hit else 0.0
    score += min(0.90, 0.45 * len(context_hits))
    score -= 0.90 if bad_kind else 0.0
    details = {
        "name_similarity": round(similarity, 6), "token_coverage": round(coverage, 6),
        "exact_label_or_alias": exact, "person_like_description": person_like,
        "country_hint_in_description": country_hit, "disallowed_entity_kind": bad_kind,
        "context_description_hits": context_hits,
    }
    return score, details


def resolve(name: str, country: str, evidence: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if name in QID_OVERRIDES:
        qid = QID_OVERRIDES[name]
        entity = fetch_entity(qid)
        return entity, {"decision": "accepted_override", "qid": qid, "candidates": []}
    if name in NO_SAFE_AUTOMATIC_NAMES:
        return None, {"decision": "no_safe_match", "qid": None, "candidates": [], "reason": "ambiguous_exact_name"}
    candidates = search(name, evidence)
    ranked = []
    for candidate in candidates:
        score, details = candidate_score(name, country, evidence, candidate)
        ranked.append({**candidate, "audit_score": round(score, 6), **details})
    ranked.sort(key=lambda item: (item["audit_score"], item.get("id", "")), reverse=True)
    top = ranked[0] if ranked else None
    accepted = bool(
        top and top["audit_score"] >= 1.45 and top["name_similarity"] >= 0.72
        and not top["disallowed_entity_kind"]
        and (top["exact_label_or_alias"] or top["person_like_description"])
    )
    decision = {
        "decision": "accepted_search" if accepted else "no_safe_match",
        "qid": top.get("id") if accepted else None,
        "candidates": ranked[:5],
    }
    return (fetch_entity(top["id"]) if accepted else None), decision


def entity_value(entity: dict[str, Any], field: str, language: str = "en") -> str | None:
    value = entity.get(field, {}).get(language)
    return value.get("value") if isinstance(value, dict) else None


def entity_aliases(entity: dict[str, Any], language: str = "en") -> list[str]:
    return [row["value"] for row in entity.get("aliases", {}).get(language, []) if row.get("value")]


def cached_entity(page: dict[str, Any]) -> dict[str, Any] | None:
    qid = page.get("wikidata_qid")
    if not qid:
        return None
    title = page.get("title") or page.get("requested_person") or qid
    sitelinks = {"enwiki": {"title": title}}
    for code, link in page.get("langlinks", {}).items():
        if link.get("title"):
            sitelinks[f"{code}wiki"] = {"title": link["title"]}
    extract = str(page.get("extract") or "").strip()
    description = extract.split(".", 1)[0][:300] if extract else None
    return {
        "id": qid, "labels": {"en": {"language": "en", "value": title}},
        "aliases": {},
        "descriptions": ({"en": {"language": "en", "value": description}} if description else {}),
        "sitelinks": sitelinks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", type=Path, default=RUNTIME / "person_localization_queue.jsonl")
    ap.add_argument("--cache", type=Path, default=RUNTIME / "person_wikipedia_full_cache.jsonl")
    ap.add_argument("--languages", type=Path, default=ROOT / "data/political_safety_collection_20260929/multilingual_pc2/pc2_languages.json")
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/jailnewsbench_person_domain_20260930/luna_review_v1/analysis_ready_samples.jsonl")
    ap.add_argument("--output", type=Path, default=RUNTIME / "entity_links_v2")
    ap.add_argument("--delay", type=float, default=0.35)
    ap.add_argument("--offline", action="store_true", help="Use only the existing cache and audited QID overrides")
    args = ap.parse_args()

    queue = list(read_jsonl(args.queue))
    people: dict[str, dict[str, Any]] = {}
    for row in queue:
        people.setdefault(row["person_id"], row)
    old = {row["person_id"]: row for row in read_jsonl(args.cache)}
    languages = json.loads(args.languages.read_text(encoding="utf-8"))["languages"]
    language_code = {row["name"]: WIKI_CODE_OVERRIDES.get(row["code"], row["code"]) for row in languages}
    evidence_by_name: dict[str, list[str]] = defaultdict(list)
    for context in read_jsonl(args.contexts):
        article = str(context.get("article_en") or "")
        for person in context.get("luna_people", []):
            canonical = str(person.get("canonical_person") or "")
            if canonical:
                evidence_by_name[norm(canonical)].append(article)

    initial_qids = sorted({
        row.get("wikipedia", {}).get("wikidata_qid") for row in old.values()
        if row.get("wikipedia", {}).get("wikidata_qid")
    } | set(QID_OVERRIDES.values()))
    entities: dict[str, dict[str, Any]] = {}
    if args.offline:
        for cached in old.values():
            entity = cached_entity(cached.get("wikipedia", {}))
            if entity:
                entities[entity["id"]] = entity
        for qid in QID_OVERRIDES.values():
            entities.setdefault(qid, {
                "id": qid, "labels": {}, "aliases": {}, "descriptions": {}, "sitelinks": {},
            })
    else:
        entities = fetch_entities(initial_qids)
    label_map = []
    decisions = []
    refresh_statuses = {"fetch_error", "page_fetch_failed", "needs_review", "not_found"}
    for index, (pid, row) in enumerate(sorted(people.items()), 1):
        name = row["canonical_name"]
        countries = row.get("profile_evidence", {}).get("countries", {})
        country = max(countries, key=countries.get) if countries else ""
        evidence = " ".join(evidence_by_name.get(norm(name), []))[:12000]
        old_page = old.get(pid, {}).get("wikipedia", {})
        must_resolve = (
            name in QID_OVERRIDES or name in FORCE_REFRESH_NAMES
            or old_page.get("status") in refresh_statuses or not old_page.get("wikidata_qid")
        )
        decision: dict[str, Any]
        entity = None
        if must_resolve:
            if args.offline and name in QID_OVERRIDES:
                qid = QID_OVERRIDES[name]
                entity = entities[qid]
                decision = {"decision": "accepted_audited_override_offline", "qid": qid, "candidates": []}
            elif args.offline:
                decision = {"decision": "pending_external_review", "qid": None, "candidates": []}
            else:
                try:
                    entity, decision = resolve(name, country, evidence)
                except Exception as exc:
                    decision = {"decision": "fetch_error", "error_type": type(exc).__name__, "error": str(exc), "qid": None, "candidates": []}
                time.sleep(args.delay)
        else:
            qid = old_page["wikidata_qid"]
            entity = entities.get(qid)
            decision = {
                "decision": "retained_verified_cache" if entity else "cache_qid_fetch_error",
                "qid": qid if entity else None, "candidates": [],
            }
        qid = decision.get("qid") if entity else None
        if qid and entity:
            entities[qid] = entity
        entity_id = qid or f"unresolved:{pid}"
        label_map.append({
            "person_id": pid, "dataset_label": name, "entity_id": entity_id,
            "wikidata_qid": qid, "country_hint": country,
            "resolution_status": decision["decision"],
        })
        decisions.append({
            "person_id": pid, "dataset_label": name, "country_hint": country,
            "previous_status": old_page.get("status"), "previous_qid": old_page.get("wikidata_qid"),
            **decision,
        })
        print(f"{index}/{len(people)} {name}: {decision['decision']} {qid or ''}", flush=True)

    aliases_by_entity: dict[str, list[str]] = defaultdict(list)
    for row in label_map:
        aliases_by_entity[row["entity_id"]].append(row["dataset_label"])
    entity_rows = []
    for entity_id, labels in sorted(aliases_by_entity.items()):
        qid = entity_id if entity_id.startswith("Q") else None
        entity = entities.get(qid, {}) if qid else {}
        enwiki = entity.get("sitelinks", {}).get("enwiki", {})
        entity_rows.append({
            "entity_id": entity_id, "wikidata_qid": qid,
            "canonical_label": entity_value(entity, "labels") or labels[0],
            "dataset_labels": sorted(labels), "english_aliases": entity_aliases(entity),
            "description": entity_value(entity, "descriptions"),
            "enwiki_title": enwiki.get("title"),
        })

    label_lookup = {row["person_id"]: row for row in label_map}
    localization_rows = []
    for row in queue:
        mapped = label_lookup[row["person_id"]]
        qid = mapped["wikidata_qid"]
        entity = entities.get(qid, {}) if qid else {}
        code = language_code[row["target_language"]]
        site = f"{code}wiki"
        sitelink = entity.get("sitelinks", {}).get(site, {})
        title = sitelink.get("title")
        item = dict(row)
        item.update({"entity_id": mapped["entity_id"], "wikidata_qid": qid})
        if title:
            url = f"https://{code}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            item.update({
                "localized_name": title, "rendering_method": "wikidata_sitelink",
                "source_title": title, "source_url": url,
                "source_accessed_at": datetime.now(timezone.utc).isoformat(),
                "identity_match": "verified_qid", "status": "verified_localized", "confidence": "high",
            })
        elif qid:
            item.update({
                "localized_name": row["canonical_name"], "rendering_method": "original_fallback",
                "identity_match": "verified_qid", "status": "fallback_original", "confidence": "medium",
            })
        else:
            item.update({
                "localized_name": row["canonical_name"], "rendering_method": "original_fallback",
                "identity_match": "unresolved", "status": "unresolved", "confidence": "low",
            })
        localization_rows.append(item)

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "label_entity_map.jsonl", label_map)
    write_jsonl(args.output / "resolution_decisions.jsonl", decisions)
    write_jsonl(args.output / "entities_deduplicated_by_qid.jsonl", entity_rows)
    write_jsonl(args.output / "person_localizations.jsonl", localization_rows)
    decision_by_pid = {row["person_id"]: row for row in decisions}
    needs_review_audit = []
    retry_audit = []
    qid_corrections = []
    for pid, cached in sorted(old.items()):
        previous = cached.get("wikipedia", {})
        resolved = label_lookup[pid]
        audit = {
            "person_id": pid, "dataset_label": resolved["dataset_label"],
            "previous_status": previous.get("status"), "previous_qid": previous.get("wikidata_qid"),
            "previous_title": previous.get("title"), "final_qid": resolved.get("wikidata_qid"),
            "final_entity_id": resolved["entity_id"], "resolution_status": resolved["resolution_status"],
            "adjudication": (
                "corrected_or_confirmed_qid" if resolved.get("wikidata_qid")
                else "rejected_previous_candidate_no_safe_local_match"
            ),
        }
        if previous.get("status") == "needs_review":
            needs_review_audit.append(audit)
        if previous.get("status") in {"fetch_error", "page_fetch_failed", "not_found"}:
            retry_audit.append(audit)
        if previous.get("wikidata_qid") and previous.get("wikidata_qid") != resolved.get("wikidata_qid"):
            qid_corrections.append(audit)
    write_jsonl(args.output / "needs_review_30_adjudicated.jsonl", needs_review_audit)
    write_jsonl(args.output / "failed_fetch_retry_audit.jsonl", retry_audit)
    write_jsonl(args.output / "qid_corrections.jsonl", qid_corrections)
    status_counts = Counter(row["resolution_status"] for row in label_map)
    localization_counts = Counter(row["status"] for row in localization_rows)
    summary = {
        "schema": "jailnews_entity_links/v2", "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_labels": len(label_map), "qid_entities": sum(row["wikidata_qid"] is not None for row in entity_rows),
        "deduplicated_entities": len(entity_rows), "unresolved_entities": sum(row["wikidata_qid"] is None for row in entity_rows),
        "resolution_status_counts": dict(status_counts), "localization_rows": len(localization_rows),
        "localization_status_counts": dict(localization_counts),
        "deduplication_key": "wikidata_qid; unresolved labels remain separate",
        "needs_review_adjudicated": len(needs_review_audit),
        "needs_review_with_qid": sum(row["final_qid"] is not None for row in needs_review_audit),
        "needs_review_rejected_without_safe_qid": sum(row["final_qid"] is None for row in needs_review_audit),
        "failed_fetch_retry_rows": len(retry_audit),
        "failed_fetch_recovered": sum(row["final_qid"] is not None for row in retry_audit),
        "failed_fetch_still_pending": sum(row["final_qid"] is None for row in retry_audit),
        "qid_corrections": len(qid_corrections),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
