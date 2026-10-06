#!/usr/bin/env python3
"""Resolve 501 people to verified Wikipedia language titles for 72 languages.

Names are localized by same-entity language links, never by blindly machine-translating
the canonical spelling.  Missing links remain explicit original-spelling fallbacks.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from fetch_jailnews_wikipedia_people import TITLE_OVERRIDES, fetch_person, request_json, tokens


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "artifacts/jailnews_bandit_20260930/runtime"
LANGUAGES = ROOT / "data/political_safety_collection_20260929/multilingual_pc2/pc2_languages.json"
WIKI_CODE_OVERRIDES = {"zh-CN": "zh", "fil": "tl"}
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "POLY-safety-evaluation/1.0 (person-language selector research)"


def wikidata_entities(titles: list[str]) -> dict[str, Any]:
    query = urllib.parse.urlencode({
        "action": "wbgetentities", "sites": "enwiki", "titles": "|".join(titles),
        "props": "sitelinks", "redirects": "yes", "normalize": 1,
        "format": "json", "formatversion": 2,
    })
    request = urllib.request.Request(f"{WIKIDATA_API}?{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response).get("entities", {})
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 5:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 2.0 ** (attempt + 1))
            time.sleep(delay)
    raise RuntimeError("unreachable Wikidata retry loop")


def fetch_person_fast(person: str, country: str) -> dict[str, Any]:
    """Resolve the common exact-title case without downloading the full article extract."""
    requested_title = TITLE_OVERRIDES.get(person, person)
    pages = request_json({
        "action": "query", "prop": "info|pageprops|langlinks", "titles": requested_title,
        "redirects": 1, "inprop": "url", "lllimit": "max", "llprop": "url",
    }).get("query", {}).get("pages", [])
    if pages and not pages[0].get("missing"):
        page = pages[0]
        overlap = len(tokens(person) & tokens(page.get("title", ""))) / max(1, len(tokens(person)))
        if "disambiguation" not in page.get("pageprops", {}) and overlap >= 0.50:
            langlinks = {
                item["lang"]: {"title": item["title"], "url": item.get("url", "")}
                for item in page.get("langlinks", [])
            }
            return {
                "status": "ok", "resolution_method": "exact_title_or_redirect_fast",
                "search_match_score": round(overlap, 6), "requested_person": person,
                "requested_country": country, "requested_title": requested_title,
                "pageid": page.get("pageid"), "wikidata_qid": page.get("pageprops", {}).get("wikibase_item"),
                "title": page.get("title"), "canonical_url": page.get("canonicalurl") or page.get("fullurl"),
                "is_disambiguation": False, "langlinks": langlinks, "langlink_count": len(langlinks),
                "search_candidates": [],
            }
    return fetch_person(person, country)


def page_result(person: str, country: str, requested_title: str, page: dict[str, Any], method: str) -> dict[str, Any] | None:
    overlap = len(tokens(person) & tokens(page.get("title", ""))) / max(1, len(tokens(person)))
    if page.get("missing") or "disambiguation" in page.get("pageprops", {}) or overlap < 0.50:
        return None
    langlinks = {
        item["lang"]: {"title": item["title"], "url": item.get("url", "")}
        for item in page.get("langlinks", [])
    }
    return {
        "status": "ok", "resolution_method": method, "search_match_score": round(overlap, 6),
        "requested_person": person, "requested_country": country, "requested_title": requested_title,
        "pageid": page.get("pageid"), "wikidata_qid": page.get("pageprops", {}).get("wikibase_item"),
        "title": page.get("title"), "canonical_url": page.get("canonicalurl") or page.get("fullurl"),
        "is_disambiguation": False, "langlinks": langlinks, "langlink_count": len(langlinks),
        "search_candidates": [],
    }


def fetch_batch_exact(items: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any], str, dict[str, Any]]]:
    requested = []
    country_by_pid = {}
    for pid, row in items:
        countries = row.get("profile_evidence", {}).get("countries", {})
        country_by_pid[pid] = max(countries, key=countries.get) if countries else ""
        requested.append(TITLE_OVERRIDES.get(row["canonical_name"], row["canonical_name"]))
    entities = wikidata_entities(requested)
    by_title = {}
    for qid, entity in entities.items():
        enwiki = entity.get("sitelinks", {}).get("enwiki", {})
        if enwiki.get("title"):
            by_title[enwiki["title"].casefold()] = (qid, entity)
    result = []
    for (pid, row), title in zip(items, requested):
        match = by_title.get(title.casefold())
        direct = None
        if match:
            qid, entity = match
            sitelinks = entity.get("sitelinks", {})
            en_title = sitelinks["enwiki"]["title"]
            langlinks = {}
            for site, link in sitelinks.items():
                if not site.endswith("wiki") or site in {"commonswiki", "specieswiki"}:
                    continue
                code = site[:-4]
                linked_title = link.get("title")
                if linked_title:
                    langlinks[code] = {
                        "title": linked_title,
                        "url": f"https://{code}.wikipedia.org/wiki/{urllib.parse.quote(linked_title.replace(' ', '_'))}",
                    }
            direct = {
                "status": "ok", "resolution_method": "wikidata_batch_exact_title",
                "search_match_score": 1.0, "requested_person": row["canonical_name"],
                "requested_country": country_by_pid[pid], "requested_title": title,
                "wikidata_qid": qid, "title": en_title,
                "canonical_url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(en_title.replace(' ', '_'))}",
                "is_disambiguation": False, "langlinks": langlinks, "langlink_count": len(langlinks),
                "search_candidates": [],
            }
        if direct is None:
            direct = fetch_person_fast(row["canonical_name"], country_by_pid[pid])
        result.append((pid, row, country_by_pid[pid], direct))
    return result


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", type=Path, default=RUNTIME / "person_localization_queue.jsonl")
    ap.add_argument("--cache", type=Path, default=RUNTIME / "person_wikipedia_full_cache.jsonl")
    ap.add_argument("--output", type=Path, default=RUNTIME / "person_localizations_resolved.jsonl")
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--max-people", type=int)
    ap.add_argument(
        "--retry-status", action="append",
        help="Wikipedia status to refresh; repeat the option. Defaults to fetch errors and review rows.",
    )
    ap.add_argument("--refresh-all", action="store_true")
    args = ap.parse_args()
    queue = list(read_jsonl(args.queue))
    people: dict[str, dict[str, Any]] = {}
    for row in queue:
        people.setdefault(row["person_id"], row)
    cached = {row["person_id"]: row for row in read_jsonl(args.cache)} if args.cache.exists() else {}
    retryable = set(args.retry_status or ["fetch_error", "page_fetch_failed", "needs_review", "not_found"])
    pending = [
        (pid, row) for pid, row in people.items()
        if args.refresh_all or pid not in cached or cached[pid].get("wikipedia", {}).get("status") in retryable
    ]
    if args.max_people is not None:
        pending = pending[:args.max_people]
    def fetch_group(group: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any], str, dict[str, Any]]]:
        try:
            return fetch_batch_exact(group)
        except Exception as exc:
            output = []
            for pid, row in group:
                countries = row.get("profile_evidence", {}).get("countries", {})
                country = max(countries, key=countries.get) if countries else ""
                output.append((pid, row, country, {
                    "status": "fetch_error", "error_type": type(exc).__name__, "error": str(exc),
                }))
            return output

    if args.workers < 1 or args.workers > 16:
        raise ValueError("--workers must be in [1,16]")
    if args.batch_size < 1 or args.batch_size > 50:
        raise ValueError("--batch-size must be in [1,50]")
    groups = [pending[start:start + args.batch_size] for start in range(0, len(pending), args.batch_size)]
    completed_people = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_group, group) for group in groups]
        for future in concurrent.futures.as_completed(futures):
            for pid, row, country, result in future.result():
                completed_people += 1
                cached[pid] = {
                    "person_id": pid, "canonical_name": row["canonical_name"],
                    "country_hint": country, "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "wikipedia": result,
                }
                print(f"{completed_people}/{len(pending)} {row['canonical_name']} {result['status']}", flush=True)
            write_jsonl(args.cache, (cached[key] for key in sorted(cached)))
            if args.delay:
                time.sleep(args.delay)
    language_rows = json.loads(LANGUAGES.read_text(encoding="utf-8"))["languages"]
    by_name_code = {item["name"]: WIKI_CODE_OVERRIDES.get(item["code"], item["code"]) for item in language_rows}
    resolved = []
    status_counts: dict[str, int] = {}
    for row in queue:
        page_row = cached.get(row["person_id"])
        page = page_row.get("wikipedia", {}) if page_row else {}
        wiki_ok = page.get("status") == "ok"
        code = by_name_code[row["target_language"]]
        link = page.get("langlinks", {}).get(code) if wiki_ok else None
        item = dict(row)
        item["wikidata_qid"] = page.get("wikidata_qid")
        if row["target_language"] == "English" and wiki_ok:
            item.update({
                "localized_name": page["title"], "rendering_method": "wikipedia_title",
                "source_title": page["title"], "source_url": page.get("canonical_url"),
                "source_accessed_at": page_row["fetched_at"], "identity_match": "verified",
                "status": "verified_localized", "confidence": "high",
            })
        elif link:
            source_url = link.get("url") or (
                f"https://{code}.wikipedia.org/wiki/"
                + urllib.parse.quote(link["title"].replace(" ", "_"))
            )
            item.update({
                "localized_name": link["title"], "rendering_method": "wikipedia_title",
                "source_title": link["title"], "source_url": source_url,
                "source_accessed_at": page_row["fetched_at"], "identity_match": "verified",
                "status": "verified_localized", "confidence": "high",
            })
        else:
            item.update({
                "localized_name": row["canonical_name"], "rendering_method": "original_fallback",
                "identity_match": "verified" if wiki_ok else "unchecked",
                "status": "fallback_original" if wiki_ok else "unresolved",
                "confidence": "medium" if wiki_ok else "low",
            })
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
        resolved.append(item)
    write_jsonl(args.output, resolved)
    summary = {
        "schema": "person_localizations_resolved/v1", "people_total": len(people),
        "people_fetched": len(cached), "rows": len(resolved), "status_counts": status_counts,
        "identity_rule": "Wikipedia same-page langlinks; missing links retain canonical spelling.",
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
