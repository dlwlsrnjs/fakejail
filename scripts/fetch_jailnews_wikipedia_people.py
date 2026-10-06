#!/usr/bin/env python3
"""Fetch auditable English Wikipedia text for person-language selector features."""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "POLY-safety-evaluation/1.0 (person-language selector research)"
TITLE_OVERRIDES = {
    "Andrés Manuel López Obrador": "Andrés Manuel López Obrador",
    "Surangel Whipps Jr": "Surangel Whipps Jr.",
    "Vieira da Silva": "José António Vieira da Silva",
    # Dataset labels and current English-Wikipedia titles that are not exact
    # matches.  These are title hints only; the v2 entity repair step still
    # validates and deduplicates the resulting Wikidata QID.
    "Prince Andrew (Duke of York)": "Andrew Mountbatten-Windsor",
    "Prince Andrew, Duke of York": "Andrew Mountbatten-Windsor",
    "Harry (son of Diana)": "Prince Harry, Duke of Sussex",
    "António Costa e Silva": "António Costa Silva",
    "Carl Philip of Sweden": "Prince Carl Philip, Duke of Värmland",
    "Tan Sri Muhyiddin": "Muhyiddin Yassin",
    "Navalny (first name not stated in the article)": "Alexei Navalny",
    "Bolsonaro (given name not established in the text)": "Jair Bolsonaro",
    "Kate Middleton": "Catherine, Princess of Wales",
    "Kate Middleton (Catherine)": "Catherine, Princess of Wales",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def request_json(params: dict[str, Any], max_retries: int = 7) -> dict[str, Any]:
    query = urllib.parse.urlencode({
        **params, "format": "json", "formatversion": 2, "utf8": 1, "maxlag": 5,
    })
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 == max_retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 3.0 * 2 ** attempt)
            time.sleep(wait)
    raise RuntimeError("unreachable retry loop")


def tokens(value: str) -> set[str]:
    stop = {"jr", "sr", "the", "da", "de", "van", "von", "and"}
    value = "".join(
        char for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )
    return {
        token.casefold() for token in re.findall(r"[^\W\d_]+", value, re.UNICODE)
        if len(token) >= 2 and token.casefold() not in stop
    }


def candidate_score(person: str, country: str, candidate: dict[str, Any]) -> tuple[float, int]:
    person_tokens = tokens(person)
    title_tokens = tokens(candidate.get("title", ""))
    overlap = len(person_tokens & title_tokens) / max(1, len(person_tokens))
    snippet = re.sub(r"<[^>]+>", " ", candidate.get("snippet", "")).casefold()
    country_bonus = 0.15 if country.casefold() in snippet else 0.0
    politician_bonus = 0.10 if any(word in snippet for word in ("politician", "minister", "president", "prime minister")) else 0.0
    exact_bonus = 0.30 if candidate.get("title", "").casefold() == person.casefold() else 0.0
    return overlap + country_bonus + politician_bonus + exact_bonus, -int(candidate.get("pageid", 0))


def fetch_person(person: str, country: str) -> dict[str, Any]:
    requested_title = TITLE_OVERRIDES.get(person, person)
    direct_pages = request_json({
        "action": "query", "prop": "extracts|info|pageprops|langlinks",
        "titles": requested_title, "redirects": 1, "explaintext": 1,
        "inprop": "url", "lllimit": "max", "llprop": "url",
    }).get("query", {}).get("pages", [])
    if direct_pages and not direct_pages[0].get("missing"):
        page = direct_pages[0]
        extract = str(page.get("extract") or "")
        is_disambiguation = "disambiguation" in page.get("pageprops", {})
        score = len(tokens(person) & tokens(page.get("title", ""))) / max(1, len(tokens(person)))
        if extract and not is_disambiguation and score >= 0.50:
            langlinks = {
                item["lang"]: {"title": item["title"], "url": item.get("url", "")}
                for item in page.get("langlinks", [])
            }
            return {
                "status": "ok",
                "resolution_method": "exact_title_or_redirect",
                "search_match_score": round(score, 6),
                "requested_person": person,
                "requested_country": country,
                "requested_title": requested_title,
                "pageid": page.get("pageid"),
                "wikidata_qid": page.get("pageprops", {}).get("wikibase_item"),
                "title": page.get("title"),
                "canonical_url": page.get("canonicalurl") or page.get("fullurl"),
                "extract": extract,
                "extract_characters": len(extract),
                "is_disambiguation": False,
                "langlinks": langlinks,
                "langlink_count": len(langlinks),
                "search_candidates": [],
            }

    query = f'"{person}" {country} politician'
    search = request_json({
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": 8, "srnamespace": 0,
    }).get("query", {}).get("search", [])
    if not search:
        search = request_json({
            "action": "query", "list": "search", "srsearch": person,
            "srlimit": 8, "srnamespace": 0,
        }).get("query", {}).get("search", [])
    if not search:
        return {"status": "not_found", "search_query": query, "candidates": []}

    ranked = sorted(search, key=lambda item: candidate_score(person, country, item), reverse=True)
    chosen = ranked[0]
    page_data = request_json({
        "action": "query", "prop": "extracts|info|pageprops|langlinks", "pageids": chosen["pageid"],
        "explaintext": 1, "inprop": "url", "lllimit": "max", "llprop": "url",
    }).get("query", {}).get("pages", [])
    if not page_data:
        return {"status": "page_fetch_failed", "search_query": query, "candidates": ranked}
    page = page_data[0]
    extract = str(page.get("extract") or "")
    score = candidate_score(person, country, chosen)[0]
    is_disambiguation = "disambiguation" in page.get("pageprops", {})
    status = "ok" if extract and not is_disambiguation and score >= 0.60 else "needs_review"
    langlinks = {
        item["lang"]: {"title": item["title"], "url": item.get("url", "")}
        for item in page.get("langlinks", [])
    }
    return {
        "status": status,
        "resolution_method": "search_fallback",
        "search_query": query,
        "search_match_score": round(score, 6),
        "requested_person": person,
        "requested_country": country,
        "pageid": page.get("pageid"),
        "wikidata_qid": page.get("pageprops", {}).get("wikibase_item"),
        "title": page.get("title"),
        "canonical_url": page.get("canonicalurl") or page.get("fullurl"),
        "extract": extract,
        "extract_characters": len(extract),
        "is_disambiguation": is_disambiguation,
        "langlinks": langlinks,
        "langlink_count": len(langlinks),
        "search_candidates": [
            {
                "pageid": item.get("pageid"),
                "title": item.get("title"),
                "score": round(candidate_score(person, country, item)[0], 6),
            }
            for item in ranked[:5]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores", type=Path,
        default=Path("data/jailnewsbench_pc2_contextual_20260929/language_scores/language_scores.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/jailnewsbench_pc2_contextual_20260929/person_wikipedia_cache.jsonl"),
    )
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    score_rows = read_jsonl(args.scores)
    prior: dict[str, dict[str, Any]] = {}
    if args.output.exists() and not args.force:
        prior = {row["pilot_id"]: row for row in read_jsonl(args.output)}

    results = []
    def persist() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for item in sorted(results, key=lambda value: value["pilot_id"]):
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    for index, row in enumerate(score_rows, 1):
        pilot_id = row["pilot_id"]
        if pilot_id in prior and prior[pilot_id].get("wikipedia", {}).get("status") == "ok":
            results.append(prior[pilot_id])
            continue
        wikipedia = fetch_person(row["person"], row["person_country_or_territory"])
        results.append({
            "pilot_id": pilot_id,
            "source_record_id": row["source_record_id"],
            "person": row["person"],
            "person_country_or_territory": row["person_country_or_territory"],
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "English Wikipedia via MediaWiki API",
            "license_note": "Wikipedia text is generally available under CC BY-SA; retain attribution URL.",
            "wikipedia": wikipedia,
        })
        persist()
        print(f"{index}/{len(score_rows)} {pilot_id} {wikipedia['status']} {wikipedia.get('title', '')}", flush=True)
        time.sleep(args.delay)

    persist()
    summary = {
        "rows": len(results),
        "status_counts": {},
    }
    for row in results:
        status = row["wikipedia"]["status"]
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
