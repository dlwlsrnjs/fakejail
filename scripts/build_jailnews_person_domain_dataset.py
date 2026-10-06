#!/usr/bin/env python3
"""Build a sample-level person/domain view of JailNewsBench.

This is a deterministic first-pass annotation layer. It preserves every source
row, keeps noisy spaCy PERSON candidates auditable, and creates an exploded
person--sample relation table plus a domain-diverse review/experiment queue.
It does not generate new attack prompts or model outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "validation", "test")

# Only unambiguous, high-frequency surface variants are normalized here. All
# other variants remain visible for later LLM/human entity linking.
MANUAL_PERSON_ALIASES = {
    "trump": "Donald Trump",
    "donald trump": "Donald Trump",
    "biden": "Joe Biden",
    "joe biden": "Joe Biden",
    "putin": "Vladimir Putin",
    "vladimir putin": "Vladimir Putin",
    "bolsonaro": "Jair Bolsonaro",
    "jair bolsonaro": "Jair Bolsonaro",
    "macron": "Emmanuel Macron",
    "emmanuel macron": "Emmanuel Macron",
    "merkel": "Angela Merkel",
    "angela merkel": "Angela Merkel",
    "lukashenko": "Alexander Lukashenko",
    "alexander lukashenko": "Alexander Lukashenko",
    "navalny": "Alexei Navalny",
    "alexei navalny": "Alexei Navalny",
    "netanyahu": "Benjamin Netanyahu",
    "benjamin netanyahu": "Benjamin Netanyahu",
    "suga": "Yoshihide Suga",
    "yoshihide suga": "Yoshihide Suga",
    "rutte": "Mark Rutte",
    "mark rutte": "Mark Rutte",
    "obama": "Barack Obama",
    "barack obama": "Barack Obama",
    "johnson": "Boris Johnson",
    "boris johnson": "Boris Johnson",
    "prince harry": "Prince Harry",
    "harry": "Prince Harry",
    "queen elizabeth": "Elizabeth II",
    "elizabeth ii": "Elizabeth II",
    "kim jong un": "Kim Jong-un",
    "kim jong-un": "Kim Jong-un",
    "zelensky": "Volodymyr Zelenskyy",
    "zelenskyy": "Volodymyr Zelenskyy",
    "volodymyr zelensky": "Volodymyr Zelenskyy",
    "volodymyr zelenskyy": "Volodymyr Zelenskyy",
}

NOISY_PERSON_EXACT = {
    "alpha", "app store", "bitcoin", "bratislava", "busan", "cnnnews",
    "cnngrnews", "darwen", "easter", "einstein", "formula", "fukushima",
    "gazeta", "gelderland", "glasgow", "guiana", "hamilton", "hyogo prefecture",
    "janssen", "kent", "law", "lazio", "lithuanian", "majesty", "messenger",
    "moderna covid-19", "moon", "neuralink", "oscar", "parler", "rivm",
    "roscosmos", "rotterdam", "santé", "schools", "seimas", "slovak",
    "sputnik", "sputnik v", "starship", "universe", "venus", "vilnius",
    "xiaomi", "chandra x-ray observatory", "manchester united", "seznam zprávy",
}

NOISY_PERSON_TOKENS = {
    "agency", "app", "bank", "city", "company", "court", "government",
    "hospital", "institute", "ministry", "mission", "news", "observatory",
    "party", "prefecture", "radio", "school", "schools", "university",
}


DOMAIN_RULES: dict[str, list[str]] = {
    "elections_and_campaigns": [
        r"\belect(?:ion|oral|ed|ing)\w*\b", r"\bvot(?:e|er|ing)\w*\b",
        r"\bballot\w*\b", r"\bcampaign\w*\b", r"\bcandidate\w*\b",
        r"\bpoll(?:s|ing)?\b", r"\breferendum\b",
    ],
    "war_conflict_and_national_security": [
        r"\bwar\b", r"\barmed conflict\b", r"\binvas(?:ion|ion)\w*\b",
        r"\bmilitar\w*\b", r"\btroops?\b", r"\barmy\b", r"\bnavy\b",
        r"\bmissile\w*\b", r"\bnuclear\w*\b", r"\bnato\b", r"\bcoup\b",
        r"\bterroris\w*\b", r"\bnational security\b", r"\bdefen[cs]e\b",
        r"\bceasefire\b", r"\bairstrike\w*\b", r"\bbomb(?:ing|ed|s)?\b",
    ],
    "international_relations_and_diplomacy": [
        r"\bdiploma\w*\b", r"\bsummit\b", r"\btreaty\b", r"\bsanction\w*\b",
        r"\bembass\w*\b", r"\bforeign (?:policy|minister|affairs)\b",
        r"\bbilateral\b", r"\binternational relations\b", r"\btrade deal\b",
        r"\beuropean union\b", r"\bgeopolit\w*\b",
    ],
    "governance_and_leadership": [
        r"\bgovernment\b", r"\bpresident\w*\b", r"\bprime minister\b",
        r"\bminister\w*\b", r"\bparliament\w*\b", r"\bcongress\b",
        r"\bsenat(?:e|or)\w*\b", r"\bcabinet\b", r"\badministration\b",
        r"\bresign(?:ed|ation|s|ing)?\b", r"\bappoint(?:ed|ment)?\b",
        r"\bpolitic(?:s|al|ian)\w*\b", r"\bmayor\b", r"\bgovernor\b",
    ],
    "law_justice_and_corruption": [
        r"\bcourt\b", r"\bjudge\b", r"\btrial\b", r"\barrest\w*\b",
        r"\bcharg(?:e|ed|es)\b", r"\bpolice\b", r"\bprosecut\w*\b",
        r"\binvestigat\w*\b", r"\bcorrupt\w*\b", r"\bbrib\w*\b",
        r"\bcrime\b", r"\bcriminal\b", r"\bprison\b", r"\blawsuit\b",
        r"\bconvict\w*\b", r"\bfraud\b",
    ],
    "public_health_policy": [
        r"\bcovid(?:-19)?\b", r"\bcoronavirus\b", r"\bpandemic\b",
        r"\bvaccin\w*\b", r"\bpublic health\b", r"\bhospital\w*\b",
        r"\bhealth minister\b", r"\bquarantine\b", r"\blockdown\b",
        r"\bmask mandate\b", r"\binfect(?:ion|ed|ions)\b",
    ],
    "economy_trade_and_labor": [
        r"\beconom\w*\b", r"\binflation\b", r"\bunemploy\w*\b",
        r"\bjobs?\b", r"\btax(?:es|ation)?\b", r"\bbudget\b", r"\bdebt\b",
        r"\bmarket\w*\b", r"\bbank\w*\b", r"\btrade\b", r"\btariff\w*\b",
        r"\blabor\b", r"\blabour\b", r"\bwage\w*\b", r"\bbitcoin\b",
    ],
    "civil_rights_identity_and_migration": [
        r"\bhuman rights?\b", r"\bcivil rights?\b", r"\braci(?:sm|st|al)\w*\b",
        r"\bdiscriminat\w*\b", r"\bimmigra\w*\b", r"\bmigrant\w*\b",
        r"\brefugee\w*\b", r"\babortion\b", r"\blgbtq?\w*\b",
        r"\breligio\w*\b", r"\bminority\b", r"\bprotest\w*\b",
    ],
    "climate_energy_and_environment": [
        r"\bclimate\b", r"\bglobal warming\b", r"\benvironment\w*\b",
        r"\bemission\w*\b", r"\bcarbon\b", r"\brenewable\w*\b",
        r"\benergy\b", r"\boil\b", r"\bgas\b", r"\bwildfire\w*\b",
    ],
    "technology_platforms_and_information": [
        r"\bsocial media\b", r"\bfacebook\b", r"\btwitter\b", r"\binstagram\b",
        r"\btiktok\b", r"\bgoogle\b", r"\bapple\b", r"\bamazon\b",
        r"\btechnology\b", r"\btech\b", r"\bartificial intelligence\b",
        r"\bcyber\w*\b", r"\bhack\w*\b", r"\bmisinformation\b",
        r"\bdisinformation\b", r"\bfake news\b", r"\bcensor\w*\b",
    ],
    "science_space_and_infrastructure": [
        r"\bspace\b", r"\bnasa\b", r"\brocket\b", r"\bsatellite\b",
        r"\bplanet\b", r"\bmoon\b", r"\bmars\b", r"\bscientist\w*\b",
        r"\bresearch\w*\b", r"\bphysics\b", r"\binfrastructure\b",
    ],
    "royalty_and_constitutional_affairs": [
        r"\bqueen\b", r"\bking\b", r"\bprince\b", r"\bprincess\b",
        r"\broyal\w*\b", r"\bmonarch\w*\b", r"\bduke\b", r"\bduchess\b",
    ],
    "public_figure_reputation_and_scandal": [
        r"\bscandal\w*\b", r"\balleg(?:e|ed|ation|ations)\w*\b",
        r"\bcontrovers\w*\b", r"\baffair\b", r"\bleak(?:ed|s)?\b",
        r"\bapolog(?:y|ize|ised|ized)\w*\b", r"\bsecret\w*\b",
    ],
    "culture_sports_and_other": [
        r"\bfootball\b", r"\bsoccer\b", r"\bolympic\w*\b", r"\bsport\w*\b",
        r"\bfilm\b", r"\bactor\b", r"\bmusic\w*\b", r"\bcelebrity\b",
        r"\baward\w*\b", r"\bmatch\b", r"\bteam\b",
    ],
}

DOMAIN_PRIORITY = list(DOMAIN_RULES)

EVENT_RULES: dict[str, list[str]] = {
    "election_activity": [r"\belect\w*\b", r"\bvot\w*\b", r"\bballot\w*\b", r"\bcampaign\w*\b"],
    "armed_or_security_action": [r"\battack\w*\b", r"\binvas\w*\b", r"\bwar\b", r"\bmilitar\w*\b", r"\bmissile\w*\b", r"\bbomb\w*\b"],
    "legal_or_investigative_action": [r"\barrest\w*\b", r"\bcourt\b", r"\btrial\b", r"\bcharg\w*\b", r"\binvestigat\w*\b"],
    "appointment_resignation_or_succession": [r"\bappoint\w*\b", r"\bresign\w*\b", r"\bsucceed\w*\b", r"\breplac\w*\b"],
    "policy_or_government_decision": [r"\bpolicy\b", r"\blaw\b", r"\bban\w*\b", r"\bmandate\b", r"\bgovernment (?:will|plans?|announc)\w*\b"],
    "diplomatic_action": [r"\bsummit\b", r"\bsanction\w*\b", r"\btreaty\b", r"\bdiploma\w*\b", r"\bmeeting\b"],
    "public_statement_or_claim": [r"\bsaid\b", r"\bsays\b", r"\bclaim\w*\b", r"\bannounc\w*\b", r"\bwarn\w*\b"],
    "protest_or_civil_unrest": [r"\bprotest\w*\b", r"\bdemonstrat\w*\b", r"\briot\w*\b", r"\bunrest\b"],
    "health_measure_or_outcome": [r"\bvaccin\w*\b", r"\blockdown\b", r"\bquarantine\b", r"\binfect\w*\b", r"\btested positive\b"],
    "economic_action_or_outcome": [r"\btax\w*\b", r"\bbudget\b", r"\btrade\b", r"\binflation\b", r"\bmarket\w*\b"],
    "reputation_or_scandal_claim": [r"\bscandal\w*\b", r"\balleg\w*\b", r"\bcontrovers\w*\b", r"\bsecret\w*\b"],
    "death_injury_or_violence": [r"\bdied\b", r"\bdeath\b", r"\bkilled\b", r"\binjur\w*\b", r"\bshoot\w*\b"],
}

SENSITIVITY_RULES: dict[str, list[str]] = {
    "election_integrity": [r"\belect\w*\b", r"\bvot\w*\b", r"\bballot\w*\b"],
    "armed_conflict": [r"\bwar\b", r"\binvas\w*\b", r"\bmilitar\w*\b", r"\bairstrike\w*\b"],
    "national_security": [r"\bnational security\b", r"\bnuclear\w*\b", r"\bterroris\w*\b", r"\bcoup\b"],
    "criminal_or_corruption_allegation": [r"\bcorrupt\w*\b", r"\bbrib\w*\b", r"\bfraud\b", r"\bcriminal\b", r"\barrest\w*\b"],
    "reputation_or_defamation": [r"\balleg\w*\b", r"\bscandal\w*\b", r"\bfake news\b", r"\bsecret\w*\b"],
    "public_health_misinformation": [r"\bcovid\w*\b", r"\bvaccin\w*\b", r"\bpandemic\b"],
    "protected_group_or_identity": [r"\braci\w*\b", r"\breligio\w*\b", r"\bminority\b", r"\blgbtq?\w*\b"],
    "information_integrity": [r"\bmisinformation\b", r"\bdisinformation\b", r"\bfake news\b", r"\bpropaganda\b"],
    "censorship_or_surveillance": [r"\bcensor\w*\b", r"\bsurveill\w*\b", r"\bspy(?:ing)?\b"],
    "violence_or_death": [r"\bkilled\b", r"\bshoot\w*\b", r"\battack\w*\b", r"\bdeath\b"],
}

CONFLICT_RULES: dict[str, list[str]] = {
    "Russia-Ukraine": [r"\brussia\w*\b.*\bukrain\w*\b", r"\bukrain\w*\b.*\brussia\w*\b", r"\bdonbas\b", r"\bcrimea\b"],
    "Israel-Palestine": [r"\bisrael\w*\b.*\bpalestin\w*\b", r"\bpalestin\w*\b.*\bisrael\w*\b", r"\bgaza\b"],
    "Afghanistan": [r"\bafghanistan\b", r"\btaliban\b"],
    "Syria": [r"\bsyria\w*\b"],
    "Nagorno-Karabakh": [r"\bnagorno[- ]karabakh\b", r"\bkarabakh\b"],
    "Iraq": [r"\biraq\w*\b"],
}

ROLE_RULES: dict[str, list[str]] = {
    "head_of_state_or_government": [
        r"\bpresident\b", r"\bprime minister\b", r"\bchancellor\b", r"\bpremier\b",
    ],
    "minister_or_senior_official": [
        r"\bminister\b", r"\bsecretary of state\b", r"\bgovernment official\b",
        r"\bcommissioner\b", r"\bgovernor\b", r"\bmayor\b",
    ],
    "legislator_candidate_or_party_leader": [
        r"\bsenator\b", r"\bmember of parliament\b", r"\bcongress(?:man|woman)?\b",
        r"\bcandidate\b", r"\bparty leader\b", r"\bopposition leader\b",
    ],
    "royalty": [r"\bqueen\b", r"\bking\b", r"\bprince\b", r"\bprincess\b", r"\bduke\b", r"\bduchess\b"],
    "business_or_technology_leader": [
        r"\bceo\b", r"\bfounder\b", r"\bentrepreneur\b", r"\bbillionaire\b",
        r"\bexecutive\b", r"\bbusinessman\b", r"\bbusinesswoman\b",
    ],
    "health_science_or_academic_expert": [
        r"\bdoctor\b", r"\bdr\.\b", r"\bscientist\b", r"\bprofessor\b",
        r"\bresearcher\b", r"\bepidemiologist\b", r"\bphysician\b",
    ],
    "activist_or_opposition_figure": [
        r"\bactivist\b", r"\bdissident\b", r"\bopposition figure\b", r"\bcampaigner\b",
    ],
    "military_security_or_legal_actor": [
        r"\bgeneral\b", r"\bcommander\b", r"\bpolice officer\b", r"\bjudge\b",
        r"\bprosecutor\b", r"\blawyer\b", r"\battorney\b",
    ],
    "journalist_or_media_figure": [r"\bjournalist\b", r"\breporter\b", r"\banchor\b", r"\beditor\b"],
    "athlete_entertainer_or_cultural_figure": [
        r"\bactor\b", r"\bactress\b", r"\bsinger\b", r"\bmusician\b",
        r"\bfootballer\b", r"\bathlete\b", r"\bplayer\b", r"\bartist\b",
    ],
}

GEOGRAPHY_ALIASES: dict[str, list[str]] = {
    "United States": ["united states", "u.s.", "usa", "american"],
    "United Kingdom": ["united kingdom", "u.k.", "britain", "british", "england"],
    "Russia": ["russia", "russian", "moscow"],
    "Ukraine": ["ukraine", "ukrainian", "kyiv", "kiev"],
    "China": ["china", "chinese", "beijing"],
    "Taiwan": ["taiwan", "taiwanese", "taipei"],
    "Germany": ["germany", "german", "berlin"],
    "France": ["france", "french", "paris"],
    "Brazil": ["brazil", "brazilian", "brasília"],
    "India": ["india", "indian", "delhi"],
    "Israel": ["israel", "israeli"],
    "Palestine": ["palestine", "palestinian", "gaza"],
    "Belarus": ["belarus", "belarusian", "minsk"],
    "Japan": ["japan", "japanese", "tokyo"],
    "South Korea": ["south korea", "south korean", "seoul"],
    "North Korea": ["north korea", "north korean", "pyongyang"],
    "European Union": ["european union", "eu ", "e.u."],
    "Afghanistan": ["afghanistan", "afghan", "kabul"],
    "Syria": ["syria", "syrian", "damascus"],
    "Iran": ["iran", "iranian", "tehran"],
    "Iraq": ["iraq", "iraqi", "baghdad"],
    "Turkey": ["turkey", "turkish", "ankara"],
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_person_surface(value: str) -> str:
    value = " ".join(value.split()).strip(" \t\n,.;:()[]{}\"'“”‘’")
    value = re.sub(r"(?:['’]s)$", "", value, flags=re.IGNORECASE)
    return " ".join(value.split())


def person_quality(value: str) -> tuple[str, list[str]]:
    normalized = normalize_person_surface(value)
    folded = normalized.casefold()
    reasons: list[str] = []
    if folded in NOISY_PERSON_EXACT:
        return "rejected_noise", ["known_ner_false_positive"]
    tokens = re.findall(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", normalized, re.UNICODE)
    if not tokens or any(token.casefold() in NOISY_PERSON_TOKENS for token in tokens):
        return "rejected_noise", ["organization_place_or_object_token"]
    if any(ch.isdigit() for ch in normalized) or len(tokens) > 7:
        return "rejected_noise", ["implausible_person_surface"]
    if folded in MANUAL_PERSON_ALIASES:
        reasons.append("curated_alias")
        return "high", reasons
    if len(tokens) >= 2:
        title_like = sum(token[:1].isupper() for token in tokens) / len(tokens)
        if title_like >= 0.67:
            return "high", ["multi_token_name_like"]
        return "medium", ["multi_token_uncertain_casing"]
    return "low", ["single_token_ambiguous"]


def canonical_person(value: str) -> tuple[str, str]:
    normalized = normalize_person_surface(value)
    alias = MANUAL_PERSON_ALIASES.get(normalized.casefold())
    if alias:
        return alias, "curated_alias_v1"
    return normalized, "surface_normalization_only"


def person_context_features(article_en: str, mention: dict[str, Any]) -> dict[str, Any]:
    """Extract conservative role evidence from a local context window."""
    start = int(mention.get("start_char", 0) or 0)
    end = int(mention.get("end_char", start) or start)
    left = max(0, start - 90)
    right = min(len(article_en), end + 90)
    context = article_en[left:right]
    role_hits = {
        role: [pattern for pattern in patterns if re.search(pattern, context, re.IGNORECASE)]
        for role, patterns in ROLE_RULES.items()
    }
    role_hits = {role: patterns for role, patterns in role_hits.items() if patterns}
    role = next(iter(role_hits), "unknown_or_not_explicit")
    confidence = "medium" if role_hits else "low"
    return {
        "person_role_category": role,
        "person_role_evidence": list(role_hits),
        "person_role_confidence": confidence,
        "person_context_window_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
    }


def matches(text: str, patterns: list[str]) -> int:
    return sum(bool(re.search(pattern, text, re.IGNORECASE | re.DOTALL)) for pattern in patterns)


def classify_text(text: str) -> dict[str, Any]:
    scores = {domain: matches(text, patterns) for domain, patterns in DOMAIN_RULES.items()}
    matched_domains = [domain for domain in DOMAIN_PRIORITY if scores[domain] > 0]
    if matched_domains:
        primary = max(matched_domains, key=lambda name: (scores[name], -DOMAIN_PRIORITY.index(name)))
        confidence = "high" if scores[primary] >= 2 else "medium"
    else:
        primary = "unclear"
        confidence = "low"

    events = [name for name, patterns in EVENT_RULES.items() if matches(text, patterns)]
    sensitivities = [name for name, patterns in SENSITIVITY_RULES.items() if matches(text, patterns)]
    conflicts = [name for name, patterns in CONFLICT_RULES.items() if matches(text, patterns)]
    geographies = []
    folded = text.casefold()
    for geography, aliases in GEOGRAPHY_ALIASES.items():
        if any(alias.casefold() in folded for alias in aliases):
            geographies.append(geography)
    return {
        "political_domain": primary,
        "political_domain_all": matched_domains,
        "political_domain_scores": {key: value for key, value in scores.items() if value},
        "political_domain_confidence": confidence,
        "event_types": events or ["other_or_unclear"],
        "sensitive_concepts": sensitivities,
        "conflict_related": bool(conflicts),
        "conflicts_detected": conflicts,
        "election_related": "election_integrity" in sensitivities,
        "war_or_security_related": any(item in sensitivities for item in ("armed_conflict", "national_security")),
        "geographies_detected": geographies,
        "annotation_method": "deterministic_english_keyword_rules_v1",
        "annotation_review_status": "automatic_needs_llm_or_human_review",
    }


def annotate_row(row: dict[str, Any]) -> dict[str, Any]:
    people = []
    article_en = str(row.get("article_en") or "")
    for mention in row.get("person_entities", []):
        surface = str(mention.get("text", ""))
        canonical, method = canonical_person(surface)
        quality, reasons = person_quality(surface)
        people.append({
            **mention,
            "surface_normalized": normalize_person_surface(surface),
            "canonical_person": canonical,
            "canonicalization_method": method,
            "candidate_quality": quality,
            "quality_reasons": reasons,
            "entity_link_review_status": "automatic_needs_llm_or_human_review",
            **person_context_features(article_en, mention),
        })
    quality_rank = {"high": 3, "medium": 2, "low": 1, "rejected_noise": 0}
    usable = [person for person in people if quality_rank[person["candidate_quality"]] > 0]
    primary = max(usable, key=lambda p: quality_rank[p["candidate_quality"]], default=None)
    article_hash = hashlib.sha256(article_en.strip().casefold().encode("utf-8")).hexdigest()
    years = sorted(set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", article_en)))
    word_count = len(re.findall(r"\b\w+\b", article_en, re.UNICODE))
    instruction = str(row.get("seed_instruction_local") or "")
    return {
        **row,
        "sample_id": row["collection_id"],
        "article_en_sha256": article_hash,
        "person_candidates": people,
        "person_candidate_count_raw": len(people),
        "person_candidate_count_usable": len(usable),
        "primary_person": primary["canonical_person"] if primary else "",
        "primary_person_quality": primary["candidate_quality"] if primary else "rejected_or_missing",
        "primary_person_role_category": primary["person_role_category"] if primary else "unknown_or_not_explicit",
        "primary_person_role_confidence": primary["person_role_confidence"] if primary else "low",
        "person_resolution_status": "automatic_candidates_need_review",
        "article_en_word_count": word_count,
        "instruction_local_word_count": len(re.findall(r"\b\w+\b", instruction, re.UNICODE)),
        "time_years_detected": years,
        **classify_text(article_en),
    }


def choose_domain_diverse(rows: list[dict[str, Any]], max_per_person: int) -> list[dict[str, Any]]:
    """Greedily select rows that add domain, split, language, and conflict coverage."""
    remaining = sorted(rows, key=lambda row: row["sample_id"])
    selected: list[dict[str, Any]] = []
    seen_domains: set[str] = set()
    seen_splits: set[str] = set()
    seen_languages: set[str] = set()
    seen_conflicts: set[str] = set()
    while remaining and len(selected) < max_per_person:
        def novelty(row: dict[str, Any]) -> tuple[int, int, int, int, str]:
            return (
                int(row["political_domain"] not in seen_domains),
                sum(item not in seen_conflicts for item in row["conflicts_detected"]),
                int(row["source_split"] not in seen_splits),
                int(row["language_code"] not in seen_languages),
                row["sample_id"],
            )
        picked = max(remaining, key=novelty)
        remaining.remove(picked)
        selected.append(picked)
        seen_domains.add(picked["political_domain"])
        seen_splits.add(picked["source_split"])
        seen_languages.add(picked["language_code"])
        seen_conflicts.update(picked["conflicts_detected"])
    return selected


def write_index_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id", "source_split", "source_record_id", "region_en", "language_code",
        "primary_person", "primary_person_quality", "person_candidate_count_usable",
        "primary_person_role_category", "primary_person_role_confidence",
        "political_domain", "political_domain_confidence", "election_related",
        "war_or_security_related", "conflict_related", "conflicts_detected",
        "event_types", "sensitive_concepts", "geographies_detected", "article_en_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = {field: row.get(field, "") for field in fields}
            for field, value in list(item.items()):
                if isinstance(value, list):
                    item[field] = json.dumps(value, ensure_ascii=False)
            writer.writerow(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("data/jailnewsbench_person_views"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/jailnewsbench_person_domain_20260930"))
    parser.add_argument("--min-samples-per-person", type=int, default=3)
    parser.add_argument("--max-samples-per-person", type=int, default=20)
    parser.add_argument("--max-people", type=int, default=500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotated: list[dict[str, Any]] = []
    for split in SPLITS:
        source = args.input_root / split / "person_records_political.jsonl"
        annotated.extend(annotate_row(row) for row in read_jsonl(source))

    # Mark exact cross-split/content duplicates without dropping them.
    hash_counts = Counter(row["article_en_sha256"] for row in annotated)
    for row in annotated:
        row["article_en_duplicate_count"] = hash_counts[row["article_en_sha256"]]
        row["article_en_is_duplicate"] = hash_counts[row["article_en_sha256"]] > 1

    write_jsonl(args.output_dir / "expanded_samples.jsonl", annotated)
    write_index_csv(args.output_dir / "expanded_samples_index.csv", annotated)

    relations: list[dict[str, Any]] = []
    for row in annotated:
        seen_people: set[str] = set()
        for person in row["person_candidates"]:
            if person["candidate_quality"] not in {"high", "medium"}:
                continue
            canonical = person["canonical_person"]
            if canonical.casefold() in seen_people:
                continue
            seen_people.add(canonical.casefold())
            relations.append({
                "relation_id": f"{row['sample_id']}::person::{hashlib.sha256(canonical.casefold().encode()).hexdigest()[:12]}",
                "sample_id": row["sample_id"],
                "canonical_person": canonical,
                "person_surface": person["text"],
                "person_candidate_quality": person["candidate_quality"],
                "canonicalization_method": person["canonicalization_method"],
                "person_role_category": person["person_role_category"],
                "person_role_evidence": person["person_role_evidence"],
                "person_role_confidence": person["person_role_confidence"],
                "source_split": row["source_split"],
                "source_record_id": row["source_record_id"],
                "region_en": row["region_en"],
                "language_code": row["language_code"],
                "political_domain": row["political_domain"],
                "political_domain_all": row["political_domain_all"],
                "event_types": row["event_types"],
                "sensitive_concepts": row["sensitive_concepts"],
                "election_related": row["election_related"],
                "war_or_security_related": row["war_or_security_related"],
                "conflict_related": row["conflict_related"],
                "conflicts_detected": row["conflicts_detected"],
                "geographies_detected": row["geographies_detected"],
                "person_country_or_territory_candidates": row["geographies_detected"],
                "person_country_inference_status": "article_context_only_not_entity_linked",
                "time_years_detected": row["time_years_detected"],
                "article_en_sha256": row["article_en_sha256"],
                "entity_link_review_status": person["entity_link_review_status"],
            })
    write_jsonl(args.output_dir / "person_sample_relations.jsonl", relations)

    sample_by_id = {row["sample_id"]: row for row in annotated}
    by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relation in relations:
        by_person[relation["canonical_person"]].append(sample_by_id[relation["sample_id"]])

    profiles = []
    for person_name, rows in by_person.items():
        profiles.append({
            "canonical_person": person_name,
            "sample_count": len(rows),
            "unique_article_count": len({row["article_en_sha256"] for row in rows}),
            "split_counts": dict(Counter(row["source_split"] for row in rows).most_common()),
            "language_counts": dict(Counter(row["language_code"] for row in rows).most_common()),
            "region_counts": dict(Counter(row["region_en"] for row in rows).most_common()),
            "political_domain_counts": dict(Counter(row["political_domain"] for row in rows).most_common()),
            "person_role_counts": dict(Counter(
                candidate["person_role_category"]
                for row in rows for candidate in row["person_candidates"]
                if candidate["canonical_person"] == person_name
            ).most_common()),
            "event_type_counts": dict(Counter(event for row in rows for event in row["event_types"]).most_common()),
            "sensitive_concept_counts": dict(Counter(item for row in rows for item in row["sensitive_concepts"]).most_common()),
            "conflict_counts": dict(Counter(item for row in rows for item in row["conflicts_detected"]).most_common()),
            "election_sample_count": sum(row["election_related"] for row in rows),
            "war_or_security_sample_count": sum(row["war_or_security_related"] for row in rows),
            "entity_link_review_status": "automatic_needs_llm_or_human_review",
        })
    profiles.sort(key=lambda row: (-row["sample_count"], row["canonical_person"]))
    write_jsonl(args.output_dir / "person_profiles.jsonl", profiles)

    eligible_profiles = [
        profile for profile in profiles
        if profile["sample_count"] >= args.min_samples_per_person
    ][: args.max_people]
    queue: list[dict[str, Any]] = []
    for profile in eligible_profiles:
        person = profile["canonical_person"]
        picked = choose_domain_diverse(by_person[person], args.max_samples_per_person)
        for rank, row in enumerate(picked, 1):
            relation = next(
                rel for rel in relations
                if rel["sample_id"] == row["sample_id"] and rel["canonical_person"] == person
            )
            queue.append({
                **relation,
                "person_sample_rank": rank,
                "person_total_sample_count": profile["sample_count"],
                "selection_method": "greedy_domain_split_language_conflict_diversity_v1",
                "review_tasks": [
                    "verify_person_identity_and_alias",
                    "verify_primary_and_secondary_domains",
                    "verify_event_and_sensitive_concepts",
                    "add_person_role_country_and_time_context",
                ],
            })
    write_jsonl(args.output_dir / "person_domain_balanced_queue.jsonl", queue)

    summary = {
        "source_rows": len(annotated),
        "source_split_counts": dict(Counter(row["source_split"] for row in annotated)),
        "raw_person_mentions": sum(row["person_candidate_count_raw"] for row in annotated),
        "usable_person_mentions": len(relations),
        "distinct_canonical_person_candidates": len(profiles),
        "person_candidates_with_at_least_3_samples": sum(profile["sample_count"] >= 3 for profile in profiles),
        "person_candidates_with_at_least_10_samples": sum(profile["sample_count"] >= 10 for profile in profiles),
        "domain_counts": dict(Counter(row["political_domain"] for row in annotated).most_common()),
        "election_related_rows": sum(row["election_related"] for row in annotated),
        "war_or_security_related_rows": sum(row["war_or_security_related"] for row in annotated),
        "conflict_related_rows": sum(row["conflict_related"] for row in annotated),
        "duplicate_article_rows": sum(row["article_en_is_duplicate"] for row in annotated),
        "balanced_queue_rows": len(queue),
        "balanced_queue_people": len(eligible_profiles),
        "caveats": [
            "Person candidates originate from spaCy NER and remain noisy until entity-link review.",
            "Domains and concepts are deterministic keyword labels for stratification, not publication-grade annotations.",
            "No translated attack prompt or model response is generated by this builder.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        "# JailNewsBench person/domain expansion\n\n"
        "This directory is a deterministic sample-level annotation layer over the political, "
        "person-centered JailNewsBench views. `expanded_samples.jsonl` is the full 24k-row "
        "table; `person_sample_relations.jsonl` explodes samples by usable person candidate; "
        "`person_profiles.jsonl` aggregates per-person coverage; and "
        "`person_domain_balanced_queue.jsonl` is a domain-diverse entity/domain review and "
        "future multilingual-experiment queue. All person linking and topic labels remain "
        "automatic and require Luna/human verification before analysis.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
