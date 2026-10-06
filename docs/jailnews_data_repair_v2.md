# JailNews entity/translation repair v2

## Contract

- Preserve the 501 dataset labels, but deduplicate entities only by Wikidata QID.
- An unresolved label remains its own `unresolved:<person_id>` entity and is never merged by spelling.
- Person redaction starts from `analysis_ready_samples.jsonl:luna_people[].surface`; a canonical label is never searched globally in article text.
- A canonical continuation is allowed only at the same Luna-surface start position when the full continuation is literally present. This repairs truncated reviewed spans such as `Ursula von der` and `Lee Jun-` while retaining an audit trail.
- `roundtrip_valid` is an automatic QA feature. `usable_for_rendering` is the authoritative final gate. Compatibility field `valid` is equal to `usable_for_rendering` in v2.

## Final entity and prompt artifacts

- `artifacts/jailnews_bandit_20260930/runtime/entity_links_v2_final/label_entity_map.jsonl`
  - 501 label rows.
- `artifacts/jailnews_bandit_20260930/runtime/entity_links_v2_final/entities_deduplicated_by_qid.jsonl`
  - 462 current entities: 451 QID-backed and 11 explicitly unresolved.
- `artifacts/jailnews_bandit_20260930/runtime/entity_links_v2_final/person_localizations.jsonl`
  - 36,072 person-label/language rows.
- `artifacts/jailnews_bandit_20260930/runtime/entity_links_v2_final/needs_review_30_adjudicated.jsonl`
  - all 30 old review rows; 20 have a corrected/confirmed QID and 10 unsafe candidates remain explicitly unresolved.
- `artifacts/jailnews_bandit_20260930/runtime/entity_links_v2_final/failed_fetch_retry_audit.jsonl`
  - all 80 old 429/fetch failures; 79 were recovered and one remains unresolved.
- `artifacts/jailnews_bandit_20260930/runtime/entity_links_v2_final/qid_corrections.jsonl`
  - 37 changed or rejected previous QIDs.
- `data/jailnewsbench_person_domain_20260930/a_middle_surface_v2/person_cases.jsonl`
  - 4,091 prepared `ARTICLE -> A -> INSTRUCTION` cases; zero preparation failures.
- `data/jailnewsbench_person_domain_20260930/concept_translations_v2/translations_verified_v2.jsonl`
  - 303,480 final rows with zero empty translations and zero `valid != usable_for_rendering` conflicts.
- `artifacts/jailnews_bandit_20260930/runtime/catalog_v2.sqlite`
  - validated final catalog: 4,091 contexts, 462 QID-deduplicated/unresolved entities, 303,480 translation pairs, 360 arms.
- `artifacts/jailnews_bandit_20260930/runtime/validation_report_v2.json`
  - final machine validation report with `passed: true`.

The context-free Wikidata retry deliberately sends only dataset person labels
and known QIDs. It does not transmit article text. Exact matches recovered by
the retry are pinned as audited QID overrides so future search ranking changes
cannot silently change an identity. Two false exact-name matches were corrected:
Scott Kelly is the astronaut (`Q362190`), and Prince Edward is the Duke of
Edinburgh (`Q154920`). Ryo Nakamura and Ladislav Dušek remain unresolved because
the exact-name search results were different people.

## Completed translation chain

1. Slurm `29738`: retry two OOM translation shards with MADLAD-400-10B batch 16.
2. Slurm `29740`: construct eight deterministic semantic-review shards.
3. Slurm `29773` and `29778`: reviewed all 74,136 old `needs_human_review` rows with eight local Qwen2.5-32B replicas on eight RTX6000 GPUs. Existing per-shard checkpoints were resumed.
4. Slurm `29784`: normalized recoverable JSON formatting mistakes and re-queried only genuinely malformed judgments. All eight shards finished with zero invalid judgments.
5. Slurm `29785`: coverage-checked, fail-closed merge; regenerated `catalog_v2.sqlite`; rebuilt surface cases; validated successfully.

The three empty translations are included in the same 74,136-row repair inventory.
MADLAD-400-10B produced non-empty target text and backtranslation for all three;
they become usable only if the Qwen semantic review accepts them.

Every valid negative Qwen decision is stored as `rejected_local_qwen32` rather
than being returned to `needs_human_review`. Malformed judge output is retried;
the final merge refuses to run if any `review_invalid_local_qwen32` remains.
Therefore the final output has neither ambiguous pending reviews nor malformed
judge results while preserving the reason each semantic rejection was excluded.

## Final verified counts

- Concept translations: 303,480 rows and 303,480 unique IDs.
- Qwen adjudication of the old review queue: 39,186 accepted and 34,950 rejected; zero rows remain in `needs_human_review`.
- Original Luna-verified rows: 35,971. Provisional automatic rows: 193,373.
- Empty translations: zero. `valid != usable_for_rendering` conflicts: zero.
- Dataset labels: 501. QID-backed entities: 451 unique QIDs. Explicit unresolved entities: 11. QID-deduplicated entity total: 462.
- Person localization pairs: 36,072 unique rows (501 × 72).
- Surface-derived prompt cases: 4,091, with zero preparation failures.
- Catalog: 4,091 contexts, 462 persons/entities, 72 languages, five methods, 360 arms, and 303,480 concept translation rows.
- Machine-readable audit and SHA-256 digests: `artifacts/jailnews_bandit_20260930/runtime/final_repair_v2_audit.json`.

## Remaining unresolved labels

Eleven labels remain explicit unresolved entities. This is preferable to
linking a byline, namesake, spouse, event, or disambiguation page. No private
article-context text was sent to Wikidata during the retry.
