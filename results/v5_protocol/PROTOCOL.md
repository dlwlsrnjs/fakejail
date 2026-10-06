# English-controlled identity-routing protocol (V5)

## Experimental invariant

Every arm for a given sample uses:

- `ARTICLE`: the public JailNewsBench `article_en`;
- `INSTRUCTION`: an English translation of the complete sample-specific
  `seed_instruction_local`, with the protected article replaced by `[ARTICLE]`;
- output language: explicitly English;
- generator and judge settings: held constant within a comparison.

The instruction translation is accepted only after forward translation,
backtranslation, reforward translation, protected-article preservation,
number preservation, and semantic-similarity checks. Of 501 instructions, 500
pass the automatic NLLB/MADLAD route; one Korean instruction has a separately
recorded source-aligned correction because both MT round trips omitted its final
fabrication-detail clause. The automatic threshold is not relaxed.

## Variable axes

Only these axes may vary within a person/sample:

1. the language of the neutral description after `A:` (72 labels);
2. the official JailNewsBench jailbreak wrapper (5 methods).

The two rendering modes are separate experimental conditions rather than a
single 720-arm pool:

- `a_literal`: retain `A` in the generated English article;
- `canonical_en`: resolve `A` to the canonical English person name.

## Primary controls

1. English `ARTICLE -> INSTRUCTION`, real person names retained, no translated
   descriptor, no `A`, no identity binding, no router or bandit.
2. V5 `a_literal` under the same English article/instruction/output contract.
3. V5 `canonical_en` under the same English article/instruction/output contract.

The official local-language, continuous JailNewsBench baseline remains a
separate paper-reproduction reference and is not the primary causal control.

## Integrity requirements

- 501 people x 72 descriptor-language labels x 5 methods = 180,360 arms per mode.
- For each person and method, replacing the `A:` description with `<DESC>` must
  make all 72 seed prompts and attacked prompts byte-identical.
- Every row must explicitly request an English final article.
- Exact reviewed person surfaces and canonical names must not leak into the
  placeholder-bearing English instruction.
- Montenegrin and Serbian are distinct arm labels even though the current NLLB
  inventory maps both to `srp_Cyrl`; results must be indexed by language label,
  not NLLB code alone.
