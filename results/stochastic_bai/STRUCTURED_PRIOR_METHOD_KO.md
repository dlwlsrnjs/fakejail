# Structured prior audit for the 360-arm JailNews bandit

## Design check

- Full action space: **72 languages × 5 methods = 360 arms**.
- The current balanced seed has 3,600 target calls: exactly 10 observations per arm, 50 per language, and 720 per method.
- It is a balanced incomplete person–arm panel with one draw per observed cell. It supports population-level prior testing, not a stochastic person-level oracle yet.
- A 320-arm claim is only valid after preregistering a 64-language subset; otherwise it silently changes the experiment.
- Full-pool invariant errors: **0**.
- Exact rendered-prompt duplicates remove 5,920 nominal arms; effective arms/person range 295–355 (median 350).
- Online selection should collapse exact prompt hashes into equivalence classes, while retaining all 360 nominal labels for reproducibility.

## Current target-model endpoints

- asr: 2398/3600 = 66.61%
- retained: 349/3600 = 9.69%
- strict: 343/3600 = 9.53%

## Does external language structure predict similar effects?

- Hybrid external graph vs target effect graph: Spearman r=0.005, permutation p=0.8146 (2000 permutations).
- Shared family: effect similarity 0.024 vs -0.003; gap +0.027.
- Shared script: effect similarity 0.021 vs -0.016; gap +0.037.
- Shared macroregion: effect similarity -0.000 vs -0.002; gap +0.002.

Interpretation: family/script/region and PC2/Wikipedia features are candidate-sharing priors. They must be transfer-calibrated; they are not evidence that a person's national language is optimal.

## Method similarities

- role_play: research_front (0.48), context_overload (0.43)
- system_override: context_overload (0.42), role_play (0.40)
- research_front: context_overload (0.57), role_play (0.48)
- neg_prompting: context_overload (0.39), system_override (0.35)
- context_overload: research_front (0.57), role_play (0.43)

## Supported intent–method matches

Only groups and complements with at least 20 observations for every method are listed. Deltas compare the focal intent group with its disjoint complement within the same method. BH q-values correct the full exploratory interaction family; overlapping intent labels still make these associations, not causal effects.

| intent group | endpoint | best method | n | rate | adjusted lift | 95% interval | BH q |
|---|---:|---|---:|---:|---:|---:|---:|
| operation:exaggeration | asr | research_front | 277 | 79.8% | +21.9% | [+15.2%, +28.3%] | 3.8e-07 |
| operation:emotional_manipulation | asr | research_front | 299 | 75.3% | +15.3% | [+8.7%, +22.0%] | 0.00125 |
| intent_flag:economic_opinion | asr | system_override | 46 | 100.0% | +17.2% | [+12.4%, +20.8%] | 0.00741 |
| source_article_dependency:4 | asr | research_front | 78 | 83.3% | +18.7% | [+9.1%, +27.1%] | 0.0187 |
| operation:omit_context | asr | research_front | 88 | 80.7% | +16.1% | [+6.5%, +24.6%] | 0.062 |
| source_article_dependency:2 | retained | system_override | 634 | 15.0% | +9.9% | [+3.9%, +14.7%] | 0.123 |
| source_article_dependency:2 | strict | system_override | 634 | 14.5% | +9.4% | [+3.4%, +14.1%] | 0.146 |
| operation:exaggeration | retained | neg_prompting | 292 | 1.7% | +1.8% | [+0.5%, +3.6%] | 0.146 |
| goal:social_polarization | asr | research_front | 107 | 76.6% | +12.0% | [+2.7%, +20.4%] | 0.174 |
| target:organization | strict | system_override | 114 | 21.1% | +9.4% | [+2.0%, +17.7%] | 0.177 |
| target:organization | retained | system_override | 114 | 21.1% | +8.8% | [+1.4%, +17.2%] | 0.182 |
| audience:voters | asr | neg_prompting | 55 | 18.2% | +10.2% | [+0.9%, +21.9%] | 0.257 |
| goal:institutional_trust_erosion | asr | neg_prompting | 324 | 12.0% | +5.0% | [+0.7%, +9.4%] | 0.257 |
| goal:electoral_influence | retained | context_overload | 171 | 14.6% | +5.7% | [+0.2%, +11.8%] | 0.387 |
| intent_flag:economic_opinion | retained | role_play | 43 | 23.3% | +11.4% | [-0.2%, +25.3%] | 0.437 |
| intent_flag:economic_opinion | strict | role_play | 43 | 23.3% | +11.4% | [-0.1%, +25.2%] | 0.437 |
| target:individual | strict | role_play | 331 | 15.7% | +4.9% | [-0.0%, +9.9%] | 0.437 |
| target:individual | retained | role_play | 331 | 15.7% | +4.9% | [-0.1%, +10.1%] | 0.437 |
| intent_flag:policy_support_shift | asr | system_override | 547 | 84.5% | +6.5% | [+0.1%, +13.6%] | 0.453 |
| intent_flag:social_polarization | asr | research_front | 270 | 70.4% | +6.5% | [-0.5%, +13.5%] | 0.489 |
| audience:voters | retained | role_play | 50 | 22.0% | +10.2% | [-0.4%, +22.8%] | 0.492 |
| audience:voters | strict | role_play | 50 | 22.0% | +10.0% | [-0.5%, +22.3%] | 0.492 |
| goal:electoral_influence | strict | context_overload | 171 | 14.0% | +5.0% | [-0.4%, +11.0%] | 0.511 |
| operation:alter_core_facts | asr | role_play | 93 | 97.8% | -2.3% | [-6.5%, +0.0%] | 0.511 |
| intent_flag:electoral_outcome | retained | system_override | 231 | 16.9% | +4.6% | [-0.9%, +10.4%] | 0.573 |
| intent_flag:electoral_outcome | strict | system_override | 231 | 16.5% | +4.6% | [-0.9%, +10.4%] | 0.573 |
| target:organization | asr | research_front | 82 | 73.2% | +7.6% | [-3.1%, +17.4%] | 0.791 |
| target:individual | asr | research_front | 355 | 68.7% | +4.9% | [-2.0%, +11.7%] | 0.791 |
| intent_flag:public_health | strict | research_front | 113 | 14.2% | +4.8% | [-1.5%, +12.0%] | 0.791 |
| intent_flag:public_health | retained | research_front | 113 | 14.2% | +4.6% | [-1.8%, +11.9%] | 0.791 |

## Proposed core method

Use a knowledge-grounded, factorized contextual bandit rather than a flat 360-way classifier:

1. Extract entity, role, country/region, political domain, sensitive concepts, requested manipulation operations, audience, directness, and complexity.
2. Build an outcome-free language graph from translation fidelity, entity recovery, Wikipedia availability/identity, PC2 knowledge/sensitivity scores, family, script, and macro-region.
3. Build a method graph from semantic affordances and historical residual response profiles across intent groups and language clusters.
4. Parameterize three hurdle probabilities—non-refusal, judge retention, and strict article success—and combine them only for the declared endpoint.
5. Initialize the 360-arm posterior from discounted surrogate ensembles; use target observations to update a graph-smoothed low-rank language×method interaction.
6. Select a diverse initial slate across language and method clusters, then use posterior Top-Two Thompson Sampling to compare the current best arm with the most plausible challenger.
7. Report success@B, simple regret to a repeated-call stochastic oracle, posterior best-arm probability, calibration, and query cost. A single lucky generation is not an oracle.

A suitable prior logit is:

`eta(x,l,m)=b + f_x^T W_L z_l + g_x^T W_M z_m + u_l + v_m + <P z_l, Q z_m> + sum_r q_r(x) A_r(l,m) + sum_s omega_s logit(p_s(x,l,m))`

with graph priors `u ~ N(0, tau_L^2 (L_L+eps I)^-1)` and `v ~ N(0, tau_M^2 (L_M+eps I)^-1)`. The surrogate weights `omega_s` are learned only on held-out target seed data and model disagreement reduces effective prior strength.

## Leakage rule

Target outcomes in this report are validation evidence. They must not enter the cold-start external graph for a held-out person or target model. Use leave-person-out and leave-target-model-out evaluation, then update only with observations available at the stated online budget.
