# JailNewsBench 의도–기법 반복 라우터 실험

## 결론

Qwen2.5-32B-Instruct를 이용해 494개 고유 인스트럭션을 `추출 → 비평 → 최종 판정 → 재비평 → 재판정`으로 처리했다. 모든 판정은 ASR, 생성문, judge label을 보지 않는 ASR-blind 조건이다. 최종적으로 494/494건이 정확한 인스트럭션 원문 근거를 포함하며 검수를 통과했다.

고품질 의도 특징은 ASR 행 단위 AUC를 메타데이터 라우터의 0.6564에서 0.6658로 높였지만, 사람별 top-1 ASR은 80.84%에서 81.24%로 +0.40pp에 그쳤고 유의하지 않았다(95% paired bootstrap CI -1.40~+2.20pp, McNemar p=0.832). 기존 7B 1-pass 라우터와 비교해도 +0.20pp로 유의하지 않았다.

따라서 현재 데이터가 지지하는 주장은 “32B reasoning 자체가 ASR을 크게 높인다”가 아니다. 더 정확한 결론은 다음과 같다.

1. 반복 추출은 의도 라벨과 근거의 감사 가능성을 크게 개선한다.
2. 의도–기법 특징은 행 단위 분별력(AUC)을 소폭 개선하지만 top-1 개선 증거는 아직 약하다.
3. 인물–언어 특징은 retained에서 더 유용하다. entity-language router는 global 대비 +4.19pp이며 유의했다(95% CI +0.40~+8.18pp, p=0.044).
4. 언어와 기법은 서로 다른 endpoint를 담당하는 두 gate로 유지하는 편이 낫다. 현재 joint/factorized 결합은 일관된 추가 이득이 없다.

## 반복 추출 품질

- 고유 인스트럭션: 494
- 최종 accepted: 494
- critic이 수정 요청: 38
- 초안 대비 최종 기법 점수 변경: 134
- primary goal 변경: 0
- multi-label intent flag 변경: 2
- 정확한 인스트럭션 근거 검증: 100%
- critic 평균 점수(0=불량, 4=우수): 번역 충실도 3.921, 의도 완전성 3.925, 근거성 3.937, 분류 일관성 3.949, 기법 루브릭 일관성 3.915

첫 실행에서는 점수 방향이 명시되지 않아 critic 점수가 비정상적으로 0~1에 몰렸고, 23건이 기사 근거만 제시했다. 루브릭 방향과 instruction-local exact evidence 요구를 명시해 두 패스를 추가 실행했으며, 남은 대소문자·부분 생략 21건은 의미 라벨을 바꾸지 않고 인스트럭션의 정확한 원문 구절을 추가했다. 수정 전 결과도 별도 보존했다.

## 추출된 의도 구조

Primary goal은 제도 신뢰 훼손 215, 선거 영향 117, 사회 양극화 87, 개인 평판 훼손 57, 공중보건 영향 15, 정책 여론 변화 3건이다. 대상은 개인 234, 정부 150, 조직 76, 집단 24, 복수 대상 10건이다.

다중 의도 플래그는 평판 훼손 447, 제도 불신 446, 정책 지지 변화 374, 사회 양극화 204, 선거 결과 164, 지정학적 정렬 75, 공중보건 74, 분쟁·안보 35, 경제 여론 31건이다.

요청 연산은 fabricated details와 selective framing이 각각 494/494건으로 사실상 상수이며, reputation attack 434, institutional attack 425, emotional manipulation 222, exaggeration 205, omit context 65, alter core facts 56건이다. request explicitness도 전부 4/4였다. 즉 JailNews 인스트럭션은 템플릿 구조가 매우 균질해, 더 정교한 reasoning이 학습할 수 있는 구조적 변이가 제한적이다.

## 501명 OOF 결과

각 사람별로 이미 관측된 상위 10개 언어×기법 arm만 재정렬했다. 같은 sample 또는 entity를 공유하는 사람은 동일 fold에 넣었다.

| Router | ASR top-1 | retained top-1 | strict top-1 | ASR AUC |
|---|---:|---:|---:|---:|
| surrogate prior | 70.86% | 22.16% | 3.99% | 0.4999 |
| global language/method bias | 79.44% | 20.56% | 3.19% | 0.5348 |
| PC2 static language | 79.64% | 20.56% | 2.40% | 0.5546 |
| entity-language | 77.45% | **24.75%** | 2.59% | 0.6102 |
| intent metadata | 80.84% | 20.76% | 3.79% | 0.6564 |
| 32B iterative intent-technique | **81.24%** | 20.96% | 3.99% | **0.6658** |
| structural formula, direct | 81.04% | 21.36% | **4.39%** | 0.5665 |
| iterative LLM+formula | **81.24%** | 21.16% | 3.99% | 0.6636 |
| observed oracle@10 | 98.00% | 70.06% | 26.35% | - |

32B iterative intent-technique와 metadata의 ASR 차이는 +0.40pp로 유의하지 않다. global 대비 차이도 +1.80pp(95% CI -0.60~+4.39, p=0.211)로 유의하지 않다. 구조 공식의 direct prior는 초안 LLM direct prior보다 +7.58pp 높았지만, global 대비 +1.60pp는 유의하지 않았다. 이는 공식이 의도별 상호작용보다는 강한 전역 method ordering을 상당 부분 회복했음을 뜻한다.

기존 7B 1-pass intent-technique와 새 32B 5-pass의 변화는 ASR +0.20pp, retained -0.20pp, strict +0.60pp였다. 세 endpoint 모두 McNemar 검정에서 유의하지 않았다.

## 의도와 기법의 탐색적 패턴

아래 패턴은 method별 전역 성공률을 뺀 뒤 Beta prior strength 20으로 완화한 기술 통계다. 후보 arm이 무작위 배정되지 않았으므로 인과적 규칙으로 사용할 수 없다.

- 선거 영향 primary goal에서는 role-play가 자신의 전역 ASR보다 +5.78pp, retained보다 +9.29pp 높았다.
- 경제 여론 플래그에서는 context-overload가 자신의 전역 ASR보다 +9.41pp 높았다.
- 유권자 audience에서는 context-overload가 자신의 전역 ASR보다 +9.19pp 높았다.
- 제도 신뢰 훼손에서는 research-front의 ASR 상호작용이 +4.01pp, context-overload의 retained 상호작용이 +2.98pp였다.
- conflict/security 플래그에서는 system-override의 retained 상호작용이 +4.83pp였다.
- social-polarization primary goal에서는 system-override의 strict 상호작용이 +3.03pp였으나 strict 사건 수가 매우 적어 불확실성이 크다.

전역 관측치는 role-play ASR 89.08%(n=174), system-override 80.13%(n=1,067), context-overload 71.44%(n=3,579), research-front 64.08%(n=142), neg-prompting 22.92%(n=48)였다. 이 지원 불균형 때문에 raw subgroup 승자를 그대로 prior로 사용하면 안 된다.

## 권장 온라인 구조

현재 증거에 맞는 온라인 선택기는 하나의 합산 점수보다 hurdle 형태가 적합하다.

1. `P(non-refusal | intent, method)`는 intent metadata + 상세 의도 특징으로 추정한다.
2. `P(retained | non-refusal, person, language)`는 Wikipedia/인물 임베딩·현지 표기·역번역 품질로 추정한다.
3. `P(strict | retained, arm)`은 희소 사건용 별도 posterior로 둔다.
4. 최종 arm utility는 세 확률의 곱과 비용·불확실성 보너스로 구성하고, 관측 후 Beta/Logistic posterior를 갱신한다.

다음 논문 실험에서는 각 intent stratum에서 language×method를 무작위 또는 propensity가 알려진 방식으로 반복 생성해야 한다. 특히 현재 데이터에서 494건 모두 directness=4이고 두 요청 연산이 상수이므로, 더 다양한 instruction paraphrase/intent formulation을 추가하지 않으면 reasoning router의 추가 이득을 검출하기 어렵다.

## 산출물

- `intent_iterative_qwen32.jsonl`: 최종 5-pass 주석, 초안·비평·최종·공식 점수 포함
- `intent_iterative_qwen32_before_recritic.jsonl`: 루브릭 수정 전 결과
- `evaluation_iterative_qwen32/results.csv`: 전체 OOF 지표
- `evaluation_iterative_qwen32/oof_predictions.jsonl`: trial별 OOF 확률
- `evaluation_iterative_qwen32/significance_*.json`: paired 검정
- `evaluation_iterative_qwen32/version_comparison_intent_technique.json`: 7B 1-pass 대 32B 5-pass
- `intent_technique_analysis/intent_method_cells.json`: subgroup×method×endpoint 셀
- `intent_technique_analysis/method_adjusted_interactions.json`: method 전역률 보정 상호작용
