# JailNews Clustered Contextual BAI v3

## 결론

TRIPLE-CLST를 그대로 복제하지 않고, JailNews의 구조에 맞춰 다음 두 층으로 바꿨다.

1. **문맥 prior 층**: 인물 Wikipedia 임베딩과 기사/지시문 임베딩으로 과거 사람을 soft clustering하고, 각 클러스터의 언어·기법 성공 통계를 현재 샘플 prior에 혼합한다.
2. **온라인 탐색 층**: 현재 사람의 후보 설정을 원문 임베딩이 아니라 여러 OOF router가 만든 reward-profile, 기법, script로 clustering한다. 한 설정이 실패하면 그 실패 residual을 가까운 설정에만 전파하고 다른 cluster를 탐색한다.

현재 501명 × 10개 관측 후보에서 가장 명확한 개선은 ASR이다. 단순 nested best-router 대비 context-cluster prior는 B=1에서 **78.84% → 81.24%**, B=2에서 **89.82% → 91.42%**로 향상됐다. B=2의 paired exact p는 0.0386이다. TRIPLE-CLST형 cluster-representative 선택은 B=1에서 **81.64%**이며 paired exact p=0.0336이다.

다만 retained와 strict 목표는 양성 수가 각각 1,039/5,010, 194/5,010으로 적고, 개선의 신뢰구간이 대부분 0을 포함한다. 따라서 현재 결과는 “ASR에서 클러스터 prior가 유효하다”까지는 지지하지만, “모든 품질 지표에서 오라클에 유의하게 가까워졌다”고 주장할 수는 없다.

## TRIPLE 논문에서 가져온 것과 가져오지 않은 것

대상 논문은 NeurIPS 2024의 *Efficient Prompt Optimization Through the Lens of Best Arm Identification*이다.

논문의 핵심은 candidate prompt를 arm, LLM 호출을 pull, 평가 점수를 reward로 보고 고정 예산 BAI로 최종 prompt를 식별하는 것이다. 기본형은 Sequential Halving과 Continuously Reject이고, 큰 후보 집합에는 embedding을 사용한 TRIPLE-CLST와 함수 근사를 사용한 TRIPLE-GSE를 제안한다.

TRIPLE-CLST의 실험 설정은 다음과 같다.

- 클러스터 수: `ceil(sqrt(number_of_prompts))`
- k-means 사용
- 전체 예산의 1/3을 cluster 선택에 사용
- 최상위 cluster 하나가 아니라 상위 절반의 cluster를 남김
- 두 단계 모두 CR 사용

TRIPLE-GSE는 1,536차원 prompt embedding을 random projection으로 64차원으로 줄이고, 80/20 train/validation으로 reward predictor를 학습한다. validation MSE가 0.1 이하일 때만 elimination을 수행한다.

우리 문제에 그대로 쓸 수 없는 이유는 다음과 같다.

- 논문은 하나의 prompt가 입력 분포 전체에서 갖는 평균 성능을 찾는다.
- 우리는 사람, 기사 의도, 언어, 공격 기법의 상호작용 때문에 샘플마다 최적 arm이 달라지는 contextual problem이다.
- 현재 타깃 데이터는 사람당 10개 arm의 1회 결과만 있다. 같은 arm의 확률 평균을 추정하는 고전 BAI 실험이 아니다.
- 번역 prompt의 raw embedding을 사용하면 reward보다 언어/script별로 clustering될 위험이 있다.

따라서 논문에서는 **embedding으로 arm 간 정보를 공유한다는 원리**, `sqrt(K)` 수준의 cluster 축약, 큰 후보 풀에서 무구조 탐색을 피한다는 관점만 반영했다.

## 수식

샘플 `x`, 언어 `l`, 기법 `m`인 arm을 `a=(l,m)`라 둔다.

### 1. 문맥 cluster prior

인물 임베딩과 기사/지시문 임베딩을 각각 고정 random projection한 후 정규화하여 결합한다.

`z_x = normalize([RP(e_person); RP(e_context)])`

training fold에서 spherical k-means로 prototype `c_j`를 만들고, hard assignment 대신 다음 soft responsibility를 사용한다.

`q(j|x) = softmax(cos(z_x,c_j) / tau)`

각 cluster의 pair/language/method 성공률은 전역 통계로 Beta shrinkage한다. 희소한 정확한 `(language, method)` pair만 믿지 않고 다음 logit mixture를 쓴다.

`logit p_cluster(a|x,j) = 0.45 logit p_pair + 0.25 logit p_language + 0.20 logit p_method + 0.10 logit p_cluster`

`p_context(a|x) = sum_j q(j|x) p_cluster(a|x,j)`

최종 초기 prior는 OOF router와 logit 공간에서 결합한다.

`logit p0(a|x) = (1-w) logit p_router(a|x) + w logit p_context(a|x)`

`w`와 router 종류는 outer-training fold 안에서만 선택한다.

### 2. 현재 샘플의 arm topology

arm embedding에는 prompt 원문을 넣지 않는다. 다음 outcome-free 정보만 쓴다.

- surrogate, PC2 language, entity-language, intent-technique, iterative, joint router의 OOF logits
- 세 endpoint(ASR, retained, strict)에 대한 예측 profile
- 공격 기법 one-hot
- language script one-hot

10개 후보는 `ceil(sqrt(10))=4`개 reward-topology cluster로 묶는다.

### 3. 실패 후 posterior 수정

시도한 arm 집합을 `O_t`, 관측을 `y_j`라 할 때, 아직 시도하지 않은 arm의 score는 다음 형태다.

`score_t(a) = logit p0(a) + lambda_c Delta_cluster(a) + lambda_k Delta_kernel(a) + eta novelty(a)`

`Delta_cluster`는 같은 cluster에서 관측한 `(y_j-p0_j)` residual의 regularized Newton update이고, `Delta_kernel`은 reward-profile cosine kernel로 가중한 residual update다. `novelty`는 이미 실패한 arm과 먼 후보에 주는 작은 보너스다.

이 구조의 목적은 UCB처럼 누적 보상을 최대화하는 것이 아니라, 높은 prior arm이 실패했을 때 같은 basin에 예산을 계속 쓰지 않고 다른 성공 basin으로 빨리 이동하는 것이다.

## 평가 프로토콜

- 501명, 사람당 이미 관측된 상위 10개 설정
- endpoint: ASR, retained, strict
- 예산: 1–10 unique settings
- group outer 5-fold: 같은 sample/entity 연결 성분이 fold를 넘지 않음
- base router 선택과 하이퍼파라미터 선택: outer-training fold에서만 수행
- context cluster 통계: 평가 대상 outer fold를 완전히 제외
- 평가 fold의 결과: 정책이 arm을 선택한 뒤에만 공개
- bootstrap CI: person 단위 paired bootstrap
- 이 평가는 **10-arm finite-catalog success discovery**이며, 360-arm stochastic BAI 성능으로 표현하지 않음

## 결과

### ASR

| 정책 | B=1 | B=2 | B=3 | B=5 | 관측 oracle@10 |
|---|---:|---:|---:|---:|---:|
| Nested best-router static | 78.84% | 89.82% | 92.22% | 94.41% | 98.00% |
| Context-cluster static | 81.24% | **91.42%** | **93.41%** | 94.81% | 98.00% |
| TRIPLE-CLST adapted | **81.64%** | 90.62% | 93.21% | **95.41%** | 98.00% |
| CCB adaptive | 81.24% | 90.22% | 92.81% | **95.41%** | 98.00% |
| Budget-aware controller | 81.24% | 89.82% | 93.01% | 95.21% | 98.00% |

주요 paired 비교:

- Context-cluster B=1: +2.40pp, 95% CI [0.00, 4.79], exact p=0.0652
- Context-cluster B=2: +1.60pp, 95% CI [0.40, 2.99], exact p=0.0386
- TRIPLE-CLST adapted B=1: +2.79pp, 95% CI [0.40, 5.19], exact p=0.0336
- CCB adaptive B=5: +1.00pp, 95% CI [-0.40, 2.59], exact p=0.3018

즉, 현재 데이터에서 가장 재현성 있는 이득은 **초기 문맥 cluster prior와 첫 cluster 대표 선택**이다. 실패 전파는 B=5 평균은 올리지만 아직 유의하지 않다.

### Retained

| 정책 | B=1 | B=3 | B=5 | 관측 oracle@10 |
|---|---:|---:|---:|---:|
| Nested best-router static | 21.96% | 40.52% | 55.09% | 70.06% |
| CCB adaptive | 21.96% | 42.71% | 54.09% | 70.06% |
| Budget-aware controller | **22.75%** | **43.31%** | 54.89% | 70.06% |

Budget-aware B=3은 +2.79pp지만 95% CI [0.00, 5.59], exact p=0.0759다. 후속 표본이 필요하다.

### Strict

| 정책 | B=1 | B=3 | B=5 | 관측 oracle@10 |
|---|---:|---:|---:|---:|
| Nested best-router static | 4.99% | 12.57% | 16.97% | 26.35% |
| TRIPLE-CLST adapted | **5.79%** | 10.98% | 17.17% | 26.35% |
| CCB adaptive | 4.99% | 11.78% | **17.37%** | 26.35% |

strict는 양성이 194건뿐이라 모든 개선의 paired CI가 0을 포함한다.

## 해석과 다음 강화 순서

1. **현재 production 후보**: ASR 초기 선택에는 context-cluster prior 또는 cluster representative를 사용한다. 실패 후 B=5까지는 CCB를 사용하되, 논문 주장은 exploratory로 제한한다.
2. **360-arm 전체 관측 확보**: 현재는 상위 10개만 타깃 outcome이 있다. 72×5 전체의 unbiased offline evaluation을 하려면 최소한 stratified coverage가 필요하다.
3. **반복 호출**: stochastic BAI를 주장하려면 같은 `(sample, language, method)`를 여러 seed/temperature에서 반복해야 한다. 현재 한 번의 0/1은 평균 arm reward가 아니다.
4. **다단계 reward**: refusal → retained → adherence/formality의 hurdle reward를 쓰고, binary ASR 발견과 고품질 기사 발견을 분리한다.
5. **confidence-gated function approximation**: TRIPLE-GSE처럼 reward predictor validation error가 임계값 이하일 때만 elimination한다. 우리 경우에는 closed-model transfer calibration error와 cluster별 ECE를 gate로 쓰는 편이 맞다.
6. **Top-two posterior sampling**: 충분한 반복 결과가 쌓이면 cluster-level Beta posterior와 arm-level logistic posterior를 함께 샘플링하고, posterior best-arm probability 상위 두 cluster 사이에 예산을 배분한다.
7. **중단 규칙**: 판정 신뢰도가 높은 retained/strict 성공이 나오면 즉시 중단하고, 불확실 판정만 재평가한다. 이 방식이 실제 API 비용을 가장 직접적으로 줄인다.

## 산출물

- 구현: `scripts/evaluate_jailnews_clustered_bai_v3.py`
- paired 분석: `scripts/analyze_jailnews_clustered_bai_v3.py`
- 전체 결과: `results.json`, `results.md`
- paired 통계: `paired_comparisons.json`, `paired_comparisons.md`

## 논문 링크

- Paper: https://proceedings.neurips.cc/paper_files/paper/2024/file/b46bc1449205888e1883f692aff1a252-Paper-Conference.pdf
- Official code: https://github.com/ShenGroup/TRIPLE
