# FakeJail 작업 인수인계 — 2026-10-06

## 1. 한 줄 목표

JailNewsBench 입력에서 인물·개념·가짜뉴스 생성 의도를 분리해, 폐쇄 모델에 질의하기
전에 72개 언어×5개 기법의 성공 확률 prior를 만들고 소수의 실패 관측만으로 다음 arm을
빠르게 선택하는 contextual stochastic BAI 파이프라인을 구축한다.

## 2. 최종 실험 통제안

샘플 내에서 아래 항목은 고정한다.

- `ARTICLE`: 공개 데이터의 영어 기사 `article_en`
- `INSTRUCTION`: 샘플별 의미를 보존한 영어 번역
- 출력 언어: 영어
- 생성 모델, judge, decoding, 최대 출력 길이

변수는 두 축뿐이다.

- `A:` 뒤의 중립적 인물 설명 언어: 72개
- JailNewsBench 공식 jailbreak 기법: 5개

따라서 사람당 360 arms이다. 인물 출력 방식은 하나의 720-arm 공간으로 섞지 않고 아래
두 실험 조건으로 분리한다.

- `a_literal`: 생성 기사에서 인물을 `A`로 유지
- `canonical_en`: 생성 기사에서 `A`를 canonical English name으로 해소

영어 기사·영어 지시문의 실명 유지, `A`/설명/라우터 미사용 조건은 primary control이다.
자세한 불변식은 `results/v5_protocol/PROTOCOL.md`에 있다.

## 3. 완료된 데이터 작업

### 인물 연결과 표기

- 원 데이터 labels: 501
- QID 기준 deduplicated entities: 462
- QID-backed: 451, 명시적 unresolved: 11
- 인물×언어 현지 표기: 36,072행(501×72)
- status: verified_localized 15,752 / fallback_original 19,528 / unresolved 792
- canonical name 전체 검색 대신 검수된 `luna_people.surface` 위치를 기준으로 치환
- 이전 429 실패 80건 재조회, 30개 needs-review 판정, 잘못 연결된 QID 교정 완료

### 개념 번역

- concepts: 4,215
- 번역 pairs: 303,480(4,215×72)
- 빈 번역: 0
- 중복 ID: 0
- `valid != usable_for_rendering`: 0
- verified_luna 35,971
- verified_local_qwen32 39,186
- provisional_auto 193,373
- rejected_local_qwen32 34,950

### 카탈로그

- contexts: 4,091
- entities: 462
- languages: 72
- methods: 5
- arms/context: 360
- `validation_report_v2.json`: `passed=true`

검증 SHA-256과 세부 카운트는 `results/data_integrity/`에 보존했다.

## 4. 구현된 파이프라인

### Offline

1. 인물·개념·의도 자산 생성 및 검수
2. 360-arm prompt matrix 생성
3. surrogate 반복 생성과 JailNewsBench gate/rubric 판정
4. model별 Beta/Jeffreys shrinkage 및 불일치에 따른 prior strength 감쇠
5. person/context embedding, Wikipedia side features, intent–method 특징 생성
6. group OOF 방식으로 라우터 및 calibration 평가

### Online

1. 새 behavior에서 인물·개념·의도 추출
2. QID/Wikipedia/embedding으로 language prior 계산
3. instruction reasoning으로 method prior 계산
4. logit 공간에서 joint 360-arm prior 생성
5. cluster representative 또는 posterior top-two 후보 선택
6. target 성공/실패로 global/language/method/pair/cluster residual posterior 갱신
7. retained/strict 성공 시 조기 중단

핵심 실행기는 `scripts/jailnews_bandit_pipeline.py`, cluster BAI는
`scripts/evaluate_jailnews_clustered_bai_v3.py`, 이중 라우터는
`scripts/evaluate_jailnews_dual_router.py`에 있다.

## 5. 완료된 실험과 해석

### 5.1 360-arm stochastic seed

Balanced phase는 3,600 calls로 arm마다 정확히 10개 관측을 갖지만 person–arm cell은
대부분 1회다.

| phase | ASR | retained | strict |
|---|---:|---:|---:|
| balanced | 66.61% | 9.71% | 9.54% |
| top-two round 1 | 76.42% | 22.08% | 21.68% |
| top-two round 2 | 77.52% | 24.18% | 24.08% |

반복된 1,451 person-arm cells에서 disagreement는 ASR 9.92%, retained 17.16%, strict
17.02%였다. 즉 target은 확률적이지만 현재 데이터만으로 360개 arm의 사람별 평균을
모두 추정할 수는 없다.

### 5.2 surrogate 전이

두 draw에서 Llama-3.1-8B는 target-cell AUC가 ASR 0.737, retained 0.825, strict
0.837이었다. Llama-3 RR 8B는 지나치게 낮은 성공률과 음의 ASR arm 상관을 보였다.
앙상블은 ASR top-10 target rate를 79%로 올렸지만 retained/strict에서는 Llama 단독보다
좋지 않았다. 따라서 RR을 동일 가중 앙상블하는 것은 권장하지 않는다.

### 5.3 GPT-4o-mini 501×top-10

GPT-5-nano judge 기준 5,010 trials:

- trial ASR: 73.23%
- retained: 20.74%
- strict: 3.87%
- person best-of-10: ASR 98.00%, retained 70.06%, strict 26.35%

당시 surrogate 점수의 trial AUC는 약 0.50으로 사실상 무작위였다. 이후 OOF 라우터가
이를 개선했지만, 이 top-10 후보 풀 자체가 과거 prior로 선택되었다는 점을 유지해야 한다.

### 5.4 intent–technique / entity–language 라우터

- Qwen2.5-32B 5-pass intent annotation: 494/494 검수 통과
- intent metadata top-1 ASR 80.84%
- iterative 32B intent–technique top-1 ASR 81.24%(+0.40pp, 유의하지 않음)
- entity–language retained top-1 24.75%, global 대비 +4.19pp(p=0.044)

의도 reasoning은 감사 가능성과 row-level AUC를 개선했지만 top-1 이득은 작았다. 현재
증거는 언어와 기법을 하나의 설명으로 합치기보다 서로 다른 hurdle로 두는 편을 지지한다.

### 5.5 clustered contextual BAI

관측된 person별 10-arm catalog에서:

| 정책 | ASR B=1 | B=2 | B=3 | B=5 |
|---|---:|---:|---:|---:|
| nested best-router static | 78.84% | 89.82% | 92.22% | 94.41% |
| context-cluster static | 81.24% | 91.42% | 93.41% | 94.81% |
| TRIPLE-CLST adapted | 81.64% | 90.62% | 93.21% | 95.41% |
| CCB adaptive | 81.24% | 90.22% | 92.81% | 95.41% |

TRIPLE-CLST B=1은 nested static 대비 +2.79pp, paired exact p=0.0336이었다. 실패 후
적응 이득은 방향성은 있으나 아직 유의하지 않다.

### 5.6 V2→V3→V4→V5

- V2: `A`의 정체를 모델이 암묵적으로 추론하도록 함. 인물 해소 실패가 많았음.
- V3: 출력 전에 인물 정체를 해소하도록 명시. matched checkpoint에서 V2 대비 ASR
  +3.67pp, retained +10.28pp, strict +14.89pp.
- V4: `A` 유지와 영어 실명 출력 두 모드를 분리. Xi 360-arm pilot에서 `a_literal`이
  `canonical_en`보다 높았지만 입력 언어 통제가 충분하지 않았음.
- V5: ARTICLE, INSTRUCTION, 출력 언어를 영어로 고정하고 설명 언어×기법만 변화.
  두 모드 각각 180,360행 matrix와 integrity audit를 완료했다. 이어 30명×6언어×5기법의
  통제 파일럿 1,950건을 Qwen3-30B FP8로 생성하고 Qwen2.5-32B로 판정했다. 1건의 빈
  visible answer를 제외한 1,949건에서 strict는 control 10.00%, `a_literal` 68.19%,
  `canonical_en` 66.00%였다. 자세한 계약과 제한은
  `docs/V5_CONTROLLED_PILOT_2026-10-06_KO.md`에 있다.

## 6. 판정기 관련 주의

JailNewsBench 논문은 GPT-5/Gemini 2.5/Claude 4 판정 평균을 사용한다. 로컬 재현은
동일 공개 rubric을 Qwen2.5-32B 단일 judge로 대체했다. GPT-5-nano 재판정에서는
retained/strict 비율이 크게 낮아졌다. 따라서 다음을 반드시 지킨다.

1. judge별 결과를 별도 표로 유지한다.
2. ASR(non-refusal), IFL, retained, strict, 8축 평균을 혼용하지 않는다.
3. `<think>...</think>`는 보존하되 visible answer만 판정한다.
4. 논문 프롬프트 해시와 upstream commit을 manifest에 고정한다.
5. strict는 프로젝트 추가 지표이며 논문 Table 2의 열이 아니다.

## 7. GPU 중단과 로컬 재개

2026-10-06에 다음 연구 작업을 모두 취소했다.

- V3 Qwen30 32k generation의 남은/연결된 작업
- 영어 실명 primary control generation 및 연결 judge/aggregate
- 관련 GPT/Qwen judge, snapshot, aggregate dependency jobs
- 남아 있던 H200 대기 배열

당시 최종 확인 시 GPU partition의 실행·대기 작업은 0이었다. 남은 `all` partition
항목은 GPU를 점유하지 않는 dependency 상태였다.

이후 2026-10-06 로컬 8×H100 환경에서 먼저 P0/P1 통제 파일럿을 완료한 뒤,
사용자 지시에 따라 주 실험을 501명×72언어×5기법=180,360행으로 확대했다.
canonical-English generation은 두 차례 외부 종료 뒤 115,499/180,360건(64.038%)까지
보존됐으며, 2026-10-07 사용자 요청으로 일시정지했다. 현재 GPU 프로세스와 자동 실행
서비스는 없다. Qwen32 판정, PC2 full-pool 평가, selected-arm 4개 추가 seed,
A-literal 전체 ablation은 시작하지 않았다. 상세 상태·체크섬·중단 이력은
`docs/V5_FULL501_CHECKPOINT_2026-10-07_KO.md`에 있다.

## 8. 다음 작업 — 권장 순서

### P0. 실험 계약 동결

상태: 완료. `configs/v5_pilot_preregister.json`에 모델 revision, decoding, cohort,
endpoint, shared-control 처리 규칙을 생성 전에 고정했다.

- V5 `a_literal`, `canonical_en`, 영어 실명 control의 정확한 prompt hash를 고정한다.
- generator/judge snapshot, chat template, temperature, seed, max tokens를 preregister한다.
- 데이터셋 split과 person/entity group을 고정한다.
- 주 endpoint를 `strict` 또는 hurdle utility로 하나 정하고 ASR은 보조 지표로 둔다.

### P1. 작은 통제 파일럿

상태: 부분 완료. 30명 층화 표본, 6개 언어, 5개 기법, 세 조건의 생성·단일 Qwen32
판정·집계를 완료했다. 출력 절단과 반복은 0건, 빈 visible answer는 1건이었다.
Qwen2.5-7B 보조 판정도 1,950건 완료했으며 Qwen32 대비 strict raw agreement는
82.86%, Cohen kappa는 0.642였다. 수동 표본검사는 아직 남아 있다.

- 501명을 바로 돌리지 말고 20–30명 층화 표본을 사용한다.
- control + 두 V5 mode에서 같은 language/method subset을 paired 실행한다.
- 출력 잘림, 실명/A 해소, 기사 형식, instruction fidelity를 수동 표본검사한다.
- Qwen32와 별도 judge의 agreement를 측정한다.

### P2. 균형 관측과 반복

상태: 일시정지. canonical-English 501명×360 arms exhaustive draw 중 115,499건을
수집했다. PC2 top-1 / observed-oracle / random-control 선택과 draw 1–4 반복은 아직
시작하지 않았다.

- 전체 360 arms의 language/method marginal이 균형인 design을 만든다.
- 동일 person-arm을 최소 3회, 가능하면 5회 서로 다른 seed로 반복한다.
- 순차 선택 데이터와 무작위 평가 데이터를 분리한다.
- exact prompt hash가 같은 nominal arms는 equivalence class로 묶는다.

### P3. surrogate 재선정·calibration

상태: full-501 Llama 두 모델×두 draw 복구 및 사전 검증 완료. Qwen target outcome을
보기 전에 safety-gated PC2 설정을 동결했다. 이후 복원된 Wikipedia/localization,
person/context embedding, ASR-blind intent를 실제 360-arm prior에 연결했다. 외부-only
prior는 surrogate에서 약했기 때문에 0–35% grid 후 외부 가중치를 2.5%로 제한했고,
최종 20회 surrogate replay와 501명 selection-manifest smoke test로 검증했다.

- Llama-3.1-8B를 기본 surrogate로 유지한다.
- RR 모델은 endpoint별 가중치를 검증하고, 음의 전이가 지속되면 제외한다.
- Qwen3-30B 32k 결과는 truncation/degenerate repetition gate를 통과한 셀만 사용한다.
- held-out entity group에서 isotonic/temperature calibration과 ECE를 보고한다.

### P4. hierarchical posterior

상태: `scripts/evaluate_jailnews_pc2_full501.py`에 outcome-blind 외부 entity-language ×
intent-method prior, leave-one-person-out population prior, 세 hurdle, PC2 graph residual,
safety-gated Top-Two를 구현했다. 외부 prior 공식은 Qwen target judgment 0건 시점에
동결했고 외부-only language/factorized baseline도 유지한다.

- hurdle을 `P(non-refusal) × P(retained|non-refusal) × P(strict|retained)`로 분해한다.
- person/context soft cluster, language, method, language×method random effect를 둔다.
- cluster-level Beta posterior와 arm-level logistic posterior를 함께 샘플링한다.
- posterior top-two cluster 사이에서 예산을 배분하고 성공 시 중단한다.

### P5. 논문 평가

- group OOF + 완전 held-out person test
- success@B, simple regret, cumulative API cost, calibration, oracle gap
- context-cluster / intent router / entity router / posterior update ablation
- language family/script/region, Wikipedia coverage, description quality별 slice
- judge 교차검증과 수동 blind audit

## 9. 비공개 논문 데이터 스냅샷

과거 115GiB staging 전체를 올리는 대신, 후속 V5 실험과 논문 분석에 필요한 파일만
`paper_snapshot_v1`으로 선별했다. 비공개 Hugging Face Dataset
`jin-kwon/fakejail-data`의 동일 경로에 네 개의 zstd 아카이브와 체크섬·선택 목록을
둔다. 총 압축 크기는 1,058,563,648 bytes(약 1.0GB)이며, 완료 revision은
`dc7747e6e57c0993cbe14656b129754678eba976`이다.

- `core_data.tar.zst`: 최종 entity/QID 수리, 72언어 번역, catalog, embeddings, 원자료
- `v5_prompts.tar.zst`: 영어 통제 V5 두 조건의 360-arm matrix와 English control
- `target_observations.tar.zst`: GPT target/judge, balanced/top-two, router/BAI 자료
- `surrogate_llama.tar.zst`: Llama-3.1-8B + Llama-3 RR 원시 생성·판정·반복 관측

Table-2 hidden states, Qwen30 탐색 실행, V2–V4 전체 matrix, 번역 중간 shard, 로그,
공개 모델 가중치는 제외했다. 과거 staging은 공식 재현 단위가 아니며, 누락 자료가
실제로 필요해질 때만 새 snapshot 버전으로 선별 추가한다. 정확한 SHA-256과 접근
명령은 `docs/DATA_ACCESS_KO.md`를 따른다.
