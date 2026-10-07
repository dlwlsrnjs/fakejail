# FakeJail: JailNewsBench multilingual transfer-bandit research

JailNewsBench의 가짜뉴스 생성 요청을 대상으로, 샘플의 인물·개념·의도를 추출하고
`72개 언어 × 5개 JailNewsBench 기법 = 360 arms` 중 적은 질의로 유효한 설정을
찾는 transfer-prior / contextual bandit 연구 코드입니다.

이 저장소에는 재현 코드, 검증 매니페스트, 집계 결과만 포함합니다. 논문 작성과 최종
V5 실험에 필요한 원자료·프롬프트·target 관측·surrogate 원시응답은 약 1.0GB의
`paper_snapshot_v1`으로 선별해 [비공개 Hugging Face Dataset](https://huggingface.co/datasets/jin-kwon/fakejail-data/tree/main/paper_snapshot_v1)에
보관합니다. 115GiB 규모의 과거 전체 staging은 탐색 실험과 중복 산출물이 많아 공식
재현 스냅샷에서 제외했습니다. 구성, 체크섬, 다운로드 방법은
[데이터 접근 문서](docs/DATA_ACCESS_KO.md)를 참조하세요.
재현용 Dataset revision은 `dc7747e6e57c0993cbe14656b129754678eba976`입니다.

## 현재 상태 (2026-10-06)

- 데이터 정합성 수리 완료: 4,091 contexts, 501 labels, QID 기준 462 entities,
  4,215 concepts, 72 languages, 5 methods, 360 arms.
- 인물 현지 표기: 36,072행(501×72), 중복 없음.
- 개념 번역: 303,480행, 빈 번역 0, usability 충돌 0.
- 샘플별 영어 기사·영어 지시문·영어 출력은 고정하고 `A:` 인물 설명 언어와
  JailNewsBench 기법만 바꾸는 V5 통제 실험 행렬 생성 완료.
- V5는 `a_literal`과 `canonical_en` 두 조건 각각 180,360행이다.
- 30명×6언어×5기법 통제 파일럿 1,950건의 생성·Qwen2.5-32B proxy 판정 완료.
  생성 invalid 1건과 judge error 0건을 기록했다.
- 501명×상위 10설정의 GPT-4o-mini 결과와 GPT-5-nano 판정 집계 완료.
- 문맥 클러스터 기반 selector와 intent–technique / entity–language 이중 라우터 구현 및
  OOF 평가 완료.
- 이전 RTX6000/H200 대량 작업은 중단 상태를 유지한다. 로컬 8×H100에서 수행한
  canonical-English 501명×360 arms 생성은 2026-10-07 사용자 요청으로 115,499/
  180,360건(64.038%)에서 일시정지했다. 판정·PC2 평가·selected-arm 반복·A-literal
  ablation은 아직 시작하지 않았다.
- 주 PC2 prior는 outcome-blind Wikipedia/localization·person/context embedding·
  ASR-blind intent 신호를 실제 언어×기법 선택에 사용하며, 외부-only baseline도
  분리 보고한다.

자세한 완료 범위, 중단 지점, 재개 순서는
[인수인계 문서](docs/HANDOFF_2026-10-06_KO.md)를 참조하세요.

## 핵심 실험 스냅샷

| 실험 | 주요 결과 | 해석 |
|---|---:|---|
| Balanced 360 seed | ASR 66.61%, retained 9.71%, strict 9.54% | 사람–arm당 1회인 불완전 패널 |
| Top-two round 2 | ASR 77.52%, retained 24.18%, strict 24.08% | 선택 편향이 있으므로 무작위 전수 결과가 아님 |
| GPT-4o-mini top-10 + GPT-5-nano judge | trial ASR 73.23%, retained 20.74%, strict 3.87% | 사람별 best-of-10은 98.00% / 70.06% / 26.35% |
| Context-cluster OOF | ASR success@1 81.24%, @2 91.42% | 관측된 10-arm catalog 평가 |
| TRIPLE-CLST OOF | ASR success@1 81.64% | nested static 대비 +2.79pp, paired p=0.0336 |
| V2 vs V3 matched checkpoint | strict +14.89pp | explicit identity resolution이 생성 품질을 높였으나 V5 통제가 필요 |
| V5 30명 통제 파일럿 + Qwen32 judge | strict control 10.00%, A-literal 68.19%, canonical-English 66.00% | 1 draw·단일 proxy judge의 탐색 결과 |

`ASR`, `retained`, `strict`는 서로 다른 endpoint입니다. 특히 판정 모델에 따라 수치가
크게 달라졌으므로 서로 다른 judge의 결과를 같은 열처럼 비교하면 안 됩니다. V5는 이
혼선을 제거하기 위해 ARTICLE, INSTRUCTION, 출력 언어, 생성·판정 설정을 고정합니다.

## 제안 메서드

1. 인물·개념·의도 추출과 QID 기반 entity linking
2. Wikipedia/인물 임베딩으로 `P(language | person, context)` 추정
3. ASR-blind instruction reasoning으로 `P(method | intent)` 추정
4. surrogate 모델의 반복 결과를 shrinkage/calibration하여 360-arm prior 구성
5. 문맥 soft-cluster와 arm reward topology로 첫 후보 및 대체 후보 선택
6. target의 성공/실패를 관측할 때 language, method, pair, cluster posterior 갱신
7. retained/strict 성공 시 조기 중단하고 불확실한 판정만 재평가

현재 증거에서 가장 안정적인 개선은 문맥 cluster prior의 첫 선택입니다. 실패 후
posterior 전파는 가능성이 보이지만, 진짜 stochastic BAI를 주장하려면 동일
`(sample, language, method)` 반복과 360-arm 균형 관측이 더 필요합니다.

## 저장소 구조

```text
fakejail/
├── configs/                 # surrogate 및 실행 설정
├── docs/                    # 프로토콜, 데이터 수리, 인수인계 문서
├── results/                 # 집계·검증 결과만 포함
├── scripts/                 # 준비, 렌더링, 생성, 판정, 라우팅, 분석 코드
├── .gitignore
└── requirements.txt
```

코드 지도를 보려면 [재현 가이드](docs/REPRODUCIBILITY_KO.md), 새 파일럿의 계약·결과·제한은
[V5 통제 파일럿 문서](docs/V5_CONTROLLED_PILOT_2026-10-06_KO.md), 501명×360 arms 주 실험은
[V5 full-501 PC2 문서](docs/V5_FULL501_PC2_2026-10-06_KO.md), 현재 체크포인트와 중단 이력은
[2026-10-07 체크포인트 문서](docs/V5_FULL501_CHECKPOINT_2026-10-07_KO.md)를 참조하세요.

## 빠른 무실행 검증

GPU나 API를 사용하지 않고 코드 문법과 공개 결과 파일을 확인할 수 있습니다.

```bash
python3 -m compileall -q scripts
python3 scripts/validate_jailnews_bandit_assets.py --help
```

실제 파이프라인은 JailNewsBench 데이터와 로컬 runtime 경로가 필요합니다. 실행 전에
`PROJECT_ROOT`와 모델 경로를 실제 환경에 맞춰야 합니다. GPU 작업은 명시적인 자원
승인 후에만 실행하세요. `jailnews_bandit_pipeline.py` 등은 `requirements.txt` 설치 후 실행하며,
V5 audit은 제외된 `artifacts/.../base_arms.jsonl`을 복원한 환경에서 수행합니다.

## 데이터 및 비밀정보 원칙

- API 키는 저장소나 명령행 인자에 저장하지 않습니다.
- `.env`, 로그, 원시 응답, `data/`, `artifacts/`, 모델 캐시는 코드 Git에서 제외하고,
  재현에 필요한 원시응답만 비공개 Dataset 스냅샷에 둡니다.
- 공개 결과는 aggregate/manifest만 사용합니다.
- 외부 API 평가에는 승인된 샘플과 예산만 전송합니다.
- JailNewsBench와 모델 체크포인트의 원 라이선스·접근 조건을 따릅니다.

## 기준 구현

JailNewsBench 공개 evaluator의 공격·판정 문자열은 upstream commit
`d89690316ad55f7c74baaf63a32b0f915c2e16c1`을 기준으로 고정했습니다.
논문의 GPT-5/Gemini 2.5/Claude 4 judge 평균 대신 Qwen2.5-32B를 사용한 결과는
논문 수치의 복제가 아니라 동일 rubric을 사용하는 proxy 재측정입니다.
