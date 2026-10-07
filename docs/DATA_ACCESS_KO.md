# 비공개 논문 데이터 및 원시 응답 접근

## 공식 스냅샷

- Hugging Face Dataset: <https://huggingface.co/datasets/jin-kwon/fakejail-data>
- 경로: `paper_snapshot_v1/`
- 소유 계정: `jin-kwon`
- 가시성: private
- 생성일: 2026-10-06 UTC
- 고정 revision: `dc7747e6e57c0993cbe14656b129754678eba976`
- 압축 크기: 1,058,563,648 bytes(약 1.0GB)
- 매니페스트: `paper_snapshot_v1/MANIFEST.json`
- 체크섬: `paper_snapshot_v1/SHA256SUMS`

이 스냅샷은 과거 115GiB 전체 staging을 그대로 보존한 것이 아니다. 논문 분석과 최종
V5 실험에 필요한 자료만 선별했으며, 공개 모델 가중치·중복 실행·폐기된 렌더링·대형
hidden-state 배열은 제외했다.

## 다운로드와 검증

접근 권한이 있는 Hugging Face 계정으로 인증한 뒤 스냅샷만 받는다.

```bash
hf auth login
hf download jin-kwon/fakejail-data \
  --repo-type dataset \
  --revision dc7747e6e57c0993cbe14656b129754678eba976 \
  --include 'paper_snapshot_v1/**' \
  --local-dir ./fakejail_private_data

cd ./fakejail_private_data/paper_snapshot_v1
sha256sum -c SHA256SUMS
```

필요한 아카이브만 받을 수도 있다.

```bash
hf download jin-kwon/fakejail-data \
  paper_snapshot_v1/core_data.tar.zst \
  --repo-type dataset \
  --revision dc7747e6e57c0993cbe14656b129754678eba976 \
  --local-dir ./fakejail_private_data
```

프로젝트 루트에서 필요한 아카이브를 해제한다. 내부 경로는 모두 프로젝트 상대 경로다.

```bash
tar --zstd -xf paper_snapshot_v1/core_data.tar.zst
tar --zstd -xf paper_snapshot_v1/v5_prompts.tar.zst
tar --zstd -xf paper_snapshot_v1/target_observations.tar.zst
tar --zstd -xf paper_snapshot_v1/surrogate_llama.tar.zst
```

## 아카이브 구성

| 아카이브 | 압축 크기 | SHA-256 | 내용 |
|---|---:|---|---|
| `core_data.tar.zst` | 231,890,622 | `6855454d1453d68c351fcc4fa376a13f41bdc4ea10f133acd2d02012729d5248` | 최종 entity/QID 수리, 72언어 번역, catalog, 임베딩, 분석용 샘플, JailNewsBench 원자료 |
| `v5_prompts.tar.zst` | 57,497,673 | `3cbda3e0334563a2d710248eeab523ff7c34dd497ee6ab077b27eb62dfee8dae` | 영어 통제 V5 `a_literal`·`canonical_en` 360-arm 행렬과 named English control |
| `target_observations.tar.zst` | 37,941,152 | `5d0ce2f693cb68107f4c7859c792f9a28a9b25e51e6a3cbb1a4613605656f073` | GPT target 응답·판정, balanced/top-two 관측, router/BAI 분석 산출물 |
| `surrogate_llama.tar.zst` | 731,234,201 | `b0245d916753810da3807966c35ae6d6ac2c96541123acf405831e3077da8b10` | Llama-3.1-8B·Llama-3 RR 생성·판정·반복 관측·전이 분석 |

각 아카이브의 정확한 선택 경로는 같은 폴더의 `*.files`에 기록했다. GitHub에도
[`PAPER_SNAPSHOT_MANIFEST.json`](PAPER_SNAPSHOT_MANIFEST.json)을 복제해 두었다.

## 명시적으로 제외한 자료

- 공개 모델 가중치와 Hugging Face 모델 캐시
- 약 56.7GB의 Table-2 self-detection hidden-state 배열
- 선택한 Llama surrogate 분석으로 대체된 Qwen30 탐색 생성물
- V2–V4 전체 prompt matrix와 기타 폐기된 렌더링
- `translations_verified_v2.jsonl` 이전 번역 수리 중간 shard
- scheduler/stdout/stderr 로그, 임시 checkpoint, 중복 tarball

과거 `/home/ljk98/POLY/fakejail_hf_upload`의 115GiB staging은 공식 배포 단위가
아니며, 논문 재현에 필요하다고 확인된 파일이 생길 때만 새 버전의 curated snapshot에
명시적으로 추가한다.

## 코드와 연결

아카이브를 코드 저장소의 상위 프로젝트 루트에 풀면 `data/`와 `artifacts/` 경로가
복원된다. 일부 스크립트에는 과거 실행 환경의 `/home/ljk98/POLY` 기본값이 있으므로
다른 환경에서는 CLI 경로 인자를 지정한다.

## V5 full-501 partial checkpoint

2026-10-07에 사용자 요청으로 일시정지한 canonical-English draw-0 generation은 기존
`paper_snapshot_v1`을 변경하지 않고 별도 경로에 보존했다.

- 경로: `run_checkpoints/v5_full501_360_pc2_20261006/canonical_en_checkpoint_115499/`
- revision: `4f779685e897ebadd835b816cf4ed03fef38234c`
- 완료: 115,499/180,360건(64.038%)
- archive SHA-256: `0b886cc8857dbdb36b26526c9ec35a21caab6b594d795a0b1f182d9b78740232`

실행 이력, invalid-output audit, shard별 체크섬과 복원 명령은
[`V5_FULL501_CHECKPOINT_2026-10-07_KO.md`](V5_FULL501_CHECKPOINT_2026-10-07_KO.md)에
있다. 이 체크포인트는 raw 생성물을 포함하므로 Dataset의 private 상태를 유지한다.

## 주의

- 원시 파일에는 정치인 이름, 가짜뉴스 생성 지시, 모델 생성물과 판정 응답이 포함된다.
- 승인된 모델 안전성 연구 목적으로만 사용한다.
- 비공개 상태를 변경하기 전에 upstream 라이선스와 harmful-content 검토를 다시 한다.
- API 키나 토큰은 포함하지 않았다. curated 원본 범위에 대해 키 패턴 검사를 수행했다.
- 새 실행은 기존 shard를 덮어쓰지 말고 새 run directory를 만들거나 정확한 manifest로
  resume한다.
