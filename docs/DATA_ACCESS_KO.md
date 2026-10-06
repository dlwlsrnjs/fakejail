# 비공개 데이터 및 원시 응답 접근

## 저장소

- Hugging Face Dataset: <https://huggingface.co/datasets/jin-kwon/fakejail-data>
- 소유 계정: `jin-kwon`
- 가시성: private
- 스냅샷: 2026-10-06
- 원본 파일: 1,480개
- 논리 크기: 123,350,291,485 bytes(약 115GiB)
- 업로드 상태: **미완료** — Hugging Face private LFS storage limit 403
- 로컬 staging: `/home/ljk98/POLY/fakejail_hf_upload`

현재 Hub에는 작은 파일만 일부 커밋되어 있으므로 아래 전체 다운로드 명령은 업로드가
완료된 뒤 사용한다. 접근 권한이 있는 계정으로 인증해야 한다.

```bash
hf auth login
hf download jin-kwon/fakejail-data \
  --repo-type dataset \
  --local-dir ./fakejail_private_data
```

필요한 하위 경로만 받을 때는 `--include`를 사용한다.

```bash
hf download jin-kwon/fakejail-data \
  --repo-type dataset \
  --include 'results-or-path-pattern' \
  --local-dir ./fakejail_private_data
```

## 구성

| 경로 | 파일 | 크기 | 내용 |
|---|---:|---:|---|
| `data/` | 603 | 7,313,761,593 bytes | JailNewsBench 원본과 인물·개념·번역 파생 데이터 |
| `artifacts/` | 713 | 54,692,962,530 bytes | prompt matrix, raw generation/judgment, prior/router/posterior 상태 |
| `baseline_runs/` | 164 | 61,343,567,362 bytes | 공식형 baseline 원시 생성·판정 |

Dataset 루트의 `DATA_MANIFEST.json`과 `README.md`에도 동일한 범위가 기록되어 있다.

## 업로드 재개

Hugging Face private storage를 증설한 경우 기존 해시 상태를 재사용한다.

```bash
hf upload-large-folder jin-kwon/fakejail-data \
  /home/ljk98/POLY/fakejail_hf_upload \
  --repo-type dataset \
  --num-workers 8
```

다른 비공개 object storage를 사용할 경우 동일 staging의 `data/`, `artifacts/`,
`baseline_runs/`, `README.md`, `DATA_MANIFEST.json`을 업로드한다. 공개 모델 가중치는
staging에 포함되어 있지 않다.

## 코드와 연결

다운로드한 루트에서 다음 경로를 코드 저장소 루트로 연결하거나, 각 스크립트의 기본
경로 인자를 지정한다.

```text
data/
artifacts/
baseline_runs/
```

코드에는 과거 실행 환경의 `/home/ljk98/POLY` 절대 경로가 일부 남아 있으므로 다른
환경에서는 해당 기본값을 수정해야 한다.

## 주의

- 원시 파일에는 정치인 이름, 가짜뉴스 생성 지시, 모델 생성물과 판정 응답이 포함된다.
- 공개 모델 가중치, Hugging Face 모델 캐시, 체크포인트 파일은 포함하지 않는다.
- 승인된 모델 안전성 연구 목적으로만 사용한다.
- 비공개 상태를 변경하기 전에 upstream 라이선스와 harmful-content 검토를 다시 한다.
- API 키나 토큰은 포함하지 않았다. 업로드 전 115GB 전체 키 패턴 검사를 수행했고
  탐지 결과는 0건이었다.
- 새 실행은 기존 shard를 덮어쓰지 말고 새 run directory를 만들거나 정확한 manifest로
  resume한다.
