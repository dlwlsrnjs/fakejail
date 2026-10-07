# V5 full-501 canonical 생성 체크포인트 — 2026-10-07

## 현재 상태

- 상태: 사용자 요청으로 일시정지
- 실행 중인 생성·판정·watcher: 없음
- GPU 점유: 없음(중지 직후 8장 모두 0 MiB, 0%)
- 목표: 501명 × 72언어 × 5기법 = 180,360건
- 완료: 115,499건(64.038%)
- 남음: 64,861건(35.962%)
- 완료 단계: `canonical_en` draw 0 partial generation만 해당
- 미시작: canonical judge/audit 후속 단계, PC2 평가, selected-arm draw 1–4,
  `a_literal` 전체 생성·판정·평가

원본 JSONL은 private runtime의
`v5_full501_360_pc2_20261006/canonical_en/generations/`에 유지한다. GitHub에는
생성문을 넣지 않고 코드·계약·집계만 저장한다. 복구용 raw checkpoint는 비공개
Hugging Face Dataset에 압축 보존한다.

## 실행 이력(KST)

1. 2026-10-06 15:09경 8×H100에서 8개 shard 생성을 시작했다.
2. 2026-10-07 00:42–00:48 사이 shard 출력이 멈췄고 103,662건이 남았다. 8개 생성
   프로세스가 거의 동시에 사라졌으며, 보존된 생성 로그에는 traceback, CUDA OOM,
   Xid가 없었다. 호스트 재부팅도 없었다.
3. 07:34에 각 partial JSONL 마지막 행을 검증한 뒤 `--resume`으로 재개했다.
4. 08:43–08:51 사이 다시 외부 종료됐고 115,499건이 남았다. 당시 GPU는 이후
   비어 있었으며 NVIDIA accounting이 비활성화돼 종료 신호의 발신자는 확정하지
   못했다.
5. 10:49에 셸 수명과 분리된 사용자 systemd 서비스로 재개를 준비했으나, 모델이
   GPU에 올라가기 전에 사용자가 실행 중지를 요청했다. 서비스 4개를 모두 중지했고
   새 행은 추가되지 않았다.

마지막 10:49 재개 시 실행 스크립트의 `>` redirection이 기존 per-shard 로그를 0바이트로
초기화했다. 따라서 원 생성 JSONL과 체크섬은 보존됐지만 앞선 상세 진행 로그는 현재
checkpoint에 포함되지 않는다. 이 제한을 숨기지 않고 기록한다.
이 체크포인트와 함께 커밋하는 네 실행기는 재개 시 기존 로그를 덮어쓰지 않도록
redirection을 append(`>>`)로 변경했다.

## shard 무결성

| shard | rows | bytes | SHA-256 |
|---:|---:|---:|---|
| 0 | 14,580 | 334,593,010 | `910b93cec741667af8fe5e3cf24e66c8a8680f4a703980e3652cb3ffdc3c9784` |
| 1 | 14,367 | 330,338,501 | `d65f0fa37b36960df6d4ce47f15485cf192634c78312f048afcc9f8df5e58194` |
| 2 | 14,367 | 330,418,648 | `abad0ae06b231793275dc0ebfd298869f9671fded10ee5597c71e0de50939618` |
| 3 | 14,373 | 330,443,768 | `761d4f91257f37c594f44737fe083edaaf409ca7d41fbe0d2a280554590b1b11` |
| 4 | 14,349 | 330,357,436 | `955b8977093f919374142f2dc59aeaa7bd93342ab936200d24ac6e23e8530348` |
| 5 | 14,556 | 334,409,039 | `7fc9b135d8e5194a82717cf46a823a42d4e446332c3378d9c48233cea0880838` |
| 6 | 14,313 | 330,255,335 | `0ca8d3df8ccfc7f326150098f9eb545bcedcf1a4e05d41cb8ff0bfb5bb654ee6` |
| 7 | 14,594 | 334,521,957 | `b3d5820c08d268089522e67b5d01c446ebbb2f38d9e6117bed641c1dbc8c9cbc` |
| 합계 | 115,499 | 2,655,337,694 | shard별 검증 |

`scripts/audit_jailnews_v5_generations.py`로 전체 115,499행을 파싱하고 응답 해시를
재계산했다.

- JSON 파싱 성공: 115,499/115,499
- 고유 trial ID: 115,499, 중복 0
- `response_sha256` 일치: 115,499/115,499
- visible answer가 빈 행: 21
- length/thinking truncation: 5
- degenerate repetition: 3(위 truncation 중 일부와 중복)
- 하나 이상의 invalid flag가 있는 행: 26

invalid 행은 삭제하거나 재생성하지 않았다. 사전등록 규칙대로 후속 분석에서 명시적으로
제외할 수 있도록 원 관측과 audit를 함께 보존한다.

## 재개 계약

재개가 다시 승인되면 `scripts/run_jailnews_v5_full501_canonical_generate.sh`의
`--resume` 경로를 사용한다. 재개 전에 다음을 다시 확인한다.

1. Hugging Face checkpoint의 archive·shard SHA-256을 검증한다.
2. 8개 JSONL의 마지막 행이 완전한 JSON인지 확인한다.
3. 총 행 수가 115,499인지 확인한다.
4. 기존 로그 보존이 필요하면 실행 스크립트의 로그를 새 timestamp 경로로 분리한다.
5. 완료 전에는 judge·PC2·후속 ablation을 실행하지 않는다.

## 비공개 Hugging Face 보존

- Dataset: `jin-kwon/fakejail-data`
- 경로: `run_checkpoints/v5_full501_360_pc2_20261006/canonical_en_checkpoint_115499/`
- 고정 revision: `4f779685e897ebadd835b816cf4ed03fef38234c`
- archive: `canonical_en_generations_115499.tar.zst`
- archive 크기: 288,110,416 bytes
- archive SHA-256: `0b886cc8857dbdb36b26526c9ec35a21caab6b594d795a0b1f182d9b78740232`
- 원격 상태: private 확인

접근 권한이 있는 계정으로 다음처럼 복원한다.

```bash
hf download jin-kwon/fakejail-data \
  --repo-type dataset \
  --revision 4f779685e897ebadd835b816cf4ed03fef38234c \
  --include 'run_checkpoints/v5_full501_360_pc2_20261006/canonical_en_checkpoint_115499/**' \
  --local-dir ./fakejail_private_checkpoint

cd ./fakejail_private_checkpoint/run_checkpoints/v5_full501_360_pc2_20261006/canonical_en_checkpoint_115499
sha256sum -c SHA256SUMS
tar --zstd -xf canonical_en_generations_115499.tar.zst
```

기계 판독용 포인터는 `docs/V5_FULL501_CHECKPOINT_MANIFEST_2026-10-07.json`에 있다.
