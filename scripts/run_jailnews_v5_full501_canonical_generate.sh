#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/home/ljk98/Documents/Codex/2026-10-06-gpu/fakejail"
RUNTIME_ROOT="/data1/users/ljk98/fakejail_runtime_20261006"
INPUT="${RUNTIME_ROOT}/artifacts/jailnews_bandit_20260930/runtime/stochastic_360_identity_v5_english/canonical_en/base_arms.jsonl"
OUTPUT_ROOT="${RUNTIME_ROOT}/artifacts/jailnews_bandit_20260930/runtime/v5_full501_360_pc2_20261006/canonical_en"
MODEL="/data1/users/ljk98/hf_cache/hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507-FP8/snapshots/60d80c83c53c3b611c642dbb8c942b3f90c5948a"
PYTHON="/data1/users/ljk98/envs/VLLM-VL-LABEL/bin/python"

export PATH="/data1/users/ljk98/envs/VLLM-VL-LABEL/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export TMPDIR="/data1/users/ljk98/runtime_cache/fakejail_tmp"
export HF_HOME="/data1/users/ljk98/hf_cache"
export TORCHINDUCTOR_CACHE_DIR="/data1/users/ljk98/runtime_cache/fakejail_torchinductor"
export VLLM_CACHE_ROOT="/data1/users/ljk98/runtime_cache/fakejail_vllm"
export TOKENIZERS_PARALLELISM="false"

mkdir -p "${OUTPUT_ROOT}/generations" "${OUTPUT_ROOT}/logs"

pids=()
for shard in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="${shard}" "${PYTHON}" scripts/jailnewsbench_table2_qwen32.py generate \
    --input "${INPUT}" \
    --model "${MODEL}" \
    --model-label "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8" \
    --output "${OUTPUT_ROOT}/generations/shard_${shard}.jsonl" \
    --transport chat \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --seed 20261100 \
    --draw-id 0 \
    --max-new-tokens 32768 \
    --chunk-size 16 \
    --tensor-parallel-size 1 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.90 \
    --num-shards 8 \
    --shard-id "${shard}" \
    --resume \
    >>"${OUTPUT_ROOT}/logs/generate_shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
