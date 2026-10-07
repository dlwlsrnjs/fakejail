#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_ROOT="/data1/users/ljk98/fakejail_runtime_20261006"
RUN_ROOT="${RUNTIME_ROOT}/artifacts/jailnews_bandit_20260930/runtime/v5_full501_360_pc2_20261006/canonical_en"
SURROGATE_ROOT="${RUNTIME_ROOT}/artifacts/jailnews_bandit_20260930/runtime"
PERSON_EMBEDDINGS="${SURROGATE_ROOT}/dual_router_v1/wikipedia_person_embeddings.npz"
CONTEXT_EMBEDDINGS="${SURROGATE_ROOT}/context_embeddings.npz"
WIKIPEDIA_CACHE="${SURROGATE_ROOT}/person_wikipedia_full_cache.jsonl"
LOCALIZATIONS="${SURROGATE_ROOT}/entity_links_v2_final/person_localizations.jsonl"
INTENT_REASONING="${SURROGATE_ROOT}/dual_router_v2/intent_iterative_qwen32.jsonl"
MODEL="/data1/users/ljk98/hf_cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
PYTHON="/data1/users/ljk98/envs/VLLM-VL-LABEL/bin/python"
EXPECTED_ROWS=180360

export PATH="/data1/users/ljk98/envs/VLLM-VL-LABEL/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export TMPDIR="/data1/users/ljk98/runtime_cache/fakejail_tmp"
export HF_HOME="/data1/users/ljk98/hf_cache"
export TORCHINDUCTOR_CACHE_DIR="/data1/users/ljk98/runtime_cache/fakejail_torchinductor"
export VLLM_CACHE_ROOT="/data1/users/ljk98/runtime_cache/fakejail_vllm"
export TOKENIZERS_PARALLELISM="false"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${RUN_ROOT}/judgments" "${RUN_ROOT}/logs" "${RUN_ROOT}/audit" "${RUN_ROOT}/evaluation"

while true; do
  total=0
  for path in "${RUN_ROOT}"/generations/shard_*.jsonl; do
    if [[ -f "${path}" ]]; then
      count=$(wc -l <"${path}")
      total=$((total + count))
    fi
  done
  if [[ "${total}" -eq "${EXPECTED_ROWS}" ]]; then
    break
  fi
  if [[ "${total}" -gt "${EXPECTED_ROWS}" ]]; then
    echo "generation row overflow: ${total}" >&2
    exit 2
  fi
  sleep 30
done

while pgrep -f '[j]ailnewsbench_table2_qwen32.py generate' >/dev/null; do
  sleep 30
done

generation_files=("${RUN_ROOT}"/generations/shard_*.jsonl)
"${PYTHON}" scripts/audit_jailnews_v5_generations.py \
  --inputs "${generation_files[@]}" \
  --output "${RUN_ROOT}/audit/generation_audit.json" \
  >>"${RUN_ROOT}/logs/generation_audit.log" 2>&1

pids=()
for shard in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="${shard}" "${PYTHON}" scripts/jailnewsbench_table2_qwen32.py judge \
    --inputs "${RUN_ROOT}/generations/shard_*.jsonl" \
    --model "${MODEL}" \
    --model-label "Qwen/Qwen2.5-32B-Instruct" \
    --output "${RUN_ROOT}/judgments/shard_${shard}.jsonl" \
    --chunk-size 2048 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --num-shards 8 \
    --shard-id "${shard}" \
    --strip-qwen-thinking \
    --resume \
    >>"${RUN_ROOT}/logs/judge_shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

"${PYTHON}" scripts/evaluate_jailnews_pc2_full501.py \
  --target "${RUN_ROOT}/judgments/shard_*.jsonl" \
  --surrogate \
    "${SURROGATE_ROOT}/full501_llama8_rr8_v1/judgments/llama31_8b_shard_*.jsonl" \
    "${SURROGATE_ROOT}/full501_llama8_rr8_v1/judgments/llama3_rr8b_shard_*.jsonl" \
    "${SURROGATE_ROOT}/full501_llama8_rr8_repeat_v1/judgments/llama31_8b_draw_1_shard_*.jsonl" \
    "${SURROGATE_ROOT}/full501_llama8_rr8_repeat_v1/judgments/llama3_rr8b_draw_1_shard_*.jsonl" \
  --output-dir "${RUN_ROOT}/evaluation" \
  --person-embeddings "${PERSON_EMBEDDINGS}" \
  --context-embeddings "${CONTEXT_EMBEDDINGS}" \
  --wikipedia-cache "${WIKIPEDIA_CACHE}" \
  --localizations "${LOCALIZATIONS}" \
  --intent-reasoning "${INTENT_REASONING}" \
  --endpoint strict \
  --budgets 1 2 3 5 10 20 30 \
  --simulations 20 \
  --seed 20261100 \
  --initial-representatives 5 \
  --top-two-beta 0.95 \
  --hurdle-weight 0.30 \
  --judge-error-policy fail \
  >>"${RUN_ROOT}/logs/evaluation.log" 2>&1
