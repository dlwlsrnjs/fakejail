#!/bin/bash
set -euo pipefail
cd /home/ljk98/POLY

JNB_PREPARE_JOB=$(sbatch --parsable slurm/jnb_table2_qwen3_8b_prepare.sbatch)
JNB_GENERATE_JOB=$(sbatch --parsable --dependency="afterok:${JNB_PREPARE_JOB}" slurm/jnb_table2_qwen3_8b_generate_array.sbatch)
JNB_JUDGE_JOB=$(sbatch --parsable --dependency="aftercorr:${JNB_GENERATE_JOB}" slurm/jnb_table2_qwen32_judge_array.sbatch)
JNB_AGGREGATE_JOB=$(sbatch --parsable --dependency="afterok:${JNB_JUDGE_JOB}" slurm/jnb_table2_qwen32_aggregate.sbatch)

echo "prepare=${JNB_PREPARE_JOB}"
echo "generate=${JNB_GENERATE_JOB}"
echo "judge=${JNB_JUDGE_JOB}"
echo "aggregate=${JNB_AGGREGATE_JOB}"
