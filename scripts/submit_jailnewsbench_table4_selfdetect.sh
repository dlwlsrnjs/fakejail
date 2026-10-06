#!/bin/bash
set -euo pipefail
cd /home/ljk98/POLY

JNB_TABLE2_JOB=${1:-}
if [[ -n "$JNB_TABLE2_JOB" ]]; then
  JNB_BUILD_JOB=$(sbatch --parsable --dependency="afterok:${JNB_TABLE2_JOB}" slurm/jnb_table4_build.sbatch)
else
  JNB_BUILD_JOB=$(sbatch --parsable slurm/jnb_table4_build.sbatch)
fi
JNB_EXTERNAL_JOB=$(sbatch --parsable --dependency="afterok:${JNB_BUILD_JOB}" slurm/jnb_table4_qwen3_8b_external_array.sbatch)
JNB_EXTRACT_JOB=$(sbatch --parsable --dependency="afterok:${JNB_BUILD_JOB}" slurm/jnb_table4_qwen3_8b_extract_array.sbatch)
JNB_EXTERNAL_SUMMARY_JOB=$(sbatch --parsable --dependency="afterok:${JNB_EXTERNAL_JOB}" slurm/jnb_table4_external_summary.sbatch)
JNB_PROBE_JOB=$(sbatch --parsable --dependency="afterok:${JNB_EXTRACT_JOB}" slurm/jnb_table4_qwen3_8b_probe.sbatch)
JNB_FINAL_JOB=$(sbatch --parsable --dependency="afterok:${JNB_EXTERNAL_SUMMARY_JOB}:${JNB_PROBE_JOB}" slurm/jnb_table4_finalize.sbatch)

echo "build=${JNB_BUILD_JOB}"
echo "external=${JNB_EXTERNAL_JOB}"
echo "extract=${JNB_EXTRACT_JOB}"
echo "external_summary=${JNB_EXTERNAL_SUMMARY_JOB}"
echo "probe=${JNB_PROBE_JOB}"
echo "final=${JNB_FINAL_JOB}"
