#!/usr/bin/env bash
set -euo pipefail
source /data1/anaconda3_new/anaconda_program/etc/profile.d/conda.sh
conda activate ultraflash
export CUDA_VISIBLE_DEVICES=7
ROOT=/Huang_group/zyyz/Projects/PS-SR/ultraflash_state_aware_adapter_6depth_v1
PY="$ROOT/scripts/continue_1000_3000.py"
mkdir -p "$ROOT/continuation_1000_3000/logs"
if ! python "$PY" audit >> "$ROOT/continuation_1000_3000/logs/audit.log" 2>&1; then
  if [[ -f "$ROOT/audit/resume_failure.json" ]] && grep -q 'optimizer state missing' "$ROOT/audit/resume_failure.json"; then
    printf 'RESUME_BLOCKED:\noptimizer state missing\n'
  else
    printf 'RESUME_AUDIT_FAILED: %s\n' "$ROOT/continuation_1000_3000/logs/audit.log"
  fi
  exit 1
fi
if ! python "$PY" equivalence >> "$ROOT/continuation_1000_3000/logs/equivalence.log" 2>&1; then
  printf 'RESUME_EQUIVALENCE_FAILED: %s\n' "$ROOT/audit/resume_equivalence.json"
  exit 1
fi
if ! python "$PY" train >> "$ROOT/continuation_1000_3000/logs/train.log" 2>&1; then
  printf 'RESUME_TRAINING_STOPPED: %s\n' "$ROOT/continuation_1000_3000/logs/train.log"
  exit 1
fi
for step in 1500 2000 2500 3000; do
  if [[ ! -f "$ROOT/metrics/val_step${step}_macro.json" ]]; then
    python "$ROOT/scripts/evaluate_continuation.py" --step "$step" >> "$ROOT/continuation_1000_3000/logs/val_step${step}.log" 2>&1
  fi
done
python "$ROOT/scripts/print_continuation.py"
