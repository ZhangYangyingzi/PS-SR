#!/usr/bin/env bash
set -euo pipefail

ROOT=/Huang_group/zyyz/Projects/PS-SR/ultraflash_state_aware_adapter_blind_final_v1
PYTHON=/Huang_group/zyyz/home_dir/.conda/envs/ultraflash/bin/python
export CUDA_VISIBLE_DEVICES=7

mkdir -p "$ROOT/logs"
if [[ ! -f "$ROOT/audit/frozen_method_before_blind_test.json" ]]; then
  "$PYTHON" "$ROOT/scripts/prepare.py" freeze >"$ROOT/logs/freeze.log" 2>&1
fi
if [[ ! -f "$ROOT/audit/blindness_audit.json" ]]; then
  "$PYTHON" "$ROOT/scripts/prepare.py" split >"$ROOT/logs/split.log" 2>&1
fi
"$PYTHON" "$ROOT/scripts/evaluate_blind.py" cache >"$ROOT/logs/cache.log" 2>&1
"$PYTHON" "$ROOT/scripts/evaluate_blind.py" stateaware_step1000 >"$ROOT/logs/stateaware.log" 2>&1
"$PYTHON" "$ROOT/scripts/evaluate_blind.py" train100_projection >"$ROOT/logs/projection.log" 2>&1
"$PYTHON" "$ROOT/scripts/evaluate_blind.py" step8000_baseline >"$ROOT/logs/step8000.log" 2>&1
"$PYTHON" "$ROOT/scripts/evaluate_blind.py" pixel >"$ROOT/logs/pixel_pssr.log" 2>&1
"$PYTHON" "$ROOT/scripts/evaluate_blind.py" score >"$ROOT/logs/score.log" 2>&1
"$PYTHON" "$ROOT/scripts/finalize.py" summarize >"$ROOT/logs/summarize.log" 2>&1
"$PYTHON" "$ROOT/scripts/finalize.py" visuals >"$ROOT/logs/visuals.log" 2>&1
"$PYTHON" "$ROOT/scripts/finalize.py" print
