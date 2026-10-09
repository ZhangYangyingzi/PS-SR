#!/usr/bin/env bash
set -euo pipefail
cd /Huang_group/zyyz/Projects/PS-SR/ultraflash_condition_projection_adaptation_v1
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/Huang_group/zyyz/home_dir/.conda/envs/ultraflash/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
/Huang_group/zyyz/home_dir/.conda/envs/ultraflash/bin/python scripts/evaluate.py generate --split val --step 0
/Huang_group/zyyz/home_dir/.conda/envs/ultraflash/bin/python scripts/evaluate.py score --split val --step 0
