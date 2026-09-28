#!/usr/bin/env bash
set -euo pipefail
cd /Huang_group/zyyz/Projects/PS-SR/ultraflash_condition_projection_adaptation_v1
export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/Huang_group/zyyz/home_dir/.conda/envs/ultraflash/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec /Huang_group/zyyz/home_dir/.conda/envs/ultraflash/bin/python scripts/run_all.py
