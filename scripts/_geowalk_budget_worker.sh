#!/bin/bash
# Worker: runs one slice of budget scaling on a single GPU.
# Called by launch_geowalk_budget_parallel.sh (3 workers per GPU).
#
# Usage: _geowalk_budget_worker.sh <gpu> <budget> <seed> [<seed> ...]
set -euo pipefail
cd /path/to/NeurIPS
export PYTHONPATH=/path/to/NeurIPS
export PYTHONUNBUFFERED=1

GPU="$1"
BUDGET="$2"
shift 2

for seed in "$@"; do
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_benchmark.py \
        --mode ours --solver geowalk --task ours \
        --seed "$seed" --budget "$BUDGET" --n-cold-start 5 2>&1 | tail -1
done
