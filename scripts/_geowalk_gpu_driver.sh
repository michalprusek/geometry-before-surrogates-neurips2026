#!/bin/bash
# Driver: runs all budgets for a single GPU, with 3 parallel workers per budget.
# Called by launch_geowalk_budget_parallel.sh (once per GPU).
#
# Usage: _geowalk_gpu_driver.sh <gpu> <budget> [<budget> ...]
set -euo pipefail
cd /path/to/NeurIPS

GPU="$1"
shift
BUDGETS=("$@")

for budget in "${BUDGETS[@]}"; do
    echo "=== GPU${GPU} BUDGET ${budget} START $(date +%H:%M:%S) ==="

    # Split 10 seeds into 3 chunks
    bash scripts/_geowalk_budget_worker.sh "$GPU" "$budget" 42 43 44 45 &
    W1=$!
    bash scripts/_geowalk_budget_worker.sh "$GPU" "$budget" 46 47 48 &
    W2=$!
    bash scripts/_geowalk_budget_worker.sh "$GPU" "$budget" 49 50 51 &
    W3=$!

    wait $W1 $W2 $W3
    echo "=== GPU${GPU} BUDGET ${budget} DONE $(date +%H:%M:%S) ==="
done

echo "=== GPU${GPU} ALL BUDGETS DONE $(date +%H:%M:%S) ==="
sleep 3600
