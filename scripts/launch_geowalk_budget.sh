#!/bin/bash
# Launch GeoWalk budget scaling: N=50,100,200,500,1000 on 12 tasks x 10 seeds.
# Splits budgets across two GPUs in a tmux session.
#
# Usage:  bash scripts/launch_geowalk_budget.sh
# Monitor: tmux attach -t gwbudget
#
# Expected runtime: ~2h total (higher budgets dominate)

set -euo pipefail
cd /path/to/NeurIPS
export PYTHONPATH=/path/to/NeurIPS
export PYTHONUNBUFFERED=1

SESSION="gwbudget"
SEEDS="42 43 44 45 46 47 48 49 50 51"

tmux kill-session -t "$SESSION" 2>/dev/null || true

# GPU 0: budgets 50, 200, 1000 (mixed sizes for load balance)
tmux new-session -d -s "$SESSION" -n "gpu0" \
  "for budget in 50 200 1000; do \
     for seed in $SEEDS; do \
       PYTHONPATH=$PYTHONPATH PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
         python scripts/run_benchmark.py --mode ours --solver geowalk \
         --task ours --seed \$seed --budget \$budget --n-cold-start 5 2>&1 | \
         tail -2; \
     done; \
   done; \
   echo '=== GPU0 GEOWALK BUDGET DONE ==='; \
   sleep 3600"

# GPU 1: budgets 100, 500
tmux new-window -t "$SESSION" -n "gpu1" \
  "for budget in 100 500; do \
     for seed in $SEEDS; do \
       PYTHONPATH=$PYTHONPATH PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
         python scripts/run_benchmark.py --mode ours --solver geowalk \
         --task ours --seed \$seed --budget \$budget --n-cold-start 5 2>&1 | \
         tail -2; \
     done; \
   done; \
   echo '=== GPU1 GEOWALK BUDGET DONE ==='; \
   sleep 3600"

echo "GeoWalk budget scaling launched in tmux session '$SESSION'"
echo "  GPU0: budgets 50, 200, 1000 (30 invocations x 12 tasks = 360 files)"
echo "  GPU1: budgets 100, 500 (20 invocations x 12 tasks = 240 files)"
echo "  Total: 600 new files in results/guacamol/geowalk/"
echo "  Monitor: tmux attach -t $SESSION"
