#!/bin/bash
# Launch parallelized GeoWalk budget scaling (b200, b500, b1000).
# Uses 3 parallel seed workers per GPU via helper scripts.
# Resumes from partial state via run_benchmark.py's skip-if-exists logic.
#
# Assignment:
#   GPU 0: b200 then b1000
#   GPU 1: b500 (single pass)
#
# Usage:   bash scripts/launch_geowalk_budget_parallel.sh
# Monitor: tmux attach -t gwbudget
#          for b in 200 500 1000; do echo -n "b$b: "; ls results/guacamol/geowalk/*_b${b}*.json 2>/dev/null | wc -l; done

set -euo pipefail
cd /path/to/NeurIPS

SESSION="gwbudget"

chmod +x scripts/_geowalk_budget_worker.sh scripts/_geowalk_gpu_driver.sh

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n "gpu0" "bash scripts/_geowalk_gpu_driver.sh 0 200 1000"
tmux new-window  -t "$SESSION" -n "gpu1" "bash scripts/_geowalk_gpu_driver.sh 1 500"

echo "Parallel GeoWalk budget scaling launched in tmux session '$SESSION'"
echo "  GPU0: b200 + b1000 (3 parallel seed workers each)"
echo "  GPU1: b500 (3 parallel seed workers)"
echo "  Monitor: tmux attach -t $SESSION"
