#!/bin/bash
# Launch COWBOYS solver for ALL 12 GuacaMol tasks × 10 seeds = 120 runs
# Re-running everything with fixed max_string_length=80 for consistency.
# Run on GPU 0. Estimated ~2.5h total (~76s per run).
set -euo pipefail

export PYTHONPATH=/path/to/NeurIPS
export CUDA_VISIBLE_DEVICES=0

# All 12 poli task names
TASKS="amlodipine_mpo deco_hop fexofenadine_mpo median_1 median_2 osimetrinib_mpo perindopril_mpo ranolazine_mpo scaffold_hop sitagliptin_mpo valsartan_smarts zaleplon_mpo"
SEEDS=$(seq 42 51)
LOGFILE="results/guacamol/spherevo/cowboys_run_all.log"

echo "Starting COWBOYS full run: 12 tasks × 10 seeds = 120 runs" > "$LOGFILE"
echo "Start time: $(date)" >> "$LOGFILE"

count=0
total=120
for task in $TASKS; do
    for seed in $SEEDS; do
        count=$((count + 1))
        echo "$(date +%H:%M:%S) [RUN $count/$total] $task seed=$seed" >> "$LOGFILE"
        if python scripts/run_benchmark.py \
            --mode ours --solver cowboys --task "$task" --seed "$seed" \
            --max-iter 25 --n-cold-start 5 \
            >> "$LOGFILE" 2>&1; then
            echo "$(date +%H:%M:%S) [OK $count/$total] $task seed=$seed" >> "$LOGFILE"
        else
            echo "$(date +%H:%M:%S) [FAIL $count/$total] $task seed=$seed" >> "$LOGFILE"
        fi
    done
done

echo "ALL COWBOYS RUNS COMPLETE" >> "$LOGFILE"
echo "End time: $(date)" >> "$LOGFILE"
