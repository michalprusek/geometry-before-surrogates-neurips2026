#!/usr/bin/env bash
# Vanilla BO 40-seed completion: run vanilla_bo_hvarfner at b=1000 for all
# (task, seed) pairs in seeds 52..81 across the 11 non-degenerate GuacaMol
# tasks. Each invocation skips if the result file already exists, so the
# script is safe to re-run after interruption.
#
# Usage:
#   GPU=0 TASKS="amlodipine_mpo median_1 osimetrinib_mpo ranolazine_mpo zaleplon_mpo scaffold_hop" \
#       bash scripts/run_vanilla_bo_40seed_completion.sh
#
# Splits across two GPUs by passing different TASKS subsets per invocation.
set -uo pipefail

export PYTHONPATH=/path/to/NeurIPS
PYTHON=/path/to/NeurIPS/.venv312/bin/python3
GPU="${GPU:-0}"
SEEDS_START="${SEEDS_START:-52}"
SEEDS_END="${SEEDS_END:-81}"
BUDGET="${BUDGET:-1000}"
# Short task names — `run_vanilla_bo_budget.py` translates to poli names via
# GUACAMOL_TO_POLI when reading the oracle.
TASKS="${TASKS:-adip fexo med1 med2 osmb pdop rano siga zale dhop shop}"

# Map short → poli for filename existence-check (output filename uses poli).
declare -A SHORT_TO_POLI=(
    [adip]=amlodipine_mpo [fexo]=fexofenadine_mpo
    [med1]=median_1       [med2]=median_2
    [osmb]=osimetrinib_mpo [pdop]=perindopril_mpo
    [rano]=ranolazine_mpo [siga]=sitagliptin_mpo
    [zale]=zaleplon_mpo   [dhop]=deco_hop
    [shop]=scaffold_hop
)

LOGDIR="/path/to/NeurIPS/logs/vanilla_bo_40seed"
mkdir -p "$LOGDIR"
RESULTS_DIR="/path/to/NeurIPS/results/guacamol/spherevo"

export CUDA_VISIBLE_DEVICES="$GPU"
echo "=== Vanilla BO 40-seed completion ==="
echo "GPU=$GPU  seeds=$SEEDS_START..$SEEDS_END  budget=$BUDGET"
echo "tasks: $TASKS"
echo "started: $(date)"
echo

total=0; done=0; skipped=0; failed=0
for short in $TASKS; do
    poli="${SHORT_TO_POLI[$short]:-$short}"
    for seed in $(seq "$SEEDS_START" "$SEEDS_END"); do
        total=$((total+1))
        out_file="$RESULTS_DIR/ours_vanilla_bo_hvarfner_${poli}_s${seed}_256d_b${BUDGET}.json"
        if [[ -f "$out_file" ]]; then
            skipped=$((skipped+1))
            continue
        fi
        log_file="$LOGDIR/${poli}_s${seed}_gpu${GPU}.log"
        echo "[$(date +%H:%M:%S)] task=$short ($poli) seed=$seed gpu=$GPU -> $log_file"
        if "$PYTHON" /path/to/NeurIPS/scripts/run_vanilla_bo_budget.py \
                --task "$short" --seed "$seed" --budget "$BUDGET" --device cuda \
                >"$log_file" 2>&1; then
            done=$((done+1))
        else
            echo "  FAILED rc=$? (see $log_file)"
            failed=$((failed+1))
        fi
    done
done

echo
echo "=== summary ==="
echo "total=$total done=$done skipped=$skipped failed=$failed"
echo "finished: $(date)"
