#!/bin/bash
# Launch GP-competitor budget scaling: N=1000 (extract N=50,100,200,500,1000 from best_trace).
#
# Usage: bash scripts/launch_gp_budget_scaling.sh <solver> [start_seed] [end_seed] [gpu]
# solver: hvarfner_scaled_gp | papenmeier_msr | xu_standard_gp

set -eo pipefail
export PYTHONPATH=/path/to/NeurIPS
PYTHON=/path/to/NeurIPS/.venv312/bin/python3

SOLVER=$1
START_SEED=${2:-42}
END_SEED=${3:-51}
export CUDA_VISIBLE_DEVICES=${4:-0}

TASKS="adip dhop fexo med1 med2 osmb pdop rano shop siga zale"
TIMEOUT_SEC=7200  # 120 min for N=1000 (GP kernel scales O(N³); plateau data can be slow)

LOG_DIR="logs/${SOLVER}_b1000_s${START_SEED}-${END_SEED}_gpu${CUDA_VISIBLE_DEVICES}"
mkdir -p "$LOG_DIR"

TOTAL=0; DONE=0; OK=0; TIMED_OUT=0; FAILED=0
for seed in $(seq $START_SEED $END_SEED); do for task in $TASKS; do TOTAL=$((TOTAL + 1)); done; done

echo "=== Launching $SOLVER N=1000 budget scaling, seeds $START_SEED-$END_SEED ($TOTAL runs) ==="
echo "Start: $(date)"
echo "Per-run logs: $LOG_DIR/"

for seed in $(seq $START_SEED $END_SEED); do
    for task in $TASKS; do
        DONE=$((DONE + 1))
        POLI=$($PYTHON -c "from src.analysis.config import GUACAMOL_TO_POLI; print(GUACAMOL_TO_POLI.get('$task',''))")
        EXISTING="results/guacamol/spherevo/ours_${SOLVER}_${POLI}_s${seed}_128d_b1000.json"
        if [ -f "$EXISTING" ]; then
            echo "[$DONE/$TOTAL] SKIP $SOLVER $task s$seed b1000 (exists)"
            OK=$((OK + 1))
            continue
        fi

        LOG="$LOG_DIR/${SOLVER}_${task}_s${seed}_b1000.log"
        echo "[$DONE/$TOTAL] RUN  $SOLVER $task s$seed b1000 -> $LOG"

        set +e
        timeout $TIMEOUT_SEC $PYTHON scripts/run_benchmark.py \
            --mode ours --solver $SOLVER --task $task --seed $seed \
            --max-iter 995 --n-cold-start 5 --budget 1000 > "$LOG" 2>&1
        RC=$?
        set -e

        if [ $RC -eq 0 ]; then
            OK=$((OK + 1))
        elif [ $RC -eq 124 ]; then
            echo "  TIMEOUT after ${TIMEOUT_SEC}s (see $LOG)"
            TIMED_OUT=$((TIMED_OUT + 1))
        else
            echo "  FAILED rc=$RC (see $LOG)"
            tail -20 "$LOG" | sed 's/^/    /'
            FAILED=$((FAILED + 1))
        fi
    done
done

echo "=== Done: $SOLVER N=1000 | OK=$OK TIMEOUT=$TIMED_OUT FAILED=$FAILED of $TOTAL | $(date) ==="

# Integrity sweep: missing JSONs are a scientific failure, not a success
MISSING=0
for seed in $(seq $START_SEED $END_SEED); do
    for task in $TASKS; do
        POLI=$($PYTHON -c "from src.analysis.config import GUACAMOL_TO_POLI; print(GUACAMOL_TO_POLI.get('$task',''))")
        EXISTING="results/guacamol/spherevo/ours_${SOLVER}_${POLI}_s${seed}_128d_b1000.json"
        [ -f "$EXISTING" ] || MISSING=$((MISSING + 1))
    done
done
if [ $MISSING -gt 0 ]; then
    echo "WARNING: $MISSING result JSONs are missing — do NOT aggregate without re-running."
fi
