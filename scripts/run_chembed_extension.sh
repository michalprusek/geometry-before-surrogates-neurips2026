#!/bin/bash
# 40-seed Chembed VAE benchmark on 11 GuacaMol tasks for 7 methods.
#
# Tests two paper claims at scale on a second VAE codec:
#   Q1: Confirm 10-seed Chembed inversion (CMA-full > GeoWalk) at 40 seeds
#   Q2: Test whether Hvarfner-on-sphere vs Hvarfner-on-box ablation
#       transfers from SELFIES to Chembed (paper §5 central claim)
#
# Usage: bash scripts/run_chembed_extension.sh
# Logs:  /tmp/chembed_extension/

set -u

PROJECT=/path/to/NeurIPS
RESULTS_BASE="$PROJECT/results/guacamol/chembed"
LOGS=/tmp/chembed_extension
PYTHON=/tmp/chembed_venv/bin/python

# Per-method job concurrency limits (GP variants tighter due to memory)
JOBS_FAST=24       # geowalk, cmaes_sub, cma_full, turbo, random_search
JOBS_GP_SPHERE=12  # hvarfner_scaled_gp (qLogEI on-sphere candidate sampling)
JOBS_GP_BOX=8      # hvarfner_scaled_gp_box (heavy L-BFGS optimize_acqf in 256D)

PER_JOB_TIMEOUT=600  # 10 min per (method, task, seed) cell

mkdir -p "$RESULTS_BASE" "$LOGS"
cd "$PROJECT"

run_one() {
    local method="$1"
    local task="$2"
    local seed="$3"

    local out_dir="$RESULTS_BASE/${method}"
    local out="$out_dir/ours_chembed_${method}_${task}_s${seed}_128d.json"
    if [ -f "$out" ]; then return 0; fi
    mkdir -p "$out_dir"

    local log="$LOGS/${method}_${task}_s${seed}.log"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        PYTHONPATH="$PROJECT" CUDA_VISIBLE_DEVICES=0 \
        timeout --kill-after=15 "$PER_JOB_TIMEOUT" \
        "$PYTHON" "$PROJECT/scripts/run_benchmark.py" \
            --mode ours --codec chembed \
            --solver "$method" --task "$task" --seed "$seed" \
            --max-iter 25 --n-cold-start 5 \
            > "$log" 2>&1

    local rc=$?
    if [ $rc -eq 124 ]; then
        echo "TIMEOUT: $method/$task/$seed (>${PER_JOB_TIMEOUT}s)" >> "$LOGS/_timeouts.log"
    elif [ $rc -ne 0 ]; then
        echo "FAIL ($rc): $method/$task/$seed" >> "$LOGS/_failures.log"
    fi
}
export -f run_one
export PROJECT RESULTS_BASE LOGS PYTHON PER_JOB_TIMEOUT

TASKS="amlodipine_mpo deco_hop fexofenadine_mpo median_1 median_2 osimetrinib_mpo perindopril_mpo ranolazine_mpo scaffold_hop sitagliptin_mpo zaleplon_mpo"

# ---------------------------------------------------------------------------
# Phase 1: Fast methods (5 of them) at 40 seeds × 11 tasks via JOBS_FAST
# ---------------------------------------------------------------------------
JOBLIST_FAST="$LOGS/_joblist_fast.txt"
> "$JOBLIST_FAST"

for method in geowalk cma_es_subspace cma_es turbo random_search; do
    for seed in $(seq 42 81); do
        for task in $TASKS; do
            echo "$method $task $seed" >> "$JOBLIST_FAST"
        done
    done
done

N_FAST=$(wc -l < "$JOBLIST_FAST")
echo "[Phase 1] Fast methods: $N_FAST jobs ($JOBS_FAST parallel, ${PER_JOB_TIMEOUT}s timeout each)"
echo "Started: $(date)"
echo "----"

xargs -P "$JOBS_FAST" -L 1 -a "$JOBLIST_FAST" bash -c 'run_one "$@"' _

echo "----"
echo "[Phase 1] Finished: $(date)"

# ---------------------------------------------------------------------------
# Phase 2: Hvarfner-on-sphere at 40 seeds × 11 tasks via JOBS_GP_SPHERE
# ---------------------------------------------------------------------------
JOBLIST_GP_SPHERE="$LOGS/_joblist_gp_sphere.txt"
> "$JOBLIST_GP_SPHERE"

for seed in $(seq 42 81); do
    for task in $TASKS; do
        echo "hvarfner_scaled_gp $task $seed" >> "$JOBLIST_GP_SPHERE"
    done
done

N_GP_SPHERE=$(wc -l < "$JOBLIST_GP_SPHERE")
echo "[Phase 2] Hvarfner-on-sphere: $N_GP_SPHERE jobs ($JOBS_GP_SPHERE parallel)"
echo "Started: $(date)"
echo "----"

xargs -P "$JOBS_GP_SPHERE" -L 1 -a "$JOBLIST_GP_SPHERE" bash -c 'run_one "$@"' _

echo "----"
echo "[Phase 2] Finished: $(date)"

# ---------------------------------------------------------------------------
# Phase 3: Hvarfner-on-box at 40 seeds × 11 tasks via JOBS_GP_BOX (slower)
# ---------------------------------------------------------------------------
JOBLIST_GP_BOX="$LOGS/_joblist_gp_box.txt"
> "$JOBLIST_GP_BOX"

for seed in $(seq 42 81); do
    for task in $TASKS; do
        echo "hvarfner_scaled_gp_box $task $seed" >> "$JOBLIST_GP_BOX"
    done
done

N_GP_BOX=$(wc -l < "$JOBLIST_GP_BOX")
echo "[Phase 3] Hvarfner-on-box: $N_GP_BOX jobs ($JOBS_GP_BOX parallel)"
echo "Started: $(date)"
echo "----"

xargs -P "$JOBS_GP_BOX" -L 1 -a "$JOBLIST_GP_BOX" bash -c 'run_one "$@"' _

echo "----"
echo "[Phase 3] Finished: $(date)"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=========================================="
echo "ALL PHASES DONE: $(date)"
echo "=========================================="
echo "File counts per method:"
for method in geowalk cma_es_subspace cma_es turbo random_search hvarfner_scaled_gp hvarfner_scaled_gp_box; do
    n=$(ls "$RESULTS_BASE/${method}/"*.json 2>/dev/null | wc -l)
    echo "  $method: $n / 440 (40 seeds × 11 tasks)"
done

if [ -f "$LOGS/_timeouts.log" ]; then
    echo ""
    echo "Timeouts: $(wc -l < "$LOGS/_timeouts.log")"
fi
if [ -f "$LOGS/_failures.log" ]; then
    echo "Failures: $(wc -l < "$LOGS/_failures.log")"
fi
