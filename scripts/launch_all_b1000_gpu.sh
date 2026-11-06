#!/bin/bash
# Launch all three GP-based methods at budget=1000 on GPU.
# Uses custom GPU runners (not poli-baselines).
# Total: 360 runs. Estimated: TuRBO ~40h, VanillaBO ~10h, BAxUS ~15h.
set -uo pipefail
# Note: NOT set -e — individual runs may segfault (rdkit/numpy compat) and we want to continue

cd /path/to/NeurIPS
source .venv/bin/activate
export PYTHONPATH=/path/to/NeurIPS
export CUDA_VISIBLE_DEVICES="1"

TASKS=(adip dhop fexo med1 med2 osmb pdop rano shop siga valt zale)
SEEDS=(42 43 44 45 46 47 48 49 50 51)

declare -A POLI_MAP
eval "$(python3 -c "
from src.analysis.config import GUACAMOL_TO_POLI
for k,v in GUACAMOL_TO_POLI.items():
    print(f'POLI_MAP[{k}]={v}')
")"

run_method() {
    local method="$1"
    local script="$2"
    local log_dir="logs/${method}_b1000"
    local file_prefix="$3"
    mkdir -p "$log_dir"

    local total=0 skipped=0 launched=0
    echo ""
    echo "========================================"
    echo "  $method  ($(date))"
    echo "========================================"

    for task in "${TASKS[@]}"; do
        poli_task="${POLI_MAP[$task]}"
        for seed in "${SEEDS[@]}"; do
            total=$((total + 1))

            # Skip if result exists
            if ls results/guacamol/spherevo/ours_${file_prefix}_${poli_task}_s${seed}_*_b1000.json 2>/dev/null | grep -q .; then
                skipped=$((skipped + 1))
                continue
            fi

            LOG="$log_dir/${task}_s${seed}.log"
            echo "[$(date +%H:%M:%S)] $method task=$task seed=$seed ..."
            if eval "$script" > "$LOG" 2>&1; then
                echo "[$(date +%H:%M:%S)] Done $method task=$task seed=$seed"
                launched=$((launched + 1))
            else
                echo "[$(date +%H:%M:%S)] FAILED $method task=$task seed=$seed (exit $?)"
            fi
        done
    done
    echo "$method: Total=$total Skipped=$skipped Launched=$launched"
}

# 1. Vanilla BO (fastest — ~5min/run)
run_method "vanilla_bo" \
    "python scripts/run_vanilla_bo_budget.py --task \$task --seed \$seed --budget 1000 --device cuda" \
    "vanilla_bo_hvarfner"

# 2. BAxUS (~7min/run)
run_method "baxus" \
    "python scripts/run_baxus_budget.py --task \$task --seed \$seed --budget 1000 --device cuda" \
    "baxus"

# 3. TuRBO (slowest — ~20min/run)
run_method "turbo" \
    "python scripts/run_turbo_budget.py --task \$task --seed \$seed --budget 1000 --device cuda" \
    "turbo"

echo ""
echo "All done! $(date)"
