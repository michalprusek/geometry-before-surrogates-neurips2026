#!/usr/bin/env bash
# Generic 11-task x 40-seed sweep for one solver, with skip-if-exists.
#
# Usage:  bash scripts/run_solver_sweep.sh GPU SOLVER [CODEC]
#   e.g.  bash scripts/run_solver_sweep.sh 0 papenmeier_msr_box_faithful
#
# Writes wherever run_benchmark.py routes the solver (spherevo/ for selfies,
# the codec-specific root otherwise). Excludes valsartan_smarts, which scores 0.
set -uo pipefail

GPU="${1:?usage: run_solver_sweep.sh GPU SOLVER [CODEC]}"
SOLVER="${2:?usage: run_solver_sweep.sh GPU SOLVER [CODEC]}"
CODEC="${3:-selfies}"
ROOT=/path/to/NeurIPS

# CODEC is the run_benchmark.py --codec value; the result root can differ from it
# (spcauchy writes to results/guacamol/spcauchy_40seed/), so keep them separate.
case "$CODEC" in
  selfies)         RESULT_ROOT="spherevo" ;;
  spcauchy)        RESULT_ROOT="spcauchy_40seed" ;;
  chembed)         RESULT_ROOT="chembed" ;;
  power_spherical) RESULT_ROOT="power_spherical" ;;
  vmf)             RESULT_ROOT="vmf" ;;
  *) echo "unknown codec '$CODEC'" >&2; exit 2 ;;
esac

if [ "$CODEC" = "selfies" ]; then
  OUT="$ROOT/results/guacamol/${RESULT_ROOT}"
  PREFIX="ours_${SOLVER}"
  CODEC_ARG=()
else
  OUT="$ROOT/results/guacamol/${RESULT_ROOT}/${SOLVER}"
  PREFIX="ours_${CODEC}_${SOLVER}"
  CODEC_ARG=(--codec "$CODEC")
fi

# Override with TASKS="a,b,c" to shard one solver across several workers.
if [ -n "${TASKS:-}" ]; then
  IFS=',' read -r -a TASKS <<< "$TASKS"
else
  TASKS=(amlodipine_mpo deco_hop fexofenadine_mpo median_1 median_2 \
         osimetrinib_mpo perindopril_mpo ranolazine_mpo scaffold_hop \
         sitagliptin_mpo zaleplon_mpo)
fi

mkdir -p "$OUT"
done_n=0; run_n=0; fail_n=0

for task in "${TASKS[@]}"; do
  for seed in $(seq 42 81); do
    f="$OUT/${PREFIX}_${task}_s${seed}_128d.json"
    if [ -s "$f" ]; then done_n=$((done_n+1)); continue; fi
    PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES="$GPU" \
      python "$ROOT/scripts/run_benchmark.py" \
        --mode ours "${CODEC_ARG[@]}" --solver "$SOLVER" \
        --task "$task" --seed "$seed" --max-iter 25 --n-cold-start 5 \
        >/dev/null 2>&1
    if [ -s "$f" ]; then run_n=$((run_n+1)); else fail_n=$((fail_n+1)); fi
    if (( (run_n+fail_n) % 20 == 0 )); then
      echo "[$SOLVER gpu$GPU] ran=$run_n skipped=$done_n failed=$fail_n"
    fi
  done
done

echo "[$SOLVER gpu$GPU] DONE ran=$run_n skipped=$done_n failed=$fail_n"
