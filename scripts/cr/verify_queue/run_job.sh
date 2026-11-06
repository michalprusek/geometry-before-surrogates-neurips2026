#!/bin/bash
# run_job.sh "<recipe> <benchmark> <seed> <budget> <batch> <n_init|-> <m_d|->" : one verification run.
# n_init "-" = harness default (cmaes: x0 = 0, no design); m_d "-" = not passed (baxus only).
# Writes status/<id>.rc = "rc start end"; the id mirrors the harness stem plus the batch size.
set -u
source "$(dirname "$0")/common.sh"
read -r RECIPE BENCH SEED BUDGET BATCH NINIT MD <<< "$1"
: "${MD:?job line needs 7 fields}"
NID=$NINIT; [ "$NINIT" = "-" ] && NID=0
ID="${RECIPE}__${BENCH//:/-}__q${BATCH}__n${NID}__s${SEED}"
RC=$RUNROOT/status/$ID.rc
# RUN_RULES 2026-10-19: verification ends with the recipes whose runs are complete; these recipes are not run.
case "$RECIPE" in
  baxus|turbo_published|cmaes)
    echo "$(date -Iseconds) $ID not run (RUN_RULES 2026-10-19)" >> "$RUNROOT/logs/not_run_2026-10-19.txt"
    exit 0;;
esac
if [ -f "$RC" ] && [ "$(cut -d' ' -f1 "$RC")" = "0" ]; then exit 0; fi
check_clone || { echo "99 $(date -Iseconds) $(date -Iseconds) refused" > "$RC"; exit 99; }
EXTRA=()
[ "$NINIT" != "-" ] && EXTRA+=(--n-init "$NINIT")
[ "$MD" != "-" ] && EXTRA+=(--m-d "$MD")
start=$(date -Iseconds)
cd "$CL"
nice -n $NICE "$PY" scripts/cr/verify_baselines.py run --recipe "$RECIPE" --benchmark "$BENCH" --seed "$SEED" \
  --budget "$BUDGET" --batch-size "$BATCH" ${EXTRA[@]+"${EXTRA[@]}"} --device cpu --out-dir "$OUT" > "$RUNROOT/logs/$ID.log" 2>&1
rc=$?
echo "$rc $start $(date -Iseconds)" > "$RC"
exit $rc
