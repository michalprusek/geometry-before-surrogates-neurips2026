#!/bin/bash
# (C) one LassoBench run of scripts/run_lassobench.py from the analysis clone, CPU only,
# 1 thread (as the stored 2026-04-27/28 runs: run_lasso_extension.sh used OMP/MKL/OPENBLAS=1,
# and as specified in cr/G_floor_lasso_geometry.md section 2), nice 10, with a provenance record
# appended to <group>/manifest.jsonl.
# usage: run_lasso.sh <group> <label> <results_subdir> <runner_script> <methods> <task> <seed> <timeout_s> [timev]
set -uo pipefail
source /path/to/cr_analysis/code/common.sh
group=$1; label=$2; sub=$3; script=$4; methods=$5; task=$6; seed=$7; tmo=$8; timev=${9:-}
G=$LCR/$group
RES=$G/$sub
mkdir -p "$RES" "$G/logs" "$G/cmds" "$G/status"
for m in ${methods//,/ }; do
  if [ -e "$RES/${m}_${task}_s${seed}.json" ]; then echo "REFUSE: $RES/${m}_${task}_s${seed}.json exists" >&2; exit 3; fi
done
check_clone || exit 4
cd "$G"
[ -e "$G/preflight_first.txt" ] || preflight "$G/preflight_first.txt"
CMD=(env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH="$CL"
     PYTHONDONTWRITEBYTECODE=1 TMPDIR=/path/to/tmp
     timeout --kill-after=10 "$tmo" nice -n "$NICE" "$PY" -B "$script"
     --methods "$methods" --tasks "$task" --seeds "$seed" --results-dir "$RES")
if [ "$timev" = "timev" ]; then CMD=(/usr/bin/time -v "${CMD[@]}"); fi
{ printf '%q ' "${CMD[@]}"; echo; } > "$G/cmds/$label.cmd"
start=$(date -Iseconds)
"${CMD[@]}" > "$G/logs/$label.log" 2>&1
rc=$?
end=$(date -Iseconds)
"$PY" -I -B "$CODE/manifest_append.py" --manifest "$G/manifest.jsonl" --group "$group" --label "$label" \
  --methods "$methods" --task "$task" --seed "$seed" --rc "$rc" --start "$start" --end "$end" \
  --cmd-file "$G/cmds/$label.cmd" --log "$G/logs/$label.log" --results-dir "$RES" --script "$script" \
  --clone "$CL" --freeze "$FREEZE" --python "$PY" --slot "${CR_ANA_SLOT:-none}" --timeout-s "$tmo" --cwd "$PWD"
echo "$rc $start $end" > "$G/status/$label.rc"
exit $rc
