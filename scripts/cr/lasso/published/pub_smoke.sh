#!/bin/bash
# Smoke at the tag: each of the ten methods once on synt_simple, seed 900, group pub_smoke (exit codes only are read).
source /path/to/cr_pubprod/code/common.sh
check_clone || exit 4
for m in $(python3 -I "$CODE/pub_jobs.py" methods); do
  echo "bash $CODE/run_lasso.sh pub_smoke ${m}_synt_simple_s900 results $CL/scripts/run_lassobench.py $m synt_simple 900 3600"
done | xargs -P 10 -I {} bash -c "{}"
date -Iseconds > "$LCR/pub_smoke/smoke.done"
