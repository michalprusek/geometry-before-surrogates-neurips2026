#!/bin/bash
# Full LassoBench rerun (RUN_RULES amendment 2026-10-12): 6 single-thread jobs in parallel, nice 10.
cd /path/to/cr_analysis || exit 1
echo "full rerun start $(date -Iseconds), $(wc -l < queue/jobs_full_rerun.txt) jobs" | tee queue/full_rerun.log
who >> queue/full_rerun.log; uptime >> queue/full_rerun.log
xargs -P 6 -I {} bash -c "{}" < queue/jobs_full_rerun.txt
echo "full rerun end $(date -Iseconds) xargs rc=$?" | tee -a queue/full_rerun.log
date -Iseconds > queue/full_rerun.done
