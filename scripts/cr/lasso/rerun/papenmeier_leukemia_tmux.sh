#!/bin/bash
# Papenmeier at D = 7129, seeds 43-81 (RUN_RULES amendment 2026-10-12: "Papenmeier at D = 7129 is run on all
# 40 seeds"). Starts when the full rerun has ended; 6 single-thread jobs in parallel, nice 10, same runner.
cd /path/to/cr_analysis || exit 1
while [ ! -e queue/full_rerun.done ]; do sleep 60; done
echo "papenmeier leukemia start $(date -Iseconds), $(wc -l < queue/jobs_papenmeier_leukemia.txt) jobs" | tee queue/papenmeier_leukemia.log
who >> queue/papenmeier_leukemia.log; uptime >> queue/papenmeier_leukemia.log
xargs -P 6 -I {} bash -c "{}" < queue/jobs_papenmeier_leukemia.txt
echo "papenmeier leukemia end $(date -Iseconds) xargs rc=$?" | tee -a queue/papenmeier_leukemia.log
date -Iseconds > queue/papenmeier_leukemia.done
