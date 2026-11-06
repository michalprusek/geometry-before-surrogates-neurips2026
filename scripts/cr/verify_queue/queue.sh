#!/bin/bash
# queue.sh <jobs file> <parallel>: run every job line through run_job.sh, longest first (file order), then flag done.
set -u
source "$(dirname "$0")/common.sh"
JOBS=$1; P=${2:-24}
{ echo "date: $(date -Iseconds)"; echo "host: $(hostname -f)"; who; uptime; nproc
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  echo "clone HEAD: $(git -C "$CL" rev-parse HEAD) tag: $(git -C "$CL" describe --tags --exact-match HEAD 2>&1)"
  echo "targets sha256: $(sha256sum "$CL/scripts/cr/baseline_targets.json" | cut -d' ' -f1)"
  echo "python: $PY $("$PY" -V 2>&1)"; echo "jobs: $(grep -c . "$JOBS") parallel: $P"; } > $RUNROOT/logs/preflight_$(date +%Y%m%d_%H%M%S).txt 2>&1
check_clone || exit 1
grep -v '^#' "$JOBS" | grep . | xargs -P "$P" -I{} bash "$(dirname "$0")/run_job.sh" "{}"
echo "$(date -Iseconds)" > $RUNROOT/queue.done
