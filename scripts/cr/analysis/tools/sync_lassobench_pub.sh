#!/bin/bash
# sync_lassobench_pub.sh -- copy the LassoBench published-method set from kajman to panda, with sha256 manifests.
#
#   bash sync_lassobench_pub.sh            copy and verify   (run from the Mac; needs sync_set.sh next to it)
#   bash sync_lassobench_pub.sh --check    only report the done flag on kajman (copies nothing)
#
# Source (read only):
#   kajman:/data2/prusek/NeurIPS-cr-pubprod/results/lassobench_pub/pub_methods/   results/*.json, manifest.jsonl,
#       status/, cmds/, preflight_first.txt (logs/ stay on kajman). RUN_RULES 2026-10-12 (3), revision 1 (f): ten
#       methods x five tasks x seeds 42-81, ten leukemia pilots (seed 42), leukemia seeds 43-81 for each method
#       whose pilot exited 0 inside the hour (crlib/lasso_cr.py, design PUB).
#   kajman:/data2/prusek/cr_pubprod/queue/   jobs_phase1.txt, jobs_phase2.txt, queue.log, queue.done (the job lists are
#       checked against the expected design by the loader); and code/ (the launcher, for the release).
# Destination: panda:/data/prusek/cr_inputs/lassobench_pub/ and lassobench_pub_queue/ (+ SOURCE_*.sha256, *.sha256).
#
# FINISHED only when kajman:/data2/prusek/cr_pubprod/queue/queue.done exists (written after phase 2 ended). The flag
# is read BEFORE anything is copied; the results are copied first, the queue set last, and the flag only if it
# existed before the copy started. Exit codes: 0 final copy; 3 partial copy (flag missing; --check: flag missing);
# 1 the flag could not be read or a set could not be copied consistently; 2 usage.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC_HOST=${LASSO_SRC_HOST:-kajman}
DST_HOST=${LASSO_DST_HOST:-panda}
SRC=${LASSO_PUB_SRC_DIR:-/data2/prusek/NeurIPS-cr-pubprod/results/lassobench_pub}
QSRC=${LASSO_PUB_QUEUE_DIR:-/data2/prusek/cr_pubprod}
DST_PARENT=${LASSO_DST_PARENT:-/data/prusek/cr_inputs}
TRIES=${LASSO_SYNC_TRIES:-10}
WAIT=${LASSO_SYNC_WAIT:-15}
FLAG=queue.done

CHECK=0
for a in "$@"; do
  case "$a" in
    --check) CHECK=1;;
    *) echo "usage: $0 [--check]" >&2; exit 2;;
  esac
done

read_flag() {   # sets MISSING=1/0; fails without a clear answer
  local i out
  for i in 1 2 3 4 5; do
    out=$(ssh -o ConnectTimeout=30 -o BatchMode=yes "$SRC_HOST" \
            "if [ -e $QSRC/queue/$FLAG ]; then echo present; else echo missing; fi" 2>/dev/null)
    case "$out" in
      present) MISSING=0; return 0;;
      missing) MISSING=1; return 0;;
    esac
    echo "  could not read the done flag on $SRC_HOST (try $i of 5), waiting $WAIT s" >&2
    sleep "$WAIT"
  done
  return 1
}

sync_one() {   # sync_one <src_dir> <name> <find expression...>
  local src=$1 name=$2; shift 2
  local i
  for i in $(seq 1 "$TRIES"); do
    if "$HERE/sync_set.sh" "$SRC_HOST" "$src" "$DST_HOST" "$DST_PARENT" "$name" -- "$@"; then return 0; fi
    echo "  $name: the copy did not verify; try $i of $TRIES" >&2
    sleep "$WAIT"
  done
  echo "FAILED: $name could not be copied consistently in $TRIES tries" >&2
  return 1
}

MISSING=1
read_flag || { echo "FAILED: the done flag $SRC_HOST:$QSRC/queue/$FLAG could not be read; nothing was copied" >&2; exit 1; }
if [ $MISSING = 0 ]; then echo "LassoBench published methods FINISHED: $QSRC/queue/$FLAG exists"
else echo "LassoBench published methods NOT FINISHED: $SRC_HOST:$QSRC/queue/$FLAG is missing"; fi
if [ $CHECK = 1 ]; then exit $((MISSING * 3)); fi

sync_one "$SRC" lassobench_pub '(' -path './pub_methods/*' ')' -not -path './pub_methods/logs/*' || exit 1
QSET=(-path './queue/jobs_phase1.txt' -o -path './queue/jobs_phase2.txt' -o -path './queue/queue.log'
      -o -path './queue/preflight_phase1.txt' -o -path './code/*')
[ $MISSING = 0 ] && QSET+=(-o -path "./queue/$FLAG")
sync_one "$QSRC" lassobench_pub_queue "${QSET[@]}" || exit 1
if [ $MISSING = 0 ]; then
  echo "done, FINAL copy: $DST_HOST:$DST_PARENT/lassobench_pub and lassobench_pub_queue"
  exit 0
fi
echo "done, PARTIAL copy (the done flag was missing when the copy started); the final run refuses it" >&2
exit 3
