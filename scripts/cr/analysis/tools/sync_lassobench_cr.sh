#!/bin/bash
# sync_lassobench_cr.sh -- copy the LassoBench camera-ready rerun sets from kajman to panda, with sha256 manifests.
#
#   bash sync_lassobench_cr.sh            copy and verify   (run from the Mac; needs sync_set.sh next to it)
#   bash sync_lassobench_cr.sh --check    only report the two done flags on kajman (copies nothing)
#
# Source (read only):  kajman:/data2/prusek/NeurIPS-cr-ana/results/lassobench_cr/
#     full_rerun/         10 methods x 5 tasks (D <= 1000) x seeds 42-81; on leukemia (D = 7129) the Hvarfner GP and
#                         Papenmeier, seeds 42-81, and the Xu pilot (seed 42). Papenmeier on leukemia: seed 42 is the
#                         pilot's job pilot_papenmeier_leukemia_s42, seeds 43-81 are the 39 jobs
#                         papenmeier_leukemia_s43 .. s81 (RUN_RULES amendment 2026-10-12, queued after the others)
#     leukemia_rerun/     the six cheap arms on leukemia (D = 7129), seeds 42-81
#     linear_bo_leukemia/ the failed Linear BO pilot on leukemia (PILOT_FAILED, no result)
#     env_provenance.txt, bitcheck_d60/compare.txt
#   of each run folder: results/*.json, manifest.jsonl (one provenance record per job), status/, cmds/ and the
#   small marker files; logs/ stay on kajman (not needed by the generator, and still being written).
#   Expected in the end: 2,322 runs in 1,481 + 40 + 1 jobs (crlib/lasso_cr.py, expected_cells / expected_jobs).
# Destination:         panda:/data/prusek/cr_inputs/lassobench_cr/   (+ SOURCE_lassobench_cr.sha256, lassobench_cr.sha256)
#                      panda:/data/prusek/cr_inputs/lassobench_cr_queue/  (job lists, queue log, done flags, runner scripts)
#
# WHEN IS THE RERUN FINISHED?  Only when BOTH done flags exist on kajman, in /data2/prusek/cr_analysis/queue/:
#     full_rerun.done            written when the original 1,442 jobs (jobs_full_rerun.txt) have ended
#     papenmeier_leukemia.done   written when the 39 Papenmeier leukemia jobs, seeds 43-81
#                                (jobs_papenmeier_leukemia.txt), have ended
# This script reads the two flags BEFORE it copies anything and says which one is missing. It then copies the
# result set first and the queue set last, and a done flag is copied only if it existed before the copy started.
# A done flag in the destination therefore means: the result set next to it was copied and verified after those
# jobs had ended (a failed or interrupted copy leaves no new flag behind). Exit codes:
#     0   both flags existed before the copy started, both sets copied and verified: the FINAL copy
#     3   the sets were copied and verified, but at least one flag was missing: a PARTIAL copy (the generator marks
#         its LassoBench tables "INCOMPLETE RERUN", and make_tables.py --lasso-require-finished / the final
#         run script refuse it). With --check: at least one flag is missing.
#     1   the flags could not be read, or a set could not be copied consistently
#     2   usage
#
# Safe to re-run: while the rerun is still running (tmux crlasso2 on kajman) the set grows and manifest.jsonl is
# appended, so a copy can catch the set in motion; sync_set.sh then refuses (manifests differ) and this script
# simply tries again. Run it once more when --check reports both flags.
# The generator's loader only uses result files that a manifest record lists with a matching sha256.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC_HOST=${LASSO_SRC_HOST:-kajman}
DST_HOST=${LASSO_DST_HOST:-panda}
SRC=${LASSO_SRC_DIR:-/data2/prusek/NeurIPS-cr-ana/results/lassobench_cr}      # the three run folders
QSRC=${LASSO_QUEUE_DIR:-/data2/prusek/cr_analysis}                            # queue/ (job lists, done flags), code/
DST_PARENT=${LASSO_DST_PARENT:-/data/prusek/cr_inputs}
TRIES=${LASSO_SYNC_TRIES:-10}
WAIT=${LASSO_SYNC_WAIT:-15}
FLAGS="full_rerun.done papenmeier_leukemia.done"

CHECK=0
for a in "$@"; do
  case "$a" in
    --check) CHECK=1;;
    *) echo "usage: $0 [--check]" >&2; exit 2;;
  esac
done

flag_jobs() {   # what a done flag stands for
  case "$1" in
    full_rerun.done)          echo "the original 1,442 jobs of the full rerun";;
    papenmeier_leukemia.done) echo "the 39 Papenmeier leukemia jobs, seeds 43-81";;
  esac
}

read_flags() {   # sets MISSING to the names of the absent done flags; fails if the source host gave no clear answer
  local i out
  for i in 1 2 3 4 5; do
    out=$(ssh -o ConnectTimeout=30 -o BatchMode=yes "$SRC_HOST" \
            "for f in $FLAGS; do if [ -e $QSRC/queue/\$f ]; then echo present \$f; else echo missing \$f; fi; done" 2>/dev/null)
    # a clear answer is exactly one 'present' or 'missing' line per flag (a dropped connection gives fewer)
    if [ "$(printf '%s\n' "$out" | grep -c -E '^(present|missing) [a-z_]+\.done$')" = 2 ] \
       && [ "$(printf '%s\n' "$out" | wc -l | tr -d ' ')" = 2 ]; then
      MISSING=$(printf '%s\n' "$out" | sed -n 's/^missing //p' | tr '\n' ' ')
      MISSING=${MISSING% }
      return 0
    fi
    echo "  could not read the done flags on $SRC_HOST (try $i of 5), waiting $WAIT s" >&2
    sleep "$WAIT"
  done
  return 1
}

report_flags() {
  local f
  for f in $FLAGS; do
    case " $MISSING " in
      *" $f "*) echo "  MISSING  $SRC_HOST:$QSRC/queue/$f   ($(flag_jobs "$f") have not ended)";;
      *)        echo "  present  $SRC_HOST:$QSRC/queue/$f   ($(flag_jobs "$f") have ended)";;
    esac
  done
}

sync_one() {   # sync_one <src_dir> <name> <find expression...>
  local src=$1 name=$2; shift 2
  local i
  for i in $(seq 1 "$TRIES"); do
    if "$HERE/sync_set.sh" "$SRC_HOST" "$src" "$DST_HOST" "$DST_PARENT" "$name" -- "$@"; then return 0; fi
    echo "  $name: the copy did not verify (the set changed while it was copied, or the network dropped); try $i of $TRIES" >&2
    sleep "$WAIT"
  done
  echo "FAILED: $name could not be copied consistently in $TRIES tries" >&2
  return 1
}

MISSING=""
read_flags || { echo "FAILED: the done flags on $SRC_HOST:$QSRC/queue could not be read; nothing was copied" >&2; exit 1; }
if [ -z "$MISSING" ]; then
  echo "LassoBench rerun FINISHED: both done flags exist"
else
  echo "LassoBench rerun NOT FINISHED: missing done flag(s): $MISSING"
fi
report_flags
if [ $CHECK = 1 ]; then
  if [ -z "$MISSING" ]; then exit 0; else exit 3; fi
fi

# 1. the results (the three run folders)
sync_one "$SRC" lassobench_cr \
  '(' -path './full_rerun/*' -o -path './leukemia_rerun/*' -o -path './linear_bo_leukemia/*' \
      -o -path './env_provenance.txt' -o -path './bitcheck_d60/compare.txt' ')' \
  -not -path './*/logs/*' || exit 1
# 2. the queue set, last; a done flag only if it existed before step 1 (a flag written during the copy would
#    otherwise sit next to results that were copied before its jobs had ended)
QSET=(-path './queue/jobs_full_rerun.txt' -o -path './queue/full_rerun.log'
      -o -path './queue/jobs_papenmeier_leukemia.txt'
      -o -path './code/run_lasso.sh' -o -path './code/manifest_append.py' -o -path './code/full_rerun_jobs.py'
      -o -path './code/full_rerun_tmux.sh' -o -path './code/common.sh' -o -path './code/with_slot.sh')
for f in $FLAGS; do
  case " $MISSING " in *" $f "*) ;; *) QSET+=(-o -path "./queue/$f");; esac
done
sync_one "$QSRC" lassobench_cr_queue "${QSET[@]}" || exit 1

if [ -z "$MISSING" ]; then
  echo "done, FINAL copy (both done flags existed before the copy started): $DST_HOST:$DST_PARENT/lassobench_cr and lassobench_cr_queue"
  exit 0
fi
echo "done, PARTIAL copy: $DST_HOST:$DST_PARENT/lassobench_cr and lassobench_cr_queue were copied and verified, but the rerun" >&2
echo "had NOT finished when the copy started. Missing done flag(s):" >&2
report_flags | grep MISSING >&2
echo "Run this script again when '$0 --check' reports both flags; the final generator run refuses this copy." >&2
exit 3
