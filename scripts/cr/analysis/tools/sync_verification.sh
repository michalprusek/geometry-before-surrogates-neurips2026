#!/bin/bash
# sync_verification.sh -- copy the verification verdicts of 2026-10-19 from kajman to panda, with sha256 manifests.
#
#   bash sync_verification.sh      (run from the Mac; needs sync_set.sh next to it)
#
# Source (read only): kajman:/data2/prusek/cr_verify/summarize_2026-10-19/{verdicts.json,summarize_console.txt}
#   (the one summarize pass of RUN_RULES 2026-10-19 on the 80 complete verification records; sha256 of verdicts.json
#   is pinned in crlib/published.py and checked by the generator).
# Destination: panda:/data/prusek/cr_inputs/verification/ (+ SOURCE_verification.sha256, verification.sha256).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC_HOST=${VERIFY_SRC_HOST:-kajman}
DST_HOST=${VERIFY_DST_HOST:-panda}
SRC=${VERIFY_SRC_DIR:-/data2/prusek/cr_verify/summarize_2026-10-19}
DST_PARENT=${VERIFY_DST_PARENT:-/data/prusek/cr_inputs}
exec "$HERE/sync_set.sh" "$SRC_HOST" "$SRC" "$DST_HOST" "$DST_PARENT" verification -- \
  '(' -path './verdicts.json' -o -path './summarize_console.txt' ')'
