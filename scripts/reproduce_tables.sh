#!/usr/bin/env bash
# Rebuild every table, figure and number of the paper from the results bundle.
#
#   bash scripts/reproduce_tables.sh <bundle_dir> <out_dir> [further make_tables.py options]
#
#   <bundle_dir>  the unpacked results bundle, i.e. the results_bundle/ directory that holds SHA256SUMS,
#                 cr2026/, cr2026_kajman/, cr2026ctl/, cr2026pub/, cr2026rd/, stored_guacamol/ and cr_inputs/
#                 (docs/camera_ready/RESULTS.md describes it)
#   <out_dir>     where the outputs go; it must lie outside this repository (make_tables.py refuses
#                 anything else)
#
# The make_tables.py arguments below are those of the run that produced the camera-ready tables, with
# the authors' directories replaced by the directories of the bundle. The expected commits are the
# commits recorded in the result files: CR and CRCPU the main campaign (tag cr-runs-2026-10-12), CRCTL
# the control block (tag cr-controls-2026-10-12), CRPUB the published-recipe block, CRRD the GuacaMol runs of
# 2026-11-02 (full-dimensional CMA-ES and BAxUS, commit 1ea2f7d), CRC4 control C4 (tag cr-controls-lbfgs-2026-10-14,
# commit 2a899ad; its files lie in cr2026ctl/ next to the control block, so CRC4 names that directory with its own
# expected commit). make_tables.py also pins CR, CRCPU, CRPUB, CRRD and CRC4
# in its code and refuses other values. Options given after <out_dir> are
# appended unchanged (for example --no-figures). PYTHON selects the interpreter (default: python).
# The environment is the one of that run: no GPU visible, four OpenMP/BLAS threads unless set.
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: bash scripts/reproduce_tables.sh <bundle_dir> <out_dir> [make_tables.py options]" >&2
    exit 2
fi

if ! B=$(cd "$1" 2>/dev/null && pwd -P); then
    echo "reproduce_tables.sh: bundle directory $1 not found" >&2
    exit 2
fi
case "$2" in /*) OUT=$2 ;; *) OUT=$PWD/$2 ;; esac    # may not exist yet; make_tables.py normalises it
shift 2

for d in cr2026 cr2026_kajman cr2026ctl cr2026pub cr2026rd stored_guacamol cr_inputs; do
    if [ ! -d "$B/$d" ]; then
        echo "reproduce_tables.sh: $B/$d not found; <bundle_dir> must be the unpacked results_bundle/" \
             "directory (docs/camera_ready/RESULTS.md)" >&2
        exit 2
    fi
done

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

exec "${PYTHON:-python}" -B scripts/cr/analysis/make_tables.py \
    --expected-commit CR=14481cb55d21a770da11c3d6f866fcae98891c18 --expected-commit CRCPU=14481cb55d21a770da11c3d6f866fcae98891c18 \
    --expected-commit CRCTL=fd8a03c5a75475e79ed8e9f460c38514a7e9717d \
    --expected-commit CRPUB=963ef040cae648d4132e1d3330a2a9c06456e97d \
    --expected-commit CRRD=1ea2f7d8fdbec151c59c91e68bc15f45bc79379e \
    --expected-commit CRC4=2a899ad301d3cb0badd906adfb4d81ad0b50f168 \
    --root CR="$B/cr2026" --root CRCPU="$B/cr2026_kajman" --root CRCTL="$B/cr2026ctl" --root CRPUB="$B/cr2026pub" \
    --root CRRD="$B/cr2026rd" --root CRC4="$B/cr2026ctl" \
    --root G="$B/stored_guacamol" --root INPUTS="$B/cr_inputs" --inputs-root "$B/cr_inputs" \
    --geometry-root "$B/cr_inputs/geometry_cr" --lasso-cr-root "$B/cr_inputs/lassobench_cr" \
    --lasso-pub-root "$B/cr_inputs/lassobench_pub" --verdicts "$B/cr_inputs/verification/verdicts.json" \
    --lasso-require-finished --p1-sel-root "$B/cr_inputs/p1_blocks_sel" \
    --outdir "$OUT" --latex-check "$@"
