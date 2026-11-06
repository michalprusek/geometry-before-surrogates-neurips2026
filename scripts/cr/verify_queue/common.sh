# Sourced by the verification job scripts (kajman, CPU only). Rules: RUN_RULES rows 2026-10-13 "Verification protocol" and its revision 1.
CL=/path/to/NeurIPS-cr-verify                 # clone of camera-ready-published, detached at the tag
EXPECT=${EXPECT:?set EXPECT to the tagged commit}
PY=/path/to/NeurIPS/.venv312/bin/python       # torch 2.11.0, botorch 0.17.2, gpytorch 1.15.2, cma 4.4.4
V=/path/to/cr_verify
RUNROOT=${RUNROOT:-$V}                             # smoke: RUNROOT=$V/smoke_final (own out/, status/, logs/)
OUT=$RUNROOT/out
mkdir -p "$OUT" "$RUNROOT/status" "$RUNROOT/logs"
export TMPDIR=/path/to/tmp TMUX_TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
NICE=10
check_clone() {
  local h p
  h=$(git -C "$CL" --no-optional-locks rev-parse HEAD)
  p=$(git -C "$CL" --no-optional-locks status --porcelain --untracked-files=no | wc -l)
  [ "$h" = "$EXPECT" ] && [ "$p" = "0" ] || { echo "REFUSE: clone HEAD $h porcelain $p (expected $EXPECT)" >&2; return 1; }
}
