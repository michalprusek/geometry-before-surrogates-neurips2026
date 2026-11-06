# Sourced by every job script of the camera-ready analysis runs (kajman, CPU only).
CL=/path/to/NeurIPS-cr-ana                      # analysis clone, detached at tag cr-runs-2026-10-12
EXPECT=14481cb55d21a770da11c3d6f866fcae98891c18
PY=/path/to/NeurIPS/.venv312/bin/python         # same interpreter as the production CPU queue
FREEZE=/path/to/cr_analysis/env/env_freeze_venv312_20261012_1311.txt
ANA=/path/to/cr_analysis
CODE=$ANA/code
LCR=$CL/results/lassobench_cr                        # gitignored (.gitignore:30 results/)
STORED_LASSO=/path/to/NeurIPS/results/lassobench
export TMPDIR=/path/to/tmp TMUX_TMPDIR=/tmp
NICE=10

preflight() {  # $1 = file
  {
    echo "date: $(date -Iseconds)"; echo "host: $(hostname -f)"
    echo "who:"; who
    echo "uptime: $(uptime)"
    echo "nvidia-smi:"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    echo "clone: $CL"
    echo "clone HEAD: $(git -C "$CL" --no-optional-locks rev-parse HEAD)"
    echo "clone tag: $(git -C "$CL" --no-optional-locks describe --tags --exact-match HEAD 2>&1)"
    echo "clone porcelain lines: $(git -C "$CL" --no-optional-locks status --porcelain | wc -l)"
    echo "env freeze: $FREEZE md5 $(md5sum < "$FREEZE" | cut -d' ' -f1)"
    echo "python: $PY ($("$PY" -V 2>&1))"
    echo "slot: ${CR_ANA_SLOT:-none}"
  } > "$1" 2>&1
}

check_clone() {
  local h p
  h=$(git -C "$CL" --no-optional-locks rev-parse HEAD)
  p=$(git -C "$CL" --no-optional-locks status --porcelain | wc -l)
  if [ "$h" != "$EXPECT" ] || [ "$p" != "0" ]; then
    echo "REFUSE: clone HEAD $h porcelain $p" >&2; return 1
  fi
}
