# Sourced by the LassoBench published-method production jobs (kajman, CPU only).
# Rules: RUN_RULES 2026-10-12 "Published-recipe block" (3) and 2026-10-13 revision 1 (f):
# ten methods, tasks synt_simple, synt_medium, dna, synt_high, synt_hard (+ leukemia by pilot), seeds 42-81,
# one thread, one hour per run, from the verification tag. Result values stay blinded until the final generator run.
CL=/path/to/NeurIPS-cr-pubprod                   # clone of camera-ready-published, detached at cr-verify-2026-10-13
EXPECT=963ef040cae648d4132e1d3330a2a9c06456e97d
PY=/path/to/NeurIPS/.venv312/bin/python         # same interpreter as the full LassoBench rerun and the verification
ANA=/path/to/cr_pubprod
CODE=$ANA/code
FREEZE=$(ls $ANA/env/env_freeze_venv312_*.txt | tail -1)
LCR=$CL/results/lassobench_pub                       # inside the clone, gitignored (.gitignore results/), as the full rerun:
                                                     # jobs run with cwd there, so the runner's git provenance finds the commit
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
