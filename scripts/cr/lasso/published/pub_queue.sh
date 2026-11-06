#!/bin/bash
# LassoBench published-method production on kajman: phase 1 (pilots + five tasks), then phase 2
# (leukemia seeds 43-81 for methods whose pilot exited 0). P single-thread jobs, nice 10.
set -uo pipefail
source /path/to/cr_pubprod/code/common.sh
P=${1:-12}
Q=$ANA/queue; G=$LCR/pub_methods
mkdir -p "$Q" "$G"
if [ -e "$Q/phase1.started" ]; then echo "REFUSE: already started" >&2; exit 3; fi
check_clone || exit 4
date -Iseconds > "$Q/phase1.started"
preflight "$Q/preflight_phase1.txt"
python3 -I "$CODE/pub_jobs.py" phase1 > "$Q/jobs_phase1.txt"
echo "phase 1 start $(date -Iseconds), $(wc -l < "$Q/jobs_phase1.txt") jobs, P=$P" | tee "$Q/queue.log"
xargs -P "$P" -I {} bash -c "{}" < "$Q/jobs_phase1.txt"
echo "phase 1 end $(date -Iseconds) xargs rc=$?" | tee -a "$Q/queue.log"
ok=()
for f in "$G"/status/pilot_*_leukemia_s42.rc; do
  m=$(basename "$f" .rc); m=${m#pilot_}; m=${m%_leukemia_s42}
  rc=$(cut -d' ' -f1 "$f")
  echo "pilot $m rc=$rc" | tee -a "$Q/queue.log"
  [ "$rc" = "0" ] && ok+=("$m")
done
if [ ${#ok[@]} -gt 0 ]; then
  python3 -I "$CODE/pub_jobs.py" phase2 "${ok[@]}" > "$Q/jobs_phase2.txt"
  echo "phase 2 start $(date -Iseconds), $(wc -l < "$Q/jobs_phase2.txt") jobs for: ${ok[*]}" | tee -a "$Q/queue.log"
  xargs -P "$P" -I {} bash -c "{}" < "$Q/jobs_phase2.txt"
  echo "phase 2 end $(date -Iseconds) xargs rc=$?" | tee -a "$Q/queue.log"
fi
date -Iseconds > "$Q/queue.done"
