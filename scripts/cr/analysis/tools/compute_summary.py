#!/usr/bin/env python3
"""Sum the recorded per-run wall-clock times of result files, per root.
Read-only. Usage: compute_summary.py LABEL=DIR [LABEL=DIR ...]. Prints label, files, files with a time,
process-hours, median seconds per run. Process-hours are the sum of per-run wall-clock times of runs that
shared their machine with other runs, not exclusive GPU or CPU hours.

The time of a run is the first of the keys elapsed_seconds, elapsed_s, wall_seconds, elapsed and
wall_s_this_process that the file has; the verification runs record only per-evaluation times, so for them it is the sum of t_eval_s.
The rule is ``crlib.runtime.run_time``, which the generator's compute table (``tab_compute``) uses too.
The compute totals of the paper's checklist (item 8) are its output over these directories of the results
bundle: cr2026, cr2026pub, cr2026rd and cr2026ctl (GPU server); cr2026_kajman, cr_inputs/lassobench_cr,
cr_inputs/lassobench_pub and cr_inputs/verification_runs (CPU server); and stored_guacamol, the stored runs of
earlier versions that the generator reads (the stored external baselines record no time)."""
import json, os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crlib.runtime import KEYS, run_time  # noqa: E402,F401  (KEYS: the keys listed above)
def times(root):
    n = 0; t = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if not f.endswith(".json"): continue
            n += 1
            try: d = json.load(open(os.path.join(dp, f)))
            except Exception: continue
            if isinstance(d, dict):
                v = run_time(d)
                if v is not None: t.append(v)
    return n, t
def main(argv):
    for arg in argv:
        lab, root = arg.split("=", 1)
        n, t = times(root)
        med = statistics.median(t) if t else float("nan")
        print(f"{lab}\t{root}\tfiles={n}\ttimed={len(t)}\tprocess_hours={sum(t)/3600:.1f}\tmedian_s={med:.1f}")
if __name__ == "__main__":
    main(sys.argv[1:])
