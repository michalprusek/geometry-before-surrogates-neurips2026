#!/usr/bin/env python
"""Append one provenance record (one JSON line) for a LassoBench run to <group>/manifest.jsonl.

scripts/run_lassobench.py writes no provenance into its result JSONs, so this sidecar is the
provenance record: the exact command, the clone's commit and tree state, the runner's sha256,
the environment (thread variables, interpreter, freeze file + md5), the LassoBench data files
read by libsvmdata, host, times, return code, and every result file the run produced (sha256,
best_score, elapsed_s, error). Appends are serialised with flock (several jobs run at once).
Run with ``python -I -B`` (no repository import).
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
import socket
import subprocess
import sys


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(clone, *args):
    try:
        r = subprocess.run(["git", "--no-optional-locks", "-C", clone, *args], capture_output=True, text=True, timeout=120)
        return r.stdout.strip() if r.returncode == 0 else f"ERR({r.returncode})"
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


ap = argparse.ArgumentParser()
for k in ("manifest", "group", "label", "methods", "task", "seed", "rc", "start", "end", "cmd_file", "log",
          "results_dir", "script", "clone", "freeze", "python", "slot", "timeout_s", "cwd"):
    ap.add_argument("--" + k.replace("_", "-"), required=k not in ("slot",), default=None)
a = ap.parse_args()

porcelain = git(a.clone, "status", "--porcelain")
files = []
for m in a.methods.split(","):
    f = os.path.join(a.results_dir, f"{m}_{a.task}_s{a.seed}.json")
    rec = {"method": m, "file": f, "exists": os.path.exists(f)}
    if rec["exists"]:
        rec["sha256"] = sha256(f)
        try:
            d = json.load(open(f))
            bs = d.get("best_score")
            rec.update({"best_score": bs, "best_score_finite": isinstance(bs, (int, float)) and math.isfinite(bs),
                        "elapsed_s": d.get("elapsed_s"), "timestamp": d.get("timestamp"),
                        "n_all_scores": len(d.get("all_scores") or []), "error": d.get("error"),
                        "surrogate_fallback_count": d.get("surrogate_fallback_count")})
        except (OSError, json.JSONDecodeError) as e:
            rec["unreadable"] = str(e)
    files.append(rec)

record = {
    "group": a.group, "label": a.label, "methods": a.methods.split(","), "task": a.task, "seed": int(a.seed),
    "rc": int(a.rc), "start": a.start, "end": a.end,
    "command": open(a.cmd_file).read().strip(), "cwd": a.cwd, "log": a.log, "timeout_s": a.timeout_s,
    "host": socket.getfqdn(), "slot": a.slot,
    "clone": a.clone, "commit": git(a.clone, "rev-parse", "HEAD"),
    "tag": git(a.clone, "describe", "--tags", "--exact-match", "HEAD"),
    "porcelain_lines_after": len([x for x in porcelain.splitlines() if x.strip()]) if not porcelain.startswith("ERR") else porcelain,
    "runner_script": a.script, "runner_script_sha256": sha256(a.script),
    "runner_script_is_committed_file": os.path.realpath(a.script) == os.path.realpath(os.path.join(a.clone, "scripts/run_lassobench.py")),
    "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": a.clone, "PYTHONDONTWRITEBYTECODE": "1", "nice": 10, "python": a.python,
            "python_version": sys.version.split()[0],
            "freeze_file": a.freeze, "freeze_md5": hashlib.md5(open(a.freeze, "rb").read()).hexdigest()},
    "result_files": files,
    "recorded": datetime.datetime.now().isoformat(timespec="seconds"),
}
with open(a.manifest, "a") as fh:
    fcntl.flock(fh, fcntl.LOCK_EX)
    fh.write(json.dumps(record) + "\n")
    fh.flush()
    fcntl.flock(fh, fcntl.LOCK_UN)
