"""Run times and hosts recorded in result files (standard library only: ``tools/compute_summary.py`` imports it).

The time of a run is the first of the keys ``elapsed_seconds``, ``elapsed_s``, ``wall_seconds``, ``elapsed`` and
``wall_s_this_process`` that the file has; the verification runs record only per-evaluation times, so for them it is
the sum of ``t_eval_s``. The host of a run is ``provenance.hostname`` (or a nested provenance, or a top-level host);
a run folder may name its host once, in the first line of ``env_provenance.txt``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

KEYS = ("elapsed_seconds", "elapsed_s", "wall_seconds", "elapsed", "wall_s_this_process")


def run_time(d: dict):
    """Wall-clock seconds of one run recorded in a result file, or None."""
    for k in KEYS:
        v = d.get(k)
        if isinstance(v, (int, float)) and v >= 0:          # the rule of the checklist totals, unchanged
            return float(v)
    t = d.get("t_eval_s")
    if isinstance(t, list):
        return float(sum(x for x in t if isinstance(x, (int, float))))
    return None


def recorded_host(d: dict):
    """Host recorded in a result file (``provenance.hostname``, nested provenance, or a top-level host), or None."""
    p = d.get("provenance")
    for q in (p, p.get("provenance") if isinstance(p, dict) else None, d):
        if isinstance(q, dict):
            for k in ("hostname", "host"):
                v = q.get(k)
                if isinstance(v, str) and v:
                    return v
    return None


_ENV_HOST_RX = re.compile(r"^recorded \S+ on (\S+)")


def folder_host(root: str):
    """Host named in the first line of ``env_provenance.txt`` of a run folder ("recorded <time> on <host>"), or None."""
    p = os.path.join(root, "env_provenance.txt")
    try:
        with open(p) as f:
            mo = _ENV_HOST_RX.match(f.readline().strip())
    except OSError:
        return None
    return mo.group(1) if mo else None


def scan(root: str) -> dict:
    """Every ``*.json`` below ``root``: {"files", "timed", "rows" (path, sha256), "hosts": {host: {"files", "times"}}}.
    A file that is not a JSON object counts as a file without a time (as in tools/compute_summary.py)."""
    out = {"root": root, "files": 0, "timed": 0, "rows": [], "hosts": {}}
    fallback = folder_host(root)
    for dp, dns, fs in os.walk(root):
        dns.sort()
        for f in sorted(fs):
            if not f.endswith(".json"):
                continue
            path = os.path.join(dp, f)
            with open(path, "rb") as fh:
                raw = fh.read()
            out["files"] += 1
            out["rows"].append((path, hashlib.sha256(raw).hexdigest()))
            try:
                d = json.loads(raw)
            except ValueError:
                d = None
            d = d if isinstance(d, dict) else None
            host = (recorded_host(d) if d is not None else None) or fallback
            e = out["hosts"].setdefault(host, {"files": 0, "times": []})
            e["files"] += 1
            t = run_time(d) if d is not None else None
            if t is not None:
                e["times"].append(t)
                out["timed"] += 1
    out["fallback_host"] = fallback
    return out
