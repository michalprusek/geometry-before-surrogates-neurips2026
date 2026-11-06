#!/usr/bin/env python
"""Camera-ready LassoBench inventory and recomputation (read-only over stored results).

1. Inventory of --results (+ any --side directories): files, seed
   coverage, finite/non-finite/error runs, timestamps, run times, per method x task.
2. Cold-start oracle agreement: for every (task, seed) the first 5 evaluations of every
   method are the same 5 points (RandomState(seed).uniform(-1, 1, (5, D))), so their losses
   must agree across methods unless the oracle (LassoBench/sklearn/celer versions) changed
   between file generations.
3. Table 3 LassoBench row (tab:crossdomain): GeoWalk vs GeoCMA on loss normalised by the
   task's random-search mean, pooled over 6 tasks x seeds (paired), Delta, paired Cohen's d
   (mean diff / SD diff, src/analysis/stats.py::cohens_d), two-sided Wilcoxon p; plus
   variants (seeds 42-51 only; task-mean aggregation; 5 tasks without leukemia) to locate
   the printed 0.790 / 0.440 / 0.350 / 0.68; plus a task-level exact Wilcoxon over 6 tasks.
4. Figure 2 numbers, replicating the plotting rule of the submitted version exactly (every
   file in the directory, grouped by the JSON field D, finite best_score, >= 3 runs,
   normalised by the random mean at that D, band = SD(ddof=0)/sqrt(n)), for the plotted
   methods plus linear_bo, turbo, baxus.
5. GeoCMA against every comparator at every D (paired by seed, Wilcoxon, d), to check
   "GeoCMA leads at every dimension and against every comparator".

Writes only into --outdir (cr_lasso.json, cr_lasso.md). --outdir is refused when it lies inside an input
directory (--results, --side) or inside any git work tree (``git rev-parse --show-toplevel`` succeeds for it
or for its nearest existing parent); paths are compared by component (``os.path.commonpath``), not by
string prefix.

The side directories are no longer built in. The audit of 2026-10-11 (scratchpad cr/cr_lasso.md, section
"Side directories") is reproduced with the three directories it listed::

    python -B scripts/cr/analysis/lassobench.py --outdir <dir outside every repository> \
        --results /data2/prusek/NeurIPS/results/lassobench \
        --side /data2/prusek/NeurIPS/results/lassobench_buggy_pre_fix \
        --side /data2/prusek/NeurIPS/results/lassobench_dsweep \
        --side /home/prusek/NeurIPS/results/lassobench

(on kajman; without the three --side arguments the "Side directories" line of the report is empty). The
camera-ready rerun (RUN_RULES amendment 2026-10-12) keeps its results in two folders, full_rerun/results and
leukemia_rerun/results; the table generator (crlib/lasso_cr.py, crlib/stored.py) reads them with the
convention of section 3 of this script. To run this script on the rerun, point --results at a folder that
links the result files of both.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

# Inputs come from the command line (--results, --side); no path is baked in.
RES = ""    # --results: the LassoBench result directory to analyse
SIDE: list = []  # --side (repeatable): further directories cross-checked for the same cells
TASKS = [("synt_simple", 60), ("synt_medium", 100), ("dna", 180), ("synt_high", 300), ("synt_hard", 1000), ("leukemia", 7129)]
TD = dict(TASKS)
SEEDS = list(range(42, 82))
FIG_METHODS = ["cmaes_sub", "geowalk", "hvarfner_gp", "papenmeier_msr", "xu_standard_gp", "random"]
EXTRA = ["linear_bo", "turbo", "baxus"]
PAT = re.compile(r"^(?P<m>.+)_(?P<t>synt_simple|synt_medium|synt_high|synt_hard|dna|leukemia)_s(?P<s>\d+)\.json$")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def inside(path, root):
    """True when ``path`` is ``root`` or lies below it (path components, not string prefixes:
    ``/x/NeurIPS-cr`` is NOT inside ``/x/NeurIPS``)."""
    a, b = os.path.realpath(path), os.path.realpath(root)
    try:
        return os.path.commonpath([a, b]) == b
    except ValueError:          # e.g. different drives
        return False


def git_toplevel(path):
    """Top level of the git work tree that contains ``path`` or its nearest existing parent ('' = none)."""
    p = os.path.realpath(path)
    while not os.path.isdir(p):
        parent = os.path.dirname(p)
        if parent == p:
            return ""
        p = parent
    try:
        r = subprocess.run(["git", "--no-optional-locks", "-C", p, "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def check_outdir(outdir, inputs):
    """Refuse an output directory inside an input directory or inside any git work tree."""
    for d in inputs:
        if d and inside(outdir, d):
            raise SystemExit(f"refusing to write inside an input directory ({d})")
    top = git_toplevel(outdir)
    if top:
        raise SystemExit(f"refusing to write inside a git work tree ({top}); choose an --outdir outside every repository")


def finite(v):
    return v is not None and isinstance(v, (int, float)) and np.isfinite(v)


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        m = PAT.match(os.path.basename(f))
        if not m:
            continue
        try:
            j = json.load(open(f))
        except (OSError, json.JSONDecodeError) as e:
            out[(m["m"], m["t"], int(m["s"]))] = {"file": f, "unreadable": str(e)}
            continue
        j["_file"] = f
        out[(m["m"], m["t"], int(m["s"]))] = j
    return out


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    p = float(stats.wilcoxon(a, b).pvalue) if (d.size >= 6 and sd > 0) else float("nan")
    return {"n": int(d.size), "mean_a": float(a.mean()), "mean_b": float(b.mean()), "delta": float(d.mean()),
            "d": float(d.mean() / sd) if sd > 0 else 0.0, "p": p, "a_better_lower": int((d < 0).sum()),
            "b_better_lower": int((d > 0).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--results", required=True,
                    help="LassoBench result directory to analyse (read only)")
    ap.add_argument("--side", action="append", default=[],
                    help="further result directory cross-checked for the same cells (repeatable)")
    args = ap.parse_args()
    global RES, SIDE
    RES = os.path.abspath(args.results)
    SIDE = [os.path.abspath(d) for d in args.side]
    outdir = os.path.abspath(args.outdir)
    check_outdir(outdir, [RES, *SIDE])
    os.makedirs(outdir, exist_ok=True)
    R = {"provenance": {"script_sha256": sha256(os.path.abspath(__file__)), "host": socket.gethostname(),
                        "python": platform.python_version(), "numpy": np.__version__,
                        "scipy": __import__("scipy").__version__, "results_dir": RES,
                        "time": datetime.datetime.now().isoformat(timespec="seconds")}}
    L = ["# LassoBench, camera-ready recomputation\n",
         f"Generated {R['provenance']['time']} on {R['provenance']['host']} (python {R['provenance']['python']}, numpy "
         f"{R['provenance']['numpy']}, scipy {R['provenance']['scipy']}); script sha256 `{R['provenance']['script_sha256'][:16]}`. "
         f"Input: `{RES}` (read only).\n"]

    data = load_dir(RES)
    methods = sorted({k[0] for k in data})
    # ── 1. inventory ───────────────────────────────────────────────────────
    inv = {}
    L.append("## 1. Inventory (files per method x task; finite best_score; seeds)\n")
    L.append("| method | " + " | ".join(f"{t} (D={D})" for t, D in TASKS) + " | timestamps | provenance block |")
    L.append("|---|" + "---|" * len(TASKS) + "---|---|")
    for m in methods:
        row, ts_all, prov = [], [], Counter()
        inv[m] = {}
        for t, D in TASKS:
            ks = [k for k in data if k[0] == m and k[1] == t]
            fin = [k for k in ks if finite(data[k].get("best_score"))]
            err = [k for k in ks if "error" in data[k]]
            seeds = sorted(k[2] for k in ks)
            fin_seeds = sorted(k[2] for k in fin)
            out_rng = [s for s in seeds if s not in SEEDS]
            missing = [s for s in SEEDS if s not in fin_seeds]
            for k in ks:
                ts_all.append(str(data[k].get("timestamp", ""))[:10])
                prov["yes" if "provenance" in data[k] else "no"] += 1
            el = [data[k].get("elapsed_s") for k in fin if data[k].get("elapsed_s") is not None]
            inv[m][t] = {"files": len(ks), "finite": len(fin), "error_files": len(err),
                         "seeds_min_max": [seeds[0], seeds[-1]] if seeds else None,
                         "seeds_outside_42_81": out_rng, "seeds_42_81_missing_or_nonfinite": missing,
                         "error_messages": sorted({str(data[k]['error'])[:120] for k in err}),
                         "elapsed_median_s": float(np.median(el)) if el else None,
                         "elapsed_max_s": float(np.max(el)) if el else None,
                         "len_all_scores": dict(Counter(len(data[k].get("all_scores") or []) for k in ks))}
            row.append(f"{len(ks)} / {len(fin)}" + (f" (seeds {seeds[0]}-{seeds[-1]})" if seeds else ""))
        ts_all = sorted(x for x in ts_all if x)
        L.append(f"| {m} | " + " | ".join(row) + f" | {ts_all[0] if ts_all else ''} .. {ts_all[-1] if ts_all else ''} | {dict(prov)} |")
    R["inventory"] = inv
    L.append("\nCell format: files / finite best_score (seed range). Details per cell (missing seeds, error messages, run times) are in the JSON.\n")
    side = {}
    for d in SIDE:
        if os.path.isdir(d):
            dd = load_dir(d)
            side[d] = dict(Counter(f"{k[0]}|{k[1]}" for k in dd))
            mts = sorted(os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d))
            side[d + "#mtime"] = [datetime.datetime.fromtimestamp(mts[0]).isoformat(timespec="minutes"),
                                  datetime.datetime.fromtimestamp(mts[-1]).isoformat(timespec="minutes")] if mts else None
        else:
            side[d] = "absent"
    R["side_directories"] = side
    L.append("Side directories: " + "; ".join(f"`{k}`: {v}" for k, v in side.items()) + "\n")

    # ── 2. cold-start oracle agreement ────────────────────────────────────
    L.append("## 2. Cold-start oracle agreement across file generations\n")
    L.append("Reference = `random` (first 5 of its 30 uniform draws are the same points as every method's cold start). "
             "Max relative difference of the first 5 losses vs the reference, per method and task; `exact` = bit-identical.\n")
    cs = {}
    L.append("| method | " + " | ".join(t for t, _ in TASKS) + " |")
    L.append("|---|" + "---|" * len(TASKS))
    for m in methods:
        cs[m] = {}
        cells = []
        for t, D in TASKS:
            rel, n, nexact = [], 0, 0
            for s in SEEDS:
                a, r = data.get((m, t, s)), data.get(("random", t, s))
                if not a or not r or "all_scores" not in a or "all_scores" not in r:
                    continue
                x, y = np.asarray(a["all_scores"][:5], float), np.asarray(r["all_scores"][:5], float)
                if x.size < 5 or y.size < 5 or not (np.isfinite(x).all() and np.isfinite(y).all()):
                    continue
                n += 1
                nexact += int(np.array_equal(x, y))
                rel.append(float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-12))))
            cs[m][t] = {"n": n, "exact": nexact, "max_rel_diff": max(rel) if rel else None,
                        "median_rel_diff": float(np.median(rel)) if rel else None}
            cells.append("-" if n == 0 else (f"exact {nexact}/{n}" if nexact == n else f"{nexact}/{n} exact; max rel {max(rel):.1e}"))
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    R["coldstart_agreement_vs_random"] = cs
    L.append("")

    # ── normalisation ──────────────────────────────────────────────────────
    rmean = {}
    for t, D in TASKS:
        v = [data[("random", t, s)]["best_score"] for s in SEEDS if ("random", t, s) in data and finite(data[("random", t, s)].get("best_score"))]
        rmean[t] = float(np.mean(v))
    R["random_mean_per_task"] = rmean

    def norm(m, t, s):
        d = data.get((m, t, s))
        if not d or not finite(d.get("best_score")):
            return None
        return d["best_score"] / rmean[t]

    # ── 3. Table 3 row ─────────────────────────────────────────────────────
    L.append("## 3. Table 3 LassoBench row (GeoWalk vs GeoCMA, loss / random mean; lower is better)\n")
    L.append("Printed in the submission: GeoWalk 0.790, GeoCMA **0.440**, Delta 0.350, |d| 0.68, p < 1e-5.\n")
    L.append("| variant | n pairs | GeoWalk | GeoCMA | Delta (GeoWalk - GeoCMA) | paired d | Wilcoxon p | GeoCMA better / worse |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---|")
    t3 = {}
    for name, tasks, seeds in [("6 tasks, seeds 42-81 (current files)", [t for t, _ in TASKS], SEEDS),
                               ("6 tasks, seeds 42-51 only", [t for t, _ in TASKS], list(range(42, 52))),
                               ("5 tasks (no leukemia), seeds 42-81", [t for t, _ in TASKS[:5]], SEEDS)]:
        a, b = [], []
        for t in tasks:
            for s in seeds:
                x, y = norm("geowalk", t, s), norm("cmaes_sub", t, s)
                if x is not None and y is not None:
                    a.append(x); b.append(y)
        pr = paired(a, b)
        t3[name] = pr
        L.append(f"| {name} | {pr['n']} | {pr['mean_a']:.4f} | {pr['mean_b']:.4f} | {pr['delta']:+.4f} | {pr['d']:+.3f} | {pr['p']:.2e} | {pr['b_better_lower']} / {pr['a_better_lower']} |")
    # task-mean aggregation and task-level test
    tm_g = [np.mean([norm("geowalk", t, s) for s in SEEDS if norm("geowalk", t, s) is not None]) for t, _ in TASKS]
    tm_c = [np.mean([norm("cmaes_sub", t, s) for s in SEEDS if norm("cmaes_sub", t, s) is not None]) for t, _ in TASKS]
    dd = np.array(tm_g) - np.array(tm_c)
    wl = stats.wilcoxon(dd, method="exact")
    t3["task_mean"] = {"geowalk_per_task": tm_g, "geocma_per_task": tm_c, "geowalk_mean": float(np.mean(tm_g)),
                       "geocma_mean": float(np.mean(tm_c)), "delta": float(dd.mean()),
                       "task_level_exact_wilcoxon_p": float(wl.pvalue), "signs_geocma_better": int((dd > 0).sum())}
    L.append(f"| task-mean of per-task means (6 tasks) | 6 tasks | {np.mean(tm_g):.4f} | {np.mean(tm_c):.4f} | {dd.mean():+.4f} | - | exact task-level {wl.pvalue:.4f} | GeoCMA better on {int((dd > 0).sum())}/6 tasks |")
    R["table3_lassobench_row"] = t3
    L.append("\nThe `GeoCMA better / worse` column counts pairs where GeoCMA's normalised loss is lower / higher. "
             "With 6 tasks the smallest attainable two-sided exact Wilcoxon p is 0.03125.\n")

    # ── 4. Figure 2 numbers ────────────────────────────────────────────────
    L.append("## 4. Figure 2 numbers (replicating the plotting rule of the submitted version)\n")
    byD = defaultdict(lambda: defaultdict(list))
    for k, d in data.items():
        if "best_score" in d and d.get("D") is not None and finite(d.get("best_score")):
            byD[d.get("method", k[0])][d["D"]].append(d["best_score"])
    rmD = {D: float(np.mean(byD["random"][D])) for _, D in TASKS}
    fig = {}
    L.append("Cells: mean normalised loss +/- SE (n). SE = SD(ddof=0)/sqrt(n) as in the plotting script; "
             "a line is drawn only where n >= 3.\n")
    L.append("| method | " + " | ".join(f"D={D}" for _, D in TASKS) + " | in submitted figure |")
    L.append("|---|" + "---|" * len(TASKS) + "---|")
    for m in FIG_METHODS + EXTRA:
        cells = []
        fig[m] = {}
        for _, D in TASKS:
            sc = np.asarray(byD[m].get(D, []), float)
            if sc.size:
                nv = sc / rmD[D]
                fig[m][D] = {"n": int(sc.size), "mean": float(nv.mean()), "se": float(nv.std() / np.sqrt(sc.size)),
                             "plotted": bool(sc.size >= 3)}
                cells.append(f"{nv.mean():.3f} +/- {nv.std() / np.sqrt(sc.size):.3f} ({sc.size})" + ("" if sc.size >= 3 else " not drawn"))
            else:
                fig[m][D] = {"n": 0}
                cells.append("none")
        L.append(f"| {m} | " + " | ".join(cells) + f" | {'yes' if m in FIG_METHODS else 'no'} |")
    R["figure2"] = {"random_mean_per_D": rmD, "methods": fig}
    L.append("")

    # ── 5. GeoCMA vs every comparator per D ───────────────────────────────
    L.append("## 5. GeoCMA vs each comparator per dimension (raw loss, paired by seed 42-81; lower is better)\n")
    L.append("Cell: Delta = comparator - GeoCMA (positive = GeoCMA better), paired d, two-sided Wilcoxon p, n. "
             "**bold** = comparator significantly better (p < 0.05).\n")
    comps = ["geowalk", "random", "hvarfner_gp", "papenmeier_msr", "xu_standard_gp", "linear_bo", "turbo", "baxus"]
    L.append("| comparator | " + " | ".join(f"{t} (D={D})" for t, D in TASKS) + " |")
    L.append("|---|" + "---|" * len(TASKS))
    vs = {}
    for c in comps:
        cells = []
        vs[c] = {}
        for t, D in TASKS:
            a, b = [], []
            for s in SEEDS:
                x, y = data.get((c, t, s)), data.get(("cmaes_sub", t, s))
                if x and y and finite(x.get("best_score")) and finite(y.get("best_score")):
                    a.append(x["best_score"]); b.append(y["best_score"])
            if len(a) < 6:
                vs[c][t] = {"n": len(a)}
                cells.append(f"n={len(a)}")
                continue
            pr = paired(a, b)
            vs[c][t] = pr
            txt = f"{pr['delta']:+.3g}, d {pr['d']:+.2f}, p {pr['p']:.1e}, n {pr['n']}"
            cells.append(f"**{txt}**" if (pr["p"] < 0.05 and pr["delta"] < 0) else txt)
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    R["geocma_vs_comparators"] = vs
    # pooled Linear BO vs GeoCMA on normalised loss over 5 tasks
    a, b = [], []
    for t, _ in TASKS[:5]:
        for s in SEEDS:
            x, y = norm("linear_bo", t, s), norm("cmaes_sub", t, s)
            if x is not None and y is not None:
                a.append(x); b.append(y)
    R["linear_bo_minus_geocma_pooled_5tasks"] = paired(a, b)
    pr = R["linear_bo_minus_geocma_pooled_5tasks"]
    L.append(f"\nPooled over the 5 tasks with Linear BO runs (normalised loss): Linear BO {pr['mean_a']:.4f} vs GeoCMA "
             f"{pr['mean_b']:.4f}, Delta {pr['delta']:+.4f}, d {pr['d']:+.3f}, Wilcoxon p {pr['p']:.3g}, n {pr['n']}.\n")

    with open(os.path.join(outdir, "cr_lasso.json"), "w") as f:
        json.dump(R, f, indent=1, default=str)
    with open(os.path.join(outdir, "cr_lasso.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
