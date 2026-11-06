"""Stored, non-rerunnable sets: NAS-Bench-101 and LassoBench (own schemas; LassoBench also as the camera-ready
rerun, loaded by crlib/lasso_cr.py), and the tables built on stored GuacaMol sets (Appendix F budget runs,
Appendix E codecs, Table 3)."""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats as sps

from . import labels as LB
from . import numbers as N
from . import stats
from .context import BOX_LABEL, Ctx
from .engine import fileset_from_paths
from .latex import COMPLETED_SHORT, SUBMITTED_SHORT, WRAP, Table, m, mb, stack, wrapp
from .numbers import camel
from .tables import conv_plural, conv_text, filt_inline, filt_text, tl_cols, tl_vals

LASSO_TASKS = [("synt_simple", 60), ("synt_medium", 100), ("dna", 180), ("synt_high", 300), ("synt_hard", 1000),
               ("leukemia", 7129)]
# the ten rerun methods; the GP and external methods are named by the print rule (crlib/labels.py)
LASSO_METHODS = [("cmaes_sub", "\\geocma{}"), ("geowalk", "\\geowalk{}"), ("random", "Random search")] + [
    (mth, LB.LASSO_DESCRIPTIVE[mth][0]) for mth in
    ("hvarfner_gp", "papenmeier_msr", "xu_standard_gp", "linear_bo", "cmaes_full", "turbo", "baxus")]
#: RUN_RULES 2026-10-12 (5): the rerun's turbo (box of the observed points), cmaes_full (population 5) and baxus
#: (not BAxUS) appear only in the correspondence table once the published versions exist (camera-ready mode)
LASSO_CORRESPONDENCE_ONLY = ("cmaes_full", "turbo", "baxus")
NAS_METHODS = ["geowalk", "cmaes_sub", "cmaes_full", "gp_bo", "random", "turbo"]
#: the full LassoBench table in the repository records (RUN_RULES 2026-10-29; records/ of a run, docs/baselines/)
LASSO_FULL_TEX = "tab_lassobench_full.tex"
#: plain names of the LassoBench tasks (captions) and the heading of the published methods (RUN_RULES 2026-10-29 (4))
LASSO_TASK_NAMES = {"synt_simple": "synthetic simple", "synt_medium": "synthetic medium", "dna": "DNA",
                    "synt_high": "synthetic high", "synt_hard": "synthetic hard", "leukemia": "leukemia"}
LASSO_PUB_GROUP_FULL = "Published methods (released-code settings; named only if verified)"
LASSO_PUB_GROUP = "Published methods (the authors' released code)"


def _fileset(paths) -> dict:
    return fileset_from_paths(paths)


# =========================================================================================
# LassoBench
# =========================================================================================
# Two generations (RUN_RULES amendments 2026-10-12, "LassoBench is rerun in full" and "Papenmeier at D = 7129
# is run on all 40 seeds"; the expected design is crlib/lasso_cr.py, expected_cells / design_text):
#   * stored  = the March/April files in <inputs-root>/lassobench (ctx.extra["lasso_dir"]). Printed in
#               --legacy mode; in camera-ready mode they appear only in the correspondence table.
#   * rerun   = the camera-ready rerun in <inputs-root>/lassobench_cr (ctx.extra["lasso_cr_dir"], loader
#               crlib/lasso_cr.py: every result joined to a clean job record, coverage reported loudly).
#               Printed in camera-ready mode (R-3). While an expected job has no job record the rerun is
#               unfinished and both tables say so ("INCOMPLETE RERUN" / "The rerun is incomplete"); timed-out
#               and non-finite runs are final outcomes: excluded, counted, and no reason for that mark.
# Statistics follow the committed scripts/cr/analysis/lassobench.py (the Table 3 convention):
#   normalised loss = best_score / mean best_score of `random` on the task (finite runs, seeds 42-81);
#   Table 3 row = GeoWalk vs GeoCMA pooled over all (task, seed) pairs where both losses are finite: flat
#   means, Delta = mean(GeoWalk - GeoCMA), paired d = mean/sd(ddof 1) of the differences, two-sided Wilcoxon
#   signed-rank test (scipy defaults) on the EXACT differences (no tie tolerance: the committed convention);
#   task level = per-task means of the paired differences (the seeds both arms completed; amendment 2026-10-12,
#   "paired tests use the seeds completed by both arms", review B2), exact two-sided Wilcoxon over the tasks.
def load_lasso(directory: str) -> dict:
    """{method: {task: {seed: (loss, path)}}} for finite best_score; every file is listed."""
    rx = re.compile(r"^(?P<m>.+)_(?P<t>synt_simple|synt_medium|synt_high|synt_hard|dna|leukemia)_s(?P<s>\d+)\.json$")
    out = defaultdict(lambda: defaultdict(dict))
    nonfinite = defaultdict(int)
    files = defaultdict(list)
    for n in sorted(os.listdir(directory)):
        mm = rx.match(n)
        if not mm:
            continue
        p = os.path.join(directory, n)
        d = json.load(open(p))
        files[mm.group("m")].append(p)
        v = d.get("best_score")
        if v is None or not np.isfinite(v):
            nonfinite[(mm.group("m"), mm.group("t"))] += 1
            continue
        out[mm.group("m")][mm.group("t")][int(mm.group("s"))] = (float(v), p)
    return {"cells": out, "nonfinite": dict(nonfinite), "files": files, "dir": directory, "generation": "stored",
            "manifests": []}


def _wilcoxon_exact_diffs(d) -> float:
    """Two-sided Wilcoxon signed-rank p on the exact differences (scipy defaults), as lassobench.py."""
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return float("nan")
    if not np.any(d != 0):
        return 1.0
    return float(sps.wilcoxon(d).pvalue)


def lasso_stats(L: dict) -> dict:
    """Every LassoBench statistic of one loaded generation (stored or rerun)."""
    cells = L["cells"]
    man = list(L.get("manifests") or [])          # the rerun's job records are part of every file set
    tasks = [(t, D) for t, D in LASSO_TASKS if cells["random"].get(t)]
    no_random = [t for t, _ in LASSO_TASKS if not cells["random"].get(t)]
    norm = {t: float(np.mean([v for v, _ in cells["random"][t].values()])) for t, _ in tasks}
    per_d = {}
    rand_files = {t: [p for _, p in cells["random"][t].values()] for t, _ in tasks}
    for meth, _ in LASSO_METHODS:
        for t, D in tasks:
            xs = [v / norm[t] for v, _ in cells[meth].get(t, {}).values()]
            if xs:
                x = np.asarray(xs)
                per_d[(meth, t)] = {"n": int(x.size), "mean": float(x.mean()),
                                    "se0": float(x.std(ddof=0) / np.sqrt(x.size)) if x.size else float("nan"),
                                    # the method's files at this D plus the random-search files that normalise them
                                    "files": [p for _, p in cells[meth][t].values()] + rand_files[t] + man}
    a, b = "geowalk", "cmaes_sub"
    pairs = [(t, s) for t, _ in tasks for s in sorted(set(cells[a].get(t, {})) & set(cells[b].get(t, {})))]
    x = np.asarray([cells[a][t][s][0] / norm[t] for t, s in pairs])
    y = np.asarray([cells[b][t][s][0] / norm[t] for t, s in pairs])
    diff = x - y
    # task level: per-task means of the paired differences, i.e. over the seeds both arms completed (review B2)
    tdm = np.asarray([float(np.mean([dd for (tt, _s), dd in zip(pairs, diff) if tt == t]))
                      for t, _ in tasks if any(tt == t for tt, _s in pairs)])
    # GeoCMA against every other method, per dimension and pooled (paired by seed; normalised loss,
    # a per-task constant that changes neither d nor the Wilcoxon test within a task)
    vs = {}
    for meth, _ in LASSO_METHODS:
        if meth == "cmaes_sub":
            continue
        pooled_d, per = [], {}
        for t, D in tasks:
            ss = sorted(set(cells[meth].get(t, {})) & set(cells["cmaes_sub"].get(t, {})))
            if len(ss) < 3:
                continue
            dd = np.asarray([(cells[meth][t][s][0] - cells["cmaes_sub"][t][s][0]) / norm[t] for s in ss])
            per[t] = {"n": len(ss), "delta": float(dd.mean()), "d": stats.cohens_d(dd), "p": _wilcoxon_exact_diffs(dd),
                      "files": ([cells[meth][t][s][1] for s in ss] + [cells["cmaes_sub"][t][s][1] for s in ss]
                                + rand_files[t] + man)}
            pooled_d.extend(dd.tolist())
        pdd = np.asarray(pooled_d)
        vs[meth] = {"per_d": per, "n": int(pdd.size), "delta": float(pdd.mean()) if pdd.size else float("nan"),
                    "d": stats.cohens_d(pdd), "p": _wilcoxon_exact_diffs(pdd),
                    "tasks": sorted(per)}
    t3_files = [p for meth in ("geowalk", "cmaes_sub", "random") for t in cells[meth]
                for _, p in cells[meth][t].values()]
    all_files = [p for ps in L["files"].values() for p in ps]
    return {"norm": norm, "per_d": per_d, "vs_geocma": vs, "n": len(pairs),
            "gw": float(x.mean()) if x.size else float("nan"), "cma": float(y.mean()) if y.size else float("nan"),
            "delta": float(diff.mean()) if diff.size else float("nan"), "d": stats.cohens_d(diff),
            "p": _wilcoxon_exact_diffs(diff),
            "tasks_cma_better": int((tdm > stats.TIE_TOL).sum()), "n_tasks": int(tdm.size), "task_diffs": tdm.tolist(),
            "p_task": stats.wilcoxon_exact_p(tdm), "nonfinite": L["nonfinite"],
            "fileset": _fileset(t3_files + man) if t3_files else None,
            "fileset_all": _fileset(all_files + man) if all_files else None,
            "generation": L.get("generation", "stored"), "tasks_without_random": no_random, "loaded": L}


def _loud(ctx: Ctx, key: str, msg: str) -> None:
    """Banner in the log + run manifest (once per run), and a skip record in every variant's dossier."""
    if not any(x["table"] == "tab_lassobench" and x["what"] == key for x in ctx.skipped):
        ctx.skip("tab_lassobench", key, msg)
    if not ctx.cache.get("warned_lasso_" + key):
        ctx.cache["warned_lasso_" + key] = True
        print("!" * 100 + "\n!!! " + msg + "\n" + "!" * 100, flush=True)
        ctx.cache.setdefault("loud_warnings", []).append(msg)


def lasso_cr_dir(ctx: Ctx) -> str:
    return ctx.extra.get("lasso_cr_dir") or os.path.join(ctx.extra.get("inputs_root") or "", "lassobench_cr")


def lasso_stored_analysis(ctx: Ctx) -> dict | None:
    """The stored March/April generation (the same numbers whatever the mode)."""
    d = ctx.extra.get("lasso_dir")
    if not d or not os.path.isdir(d):
        return None
    c = ctx.cache
    if "lasso_stored" not in c:
        c["lasso_stored"] = lasso_stats(load_lasso(d))
    return c["lasso_stored"]


def lasso_rerun_analysis(ctx: Ctx) -> dict | None:
    """The camera-ready rerun, or None (with a loud warning) when its folder is absent. A set that fails
    the provenance join raises lasso_cr.LassoLoadError (fatal: a result without a clean job record)."""
    from . import lasso_cr
    d = lasso_cr_dir(ctx)
    c = ctx.cache
    if "lasso_rerun" not in c:
        if not os.path.isdir(d) or not any(os.path.isdir(os.path.join(d, s)) for s in lasso_cr.RUN_SETS):
            c["lasso_rerun"] = None
        else:
            L = lasso_cr.load_rerun(d, allow_unclean=ctx.extra.get("lasso_cr_allow_unclean") or ())
            R = lasso_stats(L)
            R["coverage_md"] = lasso_cr.coverage_report(L)
            c["lasso_rerun"] = R
    R = c["lasso_rerun"]
    if R is None:
        _loud(ctx, "rerun_absent", f"WARNING: LassoBench rerun folder {d} is absent: the LassoBench table, figure "
                                   "and Table 3 row fall back to the STORED March/April runs")
        return None
    s = R["loaded"]["summary"]
    if not lasso_cr.coverage_is_clean(s) or R["tasks_without_random"]:
        _loud(ctx, "rerun_coverage",
              f"WARNING: LassoBench rerun coverage {s['present']} of {s['expected']} expected cells "
              f"({s['finite']} finite, {s['nonfinite']} non-finite; missing {s['missing']}: timed out {s['timeout']}, "
              f"failed {s['failed']}, no job record yet {s['pending']} ({s['running']} of them running), "
              f"result without a record {s['unrecorded']})"
              + (f"; tasks without any random-search run (dropped): {R['tasks_without_random']}"
                 if R["tasks_without_random"] else "")
              + (f"; result files OUTSIDE the expected design (used): {s['outside_design']}" if s["outside_design"] else "")
              + (f"; job records outside the expected job list: {s['unexpected_job_labels']}"
                 if s["unexpected_job_labels"] else "")
              + (f"; declared unclean job records (used): {s['declared_unclean']}" if s["declared_unclean"] else "")
              + (f"; finite runs with fewer than 30 scores: {s['short']}" if s["short"] else "")
              + ("" if s["finished"] else f"; THE RERUN IS NOT FINISHED ({s['jobs_without_record']} of "
                                          f"{s['jobs_expected']} expected jobs without a job record)")
              + ". Incomplete cells: " + " | ".join(lasso_cr.coverage_lines(R["loaded"])))
    return R


def lasso_analysis(ctx: Ctx) -> dict | None:
    """The PRINTED generation: the rerun in camera-ready mode (R-3), the stored files in --legacy mode
    (and, with a loud warning, in camera-ready mode when the rerun folder is absent)."""
    if ctx.mode == "cr":
        R = lasso_rerun_analysis(ctx)
        if R is not None:
            return R
    return lasso_stored_analysis(ctx)


def lasso_pub_dir(ctx: Ctx) -> str:
    return ctx.extra.get("lasso_pub_dir") or os.path.join(ctx.extra.get("inputs_root") or "", "lassobench_pub")


def lasso_pub_analysis(ctx: Ctx) -> dict | None:
    """The LassoBench published methods (camera-ready mode only; tag cr-verify-2026-10-13, loader design
    lasso_cr.PUB), normalised by the RERUN's random-search mean loss of each task (the same objective and
    environment). GeoWalk against each printed published method is the baseline family of RUN_RULES 2026-10-12 (4):
    per-task means over the seeds both arms completed (amendment 2026-10-12, "paired tests use the seeds completed
    by both arms"; review B2), exact two-sided Wilcoxon over the tasks, Benjamini--Hochberg within the printed
    published methods. The per-dimension cells stay means over the method's own completed seeds (n/40)."""
    from . import lasso_cr
    if ctx.mode != "cr":
        return None
    R = lasso_rerun_analysis(ctx)
    d = lasso_pub_dir(ctx)
    c = ctx.cache
    if "lasso_pub" not in c:
        if R is None or not os.path.isdir(os.path.join(d, lasso_cr.PUB_SET)):
            c["lasso_pub"] = None
        else:
            L = lasso_cr.load_rerun(d, design=lasso_cr.PUB)
            # review S7: one objective for both run sets, checked on the shared cold starts (counts only, fatal)
            chk = lasso_cr.objective_consistency(R["loaded"], L)
            print(f"  LassoBench objective check (published methods vs the rerun's GeoWalk, shared cold starts): "
                  f"{chk['compared']} runs compared, {chk['exact']} identical, {chk['within_tol']} within a relative "
                  f"{chk['rtol']:g}, {chk['beyond_tol']} beyond, {chk['without_reference']} without a GeoWalk run",
                  flush=True)
            if chk["beyond_tol"]:
                raise lasso_cr.LassoLoadError([("objective", "published methods vs the rerun's GeoWalk",
                                                f"{chk['beyond_tol']} of {chk['compared']} runs evaluate the shared "
                                                f"cold start to other losses (beyond a relative {chk['rtol']:g}; by "
                                                f"method: {chk['beyond_by_method']}): the two sets do not share one "
                                                "objective")])
            c["lasso_pub"] = _lasso_pub_stats(R, L)
            c["lasso_pub"]["coverage_md"] = lasso_cr.coverage_report(L)
            c["lasso_pub"]["objective_check"] = chk
            if not chk["compared"]:
                _loud(ctx, "objective_check", "WARNING: LassoBench objective check compared no run (no shared cold "
                                              "start with a GeoWalk run of the rerun)")
    P = c["lasso_pub"]
    if P is None:
        _loud(ctx, "pub_absent", f"WARNING: LassoBench published-method folder {d} (or the rerun) is absent: the "
                                 "LassoBench table has no published-method rows")
        return None
    s = P["loaded"]["summary"]
    if not s["finished"]:
        _loud(ctx, "pub_unfinished", f"WARNING: LassoBench published-method run set NOT FINISHED "
                                     f"({s['jobs_without_record']} of {s['jobs_expected']} expected jobs without a job record)")
    return P


def _lasso_pub_stats(R: dict, L: dict) -> dict:
    from . import lasso_cr
    cells, norm = L["cells"], R["norm"]
    man = list(L["manifests"]) + list(R["loaded"].get("manifests") or [])
    rcells = R["loaded"]["cells"]
    rand_files = {t: [p for _, p in rcells["random"][t].values()] for t in norm}
    printed = [x for x in lasso_cr.PUB_METHODS if x not in lasso_cr.PUB_UNREAD]
    per_d = {}
    for mth in printed:
        for t, _D in LASSO_TASKS:
            xs = [v / norm[t] for v, _ in cells[mth].get(t, {}).values()] if t in norm else []
            if xs:
                x = np.asarray(xs)
                per_d[(mth, t)] = {"n": int(x.size), "mean": float(x.mean()),
                                   "se0": float(x.std(ddof=0) / np.sqrt(x.size)),
                                   "files": [p for _, p in cells[mth][t].values()] + rand_files[t] + man}
    gw = rcells["geowalk"]
    fam = {}
    for mth in printed:
        # paired on the seeds both arms completed, per (method, task) (review B2)
        both = {t: sorted(set(gw.get(t, {})) & set(cells[mth].get(t, {}))) for t, _ in LASSO_TASKS if t in norm}
        tasks = [t for t, _ in LASSO_TASKS if both.get(t)]
        diff = np.asarray([float(np.mean([gw[t][s][0] - cells[mth][t][s][0] for s in both[t]])) / norm[t]
                           for t in tasks])
        files = [p for t in tasks for s in both[t] for p in (gw[t][s][1], cells[mth][t][s][1])] + \
                [p for t in tasks for p in rand_files[t]] + man
        fam[mth] = {"tasks": tasks, "n_tasks": len(tasks), "diff": diff, "n_pairs": sum(len(both[t]) for t in tasks),
                    "pairs": {t: len(both[t]) for t in tasks},
                    "delta": float(diff.mean()) if diff.size else float("nan"),
                    "gw_lower": int((diff < -stats.TIE_TOL).sum()), "mth_lower": int((diff > stats.TIE_TOL).sum()),
                    "p_task": stats.wilcoxon_exact_p(diff) if diff.size else float("nan"), "files": files}
    for mth, q in zip(printed, stats.bh([fam[x]["p_task"] for x in printed])):
        fam[mth]["p_task_bh"] = q
    return {"loaded": L, "per_d": per_d, "fam": fam, "printed": printed}


def no_result_mark(c: dict) -> str:
    """Mark of a method-dimension cell without any finite loss (review S1): 'n.a.' where the recipe is undefined,
    'pend.' only while one of its jobs has no record yet, otherwise 'n.c.' (did not complete: every seed timed out,
    failed or ended non-finite), at any dimension."""
    if c["not_applicable"] and not c["finite"]:
        return "n.a."
    if c["pending"] or c["unrecorded"]:
        return "pend."
    return "n.c."


def why_missing(c: dict) -> str:
    """'5 over the time limit, 1 failed, 2 without a finite loss' (only the causes that occurred; review M3)."""
    parts = [(len(c["timeout"]), "over the time limit"), (len(c["failed"]), "failed"),
             (len(c["nonfinite"]), "without a finite loss (a model fit failed)"),
             (len(c["pending"]) + len(c["unrecorded"]), "not run yet")]
    return ", ".join(f"{k} {txt}" for k, txt in parts if k)


#: review M3: the reason printed for an n.a. method, derived from the error text its result files carry
NA_REASONS = (("lengthscale", "In the authors' released configuration the lengthscale constraint is "
                                "$(10^{-10}, 30)$, so the initialization at $\\sqrt{D}$ lies outside it for $D>900$ and "
                                "the code stops with an error. Xu et al.\\ report a run at $D=1003$."),)


def na_reason(cov: dict, mth: str) -> str:
    errs = {e for (m, _t), c in cov.items() if m == mth and c["not_applicable"] for e in c["errors"]}
    for pat, txt in NA_REASONS:
        if errs and all(pat in e for e in errs):
            return txt
    return "The result files of the recipe mark it as not applicable at that dimension."


def lasso_gen_text(R: dict) -> str:
    if R["generation"] == "rerun":
        L = R["loaded"]
        return "rerun for this version in one environment, with one CPU thread and at most one hour per run"
    return "runs of the submitted version (March/April 2026)"


def _reg_lasso_row(ctx: Ctx, R: dict, prefix: str, table: str, gen: str) -> None:
    """The Table 3 row macros (GeoWalk vs GeoCMA) of one generation."""
    fs = R["fileset"]
    sfx = "" if gen == "stored runs" and prefix == "crLasso" else f" ({gen})"
    ctx.num(prefix + "Geowalk", R["gw"], N.f3(R["gw"]), kind="level", table=table, fileset=fs, n=R["n"],
            rule="flat mean of loss / task's random-search mean loss over paired (task, seed)" + sfx)
    ctx.num(prefix + "Geocma", R["cma"], N.f3(R["cma"]), kind="level", table=table, fileset=fs, n=R["n"],
            rule="flat mean of normalised loss" + sfx)
    ctx.num(prefix + "Delta", R["delta"], N.f3(R["delta"]), kind="delta", table=table, fileset=fs, n=R["n"],
            rule="GeoWalk - GeoCMA normalised loss (positive: GeoCMA lower)" + sfx)
    ctx.num(prefix + "D", R["d"], N.f2(R["d"]), kind="cohens_d", table=table, fileset=fs, n=R["n"],
            rule="paired cells" + sfx)
    ctx.num(prefix + "P", R["p"], N.pexp(R["p"]), kind="p_raw", table=table, fileset=fs, n=R["n"],
            rule="Wilcoxon, cells" + sfx)
    ctx.num(prefix + "N", R["n"], str(R["n"]), kind="n", table=table, fileset=fs, n=R["n"],
            rule="paired (task, seed)" + sfx)
    ctx.num(prefix + "TasksGeocma", R["tasks_cma_better"], f"{R['tasks_cma_better']}/{R['n_tasks']}", kind="tasks_won",
            table=table, fileset=fs, n=R["n_tasks"], rule="tasks with lower GeoCMA mean normalised loss" + sfx)
    ctx.num(prefix + "PTask", R["p_task"], N.psig(R["p_task"]), kind="p_task", table=table, fileset=fs, n=R["n_tasks"],
            rule="exact Wilcoxon on per-task mean differences" + sfx)


def _join_names(names: list[str]) -> str:
    """'a', 'a and b', 'a, b and c' (caption sentences that name several methods or cells)."""
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def lasso_method_label(meth: str) -> str:
    """Plain-text method label (figure legend)."""
    return {"cmaes_sub": "GeoCMA", "geowalk": "GeoWalk", "random": "Random search"}.get(meth) or LB.lasso_plain(meth, meth)


def lasso_method_tex(meth: str) -> str:
    """LaTeX method label (captions)."""
    return dict(LASSO_METHODS).get(meth) or LB.lasso_label(meth, meth)


def tab_lassobench(ctx: Ctx) -> Table | None:
    table = "tab_lassobench"
    R = lasso_analysis(ctx)
    if R is None:
        ctx.skip(table, "lassobench", "input directory absent")
        return None
    if R["fileset"] is None or not R["n"]:
        ctx.skip(table, "lassobench", "no GeoWalk/GeoCMA pair with a finite loss (table omitted)")
        return None
    rerun = R["generation"] == "rerun"
    gen = "camera-ready rerun" if rerun else "stored runs"
    cov = R["loaded"]["coverage"] if rerun else None
    leuk = R["loaded"]["leukemia"] if rerun else None
    _reg_lasso_row(ctx, R, "crLasso", table, gen)
    # count-only total of the LassoBench tasks (RUN_RULES 2026-10-22; the n of the task-level tests on LassoBench)
    ctx.num("crNTasksLasso", R["n_tasks"], str(R["n_tasks"]), kind="n_tasks", table=table, fileset=R["fileset"],
            n=R["n_tasks"], rule="LassoBench tasks with GeoWalk and GeoCMA runs (the n of the task-level tests)")
    hdr = ["Method"] + [f"$D={D}$" for _, D in LASSO_TASKS]
    t = Table("lassobench", "tab:lassobench", "", WRAP + "r" * len(LASSO_TASKS), hdr, size="\\scriptsize",
              tabcolsep="3pt")
    man_fs = _fileset(R["loaded"]["manifests"]) if rerun and R["loaded"]["manifests"] else None
    printed = [(x, lab) for x, lab in LASSO_METHODS if not (ctx.mode == "cr" and x in LASSO_CORRESPONDENCE_ONLY)]
    for meth, lab in printed:
        row = [lab]
        for task, D in LASSO_TASKS:
            x = R["per_d"].get((meth, task))
            c = cov.get((meth, task)) if cov else None
            key = f"crLasso{camel(meth)}D{camel(str(D))}"
            if c is not None and man_fs is not None:
                # coverage of the expected design: result files (clean job record) among the expected seeds
                ctx.num(key + "Runs", len(c["present"]), str(len(c["present"])), kind="n", table=table,
                        fileset=_fileset(x["files"]) if x is not None else man_fs, n=c["expected"],
                        rule=f"result files with a clean job record, of {c['expected']} expected "
                             f"seed(s){' (pilot seed only)' if c['pilot'] else ''}")
            if x is None:
                row.append("--" if c is None else no_result_mark(c))
                continue
            fsd = _fileset(x["files"])
            ctx.num(key, x["mean"], N.f3(x["mean"]), kind="level", table=table,
                    fileset=fsd, n=x["n"], rule="mean normalised loss at this D (SE ddof 0 in the figure)")
            ctx.num(key + "N", x["n"], str(x["n"]), kind="n", table=table,
                    fileset=fsd, n=x["n"], rule="seeds with a finite loss")
            if c is not None and c["pilot"]:
                ntxt = f"pilot, {x['n']}"          # a single pilot seed is never shown as full coverage
            else:
                ntxt = str(x["n"]) if (c is None or x["n"] == c["expected"] == 40) else f"{x['n']}/{c['expected']}"
            row.append(f"{m(N.f3(x['mean']))}\\,({ntxt})")
        t.add(row)
    blk = []
    for meth, lab in printed:
        v = R["vs_geocma"].get(meth)
        if not v or not v["per_d"]:
            continue
        k = f"crLassoVsGeocma{camel(meth)}"
        ctx.num(k + "Delta", v["delta"], N.s3(v["delta"]), kind="delta", table=table, fileset=R["fileset_all"], n=v["n"],
                rule=f"{meth} - GeoCMA normalised loss, pooled over the dimensions where {meth} has runs (positive: GeoCMA lower)")
        ctx.num(k + "D", v["d"], N.d2(v["d"]), kind="cohens_d", table=table, fileset=R["fileset_all"], n=v["n"], rule="pooled pairs")
        ctx.num(k + "P", v["p"], N.pfmt(v["p"]) if v["p"] >= 1e-3 else N.pexp(v["p"]), kind="p_raw", table=table,
                fileset=R["fileset_all"], n=v["n"], rule="Wilcoxon over the pooled pairs")
        ctx.num(k + "N", v["n"], str(v["n"]), kind="n", table=table, fileset=R["fileset_all"], n=v["n"], rule="pooled pairs")
        row = [lab]
        for task, D in LASSO_TASKS:
            x = v["per_d"].get(task)
            if x is None:
                row.append("--")
                continue
            fsd = _fileset(x["files"])
            ctx.num(f"{k}D{camel(str(D))}P", x["p"], N.pfmt(x["p"]), kind="p_raw", table=table, fileset=fsd, n=x["n"],
                    rule=f"Wilcoxon, {meth} - GeoCMA at D = {D}, paired by seed")
            ctx.num(f"{k}D{camel(str(D))}D", x["d"], N.d2(x["d"]), kind="cohens_d", table=table, fileset=fsd, n=x["n"],
                    rule=f"d of {meth} - GeoCMA at D = {D}")
            row.append(stack(m(N.d2(x["d"])), m("(" + N.pfmt(x["p"]) + ")"), align="r"))
        row.append(stack(m(N.d2(v['d'])), m('(' + (N.pfmt(v['p']) if v['p'] >= 1e-3 else N.pexp(v['p'])) + ')'),
                         f"$n={v['n']}$", align="r"))
        blk.append(row)
    geocma_block = None
    if blk:
        t.add_block("Each method minus \\geocma{} (normalized loss, paired by seed): $d$ and, below it, the Wilcoxon $p$ "
                    "per dimension; last column pooled over the dimensions with runs ($d$, $p$, $n$); positive $d$ = "
                    "\\geocma{} lower loss",
                    WRAP + "r" * (len(LASSO_TASKS) + 1), ["Method"] + [f"$D={D}$" for _, D in LASSO_TASKS] + ["pooled"], blk)
        geocma_block = t.blocks[-1]
    row3 = (f"In the LassoBench row of Table~\\ref{{tab:crossdomain}}, \\geowalk{{}} has a mean normalized loss of "
            f"{N.f3(R['gw'])} and \\geocma{{}} of {N.f3(R['cma'])}. Over the $n={R['n']}$ paired (task, seed) cells, "
            f"$\\Delta={N.f3(R['delta'])}$, $d={N.f2(R['d'])}$ and the Wilcoxon $p={N.pexp(R['p'])}$. "
            f"\\geocma{{}} is lower on {R['tasks_cma_better']} of the {R['n_tasks']} tasks (exact task-level "
            f"$p={N.psig(R['p_task'])}$). ")
    fixed = "The synthetic tasks use LassoBench's default seed for every run (one fixed problem per task)"
    if not rerun:
        t.caption = ("\\textbf{LassoBench (runs of the submitted version): final loss divided by the task's random-search mean loss, per "
                     "dimension, with the number of seeds that produced a finite loss in parentheses.} Lower is better. "
                     + row3 + fixed + ", and the "
                     "leukemia ($D=7129$) losses mix two oracle versions (see Appendix~\\ref{app:lassobench_phase}).")
        return t
    # ---- camera-ready rerun: coverage and the arms that did not complete at D = 7129 --------------------
    from . import lasso_cr
    L = R["loaded"]
    s = L["summary"]
    all_fs = R["fileset_all"]
    for nm, v, rule in (("CellsExpected", s["expected"], f"runs of the expected design ({lasso_cr.design_text()})"),
                        ("CellsPresent", s["present"], "expected runs with a result file and a clean job record"),
                        ("CellsMissing", s["missing"], "expected runs without a result file"),
                        ("CellsNonfinite", s["nonfinite"], "result files with a non-finite loss (a model fit failed)")):
        ctx.num("crLasso" + nm, v, N.intfmt(v), kind="count", table=table, fileset=all_fs, n=s["expected"], rule=rule)
    nc = [x for x, _ in printed if leuk[x]["status"] == "not_completed"]
    pend = [x for x, _ in printed if leuk[x]["status"] == "pending"]
    pil = [x for x, _ in printed if leuk[x]["status"] == "pilot_completed"]
    ctx.num("crLassoLeukemiaNotCompletedN", len(nc), str(len(nc)), kind="count", table=table, fileset=man_fs,
            n=len(LASSO_METHODS), rule="methods whose D = 7129 arm did not complete (pilot or runs over the time "
                                       "limit, or no finite loss)")
    ctx.num("crLassoLeukemiaNotCompleted", ", ".join(nc) or "none",
            ", ".join(lasso_method_tex(x) for x in nc) or "none", kind="text", table=table,
            fileset=man_fs, n=len(LASSO_METHODS), rule="the methods that did not complete at D = 7129")
    ctx.num("crLassoLeukemiaPendingN", len(pend), str(len(pend)), kind="count", table=table, fileset=man_fs,
            n=len(LASSO_METHODS), rule="methods whose D = 7129 arm has no job record yet (the rerun is still running)")
    inc = [(k, c) for k, c in sorted(cov.items()) if len(c["finite"]) < c["expected"] and k[0] in dict(printed)]
    cap = ("\\textbf{LassoBench: final loss divided by the task's random-search mean loss, per dimension.} The runs "
           "are the " + lasso_gen_text(R) + ". In parentheses: the number of seeds that produced a finite loss ($n/40$ "
           "where fewer than the 40 seeds 42--81 have one). Lower is better. " + row3 + fixed + ". ")
    sents = []
    for why in dict.fromkeys(leuk[x]["text"] for x in nc):      # one sentence per stated cause
        sents.append("At $D=7129$ (leukemia), " + _join_names([lasso_method_tex(x) for x in nc if leuk[x]["text"] == why])
                     + f" did not complete (n.c.; {why}).")
    if pil:
        sents.append("At $D=7129$, " + _join_names([lasso_method_tex(x) for x in pil]) + " ran one pilot seed only. "
                     "The pilot finished within the time limit, but the method was not run on the 40 seeds.")
    if pend:
        sents.append("NOT RUN YET (pend.) at $D=7129$: " + ", ".join(lasso_method_tex(x) for x in pend) + ".")
    cap += (" ".join(sents) if sents else "At $D=7129$ (leukemia) every method ran.") + " "
    ncd = [(k, c) for k, c in sorted(cov.items()) if k[0] in dict(printed) and k[1] != "leukemia"
           and R["per_d"].get(k) is None and no_result_mark(c) == "n.c."]
    if ncd:
        cap += ("Did not complete (n.c., no seed with a finite loss): " + "; ".join(
            f"{lasso_method_tex(k[0])} $D={dict(LASSO_TASKS)[k[1]]}$ ({why_missing(c)})" for k, c in ncd) + ". ")
    if s["complete"] and not s["nonfinite"]:
        cap += f"Coverage: all {N.intfmt(s['expected'])} expected runs have a finite loss."
    else:
        cap += (f"Coverage: {N.intfmt(s['present'])} of {N.intfmt(s['expected'])} expected runs have a result file, "
                f"{s['nonfinite']} of them with a non-finite loss because a model fit failed. {s['timeout']} timed "
                f"out, {s['failed']} failed and {N.intfmt(s['pending'])} were not run yet.")
        if len(inc) <= 12:
            for mth in dict.fromkeys(k[0] for k, _ in inc):        # one sentence per method
                cap += (f" {lasso_method_tex(mth)} has fewer finite losses than expected at "
                        + _join_names([f"$D={dict(LASSO_TASKS)[k[1]]}$ ({len(c['finite'])}/{c['expected']})"
                                       for k, c in inc if k[0] == mth]) + ".")
        else:
            cap += (f" {len(inc)} method--dimension cells "
                    + ("have fewer finite losses than expected" if s["finished"] else "are incomplete")
                    + " (listed in \\texttt{lassobench\\_cr\\_coverage.md}).")
    if not s["finished"]:
        # an expected job without a job record (never the number of incomplete cells) makes the table provisional
        cap += " \\textbf{INCOMPLETE RERUN: not the final table.}"
    cap += (" The rerun's trust-region, full-dimensional CMA-ES and fixed-subspace rows are not printed: the "
            "published methods below supersede them. ")
    cap += _lasso_pub_block(ctx, t, table)
    t.caption = cap
    # legibility (RUN_RULES 2026-10-22, format only): at 7 pt the three blocks and the caption do not fit one page,
    # so the paired comparisons are set as a second float right after the first; every value is unchanged
    t.continued = ("\\textbf{LassoBench, continued from Table~\\ref{tab:lassobench}: paired comparisons.} Same runs and "
                   "normalization as Table~\\ref{tab:lassobench}. Top: each method minus \\geocma{}. Bottom: "
                   "\\geowalk{} against each published method, the baseline family of LassoBench.",
                   "tab:lassobench_family")
    # RUN_RULES 2026-10-29 (2), (4): the full table -- its caption with the coverage and the time-outs of every method,
    # and the block "each method minus GeoCMA" -- is a repository record (records/tab_lassobench_full.tex, the bytes of
    # the table printed before); the printed table keeps every value of the other blocks and a short caption
    full = t
    t = Table("lassobench", "tab:lassobench", "", full.colspec, full.header, size=full.size, tabcolsep=full.tabcolsep)
    # the printed view: the group heading without the naming rule (Appendix H states it once), the family block with
    # a short title and the LassoBench-only methods marked (RUN_RULES 2026-10-29 (4)); the record keeps its bytes
    from . import published
    P0 = lasso_pub_analysis(ctx)
    only = {published.name_or_label(ctx, x, "lasso") for x in (P0["printed"] if P0 else [])
            if not published.RECIPES[published.recipe_of(x, "lasso")]["guacamol"]}
    # RUN_RULES 2026-11-02 (3): the printed view marks the recipes that were not verified; the record keeps the words
    nv = lambda r: [r[0].replace(" (not verified)", "$^{\\circ}$").replace(" (rerun)", "")] + r[1:] if not isinstance(r, tuple) else r
    t.rows = [("group", LASSO_PUB_GROUP) if isinstance(r, tuple) and r[0] == "group" and r[1] == LASSO_PUB_GROUP_FULL
              else nv(r) for r in full.rows]
    t.blocks = [("\\geowalk{} against each published method", b[1], b[2],
                 [nv(r) if isinstance(r, tuple) or r[0] not in only else nv([r[0] + "$^{\\S}$"] + r[1:]) for r in b[3]])
                for b in full.blocks if b is not geocma_block]
    t.caption = ("\\textbf{LassoBench: final loss divided by the task's random-search mean loss, per dimension.} "
                 "Each value is the mean over seeds of this ratio after 30 evaluations. A value of 1 equals random "
                 "search, and a value above 1 is worse than random search. Lower is better. The tasks are "
                 + ", ".join(f"$D={D}$ {LASSO_TASK_NAMES.get(task, task)}" for task, D in LASSO_TASKS)
                 + ". In parentheses: the seeds with a finite loss ($n/40$ where fewer than the 40 seeds 42--81 have "
                 "one). n.c.: no seed finished with a finite loss. n.a.: not applicable at this dimension, so not run. "
                 "All rows ran in one environment, with one CPU thread and at most one hour per "
                 "run. $^{\\circ}$~Not verified against a "
                 "published result (Table~\\ref{tab:verification}). The coverage and time-outs "
                 "of each method, and each method minus \\geocma{}, are in the full table released with the code "
                 "(\\texttt{docs/baselines/" + LASSO_FULL_TEX.replace("_", "\\_") + "})."
                 + "".join(f" \\textbf{{{mark}: not the final table.}}" for mark in
                           ("INCOMPLETE RERUN", "INCOMPLETE PUBLISHED-METHOD SET") if mark in full.caption))
    if t.blocks:
        fam_n = len(P0["printed"]) if P0 else 0
        t.continued = ("\\textbf{LassoBench, continued from Table~\\ref{tab:lassobench}: \\geowalk{} against each "
                       "published method.} Same runs and normalization as Table~\\ref{tab:lassobench}. These methods "
                       "form the baseline family of LassoBench. Here a task is a dimension. The difference $\\Delta$ "
                       "is \\geowalk{} minus the method, a mean over tasks of the per-task mean normalized losses. A "
                       "positive $\\Delta$ means that \\geowalk{} has the higher, worse loss. Tasks counts the tasks "
                       "on which \\geowalk{} has the lower mean loss, of those where both have runs. The column $n$ "
                       "gives the (task, seed) pairs that both completed. The column $p_{\\mathrm{task}}$ is the exact "
                       "two-sided Wilcoxon test on the per-task mean differences. The column "
                       "$p_{\\mathrm{task}}^{\\mathrm{BH}}$ is $p_{\\mathrm{task}}$ corrected by Benjamini--Hochberg "
                       f"over the {fam_n} methods. Bold marks values below 0.05. With $k$ tasks the exact two-sided test "
                       "cannot go below $2/2^k$. This floor is 0.031 for 6 tasks and 0.063 for 5 tasks. A method with "
                       "fewer than 6 tasks therefore cannot reach 0.05."
                       + (" The mark $^{\\S}$ denotes the LassoBench-only methods, which have no GuacaMol run. This "
                          f"family therefore has {fam_n} methods, and the family of Table~\\ref{{tab:published}} has "
                          f"{len(published.GUACAMOL_ROWS)} recipes." if only else ""),
                       full.continued[1])
    full.name = LASSO_FULL_TEX[len("tab_"):-len(".tex")]
    t.record_tables = [full]
    # the coverage reports next to the tables (one per output folder)
    P = lasso_pub_analysis(ctx)
    for fn, md in (("lassobench_cr_coverage.md", R["coverage_md"]), ("lassobench_pub_coverage.md",
                                                                    P["coverage_md"] if P else None)):
        if md is None:
            continue
        p = os.path.join(ctx.outdir, fn)
        with open(p, "w") as f:
            f.write(md)
        if p not in ctx.written:
            ctx.written.append(p)
    return t


def _lasso_pub_block(ctx: Ctx, t: Table, table: str) -> str:
    """Rows of the printed published methods in the LassoBench table and the baseline family (GeoWalk against
    each); returns the caption sentences. Camera-ready mode only."""
    from . import lasso_cr, published
    P = lasso_pub_analysis(ctx)
    if P is None:
        return " Published methods: not available in this run."
    L = P["loaded"]
    cov, leuk, s = L["coverage"], L["leukemia"], L["summary"]
    man_fs = _fileset(L["manifests"])
    t.group(LASSO_PUB_GROUP_FULL)
    for mth in P["printed"]:
        row = [published.name_or_label(ctx, mth, "lasso")]
        for task, D in LASSO_TASKS:
            x, c = P["per_d"].get((mth, task)), cov.get((mth, task))
            key = f"crLassoPub{camel(mth)}D{camel(str(D))}"
            ctx.num(key + "Runs", len(c["present"]), str(len(c["present"])), kind="n", table=table,
                    fileset=_fileset(x["files"]) if x is not None else man_fs, n=c["expected"],
                    rule=f"result files with a clean job record, of {c['expected']} expected seed(s)"
                         + (" (pilot seed only)" if c["pilot"] else ""))
            if x is None:
                row.append(no_result_mark(c))
                continue
            fsd = _fileset(x["files"])
            ctx.num(key, x["mean"], N.f3(x["mean"]), kind="level", table=table, fileset=fsd, n=x["n"],
                    rule="mean of loss / the task's random-search mean loss of the rerun")
            ctx.num(key + "N", x["n"], str(x["n"]), kind="n", table=table, fileset=fsd, n=x["n"],
                    rule="seeds with a finite loss")
            ntxt = (f"pilot, {x['n']}" if c["pilot"] else
                    (str(x["n"]) if x["n"] == c["expected"] == 40 else f"{x['n']}/{c['expected']}"))
            row.append(f"{m(N.f3(x['mean']))}\\,({ntxt})")
        t.add(row)
    blk = []
    fam_n = len(P["printed"])
    for mth in P["printed"]:
        f = P["fam"][mth]
        k = f"crLassoPub{camel(mth)}VsGeowalk"
        fs = _fileset(f["files"]) if f["files"] else man_fs
        ctx.num(k + "Delta", f["delta"], N.s3(f["delta"]), kind="delta", table=table, fileset=fs, n=f["n_tasks"],
                rule="mean over tasks of (GeoWalk - method) per-task mean normalised loss (negative: GeoWalk lower)")
        ctx.num(k + "Tasks", f["gw_lower"], f"{f['gw_lower']}/{f['n_tasks']}", kind="tasks_won", table=table,
                fileset=fs, n=f["n_tasks"], rule="tasks on which GeoWalk has the lower mean normalised loss")
        ctx.num(k + "N", f["n_pairs"], str(f["n_pairs"]), kind="n", table=table, fileset=fs, n=f["n_pairs"],
                rule="(task, seed) pairs completed by both GeoWalk and the method (the matched set of the test)")
        ctx.num(k + "PTask", f["p_task"], N.psig(f["p_task"]), kind="p_task", table=table, fileset=fs,
                n=f["n_tasks"], rule="exact two-sided Wilcoxon over the per-task mean differences (primary)")
        ctx.num(k + "PTaskBH", f["p_task_bh"], N.psig(f["p_task_bh"]), kind="p_task_bh", table=table, fileset=fs,
                n=f["n_tasks"], rule=f"Benjamini-Hochberg within the {fam_n} printed published methods")
        word, sent = published.reading(-f["delta"], f["p_task_bh"], f["gw_lower"], f["mth_lower"], f["n_tasks"],
                                       "\\geowalk{}", published.name_or_label(ctx, mth, "lasso"), ahead="lower loss")
        ctx.num(k + "Verdict", word, published.VERDICT_TEXT[word], kind="text", table=table, fileset=fs, n=f["n_tasks"],
                rule="RUN_RULES 2026-10-12 (4) on LassoBench (lower loss = ahead); printed text of RUN_RULES 2026-10-22")
        ctx.num(k + "Sentence", sent, sent, kind="text", table=table, fileset=fs, n=f["n_tasks"],
                rule="pre-written sentence of rule (4) chosen by the verdict above")
        blk.append([published.name_or_label(ctx, mth, "lasso"), m(N.s3(f["delta"])), f"{f['gw_lower']}/{f['n_tasks']}",
                    str(f["n_pairs"]), m(N.psig(f["p_task"])),
                    mb(N.psig(f["p_task_bh"]), np.isfinite(f["p_task_bh"]) and f["p_task_bh"] < 0.05)])
    t.add_block("\\geowalk{} against each published method (per-task means of the normalized-loss differences on the "
                "seeds both completed, $n$ = matched (task, seed) pairs; exact task-level "
                f"Wilcoxon, Benjamini--Hochberg within the {fam_n}"
                + stats.bh_note([P["fam"][x]["p_task"] for x in P["printed"]], "methods") + "; bold: "
                "$p_{\\mathrm{task}}^{\\mathrm{BH}}<0.05$; Tasks = tasks, of those where both have runs, on which "
                "\\geowalk{} has the lower mean loss)", WRAP + "rrrrr",
                ["Method", stack("$\\Delta$", "(\\geowalk{} $-$ method)"), stack("Tasks", "\\geowalk{} lower"), "$n$",
                 "$p_{\\mathrm{task}}$", "$p_{\\mathrm{task}}^{\\mathrm{BH}}$"], blk)
    # review S2: the printed counts cover the printed methods only (the full design is in the coverage report)
    pc = [c for (mm, _t), c in cov.items() if mm in P["printed"]]
    pexp = sum(c["expected"] for c in pc)
    for nm, v, rule in (("PubCellsExpected", pexp, f"runs of the {len(P['printed'])} printed published methods "
                         "(full design: " + lasso_cr.pub_design_text(L=L) + ")"),
                        ("PubCellsPresent", sum(len(c["present"]) for c in pc),
                         "expected runs of the printed methods with a result file and a clean job record"),
                        ("PubCellsTimeout", sum(len(c["timeout"]) for c in pc),
                         "expected runs of the printed methods that exceeded the one-hour limit")):
        ctx.num("crLasso" + nm, v, N.intfmt(v), kind="count", table=table, fileset=man_fs, n=pexp, rule=rule)
    nc = [x for x in P["printed"] if leuk[x]["status"] == "not_completed"]
    na = [x for x in P["printed"] if any(cov[(x, tk)]["not_applicable"] for tk, _ in LASSO_TASKS)]
    txt = (" Published methods follow the authors' released code in every tabulated setting. They have the same "
           "budget and limits (30 evaluations, one CPU thread, at most one hour per run) and are normalized by the "
           "rerun's random-search mean. The nested-subspace method evaluates its own 5-point Sobol design in its "
           "first subspace. CMA-ES evaluates $x_0=0$ first and then 29 points of its own. Neither of the two uses the "
           "shared cold starts. Every other method starts from the 5 shared cold starts and takes 25 steps. In this "
           "table, a method carries the authors' name only if it reproduced a published result (Table~\\ref{tab:verification}). "
           "Otherwise it is named by kernel, acquisition and maximizer and marked as not verified. For MSR and MLE "
           "(scaled) the printed variant learns the noise, because that variant passed the verification. ")
    for why in dict.fromkeys(leuk[x]["text"] for x in nc):      # one sentence per stated cause
        txt += ("At $D=7129$, " + _join_names([published.name_or_label(ctx, x, "lasso") for x in nc
                                              if leuk[x]["text"] == why]) + f" did not complete (n.c.; {why}). ")
    for x in na:
        dims = [D for tk, D in LASSO_TASKS if cov[(x, tk)]["not_applicable"]]
        txt += (f"{published.name_or_label(ctx, x, 'lasso')} was not run at "
                + _join_names([f"$D={D}$" for D in dims]) + f" (n.a.). {na_reason(cov, x)} ")
    ncd = [(k, c) for k, c in sorted(cov.items()) if k[0] in P["printed"] and k[1] != "leukemia"
           and P["per_d"].get(k) is None and no_result_mark(c) == "n.c."]
    if ncd:
        txt += ("Did not complete (n.c., no seed with a finite loss): " + "; ".join(
            f"{published.name_or_label(ctx, k[0], 'lasso')} $D={dict(LASSO_TASKS)[k[1]]}$ ({why_missing(c)})"
            for k, c in ncd) + ". ")
    inc = [(k, c) for k, c in sorted(cov.items()) if k[0] in P["printed"] and 0 < len(c["finite"]) < c["expected"]]
    if inc:
        # review M3: the cause of every excluded seed is the one its record or file states
        txt += ("Seeds without a finite loss are excluded, and the mean is over the completed seeds ($n/40$). "
                + " ".join(f"{published.name_or_label(ctx, k[0], 'lasso')}, $D={dict(LASSO_TASKS)[k[1]]}$: "
                           f"{len(c['finite'])}/{c['expected']} ({why_missing(c)})." for k, c in inc) + " ")
    if not s["finished"]:
        txt += "\\textbf{INCOMPLETE PUBLISHED-METHOD SET: not the final table.} "
    return txt


def _ref_printed(ids: dict) -> tuple[dict, str | None]:
    """Printed values of reference_values.json entries (the submitted Table 3 row)."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference_values.json")
    out = {}
    if not os.path.exists(p):
        return out, None
    by_id = {e["id"]: e for e in json.load(open(p))["entries"]}
    for k, i in ids.items():
        if i in by_id:
            out[k] = str(by_id[i].get("printed", ""))
    return out, p


def tab_lassobench_correspondence(ctx: Ctx) -> Table | None:
    """Camera-ready mode only: the rerun next to the stored March/April values and the submitted Table 3 row."""
    table = "tab_lassobench_correspondence"
    if ctx.mode != "cr":
        ctx.skip(table, "lassobench", "legacy mode prints the stored set itself (no correspondence table)")
        return None
    S = lasso_stored_analysis(ctx)
    R = lasso_rerun_analysis(ctx)
    if S is None or R is None or S["fileset"] is None or R["fileset"] is None or not R["n"]:
        ctx.skip(table, "lassobench", "stored set or rerun absent (table omitted)")
        return None
    _reg_lasso_row(ctx, S, "crLassoStored", table, "stored March/April runs")
    sub, ref_path = _ref_printed({"gw": "S-t3-crLassoGeowalk", "cma": "S-t3-crLassoGeocma",
                                  "delta": "S-t3-crLassoDelta", "d": "S-t3-crLassoD"})
    hdr = ["Row of Table~\\ref{tab:crossdomain}", "\\geowalk{}", "\\geocma{}", "$\\Delta$", "$d$", "$p$", "$n$",
           stack("Tasks", "\\geocma{}", "lower"), "$p_{\\mathrm{task}}$"]
    t = Table("lassobench_correspondence", "tab:lassobench_correspondence", "", WRAP + "rrrrrrrr", hdr,
              size="\\footnotesize", wide=True, tabcolsep="4pt")
    t.add(["Submitted (printed)", m(sub.get("gw", "--")), m(sub.get("cma", "--")), m(sub.get("delta", "--")),
           m(sub.get("d", "--")), m("<10^{-5}"), "--", "--", "--"])
    for lab, X in (("Files of the submitted version, recomputed", S), ("Rerun for this version (printed)", R)):
        t.add([lab, m(N.f3(X["gw"])), m(N.f3(X["cma"])), m(N.f3(X["delta"])), m(N.f2(X["d"])), m(N.pexp(X["p"])),
               str(X["n"]), f"{X['tasks_cma_better']}/{X['n_tasks']}", m(N.psig(X["p_task"]))])
    blk = []
    cov = R["loaded"]["coverage"]
    leuk = R["loaded"]["leukemia"]
    for meth, lab in LASSO_METHODS:
        row = [lab]
        for task, D in LASSO_TASKS:
            xs, xr = S["per_d"].get((meth, task)), R["per_d"].get((meth, task))
            if xs is not None:
                fsd = _fileset(xs["files"])
                ks = f"crLassoStored{camel(meth)}D{camel(str(D))}"
                ctx.num(ks, xs["mean"], N.f3(xs["mean"]), kind="level", table=table, fileset=fsd, n=xs["n"],
                        rule="mean normalised loss at this D, stored March/April runs")
                ctx.num(ks + "N", xs["n"], str(xs["n"]), kind="n", table=table, fileset=fsd, n=xs["n"],
                        rule="seeds with a finite loss, stored runs")
            # the seed count is shown only where it is not the full 40
            a = (m(N.f3(xs["mean"])) + ("" if xs["n"] == 40 else f"\\,({xs['n']})")) if xs is not None else "--"
            if xr is not None:
                b = m(N.f3(xr["mean"])) + ("" if xr["n"] == 40 else
                                           f"\\,({'pilot, ' if cov[(meth, task)]['pilot'] else ''}{xr['n']})")
            else:
                b = no_result_mark(cov[(meth, task)])
            row.append(stack(a, b, align="r"))
        blk.append(row)
    t.add_block("Mean normalized loss per method and dimension: runs of the submitted version (upper value) and the "
                "rerun for this version (lower value); in parentheses the seeds with a finite loss where fewer than "
                "40; n.c. = did not complete, pend. = not run yet",
                WRAP + "r" * len(LASSO_TASKS), ["Method"] + [f"$D={D}$" for _, D in LASSO_TASKS], blk)
    s = R["loaded"]["summary"]
    t.caption = ("\\textbf{LassoBench: the rerun for this version next to the runs of the submitted version and the "
                 "submitted LassoBench row of Table~\\ref{tab:crossdomain}.} "
                 "Submitted = the row printed in the submission; files of the submitted version = its March/April 2026 "
                 "result files, recomputed here (they mix two versions of the objective, and their GP runs could not "
                 "be reproduced, which is why LassoBench was rerun in full); rerun = " + lasso_gen_text(R) + ", the "
                 "printed results. Same convention in both recomputed rows: loss divided by the task's random-search "
                 "mean, \\geowalk{} and \\geocma{} paired over (task, seed), $\\Delta$ = \\geowalk{} $-$ \\geocma{}, "
                 "$d$ = mean/sd of the paired differences, two-sided Wilcoxon $p$; Tasks = tasks on which the mean "
                 "paired difference favors \\geocma{} (the seeds both completed), $p_{\\mathrm{task}}$ = exact Wilcoxon "
                 "over those per-task means."
                 + ("" if s["finished"] else f" \\textbf{{The rerun is incomplete}} ({N.intfmt(s['present'])} of "
                                             f"{N.intfmt(s['expected'])} expected runs): not the final table."))
    return t


# =========================================================================================
# NAS-Bench-101
# =========================================================================================
def nas_analysis(ctx: Ctx) -> dict | None:
    d = ctx.extra.get("nas_dir")
    if not d or not os.path.isdir(d):
        return None
    if "nas" in ctx.extra:
        return ctx.extra["nas"]
    rx = re.compile(r"^(?P<m>.+)_cifar10_s(?P<s>\d+)\.json$")
    cells = defaultdict(dict)
    for n in sorted(os.listdir(d)):
        mm = rx.match(n)
        if not mm:
            continue
        p = os.path.join(d, n)
        v = json.load(open(p)).get("best_score")
        cells[mm.group("m")][int(mm.group("s"))] = (float(v), p)
    out = {"means": {meth: float(np.mean([v for v, _ in cells[meth].values()])) for meth in cells},
           "n": {meth: len(cells[meth]) for meth in cells}, "pairs": {},
           "files": {meth: [p for _, p in cells[meth].values()] for meth in cells}}
    for a, b in [("cmaes_sub", "geowalk"), ("cmaes_sub", "random"), ("cmaes_sub", "turbo"), ("cmaes_sub", "gp_bo"),
                 ("geowalk", "random")]:
        seeds = sorted(set(cells[a]) & set(cells[b]))
        diff = np.asarray([cells[a][s][0] - cells[b][s][0] for s in seeds])
        out["pairs"][(a, b)] = {"n": len(seeds), "delta": float(diff.mean()), "d": stats.cohens_d(diff),
                                "p": stats.wilcoxon_p(diff),
                                "fileset": _fileset([cells[a][s][1] for s in seeds] + [cells[b][s][1] for s in seeds])}
    ctx.extra["nas"] = out
    return out


#: 2026-11-03 (authors' decision): Table 3 prints, per benchmark, the best other method that covers the row's
#: matched set, with its mean, so that the table itself shows where GeoWalk and GeoCMA stand against the field
BEST_COL = ">{\\raggedright\\arraybackslash}p{2.2cm}"
NAS_OTHER = (("random", "random search"), ("turbo", "TuRBO"), ("gp_bo", "GP"), ("cmaes_full", "CMA-ES"))
LASSO_SHORT = {"hvarfner_published": "Vanilla BO", "papenmeier_published_learned": "MSR",
               "papenmeier_mle_scaled_published_learned": "MLE (scaled)", "xu_published": "Standard GP (Xu et al.)",
               "linear_bo_published": "Linear BO", "turbo_published": "TuRBO", "baxus_official": "BAxUS",
               "cmaes_full_published": "CMA-ES", "hvarfner_gp": "RBF GP", "papenmeier_msr": "Mat\\'ern-5/2 GP ($\\ell_0=0.1\\sqrt{D}$)",
               "xu_standard_gp": "Mat\\'ern-5/2 GP ($\\ell_0=\\sqrt{D}$)", "linear_bo": "linear-kernel GP"}


def _best_cell(name: str, value: float) -> str:
    return f"{name}, {m(N.f3(value))}"


def _best_other_guacamol(ctx: Ctx, table: str) -> str:
    """The highest cross-mean among the comparators printed in Table 1, on its matched set (tab_main stashes them)."""
    rows = ctx.extra.get("main_printed")
    if not rows:
        ctx.skip(table, "best other method (GuacaMol)", "Table 1 did not run first")
        return "--"
    name, arm, lev, cells = max(rows, key=lambda r: r[2])
    ctx.num("crCrossGuacamolBestOtherMean", lev, N.f3(lev), kind="level", table=table, fileset=ctx.fs({arm: cells}),
            n=len(cells), rule=f"highest cross-mean among the printed comparators of Table 1 ({arm}) on its matched set")
    return _best_cell(name, lev)


def _best_other_nas(ctx: Ctx, R: dict, table: str) -> str:
    """The highest mean best accuracy among the other NAS-Bench-101 methods (seeds 42-51)."""
    cands = [(lab, meth, R["means"][meth]) for meth, lab in NAS_OTHER if meth in R["means"]]
    if not cands:
        return "--"
    lab, meth, val = max(cands, key=lambda c: c[2])
    ctx.num("crCrossNasBestOtherMean", val, N.f3(val), kind="level", table=table, fileset=_fileset(R["files"][meth]),
            n=R["n"][meth], rule=f"highest mean best accuracy among the other NAS-Bench-101 methods ({meth}), seeds 42-51")
    return _best_cell(lab, val)


def _best_other_lasso(ctx: Ctx, L: dict, table: str) -> str:
    """The lowest normalized loss, averaged over the six tasks, among the other LassoBench methods with every seed of
    every task (the adapted GP rows of the rerun and the published methods)."""
    P = lasso_pub_analysis(ctx)
    nrand = {t: len(L["cells"]["random"].get(t, {})) for t, _ in LASSO_TASKS} if "cells" in L else {}
    cands = []
    for src, meths in ((L["per_d"], [m_ for m_, _ in LASSO_METHODS if m_ in ("hvarfner_gp", "papenmeier_msr", "xu_standard_gp", "linear_bo")]),
                       (P["per_d"] if P else {}, list(P["printed"]) if P else [])):
        for meth in meths:
            ent = [src.get((meth, t)) for t, _ in LASSO_TASKS]
            if any(e is None for e in ent) or any(e["n"] < nrand.get(t, 40) for e, (t, _) in zip(ent, LASSO_TASKS)):
                continue
            cands.append((meth, float(np.mean([e["mean"] for e in ent])), [p for e in ent for p in e["files"]]))
    if not cands:
        ctx.skip(table, "best other method (LassoBench)", "no other method covers every seed of every task")
        return "--"
    meth, val, files = min(cands, key=lambda c: c[1])
    ctx.num("crCrossLassoBestOtherMean", val, N.f3(val), kind="level", table=table, fileset=_fileset(files),
            n=6 * nrand.get(LASSO_TASKS[0][0], 40),
            rule=f"lowest normalized loss, mean over the six tasks, among the other methods with every seed of every task ({meth})")
    return _best_cell(LASSO_SHORT.get(meth, lasso_method_label(meth)), val)


def tab_crossdomain(ctx: Ctx) -> Table | None:
    table = "tab_crossdomain"
    hdr = ["Benchmark", "\\geowalk{}", "\\geocma{}", "Better", "$|\\Delta|$", "$|d|$", stack("$p$", "(cells)"), "$n$",
           stack("Tasks", "better"), stack("$p_{\\mathrm{task}}$", "(tasks)"), "Best other method"]
    t = Table("crossdomain", "tab:crossdomain", "", WRAP + "rrlrrrrrr" + BEST_COL, hdr, size="\\footnotesize", tabcolsep="4pt")
    view = []       # RUN_RULES 2026-11-01: the rows of the printed view, in the layout of the submitted Table 3
    tl = ctx.var.tasklevel
    if ctx.has("geowalk") and ctx.has("cma_es_subspace"):
        cells = ctx.eng.joint(["geowalk", "cma_es_subspace"])
        pr = ctx.eng.paired("geowalk", "cma_es_subspace", cells)
        o = ctx.reg_paired("crCrossGuacamol", pr, "geowalk", "cma_es_subspace", cells, table)
        lg = ctx.reg_level("crCrossGuacamolGeowalk", "geowalk", cells, table, fmt=N.f3)[0]
        lc = ctx.reg_level("crCrossGuacamolGeocma", "cma_es_subspace", cells, table, fmt=N.f3)[0]
        # RUN_RULES 2026-10-22: the task-level columns print the values reg_paired registered (crCrossGuacamolTasks,
        # crCrossGuacamolPTask); Tasks counts the tasks on which the better method of the row is ahead
        tw = pr.tasks_won if pr.delta > 0 else pr.tasks_lost
        best_g = _best_other_guacamol(ctx, table)
        t.add(["GuacaMol (256-D SELFIES VAE)", m(N.f3(lg)), m(N.f3(lc)), "\\geowalk{}" if pr.delta > 0 else "\\geocma{}",
               m(N.f3(abs(pr.delta))), m(N.f2(abs(pr.d))), m(N.pexp(pr.p)), N.intfmt(pr.n),
               f"{tw}/{pr.n_tasks}", m(N.psig(pr.p_task)), best_g])
        view.append(["GuacaMol mol VAE (256D)", "$\\uparrow$", mb(N.f3(lg), pr.delta > 0), mb(N.f3(lc), pr.delta <= 0),
                     m(N.f3(abs(pr.delta))), m(N.f2(abs(pr.d))), m(N.psig(pr.p_task)) if tl else m(N.pexp(pr.p)),
                     N.intfmt(pr.n), best_g])
    else:
        ctx.skip(table, "GuacaMol row", "arm absent")
    R = nas_analysis(ctx)
    if R:
        x = R["pairs"][("cmaes_sub", "geowalk")]
        ctx.num("crNasGeowalk", R["means"]["geowalk"], N.f4(R["means"]["geowalk"]), kind="level", table=table,
                fileset=x["fileset"], n=x["n"], rule="mean best accuracy over seeds 42-51")
        ctx.num("crNasGeocma", R["means"]["cmaes_sub"], N.f4(R["means"]["cmaes_sub"]), kind="level", table=table,
                fileset=x["fileset"], n=x["n"], rule="mean best accuracy over seeds 42-51")
        ctx.num("crNasDelta", x["delta"], N.s4(x["delta"]), kind="delta", table=table, fileset=x["fileset"], n=x["n"],
                rule="GeoCMA - GeoWalk, paired by seed")
        ctx.num("crNasD", x["d"], N.f2(x["d"]), kind="cohens_d", table=table, fileset=x["fileset"], n=x["n"], rule="paired seeds")
        ctx.num("crNasP", x["p"], N.pfmt(x["p"]), kind="p_raw", table=table, fileset=x["fileset"], n=x["n"],
                rule="Wilcoxon (exact, n = 10)")
        ctx.num("crNasN", x["n"], str(x["n"]), kind="n", table=table, fileset=x["fileset"], n=x["n"], rule="seeds")
        for (a, b), y in R["pairs"].items():
            if (a, b) == ("cmaes_sub", "geowalk"):
                continue
            k = f"crNas{camel(a)}Vs{camel(b)}"
            ctx.num(k + "Delta", y["delta"], N.s4(y["delta"]), kind="delta", table=table, fileset=y["fileset"], n=y["n"],
                    rule=f"{a} - {b}, paired by seed")
            ctx.num(k + "P", y["p"], N.pfmt(y["p"]), kind="p_raw", table=table, fileset=y["fileset"], n=y["n"], rule="Wilcoxon")
        # 2026-11-03 (authors' decision): the printed row has three decimals like the other two rows and the
        # submitted version; the \crNas* macros keep four.
        best_n = _best_other_nas(ctx, R, table)
        t.add(["NAS-Bench-101 (16-D arch2vec)", m(N.f3(R["means"]["geowalk"])), m(N.f3(R["means"]["cmaes_sub"])),
               "\\geocma{}" if x["delta"] > 0 else "\\geowalk{}", m(N.f3(abs(x["delta"]))), m(N.f2(abs(x["d"]))),
               m(N.pfmt(x["p"])), str(x["n"]), "--", "--", best_n])
        view.append(["NAS-Bench-101 graph VAE (16D)", "$\\uparrow$", mb(N.f3(R["means"]["geowalk"]), x["delta"] <= 0),
                     mb(N.f3(R["means"]["cmaes_sub"]), x["delta"] > 0), m(N.f3(abs(x["delta"]))), m(N.f2(abs(x["d"]))),
                     m(N.pfmt(x["p"]) + "^{a}"), str(x["n"]), best_n])
    else:
        ctx.skip(table, "NAS-Bench row", "input directory absent")
    L = lasso_analysis(ctx)        # the printed generation: the rerun in camera-ready mode, else the stored set
    if L and L["n"]:
        lw = L["tasks_cma_better"] if L["delta"] > 0 else L["n_tasks"] - L["tasks_cma_better"]
        best_l = _best_other_lasso(ctx, L, table)
        t.add(["LassoBench ($D=60$--$7129$, normalized loss)", m(N.f3(L["gw"])), m(N.f3(L["cma"])),
               "\\geocma{}" if L["delta"] > 0 else "\\geowalk{}", m(N.f3(abs(L["delta"]))), m(N.f2(abs(L["d"]))),
               m(N.pexp(L["p"])), str(L["n"]), f"{lw}/{L['n_tasks']}", m(N.psig(L["p_task"])), best_l])
        view.append(["LassoBench, $D\\in[60,7129]$, normalized", "$\\downarrow$", mb(N.f3(L["gw"]), L["delta"] <= 0),
                     mb(N.f3(L["cma"]), L["delta"] > 0), m(N.f3(abs(L["delta"]))), m(N.f2(abs(L["d"]))),
                     m(N.psig(L["p_task"])) if tl else m(N.pexp(L["p"])), str(L["n"]), best_l])
    gsrc = "runs for this version" if ctx.mode == "cr" else "legacy files (April GeoWalk, March GeoCMA)"
    t.caption = ("\\textbf{\\geowalk{} and \\geocma{} across domains at $N=30$.} " f"GuacaMol: {gsrc}, "
                 f"{conv_text(ctx)} on the matched cells. NAS-Bench-101: runs of the submitted version (not rerun), "
                 "seeds 42--51, best validation accuracy (higher is better). LassoBench: loss divided by the "
                 "random-search mean loss of the task (lower is better). The LassoBench runs are the "
                 + (lasso_gen_text(L) if L and L["generation"] == "rerun" else "runs of the submitted version")
                 + ". Both $|\\Delta|$ and $|d|$ are paired. The cell-level $p$ is the two-sided Wilcoxon test over the "
                 "paired (task, seed) cells. Tasks = tasks on which the better method of the row has the better per-task "
                 "mean. The task-level $p_{\\mathrm{task}}$ is the exact two-sided Wilcoxon test on the per-task mean "
                 "differences, over 11 tasks on GuacaMol and 6 on LassoBench. It is the primary test of the paper and is "
                 "not corrected across the rows. NAS-Bench-101 has a single task, so it has no task-level test.")
    # ---- RUN_RULES 2026-11-01: the printed Table 3 has the layout of the submitted version (authors' decision) -------
    # its columns plus n (promise C-30); p is the task-level test; every value is one of the full table above, which
    # stays in the records with the cell-level p and the task counts
    full = t
    full.name = "crossdomain_full"
    phdr = ["Benchmark", "Dir.", "\\geowalk{}", "\\geocma{}", "$\\Delta$", "$|d|$",
            "$p_{\\mathrm{task}}$" if tl else "$p$", "$n$", "Best other method"]
    t = Table("crossdomain", "tab:crossdomain", "", WRAP + "c" + "r" * (len(phdr) - 3) + BEST_COL, phdr, size="\\small", wide=True,
              tabcolsep="3pt")
    for row in view:
        t.add(row)
    t.caption = ("\\textbf{\\geowalk{} and \\geocma{} across benchmark regimes.} $N=30$. Arrows indicate optimization "
                 f"direction, and bold marks the better mean. GuacaMol: {conv_text(ctx)} on the matched cells. "
                 "NAS-Bench-101: stored runs, seeds 42--51. LassoBench aggregates six "
                 "dimensions by normalizing each method's loss by the random-search mean loss of the task. The LassoBench "
                 "runs " + ("ran in one environment, with one CPU thread and at most one hour per run"
                           if L and L["generation"] == "rerun" else "are stored runs")
                 + ". Best other method: the best mean among the other methods that cover the row's matched set, "
                 "named as in Tables~\\ref{tab:main} and~\\ref{tab:lassobench} (NAS-Bench-101: random search, TuRBO, a GP "
                 "and full-dimensional CMA-ES). The difference $\\Delta$ is paired and favors the better method, and $|d|$ is "
                 "Cohen's $d$ of the paired cells. "
                 + ("The test $p_{\\mathrm{task}}$ is the exact two-sided Wilcoxon test on the per-task mean differences, over 11 "
                    "tasks on GuacaMol and 6 on LassoBench, not corrected across the rows. "
                    if tl else "The test $p$ is the two-sided Wilcoxon test over the paired cells. ")
                 + "The mark $^a$ denotes the paired Wilcoxon test over the matched seeds, because NAS-Bench-101 has a single task. "
                 "The full table in the repository (\\texttt{docs/baselines/tab\\_crossdomain\\_full.tex}) adds the "
                 "cell-level $p$ and the task counts.")
    t.record_tables = [full]
    return t


# =========================================================================================
# Appendix F: budget scaling (stored N = 1000 runs, relabelled)
# =========================================================================================
BUDGET_N = (30, 50, 100, 200, 500, 1000)
BUDGET_ROWS = [("b1000_geowalk", "\\geowalk{}", "Geowalk"), ("b1000_hvarfner_scaled_gp", "GP on the shell", "VanillaSphere"),
               ("b1000_cma_es_subspace", "\\geocma{}", "Geocma"), ("b1000_random_search", "Random Search (shell)", "RandomSearch"),
               ("b1000_vanilla_bo_hvarfner", "GP in the cold-start box", "TsGpBox")]
#: authors' decision of 2026-11-02: the row "GP in nested random subspaces" (b1000_baxus, the stored BAxUS runs at
#: N = 1000, which carry a known wiring defect) and its difference row are no longer printed. The arm still defines the
#: matched set (it lacks one cell, so n = 109 of 110), so that every other value of the table stays as printed before.
BUDGET_MATCHED_ONLY = ("b1000_baxus",)


def tab_budget(ctx: Ctx) -> Table | None:
    table = "tab_budget"
    rows = [r for r in BUDGET_ROWS if ctx.has(r[0]) or ctx.skip(table, r[0], "arm absent (row omitted)")]
    if len(rows) < 2:
        return None
    joint = ctx.eng.joint([a for a, _, _ in rows] + [a for a in BUDGET_MATCHED_ONLY if ctx.has(a)])
    hdr = ["Method"] + [f"$N={n}$" for n in BUDGET_N]
    t = Table("budget", "tab:budget", "", WRAP + "r" * len(BUDGET_N), hdr, size="\\small", tabcolsep="4pt")
    lev = {a: {n: ctx.eng.level(a, joint, n) for n in BUDGET_N} for a, _, _ in rows}
    for a, lab, key in rows:
        for n in BUDGET_N:
            ctx.num(f"crBudget{key}N{camel(str(n))}", lev[a][n], N.f3(lev[a][n]), kind="level", table=table,
                    fileset=ctx.fs({a: joint}), n=len(joint), rule=f"{conv_text(ctx)} of best@{n} of the N = 1000 runs")
        t.add([lab] + [mb(N.f3(lev[a][n]), lev[a][n] >= max(lev[x][n] for x, _, _ in rows) - 5e-4) for n in BUDGET_N])
    t.rule()
    # one name per row (RUN_RULES 2026-10-29 (4)): the Delta rows repeat the names of the rows above
    for b, lab, key in [("b1000_hvarfner_scaled_gp", "$\\Delta$ \\geowalk{} $-$ GP on the shell", "VsVanillaSphere"),
                        ("b1000_vanilla_bo_hvarfner", "$\\Delta$ \\geowalk{} $-$ GP in the cold-start box", "VsTsGpBox")]:
        if not (ctx.has(b) and ctx.has("b1000_geowalk")):
            continue
        cells = []
        for n in BUDGET_N:
            pr = ctx.eng.paired("b1000_geowalk", b, joint, n)
            ctx.num(f"crBudget{key}N{camel(str(n))}Delta", pr.delta, N.s3(pr.delta), kind="delta", table=table,
                    fileset=ctx.fs({"b1000_geowalk": joint, b: joint}), n=pr.n, rule=f"paired at N = {n}")
            cells.append(m(N.s3(pr.delta)))
        t.add([lab] + cells)
    ctx.num("crBudgetN", len(joint), str(len(joint)), kind="n", table=table,
            fileset=ctx.fs({a: joint for a in [x for x, _, _ in rows] + [a for a in BUDGET_MATCHED_ONLY if ctx.has(a)]}),
            n=len(joint), rule="joint matched set (of the printed rows and of BUDGET_MATCHED_ONLY)")
    spec = ctx.ds.arm(rows[0][0]).spec
    total = len(spec.tasks) * len(spec.seeds)
    seeds = sorted(spec.seeds)
    t.caption = ("\\textbf{Best-so-far of the stored $N=1000$ runs at intermediate budgets (seeds "
                 f"{seeds[0]}--{seeds[-1]}).}} One matched set of $n={len(joint)}$ of the "
                 f"{total} (task, seed) cells. Each column is the best value after $N$ evaluations of the same runs. "
                 "The $N=30$ column is therefore a prefix of these runs, not the $N=30$ results of "
                 "Table~\\ref{tab:main}. GP on the shell = " + LB.label("b1000_hvarfner_scaled_gp")
                 + ". GP in the cold-start box = a Mat\\'ern-5/2 GP (one lengthscale per dimension) with Thompson "
                 "sampling inside the bounding box of the five cold-start latents. It is not the box version of "
                 "Table~\\ref{tab:boxsphere}. Bold: the highest mean in the column. "
                 "The $\\Delta$ rows are paired differences on the matched cells.")
    return t


# =========================================================================================
# Appendix E: cross-codec scoping
# =========================================================================================
def tab_crosscodec(ctx: Ctx) -> Table | None:
    table = "tab_crosscodec"
    codecs = [("SELFIES VAE", "Selfies", "no, deterministic", {"gw": "geowalk", "cma": "cma_es_subspace",
                                                               "sph": "hvarfner_scaled_gp", "box": ctx.box("hvarfner")}),
              ("Chembed (Transformer VAE)", "Chembed", "no, deterministic",
               {"gw": "chembed_geowalk", "cma": "chembed_cma_es_subspace", "sph": "chembed_hvarfner_scaled_gp",
                "box": "chembed_hvarfner_scaled_gp_box_faithful"}),
              ("Power-Spherical VAE", "Powersph", "yes, stochastic",
               {"gw": "power_spherical_geowalk", "cma": "power_spherical_cma_es_subspace",
                "sph": "power_spherical_hvarfner_scaled_gp", "box": "power_spherical_hvarfner_scaled_gp_box_faithful"})]
    hdr = ["Codec", "$z$ normalized in decoder; decoding", "$n$", "\\geowalk{}", "\\geocma{}", stack("GP", "(shell)"),
           stack("GP", "(box)"), "$\\Delta_{\\mathrm{shell-box}}$", "$d$", "$p$"]
    t = Table("crosscodec", "tab:crosscodec_means", "", WRAP + wrapp("1.5cm") + "r" * 8, hdr, size="\\footnotesize",
              wide=True, tabcolsep="3pt")
    any_row = False
    for lab, key, normz, arms in codecs:
        if not all(ctx.has(a) for a in arms.values()):
            for a in arms.values():
                if not ctx.has(a):
                    ctx.skip(table, a, "arm absent (codec row omitted)")
            continue
        any_row = True
        joint = ctx.eng.joint(list(arms.values()))
        lv = {k: ctx.reg_level(f"crCodec{key}{camel(k)}Mean", a, joint, table)[0] for k, a in arms.items()}
        pr = ctx.eng.paired(arms["sph"], arms["box"], joint)
        ctx.reg_paired(f"crCodec{key}SphereBox", pr, arms["sph"], arms["box"], joint, table)
        pg = ctx.eng.paired(arms["gw"], arms["sph"], joint)
        ctx.reg_paired(f"crCodec{key}GwVsSphere", pg, arms["gw"], arms["sph"], joint, table)
        t.add([lab, normz, N.intfmt(len(joint)), m(N.f4(lv["gw"])), m(N.f4(lv["cma"])), m(N.f4(lv["sph"])),
               m(N.f4(lv["box"])), m(N.s4(pr.delta)), m(N.f2(pr.d)), m(N.pfmt(pr.p))])
    if not any_row:
        return None
    t.caption = ("\\textbf{Scope of the codec comparison (Appendix~\\ref{app:crosscodec}).} GP = "
                 + LB.label("hvarfner_scaled_gp").replace(", qLogEI, candidate-set argmax", "")
                 + " with qLogEI. It uses candidate-set argmax on the shell and L-BFGS-B in the box, so shell versus "
                 "box also changes the maximizer. Chembed and Power-Spherical: stored runs. Each "
                 "codec has its own box $[p_{01},p_{99}]$, computed for its own latent space. Power-Spherical "
                 "normalizes $z$ inside its decoder and decodes stochastically, so it cannot test the radial support. "
                 "Chembed is a second VAE with an empirical shell and gives a null result. Each codec row uses the "
                 f"matched cells of its four methods ({filt_inline(ctx)}). Means are {conv_plural(ctx)}. "
                 "$\\Delta_{\\mathrm{shell-box}}$ = GP (shell) minus GP (box). $d$ = Cohen's $d$ of the cell "
                 "differences. $p$ = two-sided Wilcoxon signed-rank test over the cells, not corrected. The task-level "
                 "test is given in the text.")
    return t
