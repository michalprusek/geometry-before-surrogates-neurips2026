"""Retrieval floor: E[max of 30 distinct ZINC-250k molecules] per task and per-arm verdicts.

Uses the authors' order-statistics and null-test functions from the vendored
``_vendor/zinc_retrieval_floor.py`` (sha256 d6cc4b3c..., identical to the September
worktrees and to origin/e0-instrumentation) on the cached library scores
(``scores_<task>.npy``, 249,455 molecules, commit c33f51d clean). Conventions follow
cr/G_floor_lasso_geometry.md section 1.4: rule P = exact-null Monte Carlo test (null seed 0,
both one-sided p Holm-adjusted over the tasks, each side at alpha/2); rule T = one-sample
t-test of the mean against the floor, Holm over the tasks; task level = exact Wilcoxon on the
per-task differences.

Rule P uses REPS = 1,000,000 Monte Carlo replicates of the exact null (GEN v3; GEN_verify_2
found three cells near the cut at 100,000). The vendored sampler draws its replicates in
row-major blocks from one seeded stream, so the first 100,000 of the 1,000,000 replicates ARE
the earlier 100,000-replicate computation: both verdicts are kept (``verdict_P`` and
``verdict_P_old``), the number of (arm, task) cells where they differ is registered, and so is
the number of cells whose Holm-adjusted p lies within 3 Monte Carlo standard errors of the cut
at 1,000,000 replicates. The null draws depend only on (task, number of seeds), so they are
computed once per run and shared by every arm and every variant.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os

import numpy as np
from scipy import stats as sps

from . import numbers as N
from .context import Ctx
from .engine import fileset_from_paths
from .latex import Table, m
from .numbers import camel
from .tables import filt_text

FLOOR_MODULE_SHA = "d6cc4b3cd6489af5c580cd62e7bf0a62fec0c208713278aed7a6222a3f47edd3"
B = 30
REPS = 1_000_000          # Monte Carlo replicates of the exact null (rule P)
REPS_OLD = 100_000        # the earlier computation = the first 100,000 replicates of the same stream
NULL_SEED = 0
ALPHA = 0.05


def load_floor_module():
    path = os.path.join(os.path.dirname(__file__), "_vendor", "zinc_retrieval_floor.py")
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if got != FLOOR_MODULE_SHA:
        raise RuntimeError(f"vendored floor module sha256 {got} != {FLOOR_MODULE_SHA}")
    keep = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    spec = importlib.util.spec_from_file_location("zinc_retrieval_floor_vendored", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k, v in keep.items():           # the module sets thread caps at import; restore the caller's
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return mod


# (arm or BOX:<recipe> or DEC:B0, label, group); macro keys name roles, so they do not change with the box variant.
# group: 'shell' = every proposal on the empirical shell; 'box'; 'other'. The six cells of the pre-registered
# decomposition are random_search, DEC:B0, hvarfner_scaled_gp, BOX:factorial_argmax,
# hvarfner_scaled_gp_sphere_continuous and BOX:hvarfner.
FLOOR_ARMS = [("geowalk", "\\geowalk{}", "shell"), ("cma_es_subspace", "\\geocma{}", "shell"),
              ("random_search", "RS (shell)", "shell"),
              ("hvarfner_scaled_gp", "Vanilla BO (shell, argmax)", "shell"),
              ("hvarfner_scaled_gp_sphere_continuous", "Vanilla BO (shell, Adam)", "shell"),
              ("papenmeier_msr", "Papenmeier MLE (shell)", "shell"),
              ("xu_standard_gp", "Xu (shell)", "shell"), ("linear_bo_sphere_calib", "Linear BO (shell)", "shell"),
              ("convex_hull_k1", "random ZINC latent ($k=1$)", "other"),
              ("DEC:B0", "RS (box)", "box"), ("BOX:factorial_argmax", "Vanilla BO (box, argmax)", "box"),
              ("BOX:hvarfner", "Vanilla BO (box, L-BFGS-B)", "box"),
              ("BOX:papenmeier_msr", "Papenmeier MSR (box)", "box"), ("BOX:xu", "Xu (box)", "box"),
              ("linear_bo", "Linear BO (box)", "box"),
              ("turbo_box_faithful", "TuRBO-1 (fixed box)", "box"), ("turbo", "TuRBO$^\\dagger$", "other"),
              ("baxus", "BAxUS$^\\dagger$", "other"), ("lolbo", "LOL-BO$^\\dagger$", "other"),
              ("invbo", "InvBO$^\\dagger$", "other"), ("cmaes_full", "CMA-ES$^\\dagger$", "other")]
FLOOR_KEY = {"BOX:hvarfner": "VanillaBox", "BOX:papenmeier_msr": "PapenmeierMsrBox", "BOX:xu": "XuBox",
             "BOX:factorial_argmax": "VanillaBoxArgmax", "DEC:B0": "RandomBox"}
LIMIT_TASKS = ("med1", "med2", "siga")      # the tasks of the one-sentence limitation


def _floor_tables(ctx: Ctx, zrf, fdir: str, tasks: list) -> tuple[dict, list]:
    """Per-task order statistics of the library scores (run-wide cache)."""
    key = ("floor_T", fdir, tuple(tasks))
    if key in ctx.cache:
        return ctx.cache[key]
    T, files = {}, []
    for t in tasks:
        p = os.path.join(fdir, f"scores_{t}.npy")
        s = np.load(p)
        if s.shape != (249_455,) or not np.isfinite(s).all():
            raise RuntimeError(f"{t}: unexpected floor score vector {s.shape}")
        files.append(p)
        values, cum, n = zrf._unique_cdf(s)
        Gn = zrf.max_cdf(values, cum, n, B, "norepl")
        T[t] = {"values": values, "cum": cum, "n": n, "Gn": Gn,
                "floor": zrf.exact_emax_no_replacement(values, cum, n, B),
                "floor_replace": zrf.exact_emax(values, cum, n, B)}
    summ_path = os.path.join(fdir, "zinc_floor_summary.json")
    if os.path.exists(summ_path):
        summ = json.load(open(summ_path))
        for t in tasks:
            ref = summ["tasks"][t]["emax_norepl"]["30"]
            if abs(ref - T[t]["floor"]) > 1e-12:
                raise RuntimeError(f"floor {t}: {T[t]['floor']} != September summary {ref}")
        files.append(summ_path)
    ctx.cache[key] = (T, files)
    return T, files


def floor_analysis(ctx: Ctx) -> dict | None:
    if "floor" in ctx.extra:
        return ctx.extra["floor"]
    fdir = ctx.extra.get("floor_dir")
    if not fdir or not os.path.isdir(fdir):
        return None
    zrf = load_floor_module()
    tasks = ctx.ds.config.tasks
    T, files = _floor_tables(ctx, zrf, fdir, tasks)
    floor_fs = fileset_from_paths(files)
    null_cache = ctx.cache.setdefault("floor_nulls", {})       # shared by every variant of the run
    reps = int(ctx.extra.get("floor_reps") or REPS)
    reps_old = min(REPS_OLD, reps)

    def nulls(t, nseed, seed=NULL_SEED):
        key = (fdir, t, nseed, seed, reps)
        if key not in null_cache:
            null_cache[key] = zrf.null_mean_draws(T[t]["values"], T[t]["Gn"], nseed, reps=reps, seed=seed)
        return null_cache[key]

    from .decomp import decomposition_arms
    dec_arms, _legacy_b0, _ = decomposition_arms(ctx)
    arms_out = {}
    for spec, lab, group in FLOOR_ARMS:
        if spec.startswith("BOX:"):
            arm = ctx.box(spec.split(":")[1])
        elif spec == "DEC:B0":
            arm = (dec_arms or {}).get("B0", "random_box_faithful")
            if arm != "random_box_faithful":
                lab = "RS (box, September p1 blocks)"
        else:
            arm = spec
        if not ctx.has(arm):
            ctx.skip("tab_floor", arm, "arm absent (row omitted)")
            continue
        cells = ctx.eng.elig(arm)
        vals = ctx.eng.vals(arm, cells)
        per = {}
        for t in tasks:
            x = np.asarray([vals[c] for c in sorted(cells) if c[0] == t], dtype=float)
            if x.size == 0:
                continue
            fl = T[t]["floor"]
            draws = nulls(t, x.size)
            nt = zrf.null_mean_test(draws, float(x.mean()), x.size, NULL_SEED)
            nt_old = zrf.null_mean_test(draws[:reps_old], float(x.mean()), x.size, NULL_SEED)
            tt = sps.ttest_1samp(x, fl) if x.size > 1 and x.std(ddof=1) > 0 else None
            per[t] = {"n": int(x.size), "mean": float(x.mean()), "diff": float(x.mean() - fl),
                      "p_above": nt["p_above"], "p_below": nt["p_below"],
                      "p_above_old": nt_old["p_above"], "p_below_old": nt_old["p_below"],
                      "t_p": float(tt.pvalue) if tt is not None else float("nan")}
        ts = list(per)
        ha = zrf.holm([per[t]["p_above"] for t in ts])
        hb = zrf.holm([per[t]["p_below"] for t in ts])
        hao = zrf.holm([per[t]["p_above_old"] for t in ts])
        hbo = zrf.holm([per[t]["p_below_old"] for t in ts])
        ht = zrf.holm([per[t]["t_p"] for t in ts])
        for t, a_, b_, ao, bo, c_ in zip(ts, ha, hb, hao, hbo, ht):
            r = per[t]
            r["holm_above"], r["holm_below"] = a_, b_
            r["verdict_P"] = "A" if a_ < ALPHA / 2 else ("B" if b_ < ALPHA / 2 else "i")
            r["verdict_P_old"] = "A" if ao < ALPHA / 2 else ("B" if bo < ALPHA / 2 else "i")
            r["verdict_T"] = ("i" if (c_ is None or math.isnan(c_) or c_ >= ALPHA) else ("A" if r["diff"] > 0 else "B"))
            # Monte Carlo standard error of the Holm-adjusted p of the side that is closer to the cut
            near = False
            for raw, adj in ((r["p_above"], a_), (r["p_below"], b_)):
                if raw <= 0 or adj >= 1.0:
                    continue
                se_adj = (adj / raw) * math.sqrt(raw * (1.0 - raw) / reps)
                near = near or abs(adj - ALPHA / 2) < 3.0 * se_adj
            r["near_cut"] = near
        dts = np.asarray([per[t]["diff"] for t in ts])
        nz = dts[dts != 0]
        p_task = float(sps.wilcoxon(nz, method="exact").pvalue) if nz.size >= 2 else float("nan")
        key = FLOOR_KEY.get(spec, camel(arm))
        arms_out[arm] = {"label": ctx.label(arm, lab), "key": key, "group": group, "per_task": per,
                         "n_cells": len(cells),
                         "cross_mean": float(np.mean([per[t]["mean"] for t in ts])),
                         "mean_floor": float(np.mean([T[t]["floor"] for t in ts])),
                         "counts_P": {k: sum(1 for t in ts if per[t]["verdict_P"] == k) for k in "AiB"},
                         "counts_P_old": {k: sum(1 for t in ts if per[t]["verdict_P_old"] == k) for k in "AiB"},
                         "counts_T": {k: sum(1 for t in ts if per[t]["verdict_T"] == k) for k in "AiB"},
                         "task_signs": (int((dts > 0).sum()), int((dts < 0).sum())), "p_task": p_task,
                         # an arm's floor verdicts depend on its result files AND on the floor score arrays
                         "fileset": ctx.fs_union(ctx.fs({arm: cells}), floor_fs)}
    out = {"tasks": {t: {"floor": T[t]["floor"], "floor_replace": T[t]["floor_replace"]} for t in tasks},
           "mean_floor": float(np.mean([T[t]["floor"] for t in tasks])), "arms": arms_out,
           "files": files, "module_sha256": FLOOR_MODULE_SHA, "reps": reps, "reps_old": reps_old,
           "null_seed": NULL_SEED}
    out["fileset"] = floor_fs
    ctx.extra["floor"] = out
    return out


def tab_floor(ctx: Ctx) -> Table | None:
    table = "tab_floor"
    F = floor_analysis(ctx)
    if F is None:
        ctx.skip(table, "floor cache", "input directory absent")
        return None
    tasks = list(F["tasks"])
    for t in tasks:
        ctx.num(f"crFloor{camel(t)}", F["tasks"][t]["floor"], N.f4(F["tasks"][t]["floor"]), kind="floor", table=table,
                fileset=F["fileset"], rule="E[max of 30 distinct molecules drawn uniformly from 249,455] (exact)")
    ctx.num("crFloorMean", F["mean_floor"], N.f4(F["mean_floor"]), kind="floor", table=table, fileset=F["fileset"],
            rule="mean of the per-task floors")
    hdr = ["Arm"] + tasks + ["A/i/B (P)", "A/i/B (T)", "mean $-$ floor", "$p_{\\mathrm{task}}$"]
    t = Table("floor", "tab:floor", "", "l" + "c" * len(tasks) + "rrrr", hdr, size="\\scriptsize", wide=True)
    t.add(["floor"] + [m(N.f3(F["tasks"][x]["floor"])) for x in tasks] + ["", "", m(N.f4(F["mean_floor"])), ""])
    t.rule()
    for arm, A in F["arms"].items():
        k = f"crFloorArm{A['key']}"
        cp, ct = A["counts_P"], A["counts_T"]
        d = A["cross_mean"] - A["mean_floor"]
        ctx.num(k + "CountsP", f"{cp['A']}/{cp['i']}/{cp['B']}", f"{cp['A']}/{cp['i']}/{cp['B']}", kind="counts",
                table=table, fileset=A["fileset"], n=A["n_cells"],
                rule=f"tasks above / indistinguishable / below, rule P ({F['reps']:,} null replicates, Holm over the "
                     "tasks, alpha/2 per side)")
        co = A["counts_P_old"]
        ctx.num(k + "CountsPOld", f"{co['A']}/{co['i']}/{co['B']}", f"{co['A']}/{co['i']}/{co['B']}", kind="counts",
                table=table, fileset=A["fileset"], n=A["n_cells"],
                rule=f"rule P with the first {F['reps_old']:,} null replicates (the earlier computation)")
        ctx.num(k + "CountsT", f"{ct['A']}/{ct['i']}/{ct['B']}", f"{ct['A']}/{ct['i']}/{ct['B']}", kind="counts",
                table=table, fileset=A["fileset"], n=A["n_cells"], rule="rule T (t-test, Holm)")
        ctx.num(k + "Diff", d, N.s4(d), kind="delta", table=table, fileset=A["fileset"], n=A["n_cells"],
                rule="mean of per-task means minus mean floor")
        ctx.num(k + "PTask", A["p_task"], N.psig(A["p_task"]), kind="p_task", table=table, fileset=A["fileset"],
                n=len(A["per_task"]), rule="exact Wilcoxon on the per-task differences from the floor")
        letters = []
        for x in tasks:
            r = A["per_task"].get(x)
            if r is None:
                letters.append("--")
                continue
            letters.append(r["verdict_P"] if r["verdict_P"] == r["verdict_T"] else f"{r['verdict_P']}$|${r['verdict_T']}")
            ctx.num(f"{k}{camel(x)}Diff", r["diff"], N.s4(r["diff"]), kind="delta", table=table, fileset=A["fileset"],
                    n=r["n"], rule="task mean of best@30 minus the task floor")
        t.add([A["label"]] + letters + [f"{cp['A']}/{cp['i']}/{cp['B']}", f"{ct['A']}/{ct['i']}/{ct['B']}",
                                       m(N.s4(d)), m(N.psig(A["p_task"]))])
    # ---- numbers for the one-sentence limitation ---------------------------------------------------------
    arms = F["arms"]
    fs_all = ctx.fs_union(*[A["fileset"] for A in arms.values()]) if arms else F["fileset"]
    shell = {a: A for a, A in arms.items() if A["group"] == "shell"}
    if shell:
        fs_sh = ctx.fs_union(*[A["fileset"] for A in shell.values()])
        ctx.num("crFloorShellArms", len(shell), str(len(shell)), kind="count", table=table, fileset=fs_sh,
                rule="on-shell arms in the floor table: " + ", ".join(sorted(shell)))
        for rule_key, vk in (("P", "verdict_P"), ("T", "verdict_T")):
            every = 0
            for x in LIMIT_TASKS:
                if x not in tasks:
                    continue
                below = sorted(a for a, A in shell.items() if A["per_task"].get(x, {}).get(vk) == "B")
                ctx.num(f"crFloorShellBelow{camel(x)}{rule_key}", len(below), f"{len(below)}/{len(shell)}", kind="count",
                        table=table, fileset=fs_sh, n=len(shell),
                        rule=f"on-shell arms significantly below the floor on {x}, rule {rule_key}; not below: "
                             + (", ".join(sorted(set(shell) - set(below))) or "none"))
            lim = [x for x in LIMIT_TASKS if x in tasks]
            if lim:
                every = sum(1 for A in shell.values()
                            if all(A["per_task"].get(x, {}).get(vk) == "B" for x in lim))
                ctx.num(f"crFloorShellBelowAllLimit{rule_key}", every, f"{every}/{len(shell)}", kind="count", table=table,
                        fileset=fs_sh, n=len(shell),
                        rule=f"on-shell arms significantly below the floor on every one of {', '.join(lim)}, rule {rule_key}")
    pts = {a: A["p_task"] for a, A in arms.items() if np.isfinite(A["p_task"])}
    if pts:
        sig = sorted(a for a, p in pts.items() if p < 0.05)
        ctx.num("crFloorPTaskMin", min(pts.values()), N.psig(min(pts.values())), kind="p_task", table=table,
                fileset=fs_all, n=len(pts), rule="smallest task-level p over the arms of the floor table ("
                + min(pts, key=pts.get) + ")")
        ctx.num("crFloorPTaskMax", max(pts.values()), N.psig(max(pts.values())), kind="p_task", table=table,
                fileset=fs_all, n=len(pts), rule="largest task-level p over the arms of the floor table")
        ctx.num("crFloorArmsPTaskSig", len(sig), str(len(sig)), kind="count", table=table, fileset=fs_all, n=len(pts),
                rule="arms whose task-level p is below 0.05 (uncorrected): " + (", ".join(sig) or "none"))
        rest = {a: p for a, p in pts.items() if a not in sig}
        if rest:
            ctx.num("crFloorPTaskMinRest", min(rest.values()), N.psig(min(rest.values())), kind="p_task", table=table,
                    fileset=fs_all, n=len(rest), rule="smallest task-level p among the other arms")
    changed = sorted(f"{a}:{x}" for a, A in arms.items() for x, r in A["per_task"].items()
                     if r["verdict_P"] != r["verdict_P_old"])
    near = sorted(f"{a}:{x}" for a, A in arms.items() for x, r in A["per_task"].items() if r["near_cut"])
    ctx.num("crFloorVerdictPChanged", len(changed), str(len(changed)), kind="count", table=table, fileset=fs_all,
            rule=f"(arm, task) cells whose rule-P verdict differs between {F['reps_old']:,} and {F['reps']:,} null "
                 "replicates: " + (", ".join(changed) or "none"))
    ctx.num("crFloorNearCut", len(near), str(len(near)), kind="count", table=table, fileset=fs_all,
            rule=f"(arm, task) cells whose Holm-adjusted p is within 3 Monte Carlo SE of alpha/2 at {F['reps']:,} "
                 "replicates: " + (", ".join(near) or "none"))
    ctx.num("crFloorReps", F["reps"], N.intfmt(F["reps"]), kind="count", table=table, fileset=F["fileset"],
            rule="Monte Carlo replicates of the exact null in rule P")
    t.caption = ("\\textbf{Retrieval floor: the expected best of 30 distinct ZINC-250k molecules drawn uniformly at "
                 "random, and each arm's 40-seed mean against it.} A/i/B = significantly above / indistinguishable / "
                 f"below; rule P: exact-null Monte Carlo test of the seed mean ({N.intfmt(F['reps'])} replicates), both "
                 "one-sided $p$ Holm-adjusted over the tasks at $\\alpha/2$ each; rule T: one-sample $t$-test, Holm; "
                 "where the rules differ the cell shows P$|$T. $p_{\\mathrm{task}}$: exact Wilcoxon test of the per-task "
                 f"differences (task as the unit). Arms on their own {filt_text(ctx)}, where a file without a status "
                 "field counts as completed for the stored external harness and for the March \\geocma{} files "
                 "(intention to treat); $^\\dagger$ stored runs. "
                 f"{len(changed)} (arm, task) verdicts of rule P differ from the earlier {N.intfmt(F['reps_old'])}-replicate "
                 f"computation; {len(near)} lie within 3 Monte Carlo standard errors of the cut.")
    return t
