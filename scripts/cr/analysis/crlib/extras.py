"""Additions of the framing synthesis (cr/framing/SYNTHESIS_cz.md section 9.1) that rest on the camera-ready arms.

* tab_decomposition_robustness  for every pre-registered decomposition contrast: the seed-block permutation p
                                (one sign per seed, jointly across tasks) and the shared-cold-start correlation
                                r-bar next to the exact task-level p (G2)
* tab_anytime                   anytime metrics at the task level (G3): normalised cumulative simple regret over
                                evaluations 6-30 (nCSR; the area between f* and the best-so-far curve), the mean
                                best-so-far over the same evaluations, and the mean score per query, for the six
                                decomposition cells, GeoWalk, Linear BO shell/box and TuRBO-1 (fixed box)
* tab_linear_bo                 Linear BO disclosure rows (G13): GeoWalk against Linear BO in the box and on the
                                shell, Linear BO in the box against uniform shell sampling, and the stored
                                uncalibrated shell port next to the calibrated one
* anytime_legacy_numbers        the July rebuttal's anytime numbers under their own conventions (legacy checks)

Definitions of the anytime metrics (those of the July rebuttal, kept; its script is not part of this release):
  nCSR(cell)      = mean over evaluations i = 6..30 of (f*_t - b_i) / f*_t, b_i = best score after i evaluations
                    (best_trace), f*_t = the largest best@30 of task t; lower is better. f*_t is taken over the
                    rows and the matched cells of THIS table (the rebuttal took it over its own seven methods), so
                    nCSR values are comparable within the table only.
  mean best-so-far = mean over i = 6..30 of b_i (no reference value needed); higher is better.
  per-query mean  = mean of the 25 proposal scores (all_scores[5:30]); higher is better. A proposal that decoded
                    to nothing is stored as -1.0; it counts as 0 here (the score of an invalid molecule) and the
                    raw mean with -1 is registered beside it (the rebuttal's "strict" proxy used the raw values).
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import numbers as N
from . import stats
from .context import Ctx
from .decomp import DEC_CONTRASTS, DEC_ARMS, ci_text, dec_compute, decomposition_arms, _legacy_note, stack
from .latex import Table, m, mb
from .tables import conv_text, filt_text

N_COLD = 5


# =========================================================================================
# seed-block permutation and shared-cold-start correlation of the decomposition contrasts
# =========================================================================================
def tab_decomposition_robustness(ctx: Ctx) -> Table | None:
    table = "tab_decomposition_robustness"
    arms, legacy_b0, missing = decomposition_arms(ctx)
    if arms is None:
        ctx.skip(table, ",".join(missing), "decomposition arm absent (table omitted)")
        return None
    R = dec_compute(ctx, arms)
    cells = R["cells"]
    if not cells:
        return None
    fs = ctx.fs({a: cells for a in arms.values()})
    hdr = ["Contrast", "$\\Delta$", "Tasks", stack("$p_{\\mathrm{task}}$", "(exact)"), stack("$p$, seed-block", "permutation"),
           "$\\bar r$", stack("Task", "pairs"), stack("$r$", "min / max")]
    t = Table("decomposition_robustness", "tab:decomposition_robustness", "", "p{4.6cm}rrrrrrr", hdr,
              size="\\footnotesize", wide=True)
    complete = True
    for k, lab, _, _ in DEC_CONTRASTS:
        pr = R["pr"][k]
        sb = stats.seed_block_permutation(pr.cell_diffs)
        rb = stats.shared_coldstart_rbar(pr.cell_diffs)
        complete = complete and bool(sb.get("complete"))
        ctx.num(f"crDec{k}PSeedBlock", sb["p"], N.psig(sb["p"]), kind="p_perm", table=table, fileset=fs, n=pr.n_tasks,
                rule=f"two-sided seed-block permutation p of the task-level signed-rank statistic ({sb['n_perm']:,} "
                     f"permutations, default_rng({sb['rng']}); one sign per seed applied to the seed's cell contrasts "
                     "in every task)")
        ctx.num(f"crDec{k}Rbar", rb["rbar"], N.s3(rb["rbar"]), kind="correlation", table=table, fileset=fs,
                n=rb["n_pairs"], rule="shared-cold-start correlation: mean over task pairs of the across-seed "
                                      "correlation of the two tasks' cell contrasts")
        t.add([lab, m(N.s4(pr.delta)), f"{pr.tasks_won}/{pr.n_tasks}", m(N.psig(pr.p_task)), m(N.psig(sb["p"])),
               m(N.s3(rb["rbar"])), str(rb["n_pairs"]), m(f"{N.s3(rb['r_min'])}\\,/\\,{N.s3(rb['r_max'])}")])
    rbs = [ctx.reg.value(f"crDec{k}Rbar") for k, _, _, _ in DEC_CONTRASTS]
    rbs = [v for v in rbs if v is not None]
    if rbs:
        ctx.num("crDecRbarMin", min(rbs), N.s3(min(rbs)), kind="correlation", table=table, fileset=fs,
                rule="smallest r-bar over the nine contrasts")
        ctx.num("crDecRbarMax", max(rbs), N.s3(max(rbs)), kind="correlation", table=table, fileset=fs,
                rule="largest r-bar over the nine contrasts")
    t.caption = ((_legacy_note(arms) if legacy_b0 else "")
                 + "\\textbf{Dependence between tasks: seed-block permutation test and shared-cold-start correlation "
                 "for the decomposition contrasts.} A seed fixes the five cold-start molecules of every task, so the "
                 "cells of one seed are not independent across tasks. Seed-block permutation: one random sign per "
                 "seed is applied to that seed's cell contrasts in all tasks (equivalent to permuting the arm labels "
                 "within the seed, jointly across tasks); statistic $|W^+-n(n+1)/4|$ of the signed-rank test on the "
                 f"per-task means; $p=(1+\\#\\{{\\mathrm{{perm}}\\ge\\mathrm{{obs}}\\}})/({N.intfmt(stats.PERM_N)}+1)$, "
                 f"{N.intfmt(stats.PERM_N)} permutations, fixed seed. $\\bar r$: mean over task pairs of the correlation, "
                 "across the seeds both tasks have, of the two tasks' cell contrasts. Both follow the September "
                 "analysis plan (section 2.5), on the native score scale"
                 + ("" if complete else " and, because the matched set lacks some cells, with per-task means over the "
                                        "cells present")
                 + f". Same matched set as Table~\\ref{{tab:decomposition_contrasts}} ($n={len(cells)}$, "
                 f"{filt_text(ctx)}); $\\Delta$ = {conv_text(ctx)}. Neither column is part of the pre-registered "
                 "family; they are reported for robustness.")
    return t


# =========================================================================================
# anytime metrics
# =========================================================================================
ANY_ROWS = [("S0", "Shell, no surrogate", "ShellNone"), ("SA", "Shell, GP + argmax", "ShellArgmax"),
            ("SC", "Shell, GP + projected Adam", "ShellCont"), ("B0", "Box, no surrogate", "BoxNone"),
            ("BA", "Box, GP + argmax", "BoxArgmax"), ("BC", "Box, GP + L-BFGS-B", "BoxCont"),
            ("GW", "\\geowalk{}", "Geowalk"), ("LS", "Linear BO (shell, calibrated)", "LinearSphere"),
            ("LB", "Linear BO (box)", "LinearBox"), ("TB", "TuRBO-1 (fixed box)", "TurboBox")]
ANY_EXTRA = {"GW": "geowalk", "LS": "linear_bo_sphere_calib", "LB": "linear_bo", "TB": "turbo_box_faithful"}


def cell_anytime(cell, fstar: float | None = None) -> dict:
    """Anytime metrics of one 'ours' cell (30 evaluations: 5 cold start + 25 proposals)."""
    b = np.asarray(cell.trace[:30], dtype=float)
    sc = np.asarray(cell.scores[:30], dtype=float)
    q = sc[N_COLD:30]
    out = {"auc": float(b[N_COLD:30].mean()),
           "perquery_raw": float(np.nansum(q) / (30 - N_COLD)),
           "perquery": float(np.nansum(np.where(q == -1.0, 0.0, q)) / (30 - N_COLD))}
    if fstar:
        out["ncsr"] = float(np.mean(fstar - b[N_COLD:30]) / fstar)
    return out


def tab_anytime(ctx: Ctx) -> Table | None:
    table = "tab_anytime"
    dec, legacy_b0, _ = decomposition_arms(ctx)
    roles = {}
    for k, lab, _ in ANY_ROWS:
        a = (dec or DEC_ARMS).get(k) if k in DEC_ARMS else ANY_EXTRA[k]
        if a and ctx.has(a) and ctx.ds.arm(a).spec.fmt == "ours":
            roles[k] = a
        else:
            ctx.skip(table, a or k, "arm absent (row omitted)")
    if "GW" not in roles or len(roles) < 2:
        ctx.skip(table, "geowalk", "reference arm absent (table omitted)")
        return None
    cells = ctx.eng.joint(list(roles.values()))
    cells = {c for c in cells if all(ctx.ds.arm(a).cells[c].scores is not None
                                     and len(ctx.ds.arm(a).cells[c].scores) >= 30 for a in roles.values())}
    if not cells:
        return None
    tasks = sorted({c[0] for c in cells})
    fstar = {t: max(ctx.ds.arm(a).cells[c].trace[29] for a in roles.values() for c in cells if c[0] == t)
             for t in tasks}
    met = {k: {c: cell_anytime(ctx.ds.arm(a).cells[c], fstar[c[0]]) for c in cells} for k, a in roles.items()}

    def col(k, key):
        return {c: met[k][c][key] for c in cells}

    hdr = ["Arm", "nCSR $\\downarrow$", stack("Tasks", "(row better)"), "$p_{\\mathrm{task}}$",
           stack("Mean best-", "so-far $\\uparrow$"), stack("Per-query", "mean $\\uparrow$"), stack("Tasks", "(row better)"),
           "$p_{\\mathrm{task}}$"]
    t = Table("anytime", "tab:anytime", "", "lrrrrrrr", hdr, size="\\footnotesize", wide=True)
    ref = "GW"
    best = {key: (min if key == "ncsr" else max)(stats.level(col(k, key), cells, ctx.var.conv) for k in roles)
            for key in ("ncsr", "auc", "perquery")}
    fs_all = ctx.fs({a: cells for a in roles.values()})
    for k, lab, role in ANY_ROWS:
        if k not in roles:
            continue
        a = roles[k]
        fs = ctx.fs({a: cells})
        lv = {key: stats.level(col(k, key), cells, ctx.var.conv) for key in ("ncsr", "auc", "perquery", "perquery_raw")}
        ctx.num(f"crAny{role}Ncsr", lv["ncsr"], N.f4(lv["ncsr"]), kind="level", table=table, fileset=fs_all, n=len(cells),
                rule=f"{conv_text(ctx)} of the normalised cumulative simple regret over evaluations 6-30 "
                     "(f* = largest best@30 of the task over the rows and matched cells of the table)")
        ctx.num(f"crAny{role}Auc", lv["auc"], N.f4(lv["auc"]), kind="level", table=table, fileset=fs, n=len(cells),
                rule=f"{conv_text(ctx)} of the mean best-so-far over evaluations 6-30")
        ctx.num(f"crAny{role}PerQuery", lv["perquery"], N.f4(lv["perquery"]), kind="level", table=table, fileset=fs,
                n=len(cells), rule=f"{conv_text(ctx)} of the mean score of the 25 proposals (empty decode = 0)")
        ctx.num(f"crAny{role}PerQueryRaw", lv["perquery_raw"], N.f4(lv["perquery_raw"]), kind="level", table=table,
                fileset=fs, n=len(cells), rule=f"{conv_text(ctx)} of the mean score of the 25 proposals as stored "
                                               "(empty decode = -1)")
        row = [ctx.label(a, lab), mb(N.f4(lv["ncsr"]), abs(lv["ncsr"] - best["ncsr"]) < 5e-5)]
        tests = {}
        for key, better in (("ncsr", -1), ("perquery", +1)):
            if k == ref:
                tests[key] = None
                continue
            # contrast oriented so that positive = the row is better than GeoWalk
            x = {c: better * (met[k][c][key] - met[ref][c][key]) for c in cells}
            pr = stats.paired(x, {c: 0.0 for c in cells}, cells, ctx.var.conv)
            tests[key] = pr
            suf = "Ncsr" if key == "ncsr" else "PerQuery"
            fs2 = fs_all if key == "ncsr" else ctx.fs({a: cells, roles[ref]: cells})
            ctx.num(f"crAny{role}{suf}VsGwDelta", better * pr.delta, N.s4(better * pr.delta), kind="delta", table=table,
                    fileset=fs2, n=pr.n, rule=f"row minus GeoWalk, {conv_text(ctx)}")
            ctx.num(f"crAny{role}{suf}VsGwTasks", pr.tasks_won, f"{pr.tasks_won}/{pr.n_tasks}", kind="tasks_won",
                    table=table, fileset=fs2, n=pr.n_tasks, rule="tasks on which the row is better than GeoWalk")
            ctx.num(f"crAny{role}{suf}VsGwPTask", pr.p_task, N.psig(pr.p_task), kind="p_task", table=table, fileset=fs2,
                    n=pr.n_tasks, rule="exact two-sided Wilcoxon on the per-task mean differences from GeoWalk")
        tn, tq = tests["ncsr"], tests["perquery"]
        row += ([f"{tn.tasks_won}/{tn.n_tasks}", m(N.psig(tn.p_task))] if tn else ["ref.", "--"])
        row += [mb(N.f4(lv["auc"]), abs(lv["auc"] - best["auc"]) < 5e-5),
                mb(N.f4(lv["perquery"]), abs(lv["perquery"] - best["perquery"]) < 5e-5)]
        row += ([f"{tq.tasks_won}/{tq.n_tasks}", m(N.psig(tq.p_task))] if tq else ["ref.", "--"])
        t.add(row)
    ctx.num("crAnyN", len(cells), N.intfmt(len(cells)), kind="n", table=table, fileset=fs_all, n=len(cells),
            rule="joint matched set of the anytime table")
    t.caption = ((_legacy_note(dec) if legacy_b0 and "B0" in roles else "")
                 + "\\textbf{Anytime performance over evaluations 6--30, with the task as the unit.} nCSR = normalised "
                 "cumulative simple regret, the mean over evaluations $i=6,\\dots,30$ of $(f^\\ast_t-b_i)/f^\\ast_t$ with "
                 "$b_i$ the best score after $i$ evaluations and $f^\\ast_t$ the largest best@30 of task $t$ among the rows "
                 "and matched cells of this table (lower is better; values are comparable within the table only); mean "
                 "best-so-far = mean of $b_i$ over the same evaluations; per-query mean = mean score of the 25 proposals "
                 "(a proposal that decodes to nothing counts as 0). All columns are " + conv_text(ctx) + " on one "
                 f"matched set of all rows, $n={len(cells)}$ cells ({filt_text(ctx)}). Tasks = tasks on which the row is "
                 "better than \\geowalk{} on the metric to its left; $p_{\\mathrm{task}}$ = exact two-sided Wilcoxon "
                 "signed-rank test on the per-task mean differences from \\geowalk{} (uncorrected). The five cold-start "
                 "molecules are shared by every arm and are excluded from all three metrics.")
    return t


def anytime_legacy_numbers(ctx: Ctx) -> None:
    """The July rebuttal's anytime numbers under their own conventions (each arm's own clean cells; f* per task
    read from the stored rebuttal_anytime.json; per-query mean with empty decodes as stored): GeoWalk and Vanilla BO
    on the shell. Registered in legacy mode only (legacy checks A3-any-*)."""
    T = "numbers_anytime_legacy"
    if ctx.mode != "legacy":
        return
    p = os.path.join(ctx.extra.get("inputs_root", ""), "precomputed", "rebuttal_anytime.json")
    if not os.path.exists(p):
        ctx.skip(T, "rebuttal_anytime.json", "input absent")
        return
    from . import TASKS_T11
    fstar = json.load(open(p))["fstar_per_task"]
    fs_in = ctx.fs_paths([p])
    for arm, role in (("geowalk", "Geowalk"), ("hvarfner_scaled_gp", "VanillaSphere")):
        if not ctx.has(arm):
            ctx.skip(T, arm, "arm absent")
            continue
        cells = ctx.eng.elig(arm, "clean")
        A = ctx.ds.arm(arm)
        cells = {c for c in cells if TASKS_T11[c[0]] in fstar and A.cells[c].scores is not None
                 and len(A.cells[c].scores) >= 30}
        if not cells:
            continue
        mm = {c: cell_anytime(A.cells[c], fstar[TASKS_T11[c[0]]]) for c in cells}
        fs = ctx.fs_union(ctx.fs({arm: cells}), fs_in)
        v = stats.level({c: mm[c]["ncsr"] for c in cells}, cells, "taskmean")
        ctx.num(f"crAnyLegacy{role}Ncsr", v, N.f4(v), kind="level", table=T, fileset=fs, n=len(cells),
                rule="July convention: mean of per-task means of nCSR on the arm's own clean cells, f* per task from "
                     "rebuttal_anytime.json")
        v = stats.level({c: mm[c]["perquery_raw"] for c in cells}, cells, "taskmean")
        ctx.num(f"crAnyLegacy{role}PerQuery", v, N.f4(v), kind="level", table=T, fileset=ctx.fs({arm: cells}),
                n=len(cells), rule="July convention: mean of per-task means of the per-query mean score (empty decode "
                                   "= -1 as stored) on the arm's own clean cells")


# =========================================================================================
# Linear BO disclosure rows
# =========================================================================================
LIN_ROWS = [("GwVsLinearBox", "geowalk", "linear_bo", "\\geowalk{} $-$ Linear BO (box)"),
            ("GwVsLinearSphere", "geowalk", "linear_bo_sphere_calib", "\\geowalk{} $-$ Linear BO (shell, calibrated)"),
            ("LinearBoxVsRs", "linear_bo", "random_search", "Linear BO (box) $-$ uniform shell sampling"),
            ("LinearSphereVsRs", "linear_bo_sphere_calib", "random_search",
             "Linear BO (shell, calibrated) $-$ uniform shell sampling"),
            ("LinearSphereVsBox", "linear_bo_sphere_calib", "linear_bo", "Linear BO: shell (calibrated) $-$ box")]
LIN_STORED = ("LinearCalibVsUncalib", "linear_bo_sphere_calib", "linear_bo_sphere",
              "Linear BO shell: calibrated $-$ uncalibrated$^\\dagger$")


def tab_linear_bo(ctx: Ctx) -> Table | None:
    table = "tab_linear_bo"
    rows = [r for r in LIN_ROWS if (ctx.has(r[1]) and ctx.has(r[2])) or ctx.skip(table, r[2], "arm absent (row omitted)")]
    if not rows:
        return None
    arms = sorted({a for r in rows for a in r[1:3]})
    cells = ctx.eng.joint(arms)
    if not cells:
        return None
    hdr = ["Comparison", "$n$", "First", "Second", "$\\Delta$", "95\\,\\% CI", "Tasks", "$p_{\\mathrm{task}}$", "$d$"]
    t = Table("linear_bo", "tab:linear_bo", "", "p{5.2cm}rrrrrrrr", hdr, size="\\footnotesize", wide=True)

    def add(key, a, b, lab, cs):
        pr = ctx.eng.paired(a, b, cs)
        fs = ctx.fs({a: cs, b: cs})
        lo, hi = pr.ci()
        p = f"crLin{key}"
        ctx.num(p + "First", pr.level_a, N.f4(pr.level_a), kind="level", table=table, fileset=ctx.fs({a: cs}), n=pr.n,
                rule=f"{conv_text(ctx)} of best@30 of {a}")
        ctx.num(p + "Second", pr.level_b, N.f4(pr.level_b), kind="level", table=table, fileset=ctx.fs({b: cs}), n=pr.n,
                rule=f"{conv_text(ctx)} of best@30 of {b}")
        ctx.num(p + "Delta", pr.delta, N.s4(pr.delta), kind="delta", table=table, fileset=fs, n=pr.n,
                rule=f"{conv_text(ctx)} of the paired differences {a} - {b}")
        ctx.num(p + "CiLo", lo, N.s4(lo), kind="ci", table=table, fileset=fs, n=pr.n_tasks, rule="95 % task bootstrap, lower end")
        ctx.num(p + "CiHi", hi, N.s4(hi), kind="ci", table=table, fileset=fs, n=pr.n_tasks, rule="95 % task bootstrap, upper end")
        ctx.num(p + "Tasks", pr.tasks_won, f"{pr.tasks_won}/{pr.n_tasks}", kind="tasks_won", table=table, fileset=fs,
                n=pr.n_tasks, rule="tasks on which the first arm has the higher mean")
        ctx.num(p + "PTask", pr.p_task, N.psig(pr.p_task), kind="p_task", table=table, fileset=fs, n=pr.n_tasks,
                rule="exact two-sided Wilcoxon on the per-task mean differences")
        ctx.num(p + "D", pr.d, N.d2(pr.d), kind="cohens_d", table=table, fileset=fs, n=pr.n, rule="cells (secondary)")
        ctx.num(p + "N", pr.n, N.intfmt(pr.n), kind="n", table=table, fileset=fs, n=pr.n, rule="matched cells of the row")
        t.add([lab, N.intfmt(pr.n), m(N.f4(pr.level_a)), m(N.f4(pr.level_b)), m(N.s4(pr.delta)), m(ci_text(lo, hi)),
               f"{pr.tasks_won}/{pr.n_tasks}", m(N.psig(pr.p_task)), m(N.d2(pr.d))])

    for key, a, b, lab in rows:
        add(key, a, b, lab, cells)
    key, a, b, lab = LIN_STORED
    stored_txt = ""
    if ctx.has(a) and ctx.has(b):
        cs = ctx.eng.joint([a, b])
        if cs:
            t.rule()
            add(key, a, b, lab, cs)
            gen = ("stored July files of both arms" if ctx.ds.arm(a).spec.kind != "cr"
                   else "camera-ready calibrated arm against the stored July files of the uncalibrated arm, which was not "
                        "rerun: the row crosses code generations and is descriptive")
            stored_txt = (" $^\\dagger$ The first shell port bounded the kernel's projection by "
                          "$\\pm1.1\\max(\\max|x|, r)$; after its first run the bound was changed to "
                          "$\\pm1.05\\max(5r/\\sqrt{D}, \\max|x|)$ (the calibrated arm, the one printed elsewhere). Last "
                          f"row: {gen}.")
    else:
        ctx.skip(table, b, "stored uncalibrated arm absent (disclosure row omitted)")
    t.caption = ("\\textbf{Linear BO: comparisons with the task as the unit, and what was changed relative to the "
                 "published method.} Linear BO (box) uses the published spherically projected linear kernel with "
                 "analytic LogEI, but in our per-coordinate $[p_{01},p_{99}]$ box rescaled to $[0,1]^D$ (the released code "
                 "searches $[-8,8]^{256}$ in raw latent units) and from the 5 shared cold-start encodings (the "
                 "published initial design has 30 Sobol points). Linear BO (shell) is our adaptation: the same kernel, "
                 "qLogEI maximised over 2,048 uniform shell points. "
                 f"Rows above the rule share one matched set, $n={len(cells)}$ cells ({filt_text(ctx)}); $\\Delta$ = first "
                 f"minus second arm, {conv_text(ctx)}; 95\\,\\% CI from 10,000 task-bootstrap resamples; Tasks = tasks with "
                 "the first arm ahead; $p_{\\mathrm{task}}$ = exact two-sided Wilcoxon over tasks (uncorrected); $d$ over "
                 "cells." + stored_txt)
    return t
