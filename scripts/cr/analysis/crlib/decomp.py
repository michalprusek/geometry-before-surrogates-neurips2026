"""The pre-registered decomposition (RUN_RULES amendment 2026-10-12) and the tables built on it.

* tab_decomposition            2 x 3 levels {shell, [p01,p99] box} x {no surrogate (one uniform draw),
                               GP + candidate-set argmax, GP + continuous maximisation (projected Adam on
                               the shell, L-BFGS-B in the box)} on ONE joint matched set of the six arms,
                               and the gain of each cell over the best of the 5 shared cold-start molecules
* tab_decomposition_contrasts  exactly the pre-registered 9-contrast family: Delta, task-bootstrap 95 %
                               interval, tasks won, exact two-sided task-level Wilcoxon p (primary), BH
                               within the family, cell-level Wilcoxon p and d (secondary), W/L, n
* tab_decomposition_contrasts_byN  the same family at N = 10 and N = 20 (best_trace)
* tab_decomposition_pertask    per-task sign/Delta map, 11 tasks x 9 contrasts
* tab_factorial_by_box         support x maximiser 2 x 2 with the [p01,p99] and the [min,max] box, the
                               interaction per box, and the posted (flat, 423) and E-1 (task-mean, 426)
                               generations for [p01,p99]
* tab_equivalence              TOST at margin 0.01 for GeoWalk - Vanilla BO (shell) and GeoWalk - Linear BO
                               (shell), 90 % task-bootstrap interval decides; N = 10, 20 for information
* tab_distinct                 distinct proposals of 25 per arm (score-equality proxy) and a sensitivity of
                               the maximiser-on-shell contrast to the shell x Adam cell's repeats
* tab_gp_radius                GP + argmax at 0.5/1/1.5/2 x the cold-start radius and with N(0,I)
                               candidates, next to the GeoWalk radius sweep, task-level tests against 1x

The decomposition is defined on the [p01,p99] box whatever the R2 variant (pre-registered), and keeps its
task-level columns in R7 (the task-level test is its pre-registered primary test).

Legacy mode: the decomposition needs uniform box sampling (random_box_faithful), which has no pre-camera-ready
per-cell generation. In --legacy runs it then reads the September p1 blocks (arm p1_random_box_faithful,
commit 4780475, batch runner) and every caption says LEGACY VARIANT: the six cells come from three code
states, so this variant is descriptive and is never the printed generation.
"""
from __future__ import annotations

import numpy as np

from . import numbers as N
from . import stats
from .context import Ctx
from .fidelity import cell_fidelity
from .latex import Table, m, mb
from .numbers import camel
from .tables import conv_text, filt_text

DEC_ARMS = {"S0": "random_search", "B0": "random_box_faithful", "SA": "hvarfner_scaled_gp",
            "BA": "hvarfner_scaled_gp_box_argmax", "SC": "hvarfner_scaled_gp_sphere_continuous",
            "BC": "hvarfner_scaled_gp_box_faithful"}
LEGACY_B0 = "p1_random_box_faithful"
CELL_ROLE = {"S0": "ShellNone", "B0": "BoxNone", "SA": "ShellArgmax", "BA": "BoxArgmax", "SC": "ShellCont",
             "BC": "BoxCont"}
# Pre-registered family, in the order of the amendment log (2026-10-12).
DEC_CONTRASTS = [
    ("SupportArgmax", "support under candidate-set argmax (shell $-$ box)", "Supp.\\,argmax", ("SA", "BA")),
    ("SupportCont", "support under continuous maximisation (shell $-$ box)", "Supp.\\,cont.", ("SC", "BC")),
    ("SupportNone", "support with no surrogate (shell $-$ box)", "Supp.\\,none", ("S0", "B0")),
    ("MaxShell", "maximiser on the shell (argmax $-$ projected Adam)", "Max.\\,shell", ("SA", "SC")),
    ("MaxBox", "maximiser in the box (argmax $-$ L-BFGS-B)", "Max.\\,box", ("BA", "BC")),
    ("IntMax", "support $\\times$ maximiser interaction", "Int.\\,max.", (("SA", "BA"), ("SC", "BC"))),
    ("SurrShell", "surrogate on the shell (GP argmax $-$ uniform)", "Surr.\\,shell", ("SA", "S0")),
    ("SurrBox", "surrogate in the box (GP argmax $-$ uniform)", "Surr.\\,box", ("BA", "B0")),
    ("IntSurr", "support $\\times$ surrogate interaction", "Int.\\,surr.", (("SA", "BA"), ("S0", "B0"))),
]
PREREG = ("pre-registered family (amendment of 2026-10-12 to the run rules released with the code): primary test = exact two-sided Wilcoxon "
          "signed-rank test on the per-task mean differences, Benjamini--Hochberg within these nine contrasts; "
          "cell-level Wilcoxon $p$ (uncorrected) and $d$ are secondary")


def stack(*lines: str) -> str:
    """A multi-line header cell (the empty group after the line break keeps a line that starts with '[' from
    being read as the optional argument of the line break)."""
    return "\\begin{tabular}[b]{@{}c@{}}" + "\\\\{}".join(lines) + "\\end{tabular}"


def ci_text(lo: float, hi: float, fmt=N.s3) -> str:
    return f"[{fmt(lo)}, {fmt(hi)}]"


def reg_contrast(ctx: Ctx, prefix: str, pr, fs: dict, table: str, what: str, ci_level: float = 0.95) -> tuple:
    """Delta, task-bootstrap CI, tasks won, p_task (+ BH), cell p, d, W/L (+ exact), n."""
    lo, hi = pr.ci(ci_level)
    pct = int(round(100 * ci_level))
    ctx.num(prefix + "Delta", pr.delta, N.s4(pr.delta), kind="delta", table=table, fileset=fs, n=pr.n,
            rule=f"{conv_text(ctx)} of the paired cell contrast: {what}")
    ctx.num(prefix + "CiLo", lo, N.s4(lo), kind="ci", table=table, fileset=fs, n=pr.n_tasks,
            rule=f"lower end of the {pct} % task-bootstrap interval (10,000 resamples of tasks, seed {stats.BOOT_SEED})")
    ctx.num(prefix + "CiHi", hi, N.s4(hi), kind="ci", table=table, fileset=fs, n=pr.n_tasks,
            rule=f"upper end of the {pct} % task-bootstrap interval")
    ctx.num(prefix + "Tasks", pr.tasks_won, f"{pr.tasks_won}/{pr.n_tasks}", kind="tasks_won", table=table, fileset=fs,
            n=pr.n_tasks, rule="tasks with a positive per-task mean contrast")
    ctx.num(prefix + "PTask", pr.p_task, N.psig(pr.p_task), kind="p_task", table=table, fileset=fs, n=pr.n_tasks,
            rule="exact two-sided Wilcoxon signed-rank test on the per-task mean contrasts (primary)")
    if np.isfinite(pr.p_task_bh) or not np.isfinite(pr.p_task):
        ctx.num(prefix + "PTaskBH", pr.p_task_bh, N.psig(pr.p_task_bh), kind="p_task_bh", table=table, fileset=fs,
                n=pr.n_tasks, rule="Benjamini-Hochberg within the family, on the task-level p")
    ctx.num(prefix + "P", pr.p, N.pfmt(pr.p), kind="p_raw", table=table, fileset=fs, n=pr.n,
            rule="two-sided Wilcoxon signed-rank test over cells (secondary, uncorrected; |diff| <= 1e-6 = zero)")
    ctx.num(prefix + "PExact", pr.p_exact, N.pfmt(pr.p_exact), kind="p_raw_exact", table=table, fileset=fs, n=pr.n,
            rule="old rule: the same test on the exact float differences")
    ctx.num(prefix + "D", pr.d, N.d2(pr.d), kind="cohens_d", table=table, fileset=fs, n=pr.n,
            rule="Cohen's d = mean/sd of the cell contrasts (secondary)")
    ctx.num(prefix + "WL", f"{pr.wins}/{pr.losses}", f"{pr.wins}/{pr.losses}", kind="wins_losses", table=table,
            fileset=fs, n=pr.n, rule="cells with contrast > +1e-6 / < -1e-6")
    ctx.num(prefix + "WLExact", f"{pr.wins_exact}/{pr.losses_exact}", f"{pr.wins_exact}/{pr.losses_exact}",
            kind="wins_losses_exact", table=table, fileset=fs, n=pr.n, rule="old W/L rule (exact float differences)")
    ctx.num(prefix + "N", pr.n, N.intfmt(pr.n), kind="n", table=table, fileset=fs, n=pr.n, rule="matched cells")
    return lo, hi


# =========================================================================================
# the six cells
# =========================================================================================
def decomposition_arms(ctx: Ctx) -> tuple[dict | None, bool, list]:
    arms = dict(DEC_ARMS)
    legacy_b0 = False
    if not ctx.has(arms["B0"]) and ctx.mode == "legacy" and ctx.has(LEGACY_B0):
        arms["B0"] = LEGACY_B0
        legacy_b0 = True
    missing = [a for a in arms.values() if not ctx.has(a)]
    return (None if missing else arms), legacy_b0, missing


def dec_compute(ctx: Ctx, arms: dict, N_eval: int = 30) -> dict:
    """Levels and the nine contrasts on the joint matched set of the six arms (cached per variant)."""
    key = ("dec", N_eval, tuple(sorted(arms.items())))
    store = ctx.extra.setdefault("_dec_cache", {})
    if key in store:
        return store[key]
    cells = ctx.eng.joint(list(arms.values()))
    vals = {k: ctx.eng.vals(a, cells, N_eval) for k, a in arms.items()}
    out = {"cells": cells, "levels": {k: stats.level(vals[k], cells, ctx.var.conv) for k in arms}, "pr": {}}
    for k, _, _, pair in DEC_CONTRASTS:
        if isinstance(pair[0], tuple):
            (a1, b1), (a2, b2) = pair
            x = {c: (vals[a1][c] - vals[b1][c]) - (vals[a2][c] - vals[b2][c]) for c in cells}
            y = {c: 0.0 for c in cells}
        else:
            x, y = vals[pair[0]], vals[pair[1]]
        out["pr"][k] = stats.paired(x, y, cells, ctx.var.conv)
    q = stats.bh([out["pr"][k].p_task for k, _, _, _ in DEC_CONTRASTS])
    for (k, _, _, _), qq in zip(DEC_CONTRASTS, q):
        out["pr"][k].p_task_bh = qq
    store[key] = out
    return out


def _legacy_note(arms: dict) -> str:
    return ("\\textbf{LEGACY VARIANT, not the camera-ready generation:} the box cell without a surrogate is read from "
            "the September p1 blocks (\\texttt{random\\_box\\_faithful}, commit 4780475, batch runner, greedy "
            "decoding), the shell cell without a surrogate from the March random-search files (stochastic decoding), "
            "and the four GP cells from the posted generation; the six cells come from three code states, so every "
            "contrast that involves the no-surrogate column crosses code states and is descriptive only. ")


def _dec_fs(ctx: Ctx, arms: dict, cells, keys) -> dict:
    return ctx.fs({arms[k]: cells for k in keys})


def tab_decomposition(ctx: Ctx) -> Table | None:
    table = "tab_decomposition"
    arms, legacy_b0, missing = decomposition_arms(ctx)
    if arms is None:
        for a in missing:
            ctx.skip(table, a, "decomposition arm absent (table omitted)")
        return None
    R = dec_compute(ctx, arms)
    cells = R["cells"]
    if not cells:
        ctx.skip(table, "matched set", "empty")
        return None
    R5 = dec_compute(ctx, arms, 5)
    L, L5 = R["levels"], R5["levels"]
    best5 = L5["S0"]
    for k in DEC_ARMS:
        fs = ctx.fs({arms[k]: cells})
        ctx.num(f"crDecLevel{CELL_ROLE[k]}", L[k], N.f4(L[k]), kind="level", table=table, fileset=fs, n=len(cells),
                rule=f"{conv_text(ctx)} of best@30 on the joint matched set of the six decomposition arms")
        ctx.num(f"crDecGain{CELL_ROLE[k]}", L[k] - L5[k], N.s4(L[k] - L5[k]), kind="delta", table=table, fileset=fs,
                n=len(cells), rule=f"{conv_text(ctx)} of best@30 - best@5 (gain over the 5 shared cold-start molecules)")
    fs_all = _dec_fs(ctx, arms, cells, DEC_ARMS)
    ctx.num("crDecBestFive", best5, N.f4(best5), kind="level", table=table, fileset=ctx.fs({arms["S0"]: cells}),
            n=len(cells), rule="best of the 5 shared cold-start molecules (best@5; identical across arms up to oracle "
                               "round-off), the uniform-shell arm's value")
    ctx.num("crDecN", len(cells), N.intfmt(len(cells)), kind="n", table=table, fileset=fs_all, n=len(cells),
            rule="joint matched set of the six decomposition arms")
    gs, ga = L["SA"] - L5["SA"], L["S0"] - L5["S0"]
    if abs(gs) > 0:
        ctx.num("crDecShareUniformShell", 100 * ga / gs, f"{100 * ga / gs:.0f}", kind="percent", table=table,
                fileset=ctx.fs({arms["S0"]: cells, arms["SA"]: cells}), n=len(cells),
                rule="gain of uniform shell sampling as a percentage of the GP-argmax shell cell's gain over best@5")
    best = max(L.values())
    hdr = ["Support", stack("No surrogate", "(one uniform draw)"), stack("GP + candidate-set", "argmax (2,048 draws)"),
           stack("GP + continuous maximisation:", "projected Adam (shell),", "L-BFGS-B (box)")]
    t = Table("decomposition", "tab:decomposition", "", "lrrr", hdr, size="\\small")
    for sup, ks in (("Empirical shell ($r$ per run)", ("S0", "SA", "SC")), ("Box $[p_{01},p_{99}]$", ("B0", "BA", "BC"))):
        t.add([sup] + [mb(N.f4(L[k]), abs(L[k] - best) < 5e-5) for k in ks])
    t.add_block(f"Gain over the best of the 5 shared cold-start molecules (best@5 $={N.f4(best5)}$)", "lrrr",
                ["Support", "No surrogate", "GP + argmax", "GP + continuous"],
                [["Empirical shell"] + [m(N.s4(L[k] - L5[k])) for k in ("S0", "SA", "SC")],
                 ["Box $[p_{01},p_{99}]$"] + [m(N.s4(L[k] - L5[k])) for k in ("B0", "BA", "BC")]])
    t.caption = ((_legacy_note(arms) if legacy_b0 else "")
                 + "\\textbf{Where the gain at $N=30$ comes from: support $\\times$ selection layer (same GP constructor "
                 "in the four GP cells; box inputs rescaled per coordinate, shell inputs raw).} Best@30, " + conv_text(ctx) + f", on one joint matched set of the six arms, $n={len(cells)}$ "
                 f"cells ({filt_text(ctx)}). No surrogate: each of the 25 proposals is one uniform draw from the support "
                 "(\\texttt{random\\_search}, \\texttt{random\\_box\\_faithful}); candidate-set argmax: qLogEI maximised "
                 "over 2,048 fresh support draws per step; continuous: qLogEI maximised by projected Adam on the shell "
                 "(10 starts from 512 raw, 200 steps) and by L-BFGS-B in the box (BoTorch \\texttt{optimize\\_acqf}, "
                 "10 restarts, 512 raw). Lower block: best@30 minus best@5 of the same arm (the five cold-start "
                 "molecules are shared by every arm). Contrasts and tests: Table~\\ref{tab:decomposition_contrasts}.")
    return t


def _contrast_rows(ctx, R, table, prefix, fs, ci_level=0.95, register=True) -> list:
    rows = []
    for k, lab, _, _ in DEC_CONTRASTS:
        pr = R["pr"][k]
        if register:
            lo, hi = reg_contrast(ctx, f"{prefix}{k}", pr, fs, table, lab, ci_level)
        else:
            lo, hi = pr.ci(ci_level)
        rows.append((k, lab, pr, lo, hi))
    return rows


def tab_decomposition_contrasts(ctx: Ctx) -> Table | None:
    table = "tab_decomposition_contrasts"
    arms, legacy_b0, missing = decomposition_arms(ctx)
    if arms is None:
        ctx.skip(table, ",".join(missing), "decomposition arm absent (table omitted)")
        return None
    R = dec_compute(ctx, arms)
    cells = R["cells"]
    if not cells:
        return None
    fs = _dec_fs(ctx, arms, cells, DEC_ARMS)
    rows = _contrast_rows(ctx, R, table, "crDec", fs)
    hdr = ["Contrast", "$\\Delta$", "95\\,\\% CI (tasks)", "Tasks", "$p_{\\mathrm{task}}$", "$p_{\\mathrm{task}}^{\\mathrm{BH}}$",
           "$p$ (cells)", "$d$", "W/L", "$n$"]
    t = Table("decomposition_contrasts", "tab:decomposition_contrasts", "", "p{4.6cm}rrrrrrrrr", hdr,
              size="\\footnotesize", wide=True)
    for k, lab, pr, lo, hi in rows:
        t.add([lab, m(N.s4(pr.delta)), m(ci_text(lo, hi)), f"{pr.tasks_won}/{pr.n_tasks}", m(N.psig(pr.p_task)),
               mb(N.psig(pr.p_task_bh), pr.p_task_bh < 0.05), m(N.pfmt(pr.p)), m(N.d2(pr.d)), f"{pr.wins}/{pr.losses}",
               N.intfmt(pr.n)])
    t.caption = ((_legacy_note(arms) if legacy_b0 else "")
                 + "\\textbf{The decomposition contrasts at $N=30$.} " + PREREG[0].upper() + PREREG[1:] + ". "
                 f"$\\Delta$ = {conv_text(ctx)} of the paired cell contrast on the joint matched set of the six arms "
                 f"($n={len(cells)}$, {filt_text(ctx)}); the interactions are computed per cell, "
                 "$(SA-BA)-(SC-BC)$ and $(SA-BA)-(S0-B0)$ (S/B = shell/box; 0 = no surrogate, A = GP + argmax, "
                 "C = GP + continuous). 95\\,\\% CI: percentile interval over 10,000 bootstrap resamples of the tasks "
                 f"(seed {stats.BOOT_SEED}). Tasks = tasks with a positive per-task contrast; ties "
                 "($|\\mathrm{diff}|\\le10^{-6}$) are excluded from W/L and enter the cell-level Wilcoxon test as zero "
                 "differences. Bold: $p_{\\mathrm{task}}^{\\mathrm{BH}}<0.05$.")
    return t


def tab_decomposition_contrasts_byN(ctx: Ctx) -> Table | None:
    table = "tab_decomposition_contrasts_byN"
    arms, legacy_b0, missing = decomposition_arms(ctx)
    if arms is None:
        ctx.skip(table, ",".join(missing), "decomposition arm absent (table omitted)")
        return None
    res = {}
    for n_eval in (10, 20):
        R = dec_compute(ctx, arms, n_eval)
        if not R["cells"]:
            return None
        fs = _dec_fs(ctx, arms, R["cells"], DEC_ARMS)
        res[n_eval] = _contrast_rows(ctx, R, table, f"crDecN{camel(str(n_eval))}", fs)
    hdr = ["Contrast"] + [x for n_eval in (10, 20) for x in
                          (f"$\\Delta_{{{n_eval}}}$", "95\\,\\% CI", "Tasks", "$p_{\\mathrm{task}}^{\\mathrm{BH}}$")]
    t = Table("decomposition_contrasts_byN", "tab:decomposition_contrasts_byN", "", "p{4.2cm}" + "r" * 8, hdr,
              size="\\footnotesize", wide=True)
    for i, (k, lab, _, _) in enumerate(DEC_CONTRASTS):
        row = [lab]
        for n_eval in (10, 20):
            _, _, pr, lo, hi = res[n_eval][i]
            row += [m(N.s4(pr.delta)), m(ci_text(lo, hi)), f"{pr.tasks_won}/{pr.n_tasks}",
                    mb(N.psig(pr.p_task_bh), pr.p_task_bh < 0.05)]
        t.add(row)
    n = len(dec_compute(ctx, arms, 10)["cells"])
    t.caption = ((_legacy_note(arms) if legacy_b0 else "")
                 + "\\textbf{The decomposition contrasts after $N=10$ and $N=20$ evaluations} (best-so-far from "
                 "\\texttt{best\\_trace}; 5 shared cold-start molecules + $N-5$ proposals), same cells as at $N=30$ "
                 f"($n={n}$). The $N=30$ family is the pre-registered one; these are reported for information, with "
                 "Benjamini--Hochberg within each $N$ on the exact task-level Wilcoxon $p$.")
    return t


def tab_decomposition_pertask(ctx: Ctx) -> Table | None:
    table = "tab_decomposition_pertask"
    arms, legacy_b0, missing = decomposition_arms(ctx)
    if arms is None:
        ctx.skip(table, ",".join(missing), "decomposition arm absent (table omitted)")
        return None
    R = dec_compute(ctx, arms)
    cells = R["cells"]
    if not cells:
        return None
    tasks = sorted({c[0] for c in cells})
    hdr = ["Task", "$n$"] + [short for _, _, short, _ in DEC_CONTRASTS]
    t = Table("decomposition_pertask", "tab:decomposition_pertask", "", "lr" + "r" * len(DEC_CONTRASTS), hdr,
              size="\\scriptsize", wide=True)
    for task in tasks:
        tc = {c for c in cells if c[0] == task}
        fs = _dec_fs(ctx, arms, tc, DEC_ARMS)
        row = [task, str(len(tc))]
        for k, lab, _, _ in DEC_CONTRASTS:
            v = R["pr"][k].task_diffs.get(task, float("nan"))
            ctx.num(f"crDecTask{camel(task)}{k}", v, N.s3(v), kind="delta", table=table, fileset=fs, n=len(tc),
                    rule=f"per-task mean contrast on {task}: {lab}")
            row.append(m(N.s3(v)) if v <= 0 else m("\\mathbf{" + N.s3(v) + "}"))
        t.add(row)
    t.rule()
    t.add(["tasks $>0$", ""] + [f"{R['pr'][k].tasks_won}/{R['pr'][k].n_tasks}" for k, _, _, _ in DEC_CONTRASTS])
    legend = "; ".join(f"{short} = {lab}" for _, lab, short, _ in DEC_CONTRASTS)
    t.caption = ((_legacy_note(arms) if legacy_b0 else "")
                 + "\\textbf{Per-task decomposition contrasts at $N=30$} (mean over the task's matched seeds; positive "
                 f"values in bold). {legend}. Same matched set as Table~\\ref{{tab:decomposition_contrasts}} "
                 f"($n={len(cells)}$).")
    return t


# =========================================================================================
# support x maximiser by box, next to the posted generation and E-1
# =========================================================================================
FBB_ROWS = [("Cr", "Pct"), ("Cr", "Minmax"), ("Posted", "Pct"), ("Eone", "Pct")]


def _fbb_arms(ctx: Ctx, gen: str, box: str) -> dict | None:
    if gen == "Cr":
        arms = {"SA": "hvarfner_scaled_gp", "SC": "hvarfner_scaled_gp_sphere_continuous",
                "BA": "hvarfner_scaled_gp_box_argmax" if box == "Pct" else "hvarfner_scaled_gp_norm_box_argmax",
                "BC": "hvarfner_scaled_gp_box_faithful" if box == "Pct" else "hvarfner_scaled_gp_norm_box"}
    elif gen == "Posted":
        pre = "posted:" if ctx.mode == "cr" else ""
        arms = {"SA": pre + "hvarfner_scaled_gp", "SC": pre + "hvarfner_scaled_gp_sphere_continuous",
                "BA": pre + "hvarfner_scaled_gp_box_argmax", "BC": pre + "hvarfner_scaled_gp_box_faithful"}
    else:
        arms = {"SA": "e1_hvarfner_scaled_gp", "SC": "e1_hvarfner_scaled_gp_sphere_continuous",
                "BA": "e1_hvarfner_scaled_gp_box_argmax", "BC": "e1_hvarfner_scaled_gp_box_faithful"}
    return arms if all(ctx.has(a) for a in arms.values()) else None


def tab_factorial_by_box(ctx: Ctx) -> Table | None:
    table = "tab_factorial_by_box"
    got = {}
    for gen, box in FBB_ROWS:
        arms = _fbb_arms(ctx, gen, box)
        if arms is None:
            ctx.skip(table, f"{gen} {box}", "factorial arms absent (row omitted)")
            continue
        got[(gen, box)] = arms
    if not any(g == "Cr" for g, _ in got):
        return None
    # the camera-ready rows share one matched set over both boxes (R-5); posted and E-1 keep their own
    # pre-specified conventions: posted = flat mean, clean cells (423); E-1 = task-mean, clean cells (426)
    cr_arms = sorted({a for (g, _), arms in got.items() if g == "Cr" for a in arms.values()})
    cr_cells = ctx.eng.joint(cr_arms)
    conv_of = {"Cr": ctx.var.conv, "Posted": "flat", "Eone": "taskmean"}
    filt_of = {"Cr": None, "Posted": "clean", "Eone": "clean"}
    gen_lab = {"Cr": "camera-ready" if ctx.mode == "cr" else "legacy (posted files, this table's conventions)",
               "Posted": "posted generation, flat mean, clean cells", "Eone": "E-1 rerun, task-mean, clean cells"}
    box_lab = {"Pct": "$[p_{01},p_{99}]$", "Minmax": "$[\\min,\\max]$"}
    hdr = ["Generation", "Box", "$n$", "SA", "SC", "BA", "BC", stack("Shell$-$box", "argmax"), stack("Shell$-$box", "cont."),
           stack("Inter-", "action"), "Tasks", "$p_{\\mathrm{task}}$"]
    t = Table("factorial_by_box", "tab:factorial_by_box", "", "llr" + "r" * 9, hdr, size="\\footnotesize", wide=True)
    for (gen, box), arms in got.items():
        cells = cr_cells if gen == "Cr" else ctx.eng.joint(list(arms.values()), filt_of[gen])
        conv = conv_of[gen]
        if not cells:
            continue
        v = {k: ctx.eng.vals(a, cells) for k, a in arms.items()}
        lev = {k: stats.level(v[k], cells, conv) for k in arms}
        pa = stats.paired(v["SA"], v["BA"], cells, conv)
        pc = stats.paired(v["SC"], v["BC"], cells, conv)
        inter = {c: (v["SA"][c] - v["BA"][c]) - (v["SC"][c] - v["BC"][c]) for c in cells}
        pi = stats.paired(inter, {c: 0.0 for c in cells}, cells, conv)
        fs = ctx.fs({a: cells for a in arms.values()})
        p = f"crFbb{gen}{box}"
        names = {"SA": "ShellCand", "SC": "ShellCont", "BA": "BoxCand", "BC": "BoxCont"}
        for k in ("SA", "SC", "BA", "BC"):
            ctx.num(p + names[k], lev[k], N.f4(lev[k]), kind="level", table=table, fileset=ctx.fs({arms[k]: cells}),
                    n=len(cells), rule=f"{'flat mean' if conv == 'flat' else 'mean of per-task means'} of best@30 ({gen_lab[gen]})")
        ctx.num(p + "N", len(cells), N.intfmt(len(cells)), kind="n", table=table, fileset=fs, n=len(cells),
                rule="matched cells of this row")
        for nm, pr in (("SupportCand", pa), ("SupportCont", pc), ("Interaction", pi)):
            dl = pr.delta_taskmean if conv == "taskmean" else pr.delta_flat
            ctx.num(p + nm + "Delta", dl, N.s4(dl), kind="delta", table=table, fileset=fs, n=pr.n,
                    rule=f"{'flat' if conv == 'flat' else 'task-mean'} paired contrast ({gen_lab[gen]})")
        ctx.num(p + "InteractionTasks", pi.tasks_won, f"{pi.tasks_won}/{pi.n_tasks}", kind="tasks_won", table=table,
                fileset=fs, n=pi.n_tasks, rule="tasks with a positive per-task interaction")
        ctx.num(p + "InteractionPTask", pi.p_task, N.psig(pi.p_task), kind="p_task", table=table, fileset=fs,
                n=pi.n_tasks, rule="exact two-sided Wilcoxon on the per-task interactions")
        ctx.num(p + "InteractionD", pi.d, N.d2(pi.d), kind="cohens_d", table=table, fileset=fs, n=pi.n,
                rule="d of the per-cell interaction")
        dl = (lambda pr: pr.delta_taskmean if conv == "taskmean" else pr.delta_flat)
        short = {"Cr": "camera-ready" if ctx.mode == "cr" else "legacy files", "Posted": "posted (flat)",
                 "Eone": "E-1"}[gen]
        t.add([short, box_lab[box], N.intfmt(len(cells))]
              + [m(N.f4(lev[k])) for k in ("SA", "SC", "BA", "BC")]
              + [m(N.s4(dl(pa))), m(N.s4(dl(pc))), m(N.s4(dl(pi))), f"{pi.tasks_won}/{pi.n_tasks}", m(N.psig(pi.p_task))])
    t.caption = ("\\textbf{Support $\\times$ acquisition maximisation with two box definitions, next to the earlier "
                 "generations.} SA/SC = empirical shell with candidate-set argmax / continuous maximisation (projected "
                 "Adam); BA/BC = box with candidate-set argmax / continuous maximisation (L-BFGS-B); the same GP "
                 "constructor in every cell (box inputs rescaled per coordinate, shell inputs raw). The camera-ready rows share one matched set over both boxes "
                 f"(n = {len(cr_cells)}, {filt_text(ctx)}, {conv_text(ctx)}); the posted generation is recomputed under "
                 "its posted convention (flat mean over the clean cells of its four arms) and E-1 under its own (mean "
                 "of per-task means, clean cells). Interaction = $(SA-BA)-(SC-BC)$ per cell; Tasks and "
                 "$p_{\\mathrm{task}}$ (exact two-sided Wilcoxon over tasks) refer to the interaction.")
    return t


# =========================================================================================
# equivalence (TOST, margin 0.01)
# =========================================================================================
EQ_PAIRS = [("geowalk", "hvarfner_scaled_gp", "Vanilla", "\\geowalk{} $-$ Vanilla BO (shell)"),
            ("geowalk", "linear_bo_sphere_calib", "Linear", "\\geowalk{} $-$ Linear BO (shell, calibrated)")]


def tab_equivalence(ctx: Ctx) -> Table | None:
    table = "tab_equivalence"
    pairs = [p for p in EQ_PAIRS if ctx.has(p[0]) and ctx.has(p[1]) or ctx.skip(table, p[1], "arm absent (row omitted)")]
    if not pairs:
        return None
    arms = sorted({a for p in pairs for a in p[:2]})
    cells = ctx.eng.joint(arms)
    if not cells:
        return None
    mg = stats.EQUIV_MARGIN
    hdr = ["Comparison", "$N$", "$\\Delta$", "90\\,\\% CI (tasks)", "95\\,\\% CI (tasks)", "$p_{\\mathrm{TOST}}$ ($t$)",
           "$p_{\\mathrm{TOST}}$ (Wilcoxon)", "Tasks", "Verdict"]
    t = Table("equivalence", "tab:equivalence", "", "lrrrrrrrl", hdr, size="\\footnotesize", wide=True)
    for n_eval in (30, 10, 20):
        if n_eval == 10:
            t.group("For information only (not pre-registered claims): $N=10$ and $N=20$")
        for a, b, key, lab in pairs:
            pr = ctx.eng.paired(a, b, cells, n_eval)
            T = stats.tost(pr.task_diffs, mg, pr.task_n if ctx.var.conv == "flat" else None)
            lo, hi = T["ci90"]
            fs = ctx.fs({a: cells, b: cells})
            p = f"crEquiv{key}N{camel(str(n_eval))}"
            verdict = T["verdict"]
            ctx.num(p + "Delta", pr.delta, N.s4(pr.delta), kind="delta", table=table, fileset=fs, n=pr.n,
                    rule=f"{conv_text(ctx)} paired difference at N = {n_eval}")
            ctx.num(p + "CiLo", lo, N.s4(lo), kind="ci", table=table, fileset=fs, n=pr.n_tasks,
                    rule="90 % task-bootstrap interval, lower end (10,000 resamples)")
            ctx.num(p + "CiHi", hi, N.s4(hi), kind="ci", table=table, fileset=fs, n=pr.n_tasks,
                    rule="90 % task-bootstrap interval, upper end")
            ctx.num(p + "PTostT", T["p_t"], N.psig(T["p_t"]), kind="p_tost", table=table, fileset=fs, n=pr.n_tasks,
                    rule=f"TOST, two one-sided t-tests on the per-task differences, margin {mg}")
            ctx.num(p + "PTostW", T["p_w"], N.psig(T["p_w"]), kind="p_tost", table=table, fileset=fs, n=pr.n_tasks,
                    rule=f"TOST, two one-sided exact Wilcoxon tests on the per-task differences, margin {mg}")
            ctx.num(p + "Tasks", pr.tasks_won, f"{pr.tasks_won}/{pr.n_tasks}", kind="tasks_won", table=table,
                    fileset=fs, n=pr.n_tasks, rule="tasks on which GeoWalk has the higher mean")
            ctx.num(p + "Verdict", verdict, verdict, kind="verdict", table=table, fileset=fs, n=pr.n_tasks,
                    rule=f"'equivalent' iff the 90 % task-bootstrap interval lies inside (-{mg}, +{mg}) (pre-registered)")
            lo95, hi95 = pr.ci(0.95)
            ctx.num(p + "CiLoNinetyFive", lo95, N.s4(lo95), kind="ci", table=table, fileset=fs, n=pr.n_tasks,
                    rule="95 % task-bootstrap interval, lower end (10,000 resamples; for information, the verdict "
                         "uses the 90 % interval)")
            ctx.num(p + "CiHiNinetyFive", hi95, N.s4(hi95), kind="ci", table=table, fileset=fs, n=pr.n_tasks,
                    rule="95 % task-bootstrap interval, upper end")
            t.add([lab, str(n_eval), m(N.s4(pr.delta)), m(ci_text(lo, hi, N.s4)), m(ci_text(lo95, hi95, N.s4)),
                   m(N.psig(T["p_t"])), m(N.psig(T["p_w"])), f"{pr.tasks_won}/{pr.n_tasks}",
                   ("\\textbf{equivalent}" if T["equivalent"] else verdict)])
    ctx.num("crEquivN", len(cells), N.intfmt(len(cells)), kind="n", table=table, fileset=ctx.fs({a: cells for a in arms}),
            n=len(cells), rule="joint matched set of the equivalence table")
    rec_txt = _recovered_block(ctx, t, table, pairs)
    t.caption = ("\\textbf{Equivalence of \\geowalk{} with the GP recipes on the shell} (pre-registered, amendment of 2026-10-12 "
                 f"to the run rules released with the code): two one-sided tests on the per-task mean differences with margin $\\pm{mg}$ "
                 "(the size of the GP's gain over uniform sampling on the shell); `equivalent' is printed only if the "
                 "90\\,\\% task-bootstrap interval (10,000 resamples of the tasks) lies inside $\\pm" + f"{mg}" + "$. "
                 f"One matched set of $n={len(cells)}$ cells ({filt_text(ctx)}); $\\Delta$ = {conv_text(ctx)} of the paired "
                 "differences; $p_{\\mathrm{TOST}}$ = the larger of the two one-sided $p$ values ($t$-test with "
                 "10 degrees of freedom; exact Wilcoxon signed-rank test). Tasks = tasks with the higher \\geowalk{} mean. "
                 "The 95\\,\\% interval is shown for information; the verdict uses the 90\\,\\% interval." + rec_txt)
    return t


REC_REF = "random_search"


def _recovered_block(ctx: Ctx, t: Table, table: str, pairs) -> str:
    """Recovered fraction (GeoWalk - RS) / (GP - RS): how much of a GP recipe's gain over uniform shell sampling
    GeoWalk reaches without a surrogate. Own matched set (the equivalence arms plus random_search)."""
    if not ctx.has(REC_REF):
        ctx.skip(table, REC_REF, "arm absent (recovered-fraction block omitted)")
        return ""
    arms = sorted({a for p in pairs for a in p[:2]} | {REC_REF})
    cells = ctx.eng.joint(arms)
    if not cells:
        return ""
    v = {a: ctx.eng.vals(a, cells) for a in arms}
    fs_all = ctx.fs({a: cells for a in arms})
    ctx.num("crRecovN", len(cells), N.intfmt(len(cells)), kind="n", table=table, fileset=fs_all, n=len(cells),
            rule="joint matched set of the recovered-fraction block (equivalence arms and random_search)")
    gw = "geowalk"
    pg = stats.paired(v[gw], v[REC_REF], cells, ctx.var.conv)
    lo, hi = pg.ci()
    fs = ctx.fs({gw: cells, REC_REF: cells})
    for suf, val, txt, kind, rule in (
            ("Delta", pg.delta, N.s4(pg.delta), "delta", f"{conv_text(ctx)} of GeoWalk - uniform shell sampling"),
            ("CiLo", lo, N.s4(lo), "ci", "95 % task bootstrap, lower end"), ("CiHi", hi, N.s4(hi), "ci", "95 % task bootstrap, upper end"),
            ("Tasks", pg.tasks_won, f"{pg.tasks_won}/{pg.n_tasks}", "tasks_won", "tasks with GeoWalk ahead of uniform shell sampling"),
            ("PTask", pg.p_task, N.psig(pg.p_task), "p_task", "exact two-sided Wilcoxon on the per-task mean differences")):
        ctx.num("crRecovGwMinusRs" + suf, val, txt, kind=kind, table=table, fileset=fs, n=pg.n_tasks, rule=rule)
    rows = [["\\geowalk{} $-$ uniform shell sampling", m(N.s4(pg.delta)), m(ci_text(lo, hi, N.s4)),
             f"{pg.tasks_won}/{pg.n_tasks}", m(N.psig(pg.p_task)), "--", "--"]]
    for a, b, key, lab in pairs:
        if a != gw:
            continue
        pb = stats.paired(v[b], v[REC_REF], cells, ctx.var.conv)
        lo_b, hi_b = pb.ci()
        fsb = ctx.fs({b: cells, REC_REF: cells})
        p = f"crRecov{key}"
        ctx.num(p + "GpMinusRsDelta", pb.delta, N.s4(pb.delta), kind="delta", table=table, fileset=fsb, n=pb.n,
                rule=f"{conv_text(ctx)} of {b} - uniform shell sampling (the recipe's gain on the shell)")
        ctx.num(p + "GpMinusRsTasks", pb.tasks_won, f"{pb.tasks_won}/{pb.n_tasks}", kind="tasks_won", table=table,
                fileset=fsb, n=pb.n_tasks, rule="tasks with the recipe ahead of uniform shell sampling")
        ctx.num(p + "GpMinusRsPTask", pb.p_task, N.psig(pb.p_task), kind="p_task", table=table, fileset=fsb,
                n=pb.n_tasks, rule="exact two-sided Wilcoxon on the per-task mean differences")
        if ctx.var.conv == "taskmean":
            rb = stats.ratio_bootstrap(pg.task_diffs, pb.task_diffs)
        else:       # flat convention: cell-weighted task means in numerator and denominator
            w = pg.task_n
            rb = stats.ratio_bootstrap({k_: pg.task_diffs[k_] * w[k_] for k_ in w}, {k_: pb.task_diffs[k_] * w[k_] for k_ in w})
        fsr = ctx.fs({gw: cells, b: cells, REC_REF: cells})
        ctx.num(p + "Fraction", rb["ratio"], N.f2(rb["ratio"]), kind="ratio", table=table, fileset=fsr, n=rb["n_tasks"],
                rule=f"recovered fraction (GeoWalk - RS) / ({b} - RS), ratio of the {conv_text(ctx)} differences")
        ctx.num(p + "FractionCiLo", rb["lo"], N.f2(rb["lo"]), kind="ci", table=table, fileset=fsr, n=rb["n_tasks"],
                rule="95 % task-bootstrap interval of the ratio, lower end (10,000 resamples; the same resampled tasks "
                     "in numerator and denominator)")
        ctx.num(p + "FractionCiHi", rb["hi"], N.f2(rb["hi"]), kind="ci", table=table, fileset=fsr, n=rb["n_tasks"],
                rule="95 % task-bootstrap interval of the ratio, upper end")
        ctx.num(p + "FractionDenNonpositive", rb["den_nonpositive"], str(rb["den_nonpositive"]), kind="count",
                table=table, fileset=fsr, n=rb["n_tasks"],
                rule="bootstrap resamples (of 10,000) in which the recipe's gain over uniform sampling is <= 0 "
                     "(their ratio has no meaning; a nonzero count makes the interval unreliable)")
        frac = (f"{N.f2(rb['ratio'])}\\;[{N.f2(rb['lo'])}, {N.f2(rb['hi'])}]"
                + ("^{\\ddagger}" if rb["den_nonpositive"] else ""))
        rows.append([lab.split(" $-$ ")[1] + " $-$ uniform shell sampling", m(N.s4(pb.delta)), m(ci_text(lo_b, hi_b, N.s4)),
                     f"{pb.tasks_won}/{pb.n_tasks}", m(N.psig(pb.p_task)), m(frac), str(rb["den_nonpositive"])])
    t.add_block(f"Gain over uniform shell sampling (random search) and the fraction of a recipe's gain that "
                f"\\geowalk{{}} recovers, matched set $n={len(cells)}$",
                "lrrrrrr", ["Contrast", "$\\Delta$", "95\\,\\% CI", "Tasks", "$p_{\\mathrm{task}}$",
                            stack("Recovered fraction", "[95\\,\\% CI]"), stack("Resamples with", "gain $\\le 0$")], rows)
    return (" Lower block: each arm's gain over uniform shell sampling on one matched set of the equivalence arms "
            f"and random search ($n={len(cells)}$), and the recovered fraction (\\geowalk{{}} $-$ RS)/(recipe $-$ RS) "
            "of the task-mean gains, with a percentile interval over 10,000 task-bootstrap resamples that use the "
            "same resampled tasks in numerator and denominator; the last column counts resamples in which the "
            "recipe's gain is not positive ($^{\\ddagger}$: the interval is then unreliable).")


# =========================================================================================
# distinct proposals
# =========================================================================================
DIST_ROWS = [("S0", "Shell, no surrogate (uniform)"), ("B0", "Box $[p_{01},p_{99}]$, no surrogate"),
             ("SA", "Shell, GP + argmax"), ("BA", "Box $[p_{01},p_{99}]$, GP + argmax"),
             ("SC", "Shell, GP + projected Adam"), ("BC", "Box $[p_{01},p_{99}]$, GP + L-BFGS-B"),
             ("BAm", "Box $[\\min,\\max]$, GP + argmax"), ("BCm", "Box $[\\min,\\max]$, GP + L-BFGS-B"),
             ("GW", "\\geowalk{}")]
DIST_EXTRA = {"BAm": "hvarfner_scaled_gp_norm_box_argmax", "BCm": "hvarfner_scaled_gp_norm_box", "GW": "geowalk"}
DIST_ROLE = dict(CELL_ROLE, BAm="BoxMinmaxArgmax", BCm="BoxMinmaxCont", GW="Geowalk")


def distinct_counts(ctx: Ctx, arm: str, cells) -> dict:
    """(task, seed) -> distinct proposals of 25 = 25 - empty - nonzero repeats (score-equality proxy)."""
    a = ctx.ds.arm(arm)
    out = {}
    for k in cells:
        e, r = cell_fidelity(a, a.cells[k])
        out[k] = 25 - e - r
    return out


def tab_distinct(ctx: Ctx) -> Table | None:
    table = "tab_distinct"
    arms, legacy_b0, missing = decomposition_arms(ctx)
    roles = {}
    for k, _ in DIST_ROWS:
        a = (arms or DEC_ARMS).get(k) if k in DEC_ARMS else DIST_EXTRA[k]
        if a and ctx.has(a):
            roles[k] = a
        else:
            ctx.skip(table, a or k, "arm absent (row omitted)")
    if "SA" not in roles or "SC" not in roles:
        return None
    cells = ctx.eng.joint(list(roles.values()))
    if not cells:
        return None
    hdr = ["Arm", "Mean", "Median", "Min", "Cells with 25", "Repeats / cell", "Empty / cell"]
    t = Table("distinct", "tab:distinct", "", "lrrrrrr", hdr, size="\\footnotesize")
    dc = {}
    for k, lab in DIST_ROWS:
        if k not in roles:
            continue
        a = roles[k]
        cnt = distinct_counts(ctx, a, cells)
        dc[k] = cnt
        arr = np.asarray(list(cnt.values()), dtype=float)
        A = ctx.ds.arm(a)
        reps = np.mean([cell_fidelity(A, A.cells[c])[1] for c in cells])
        emps = np.mean([cell_fidelity(A, A.cells[c])[0] for c in cells])
        fs = ctx.fs({a: cells})
        p = f"crDistinct{DIST_ROLE[k]}"
        for suf, v, txt in (("Mean", arr.mean(), N.f2(arr.mean())), ("Median", float(np.median(arr)), f"{np.median(arr):.0f}"),
                            ("Min", arr.min(), f"{arr.min():.0f}"), ("Full", int((arr == 25).sum()), str(int((arr == 25).sum()))),
                            ("Repeats", reps, N.f2(reps)), ("Empty", emps, N.f2(emps))):
            ctx.num(p + suf, v, txt, kind="fidelity", table=table, fileset=fs, n=len(cells),
                    rule=f"distinct proposals of 25 (score-equality proxy): {suf.lower()} over the matched cells")
        t.add([ctx.label(a, lab) + ("$^\\ddagger$" if a == LEGACY_B0 else ""), m(N.f2(arr.mean())),
               f"{np.median(arr):.0f}", f"{arr.min():.0f}", str(int((arr == 25).sum())), m(N.f2(reps)), m(N.f2(emps))])
    # sensitivity: maximiser on the shell (SA - SC) restricted to cells where the Adam cell has at least the
    # median number of distinct proposals
    sc = dc["SC"]
    thr = float(np.median(list(sc.values())))
    sub = {c for c in cells if sc[c] >= thr}
    va, vc = ctx.eng.vals(roles["SA"], cells), ctx.eng.vals(roles["SC"], cells)
    blk = []
    for key, lab, cs in (("All", "all matched cells", cells), ("Restricted", f"cells with $\\ge{thr:.0f}$ distinct (shell $\\times$ Adam)", sub)):
        pr = stats.paired({c: va[c] for c in cs}, {c: vc[c] for c in cs}, cs, ctx.var.conv)
        lo, hi = pr.ci()
        fs = ctx.fs({roles["SA"]: cs, roles["SC"]: cs})
        p = f"crDistinctSens{key}"
        ctx.num(p + "N", pr.n, N.intfmt(pr.n), kind="n", table=table, fileset=fs, n=pr.n, rule=f"cells: {lab}")
        ctx.num(p + "NTasks", pr.n_tasks, str(pr.n_tasks), kind="n", table=table, fileset=fs, n=pr.n_tasks,
                rule="tasks represented")
        ctx.num(p + "Delta", pr.delta, N.s4(pr.delta), kind="delta", table=table, fileset=fs, n=pr.n,
                rule=f"maximiser on the shell (GP argmax - projected Adam), {conv_text(ctx)}")
        ctx.num(p + "CiLo", lo, N.s4(lo), kind="ci", table=table, fileset=fs, n=pr.n_tasks, rule="95 % task bootstrap")
        ctx.num(p + "CiHi", hi, N.s4(hi), kind="ci", table=table, fileset=fs, n=pr.n_tasks, rule="95 % task bootstrap")
        ctx.num(p + "Tasks", pr.tasks_won, f"{pr.tasks_won}/{pr.n_tasks}", kind="tasks_won", table=table, fileset=fs,
                n=pr.n_tasks, rule="tasks with argmax ahead")
        ctx.num(p + "PTask", pr.p_task, N.psig(pr.p_task), kind="p_task", table=table, fileset=fs, n=pr.n_tasks,
                rule="exact two-sided Wilcoxon over the per-task contrasts")
        ctx.num(p + "D", pr.d, N.d2(pr.d), kind="cohens_d", table=table, fileset=fs, n=pr.n, rule="cells")
        blk.append([lab, N.intfmt(pr.n), str(pr.n_tasks), m(N.s4(pr.delta)), m(ci_text(lo, hi)),
                    f"{pr.tasks_won}/{pr.n_tasks}", m(N.psig(pr.p_task)), m(N.d2(pr.d))])
    ctx.num("crDistinctSensThreshold", thr, f"{thr:.0f}", kind="count", table=table,
            fileset=ctx.fs({roles["SC"]: cells}), n=len(cells),
            rule="median distinct proposals of the shell x projected-Adam cell over the matched cells")
    ctx.num("crDistinctN", len(cells), N.intfmt(len(cells)), kind="n", table=table,
            fileset=ctx.fs({a: cells for a in roles.values()}), n=len(cells), rule="joint matched set of the table")
    t.add_block("Maximiser on the shell (GP argmax $-$ projected Adam), all cells and cells where the Adam arm "
                "proposed at least the median number of distinct molecules", "lrrrrrrr",
                ["Cells", "$n$", "Tasks in set", "$\\Delta$", "95\\,\\% CI", "Tasks", "$p_{\\mathrm{task}}$", "$d$"], blk)
    t.caption = (("$^\\ddagger$ " + _legacy_note(arms) if legacy_b0 and "B0" in roles else "")
                 + "\\textbf{Distinct proposals among the 25 per cell (score-equality proxy).} A proposal counts as "
                 "distinct unless it decoded to nothing (score $-1$) or its nonzero score equals an earlier score of "
                 "the same cell (cold start included); the proxy misses repeated zero-score molecules and can merge "
                 "distinct molecules with equal scores. One matched set of all rows, "
                 f"$n={len(cells)}$ ({filt_text(ctx)}). Lower block: the maximiser-on-the-shell contrast on all cells "
                 "and restricted to the cells where the shell $\\times$ projected-Adam arm proposed at least its median "
                 "number of distinct molecules (task-level tests as in Table~\\ref{tab:decomposition_contrasts}).")
    return t


# =========================================================================================
# GP radius dose-response next to the GeoWalk radius sweep
# =========================================================================================
GP_RAD = [("hvarfner_scaled_gp_sphere05", "Half", 0.5), ("hvarfner_scaled_gp", "One", 1.0),
          ("hvarfner_scaled_gp_sphere15", "OneHalf", 1.5), ("hvarfner_scaled_gp_sphere20", "Two", 2.0),
          ("hvarfner_scaled_gp_prior", "Prior", None)]
GW_RAD = [("geowalk_r4", "RFour", 4), ("geowalk_r6", "RSix", 6), ("geowalk_r8", "REight", 8), ("geowalk", "Default", None),
          ("geowalk_r10", "RTen", 10), ("geowalk_r12", "RTwelve", 12), ("geowalk_r16", "RSixteen", 16)]


def tab_gp_radius(ctx: Ctx) -> Table | None:
    table = "tab_gp_radius"
    from .geometry import geometry_inputs
    from .supports import get_supports
    gp = [x for x in GP_RAD if ctx.has(x[0]) or ctx.skip(table, x[0], "arm absent (row omitted)")]
    gw = [x for x in GW_RAD if ctx.has(x[0]) or ctx.skip(table, x[0], "arm absent (row omitted)")]
    if not any(x[0] == "hvarfner_scaled_gp" for x in gp):
        return None
    arms = [x[0] for x in gp + gw]
    cells = ctx.eng.joint(arms)
    if not cells:
        return None
    G = geometry_inputs(ctx)
    rbar = G["default"]["g"]["mu_cloud"]["norm"]["mean"] if "default" in G else None
    rbar_fs = ctx.fs_paths([G["default"]["json"]]) if "default" in G else None
    S = get_supports(ctx) if "default" in G else {}
    hdr = ["Candidates", "$r/\\bar r$", "Mean", "$\\Delta$ vs $1\\times$", "95\\,\\% CI", "Tasks", "$p_{\\mathrm{task}}$",
           "$p_{\\mathrm{task}}^{\\mathrm{BH}}$", "$d$"]
    t = Table("gp_radius", "tab:gp_radius", "", "lrrrrrrrr", hdr, size="\\footnotesize", wide=True)

    def block(rows, ref, prefix, title, label_of, ratio_of):
        prs = [(a, key, val, ctx.eng.paired(a, ref, cells)) for a, key, val in rows if a != ref]
        q = stats.bh([pr.p_task for _, _, _, pr in prs])
        for x, qq in zip(prs, q):
            x[3].p_task_bh = qq
        out = []
        for a, key, val in rows:
            lv = ctx.eng.level(a, cells)
            fs = ctx.fs({a: cells})
            ctx.num(f"{prefix}{key}Mean", lv, N.f4(lv), kind="level", table=table, fileset=fs, n=len(cells),
                    rule=f"{conv_text(ctx)} of best@30, joint matched set of the radius table")
            rt, rfs = ratio_of(a, val)
            if rt is not None:
                ctx.num(f"{prefix}{key}Ratio", rt, f"{rt:.2f}", kind="ratio", table=table, fileset=rfs, n=None,
                        rule="candidate radius / r-bar (r-bar = mean encoder-mean norm of the full ZINC set)")
            if a == ref:
                out.append([label_of(a, val) + " (reference)", m(f"{rt:.2f}") if rt is not None else "--",
                            m(N.f4(lv)), "--", "--", "--", "--", "--", "--"])
                continue
            pr = next(x[3] for x in prs if x[0] == a)
            fs2 = ctx.fs({a: cells, ref: cells})
            lo, hi = reg_contrast(ctx, f"{prefix}{key}", pr, fs2, table, f"{a} - {ref}")
            out.append([label_of(a, val), m(f"{rt:.2f}") if rt is not None else "--", m(N.f4(lv)), m(N.s4(pr.delta)),
                        m(ci_text(lo, hi)), f"{pr.tasks_won}/{pr.n_tasks}", m(N.psig(pr.p_task)),
                        m(N.psig(pr.p_task_bh)), m(N.d2(pr.d))])
        t.group(title)
        for r in out:
            t.add(r)

    def gp_label(a, c):
        return f"GP + argmax, shell at ${c:g}\\,r$" if c is not None else "GP + argmax, candidates $N(0,I_D)$"

    def gp_ratio(a, c):
        if c is not None:        # the multiple of the run's cold-start radius is the arm's definition
            return c, ctx.fs({a: cells})
        if "prior" in S:
            return S["prior"]["ratio"], ctx.fs_paths(S["prior"]["inputs"])
        return None, None

    def gw_label(a, r):
        return f"\\geowalk{{}}, fixed $r={r}$" if r is not None else "\\geowalk{}, $r$ per run (default)"

    def gw_ratio(a, r):
        if r is None:
            return 1.0, ctx.fs({a: cells})
        if rbar is None:
            return None, None
        return r / rbar, rbar_fs

    block(gp, "hvarfner_scaled_gp", "crGpRad", "Vanilla BO GP, qLogEI argmax over 2,048 candidates; reference: the shell "
          "at the cold-start radius ($1\\times$)", gp_label, gp_ratio)
    if any(x[0] == "geowalk" for x in gw):
        block(gw, "geowalk", "crGwRad", "\\geowalk{} radius sweep (no surrogate); reference: the per-run radius",
              gw_label, gw_ratio)
    ctx.num("crRadN", len(cells), N.intfmt(len(cells)), kind="n", table=table, fileset=ctx.fs({a: cells for a in arms}),
            n=len(cells), rule="joint matched set of the radius table")
    t.caption = ("\\textbf{Radial scale under GP selection and without a surrogate.} Top: the Vanilla BO GP with "
                 "candidate-set argmax on spheres of 0.5, 1, 1.5 and 2 times the run's cold-start radius $r$, and with "
                 "candidates from the VAE prior $N(0,I_D)$ (typical norm from the Monte Carlo of "
                 "Table~\\ref{tab:support_geometry}); bottom: \\geowalk{} at fixed radii, $r/\\bar r$ with "
                 "$\\bar r$ the full-set mean encoder-mean norm. One matched set of all rows, "
                 f"$n={len(cells)}$ ({filt_text(ctx)}); $\\Delta$ = row minus the block's $1\\times$ reference "
                 f"({conv_text(ctx)}); Tasks = tasks with the row ahead; $p_{{\\mathrm{{task}}}}$ = exact two-sided "
                 "Wilcoxon over tasks, BH within each block; 95\\,\\% CI from 10,000 task-bootstrap resamples.")
    return t
