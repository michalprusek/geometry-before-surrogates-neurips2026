"""Figures (PDF, TrueType fonts embedded as Type 42; no Type 3)."""
from __future__ import annotations

import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import labels as LB  # noqa: E402
from . import published  # noqa: E402
from . import stats  # noqa: E402
from .context import BOX_LABEL, Ctx  # noqa: E402
from .stored import (LASSO_CORRESPONDENCE_ONLY, LASSO_METHODS, LASSO_TASKS, lasso_analysis,  # noqa: E402
                     lasso_method_label, lasso_pub_analysis)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8,
                     "axes.titlesize": 8, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})

#: the published Linear BO recipe added to the convergence figure (RUN_RULES 2026-10-29 (3)); the trajectory figure,
#: a record since that row, keeps its curves
PUB_LINEAR = "linear_bo_box8"
#: authors' decision of 2026-11-02: in the convergence figure the stored TuRBO and BAxUS curves are replaced by the runs
#: that Table 1 prints under these names (camera-ready mode); the BAxUS rerun covers 439 of the 440 cells (one run
#: crashed), so it stays out of the figure's matched set and is drawn on the cells of that set it covers. The
#: trajectory figure (a repository record) keeps the stored curves.
REPLACED_CURVES = {"turbo": "turbo_published", "baxus": "baxus_cr"}
PARTIAL_CURVES = ("baxus_cr",)
#: 2026-11-03 (external review of the submission-ready PDF): the grid is included at \textwidth = 345 pt, so a
#: 7-inch canvas prints at 0.68 of its size and the 6 pt legend reached the reader at about 4 pt. The text of
#: this figure is set larger, and the three curves that the main text discusses are drawn heavier than the rest.
EMPH_CURVES = ("geowalk", "hvarfner_scaled_gp", "random_search")
#: 2026-11-06 (authors' request): the printed convergence figure names its curves as the printed tables do, not by
#: their implementation. A row of Table 1 carries its Table 1 name; an adapted recipe on either support carries its
#: Table 2 name; the two ports of Linear BO and the re-implementation of TuRBO-1, which only the trajectory table of
#: Appendix D and the overview of Appendix H print, carry short names that say what they are. An arm that is not listed keeps its descriptive label.
TABLE_NAMES = {
    "geowalk": "\\geowalk{}",
    "hvarfner_scaled_gp": "GP on the shell",
    "papenmeier_msr": "Mat\\'ern-5/2 GP ($\\ell_0=0.1\\sqrt{D}$), shell",
    "xu_standard_gp": "Mat\\'ern-5/2 GP ($\\ell_0=\\sqrt{D}$), shell",
    "linear_bo_sphere_calib": "Linear BO port to the shell (calibrated bounds)",
    "random_search": "Random Search (shell)",
    # 2026-11-06 21:10 (first author): the box version of the GP on the shell is "GP in the box" everywhere (Tables 2, 8 to 10)
    "hvarfner_scaled_gp_box_faithful": "GP in the box",
    "hvarfner_scaled_gp_norm_box": "Mat\\'ern-5/2 GP (DSP prior), box $[\\min,\\max]$",
    "linear_bo": "Linear BO port, box $[p_{01},p_{99}]$",
    PUB_LINEAR: "Linear BO",
    "turbo_box_faithful": "TuRBO-1 re-implementation, fixed box $[p_{01},p_{99}]$",
    "turbo_published": "TuRBO",
    "baxus_cr": "BAxUS",
    "cmaes_full_rd_pop5": "CMA-ES",
}

TASK_TITLES = {"adip": "Amlodipine MPO", "dhop": "Deco Hop", "fexo": "Fexofenadine MPO", "med1": "Median 1",
               "med2": "Median 2", "osmb": "Osimertinib MPO", "pdop": "Perindopril MPO", "rano": "Ranolazine MPO",
               "shop": "Scaffold Hop", "siga": "Sitagliptin MPO", "zale": "Zaleplon MPO"}


def mpl_label(latex: str) -> str:
    """A table's LaTeX row label as a matplotlib label (mathtext for the $...$ parts), so that a legend prints the
    same words as the tables (RUN_RULES 2026-10-22: figures change labels only)."""
    out = latex
    for a, b in (("Mat\\'ern", "Matérn"), ("\\geowalk{}", "GeoWalk"), ("\\geocma{}", "GeoCMA"), ("\\,", " "),
                 ("--", "–"), ("\\ ", " ")):
        out = out.replace(a, b)
    return out


def _with_support(label: str, support: str, submitted: bool = False) -> str:
    """'<label> (<support>)' as Table 1 prints it (label, Support column); a run of the submitted version says so."""
    base = label.replace(" (submitted version)", "")
    tail = support + ("; stored run" if submitted else "")
    return f"{base} ({tail})" if tail else base


def curve_arms(ctx: Ctx, stored_turbo_baxus: bool = False) -> list[tuple[str, str, dict]]:
    """Legend labels are the Table 1 row labels with Table 1's support (crlib/labels.py, crlib/tables.py tab_main):
    kernel, acquisition and maximizer, plus the support. In camera-ready mode the stored TuRBO and BAxUS curves are
    replaced (REPLACED_CURVES) unless ``stored_turbo_baxus`` (the trajectory figure, a repository record)."""
    bl = BOX_LABEL[ctx.var.box]
    L = LB.label
    rows = [
        ("geowalk", "\\geowalk{} (shell)", dict(color="#c0392b", lw=1.8, zorder=10)),
        ("hvarfner_scaled_gp", _with_support(L("hvarfner_scaled_gp"), "shell"), dict(color="#2471a3", lw=1.3)),
        ("papenmeier_msr", _with_support(L("papenmeier_msr"), "shell"), dict(color="#5dade2", lw=1.0)),
        ("xu_standard_gp", _with_support(L("xu_standard_gp"), "shell"), dict(color="#1abc9c", lw=1.0)),
        ("linear_bo_sphere_calib", _with_support(L("linear_bo_sphere_calib"), "shell"), dict(color="#8e44ad", lw=1.2)),
        ("random_search", "Random Search (shell)", dict(color="#7f8c8d", lw=1.2, ls="--")),
        (ctx.box("hvarfner"), _with_support(L(ctx.box("hvarfner")), bl), dict(color="#2471a3", lw=1.2, ls=":")),
        ("linear_bo", _with_support(L("linear_bo"), "$[p_{01},p_{99}]$"), dict(color="#8e44ad", lw=1.2, ls=":")),
    ] + ([
        # RUN_RULES 2026-10-29 (3), promise C-32: the verified Linear BO recipe (published-recipe block), in its box
        (PUB_LINEAR, _with_support(published.name_or_label(ctx, PUB_LINEAR), published.support(PUB_LINEAR)),
         dict(color="#4a235a", lw=1.2, ls="--")),
    ] if ctx.mode == "cr" else []) + ([
        # RUN_RULES 2026-11-06 21:30: the TuRBO-1 re-implementation is a record only (the trajectory figure of the
        # repository); the printed figure shows the published TuRBO run
        ("turbo_box_faithful", _with_support(L("turbo_box_faithful"), "fixed box $[p_{01},p_{99}]$"),
         dict(color="#d35400", lw=1.0, ls="-.")),
    ] if stored_turbo_baxus or ctx.mode != "cr" else []) + ([
        ("turbo_published", _with_support(L("turbo_published"), published.support("turbo_published")),
         dict(color="#e67e22", lw=1.0, ls="--")),
        ("baxus_cr", _with_support(L("baxus_cr"), "nested subspaces of $[p_{01},p_{99}]$"),
         dict(color="#a04000", lw=1.0, ls="--")),
    ] if ctx.mode == "cr" and not stored_turbo_baxus else [
        ("turbo", _with_support(L("turbo"), "box of observed latents", True), dict(color="#e67e22", lw=1.0, ls=":")),
        ("baxus", _with_support(L("baxus"), "nested subspaces, box", True), dict(color="#a04000", lw=1.0, ls=":")),
    ]) + ([
        # RUN_RULES 2026-11-02: the printed figure has no stored curve of LOL-BO, InvBO or CMA-ES; CMA-ES is the rerun
        ("cmaes_full_rd_pop5", _with_support(L("cmaes_full_rd_pop5"), "unbounded $\\mathbb{R}^D$"),
         dict(color="#566573", lw=1.0, ls="-.")),
    ] if ctx.mode == "cr" and not stored_turbo_baxus else [
        ("lolbo", _with_support(L("lolbo"), "latent trust region", True), dict(color="#27ae60", lw=1.0, ls=":")),
        ("invbo", _with_support(L("invbo"), "latent trust region", True), dict(color="#f1c40f", lw=1.0, ls=":")),
        ("cmaes_full", _with_support(L("cmaes_full"), "unbounded $\\mathbb{R}^D$", True),
         dict(color="#566573", lw=1.0, ls=":")),
    ])
    return [(a, mpl_label(lab), sty) for a, lab, sty in rows]


def _with_count(label: str, count: str) -> str:
    """'<label> (<support>; <count>)': the cells a partial curve covers, inside the support parentheses."""
    return label[:-1] + f"; {count})" if label.endswith(")") else f"{label} ({count})"


def _wrap(label: str, width: int) -> str:
    """Wrap a legend label without breaking inside a $...$ part."""
    cur, out = "", []
    for w in label.split(" "):
        cand = (cur + " " + w).strip()
        if len(cand.replace("$", "").replace("\\", "")) > width and cur and cur.count("$") % 2 == 0:
            out.append(cur)
            cur = w
        else:
            cur = cand
    out.append(cur)
    return "\n".join(out)


def _present(ctx: Ctx, arms, fig: str):
    out = []
    for a, lab, sty in arms:
        if ctx.has(a):
            out.append((a, lab, sty))
        else:
            ctx.skip(fig, a, "arm absent (curve omitted)")
    return out


def _curves(ctx: Ctx, arm: str, cells) -> np.ndarray:
    a = ctx.ds.arm(arm)
    return np.asarray([a.cells[c].trace[:30] for c in sorted(cells)], dtype=float)


def fig_convergence_grid(ctx: Ctx) -> str | None:
    name = "fig_convergence_grid"
    arms = _present(ctx, curve_arms(ctx), name)
    if not arms:
        return None
    # the printed figure names its curves as Tables 1 and 2 do (TABLE_NAMES, 2026-11-06)
    arms = [(a, mpl_label(TABLE_NAMES.get(a, lab)), sty) for a, lab, sty in arms]
    # a curve of a run that covers only part of the matched set (PARTIAL_CURVES) stays out of the intersection and is
    # drawn on the cells of the set it covers; its legend says how many
    joint = ctx.eng.joint([a for a, _, _ in arms if a not in PARTIAL_CURVES])
    cells_of = {a: (joint & ctx.eng.elig(a)) if a in PARTIAL_CURVES else joint for a, _, _ in arms}
    arms = [(a, _with_count(lab, f"{len(cells_of[a])} of the {len(joint)} cells") if cells_of[a] != joint else lab, sty)
            for a, lab, sty in arms]
    tasks = sorted({c[0] for c in joint})
    # 2026-11-06 (external review): the titles of the right column were cut at the page edge. When every task has
    # the same number of cells, the count moves from the panel titles to the title of the figure.
    per_task = {t: len({c for c in joint if c[0] == t}) for t in tasks}
    same_n = len(set(per_task.values())) == 1
    nrow, ncol = 4, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 10.0), squeeze=False)
    x = np.arange(1, 31)
    for i, t in enumerate(tasks[: nrow * ncol - 1]):
        ax = axes[i // ncol][i % ncol]
        cells = {c for c in joint if c[0] == t}
        for a, lab, sty in arms:
            Y = _curves(ctx, a, cells & cells_of[a])
            mu = Y.mean(0)
            se = Y.std(0, ddof=1) / np.sqrt(Y.shape[0]) if Y.shape[0] > 1 else np.zeros_like(mu)
            emph = dict(sty)
            if a in EMPH_CURVES:
                emph["lw"] = max(float(emph.get("lw", 1.0)), 1.6)
                emph["zorder"] = float(emph.get("zorder", 5)) + 5
            else:
                emph["alpha"] = 0.8
            ax.plot(x, mu, label=lab, **emph)
            ax.fill_between(x, mu - se, mu + se, color=sty.get("color"), alpha=0.12, lw=0)
        ax.set_title(TASK_TITLES.get(t, t) if same_n else f"{TASK_TITLES.get(t, t)} (n={len(cells)})", fontsize=11)
        ax.axvline(5.5, color="0.8", lw=0.6, zorder=0)
        ax.tick_params(labelsize=9.5)
        ax.set_xlim(1, 30)
        if i % ncol == 0:
            ax.set_ylabel("best so far", fontsize=11)
        if i // ncol == nrow - 1 or i >= len(tasks) - ncol:
            ax.set_xlabel("evaluations", fontsize=11)
    for j in range(len(tasks), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    h, l_ = axes[0][0].get_legend_handles_labels()
    # the legend sits below the grid in two columns and wraps long labels, so that no label is cut at the edge
    fig.legend(h, [_wrap(x, 42) for x in l_], loc="lower center", frameon=False, ncol=2, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=1.2)
    tail = f", {per_task[tasks[0]]} per task)" if same_n else ")"
    fig.suptitle(f"Best-so-far over 30 evaluations, mean ± SE over the matched seeds (n = {len(joint)} cells" + tail,
                 fontsize=10)  # 10 pt: the title is the widest element, so it sets the print scale
    # the legend holds the short names of TABLE_NAMES (2026-11-06), so it needs less of the page than before
    fig.tight_layout(rect=(0, 0.19, 1, 0.98))
    path = os.path.join(ctx.outdir, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    ctx.num("crFigConvergenceN", len(joint), str(len(joint)), kind="n", table=name,
            fileset=ctx.fs({a: joint for a, _, _ in arms if a not in PARTIAL_CURVES}), n=len(joint),
            rule="joint matched set of all curves that cover every cell (PARTIAL_CURVES: their part of it)")
    return path


def fig_trajectories(ctx: Ctx) -> str | None:
    name = "fig_trajectories"
    want = {"geowalk", "hvarfner_scaled_gp", "linear_bo_sphere_calib", "random_search", ctx.box("hvarfner"),
            "linear_bo", "turbo_box_faithful", "turbo", "baxus"}
    # a repository record: it keeps the stored TuRBO and BAxUS curves (REPLACED_CURVES applies to the convergence figure)
    arms = _present(ctx, [x for x in curve_arms(ctx, stored_turbo_baxus=True) if x[0] in want], name)
    if not arms:
        return None
    joint = ctx.eng.joint([a for a, _, _ in arms])
    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    x = np.arange(1, 31)
    for a, lab, sty in arms:
        mus, ses = [], []
        for n in x:
            v = ctx.eng.vals(a, joint, int(n))
            mus.append(stats.level(v, joint, ctx.var.conv))
            ses.append(stats.level_se(v, joint))
        mus, ses = np.asarray(mus), np.asarray(ses)
        ax.plot(x, mus, label=lab, **sty)
        ax.fill_between(x, mus - ses, mus + ses, color=sty.get("color"), alpha=0.12, lw=0)
    ax.axvline(5.5, color="0.8", lw=0.6, zorder=0)
    ax.set_xlim(1, 30)
    ax.set_xlabel("evaluations (5 shared cold start + proposals)")
    ax.set_ylabel("best so far, mean of per-task means" if ctx.var.conv == "taskmean" else "best so far, mean over cells")
    ax.set_title(f"Aggregate best-so-far, one matched set (n = {len(joint)} cells)")
    # the legend sits in its own band below the axes (figure coordinates), so it never covers the axis label
    fig.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.42)
    hs, ls_ = ax.get_legend_handles_labels()
    fig.legend(hs, [_wrap(x, 90) for x in ls_], frameon=False, fontsize=6, loc="upper center",
               bbox_to_anchor=(0.5, 0.31), ncol=1)
    path = os.path.join(ctx.outdir, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    ctx.num("crFigTrajectoriesN", len(joint), str(len(joint)), kind="n", table=name,
            fileset=ctx.fs({a: joint for a, _, _ in arms}), n=len(joint), rule="joint matched set of all curves")
    return path


def fig_lassobench(ctx: Ctx) -> str | None:
    """Normalised loss against the dimension for every LassoBench arm (all ten methods, Linear BO and
    full-D CMA-ES included). Camera-ready mode: the rerun (R-3); --legacy: the stored runs. A point is drawn
    where at least 3 seeds have a finite loss; error bars are SD(ddof 0)/sqrt(n) as in the submitted figure."""
    name = "fig_lassobench"
    R = lasso_analysis(ctx)
    if R is None:
        ctx.skip(name, "lassobench", "input directory absent")
        return None
    rerun = R.get("generation") == "rerun"
    styles = {"cmaes_sub": ("#c0392b", "o", "-"), "geowalk": ("#e67e22", "s", "-"), "random": ("#7f8c8d", "x", "--"),
              "hvarfner_gp": ("#2471a3", "^", ":"), "papenmeier_msr": ("#5dade2", "v", ":"),
              "xu_standard_gp": ("#1abc9c", "D", ":"), "linear_bo": ("#8e44ad", "P", "-."),
              "cmaes_full": ("#566573", "*", "--"), "turbo": ("#d35400", "<", ":"), "baxus": ("#a04000", ">", ":")}
    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    files, drawn = [], 0
    curves = [(meth, R["per_d"], mpl_label(tex), styles[meth]) for meth, tex in LASSO_METHODS
              if not (ctx.mode == "cr" and meth in LASSO_CORRESPONDENCE_ONLY)]
    P = lasso_pub_analysis(ctx) if ctx.mode == "cr" else None
    if P is not None:
        from . import published
        pub_styles = ["#1f618d", "#76448a", "#117a65", "#b9770e", "#6e2c00", "#2e4053", "#943126", "#7d6608"]
        for i, mth in enumerate(P["printed"]):
            curves.append((mth, P["per_d"], mpl_label(published.name_or_label(ctx, mth, "lasso")),
                           (pub_styles[i % len(pub_styles)], "o", "-.")))
    for meth, per_d, txt, sty in curves:
        xs, ys, es, ns = [], [], [], []
        for task, D in LASSO_TASKS:
            x = per_d.get((meth, task))
            if x is None or x["n"] < 3:
                continue
            xs.append(D)
            ys.append(x["mean"])
            es.append(x["se0"])
            ns.append(x["n"])
            files += x["files"]
        if not xs:
            ctx.skip(name, meth, "no dimension with at least 3 finite seeds (curve omitted)")
            continue
        drawn += 1
        c, mk, ls = sty
        partial = any(n < 40 for n in ns) or len(xs) < len(LASSO_TASKS)
        ax.errorbar(xs, ys, yerr=es, color=c, marker=mk, ls=ls, ms=3.5, lw=1.0, capsize=2,
                    label=txt + (" (partial)" if partial else ""))
    if not drawn:
        plt.close(fig)
        ctx.skip(name, "lassobench", "no method has 3 finite seeds at any dimension (figure omitted)")
        return None
    ax.set_xscale("log")
    ax.set_xlabel("dimension D")
    ax.set_ylabel("loss / random-search loss (lower is better)")
    ax.set_title("LassoBench, N = 30, " + ("rerun" if rerun else "runs of the submitted version")
                 + " (points with n ≥ 3 seeds)")
    hs, ls_ = ax.get_legend_handles_labels()
    # legend in its own band below the axes (figure coordinates), two columns, full labels
    fig.subplots_adjust(left=0.13, right=0.97, top=0.94, bottom=0.42)
    fig.legend(hs, [_wrap(x, 60) for x in ls_], frameon=False, fontsize=5.5, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 0.32))
    path = os.path.join(ctx.outdir, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    if files:
        from .engine import fileset_from_paths
        ctx.num("crFigLassobenchMethods", drawn, str(drawn), kind="count", table=name,
                fileset=fileset_from_paths(sorted(set(files))), n=len(curves),
                rule="methods drawn in the LassoBench figure (at least one dimension with >= 3 finite seeds); "
                     + ("rerun" if rerun else "runs of the submitted version"))
    return path
