"""GuacaMol table builders. Each builder registers every number it prints and returns a Table
(or None when a required arm is absent; the skip is recorded in ctx.skipped)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from scipy import stats as sps

from . import labels as LB
from . import published
from . import numbers as N
from . import stats
from .context import BOX_LABEL, Ctx
from .latex import (COMPLETED_FULL, COMPLETED_SHORT, SUBMITTED_FULL, SUBMITTED_SHORT, WRAP, Table, m, mb, rules,
                    stack, wrapp)
from .numbers import camel


@dataclass
class Row:
    arm: str
    label: str
    support: str = ""
    stored: bool = False
    subset: bool = False
    key: str = ""
    # a row that never shrinks the matched set: if it lacks a cell of the intersection of the other rows (a run that did
    # not complete), it becomes a subset row, paired on the cells it covers (the BAxUS rerun of 2026-11-02)
    partial: bool = False

    def k(self) -> str:
        return self.key or camel(self.arm)


def conv_text(ctx: Ctx) -> str:
    return "mean of per-task means" if ctx.var.conv == "taskmean" else "flat mean over cells"


def conv_plural(ctx: Ctx) -> str:
    """'means of per-task means' (plural, for 'Means are ...'; RUN_RULES 2026-10-29 (4))."""
    return "means of per-task means" if ctx.var.conv == "taskmean" else "flat means over cells"


def filt_text(ctx: Ctx, full: bool = False) -> str:
    """The cell filter in reader words (RUN_RULES 2026-10-22: no rule numbers in captions); Table 1 defines it in
    full, every other caption points to Table 1."""
    if ctx.eng.cell_filter == "completed":
        return COMPLETED_FULL if full else COMPLETED_SHORT
    return "clean cells (status completed, no surrogate-fallback step)"


def no_stored_suffix(label: str) -> str:
    """A row label without the run-origin suffix, where the row's dagger already states it."""
    return label.replace(" (submitted version)", "")


def tl_cols(ctx: Ctx, who: str = "") -> list[str]:
    """Task-level columns; ``who`` names the side the Tasks column counts (RUN_RULES 2026-10-22: every Tasks header
    states what it counts and in which direction)."""
    if not ctx.var.tasklevel:
        return []
    return [stack("Tasks", *who.split("|")) if who else "Tasks", "$p_{\\mathrm{task}}$"]


def tl_vals(ctx: Ctx, pr) -> list[str]:
    if not ctx.var.tasklevel:
        return []
    return [f"{pr.tasks_won}/{pr.n_tasks}", m(N.psig(pr.p_task))]


def tl_vals_bh(ctx: Ctx, pr) -> list[str]:
    """The task count and the corrected task-level p of a recipe row of the printed Table 2 (bold: p < 0.05)."""
    if not ctx.var.tasklevel:
        return []
    return [f"{pr.tasks_won}/{pr.n_tasks}", mb(N.psig(pr.p_task_bh), np.isfinite(pr.p_task_bh) and pr.p_task_bh < 0.05)]


def mark_partial(ctx: Ctx, base: list[str], rows: list[Row]) -> None:
    """A row with ``partial`` becomes a subset row if it lacks a cell of the intersection of ``base`` and of the rows
    that are neither subset nor partial; otherwise it covers that intersection and joins it (which leaves it as it is)."""
    joint0 = ctx.eng.joint(base + [r.arm for r in rows if not r.subset and not r.partial])
    for r in rows:
        if r.partial and not r.subset:
            r.subset = bool(joint0 - ctx.eng.elig(r.arm))


def compare_block(ctx: Ctx, table: str, prefix: str, ref: str, rows: list[Row], direction: str = "ref-row",
                  family: str = "") -> tuple[set, list]:
    present = []
    nseeds = len(ctx.ds.config.seeds)
    for r in rows:
        if ctx.has(r.arm):
            spec = ctx.ds.arm(r.arm).spec
            # an arm that covers fewer seeds than the run (10-seed arms) is paired inside the
            # matched set and never shrinks it; stored arms carry the dagger
            r.subset = r.subset or len(spec.seeds) < nseeds
            r.stored = r.stored or spec.kind != "cr"
            present.append(r)
        else:
            ctx.skip(table, r.arm, "arm absent (row omitted)")
    if present and all(r.subset for r in present):
        for r in present:          # every row covers the same reduced seed set: they form the matched set
            r.subset = False
    if not ctx.has(ref):
        ctx.skip(table, ref, "reference arm absent (table omitted)")
        return set(), []
    mark_partial(ctx, [ref], present)
    joint = ctx.eng.joint([ref] + [r.arm for r in present if not r.subset])
    res = []
    for r in present:
        cells = joint if not r.subset else (joint & ctx.eng.elig(r.arm))
        a, b = (ref, r.arm) if direction == "ref-row" else (r.arm, ref)
        pr = ctx.eng.paired(a, b, cells)
        res.append({"row": r, "cells": cells, "pr": pr, "a": a, "b": b,
                    "lev_row": ctx.eng.level(r.arm, cells) if cells else float("nan"),
                    "lev_ref": ctx.eng.level(ref, cells) if cells else float("nan")})
    for x, q in zip(res, stats.bh([x["pr"].p for x in res])):
        x["pr"].p_bh = q
    for x, q in zip(res, stats.bh([x["pr"].p_exact for x in res])):
        x["pr"].p_bh_exact = q
    fam = family or f"the {len(res)} comparisons of this table"
    for x in res:
        r, pr, cells = x["row"], x["pr"], x["cells"]
        p = prefix + r.k()
        ctx.reg_level(p + "Mean", r.arm, cells, table)
        out = ctx.reg_paired(p, pr, x["a"], x["b"], cells, table)
        ctx.reg_pbh(p, pr, out["_fs"], table, fam)
    if joint:
        ctx.reg_level(prefix + camel(ref) + "RefMean", ref, joint, table)
    ctx.num(prefix + "MatchedN", len(joint), N.intfmt(len(joint)), kind="n", table=table,
            fileset=ctx.fs({a: joint for a in [ref] + [r.arm for r in present if not r.subset]}),
            n=len(joint), rule="joint intersection of eligible cells of all non-subset arms of the table")
    return joint, res


def _bold_best(values: list[float]) -> list[bool]:
    arr = np.asarray([v if np.isfinite(v) else -np.inf for v in values])
    if not np.isfinite(arr).any():
        return [False] * len(values)
    mx = arr.max()
    return [bool(np.isfinite(v) and abs(v - mx) < 5e-5) for v in arr]


#: Table 1 footnote: every abbreviation of the row labels, with the definition the code implements (RUN_RULES
#: 2026-10-22; sources: crlib/labels.py, src/spherevo/gp_competitors.py, src/spherevo/published_recipes.py)
GLOSSARY = ("Abbreviations: DSP prior is the dimension-scaled lengthscale prior "
            "$\\mathrm{LogNormal}(\\sqrt2+\\tfrac12\\ln D,\\sqrt3)$ of Hvarfner et al. $\\ell_0$ is the initial "
            "lengthscale before the maximum-likelihood fit. EI, LogEI and qLogEI are expected improvement, its "
            "logarithm (analytic) and its Monte Carlo batch version. Candidate-set argmax: the acquisition is "
            "maximized over a finite set of random candidates on the support. Sample-around-best: L-BFGS-B restarts "
            "include points sampled around the best observations. SAAS is the sparse axis-aligned subspace prior. "
            "MSR and MLE (scaled) are the two recipes of Papenmeier et al., with maximum-likelihood lengthscales "
            "initialized at $0.1\\sqrt D$. MSR adds RAASP perturbations in the acquisition maximizer, and MLE (scaled) "
            "does not. The $[p_{01},p_{99}]$ box spans the per-coordinate 1st to 99th percentile of encoded ZINC "
            "latents. $\\bar r$ is the shell radius, the mean norm of the run's five cold-start latents. $S^{D-1}$ is "
            "the unit sphere.")


#: RUN_RULES 2026-10-29 (1): GP-recipe rows of the full Table 1 that the printed table leaves out (Table 2 prints their
#: supports, or the text does not name them): the box rows of these recipes (ctx.box) and these shell arms
MAIN_NOT_PRINTED_BOX = ("papenmeier_msr", "papenmeier_mle", "xu")
MAIN_NOT_PRINTED_SHELL = ("papenmeier_msr", "xu_standard_gp")
#: RUN_RULES 2026-11-01 (authors' decision): the printed Table 1 in the layout of the submitted version. (arm, or
#: "key:" + row key of a published recipe; printed name; citation keys). Rows of the submitted version, then the rows
#: the rebuttal promised (C-31 Linear BO) or posted (the GP on the shell, Linear BO on the shell); COWBOYS is not a
#: published method (ROZ-C). Every other comparator row stays in the full table of the records.
#: Authors' decision of 2026-11-02: LOL-BO and InvBO are not printed (the paper keeps the VAE fixed, so methods that
#: update it are outside the comparison; both rows stay in the full table); CMA-ES, TuRBO and BAxUS print the runs of
#: this version instead of the stored runs: the rerun of 2026-11-02 (root CRRD; CMA-ES with population 5, the population
#: of GeoCMA) and the TuRBO-1 recipe of the published-recipe block. The stored rows stay in the full table.
#: Authors' decision of 2026-11-04 (RUN_RULES 2026-11-04 15:30): the Linear BO port on the shell (linear_bo_sphere_calib,
#: posted in the rebuttal) is not printed; it stays in the full table, in the trajectory table and the convergence figure
#: of Appendix D and in the overview of Appendix H. The caption of Table 1 no longer states the partial coverage of a
#: row (Section 3.1 and the notes to Table 1 do).
#: Authors' decision of 2026-11-06 (RUN_RULES 2026-11-06): the GP on the shell (hvarfner_scaled_gp) is the GP of the
#: paper's own on-sphere scaffold, not a baseline, so it is not a printed row of Table 1. Table 2 prints it beside its
#: box version; Tables 4, 9, 12 and 15 and the convergence figure keep it as the comparator of GeoWalk.
PRINTED_MAIN = (("cmaes_full_rd_pop5", "CMA-ES", "hansen2001cmaes,hansen2019pycma"),
                ("key:PubTurbo", "TuRBO", "eriksson2019turbo"),
                ("baxus_cr", "BAxUS", "papenmeier2022baxus"),
                ("rembo", "REMBO", "wang2016rembo"), ("hesbo", "HeSBO", "nayebi2019hesbo"),
                ("key:PubHvarfner", "Vanilla BO", "hvarfner2024vanilla"),
                ("key:PubMsr", "Papenmeier MSR", "papenmeier2025understanding"),
                ("key:PubXu", "Xu Standard GP", "xu2025standardgp"), ("key:PubLinear", "Linear BO", "doumont2026still"),
                ("gabo", "GaBO", "jaquier2020gabo"), ("matern_gabo", "Riemannian Mat\\'ern GP", "jaquier2022geometry"))
#: the GP on the shell: the comparator of Result 1, printed in Table 2 (not in Table 1 since 2026-11-06)
SHELL_GP, SHELL_GP_NAME = "hvarfner_scaled_gp", "GP on the shell"
PRINTED_MAIN_THIS_PAPER = (("random_search", "Random Search (shell)"), ("cma_es_subspace", "\\geocma{}"))
#: the full Table 1 in the repository records (records/ of a run, committed as docs/baselines/)
TABLE1_FULL_MD, TABLE1_FULL_TEX = "TABLE1_FULL.md", "tab_main_full.tex"
_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def _count_word(k: int) -> str:
    return _WORDS.get(k, str(k))


#: the glossary of the printed Table 1 notes (RUN_RULES 2026-10-29 (4)): plain sentences, the DSP prior written as in
#: Table 13; the full table in the records keeps GLOSSARY
PRINTED_GLOSSARY = ("Abbreviations: DSP prior is the dimension-scaled lengthscale prior "
                    "$\\mathrm{LogNormal}(\\sqrt{2}+0.5\\ln D,\\sqrt{3})$ of Hvarfner et al. The value $\\ell_0$ is the "
                    "initial lengthscale before the maximum-likelihood fit. EI, LogEI and qLogEI are expected improvement, its "
                    "logarithm (analytic) and its Monte Carlo batch version. Candidate-set argmax: the acquisition is "
                    "maximized over a finite set of random candidates on the support. Sample-around-best: L-BFGS-B "
                    "restarts include points sampled around the best observations. SAAS is the sparse axis-aligned "
                    "subspace prior. MSR and MLE (scaled) are the two recipes of Papenmeier et al., with "
                    "maximum-likelihood lengthscales initialized at $0.1\\sqrt{D}$. MSR adds RAASP perturbations (random "
                    "axis-aligned subspace perturbations) in the acquisition maximizer, and MLE (scaled) does not. The "
                    "$[p_{01},p_{99}]$ box spans the per-coordinate 1st to 99th percentile of encoded ZINC latents. "
                    "The radius $\\bar r$ of the shell is the mean norm of the run's five cold-start latents. The set "
                    "$S^{D-1}$ is the unit sphere.")
#: plain names of the GuacaMol task codes (the 11 tasks of the protocol; GuacaMol benchmark names)
TASK_KEY = (("adip", "Amlodipine MPO"), ("dhop", "Deco Hop"), ("fexo", "Fexofenadine MPO"), ("med1", "Median molecules 1"),
            ("med2", "Median molecules 2"), ("osmb", "Osimertinib MPO"), ("pdop", "Perindopril MPO"),
            ("rano", "Ranolazine MPO"), ("shop", "Scaffold Hop"), ("siga", "Sitagliptin MPO"), ("zale", "Zaleplon MPO"))


def filt_inline(ctx: Ctx) -> str:
    """The cell filter for a caption sentence, without nested parentheses (RUN_RULES 2026-10-29 (4))."""
    if ctx.eng.cell_filter == "completed":
        return "completed cells, as in Table~\\ref{tab:main}"
    return "clean cells (status completed, no surrogate-fallback step)"


def stat_defs(delta: str | None, family: str, tasks: str | None,
              ptask: str = "the 11 per-task mean differences") -> str:
    """Definitions of the paired columns, one short sentence per symbol, each starting with a word (third pass of
    RUN_RULES 2026-10-29 (4))."""
    out = (f"The difference $\\Delta$ is {delta}. " if delta else "")
    out += ("The effect size $d$ is Cohen's $d$ of the cell differences. The column $p_{\\mathrm{BH}}$ is the "
            f"two-sided Wilcoxon signed-rank test over the cells. It is Benjamini--Hochberg corrected {family}.")
    if tasks:
        out += (f" Tasks counts the tasks on which {tasks}. The column $p_{{\\mathrm{{task}}}}$ is the exact "
                f"two-sided Wilcoxon test on {ptask}, not corrected.")
    return out


def same_n_sentence(ctx: Ctx, prefixes: list, what: str = "All blocks") -> str:
    """'All blocks use the same n = 440 completed cells, as in Table 1.' when the matched sets of the blocks have one
    size, else ''."""
    ns = {ctx.reg.records[p + "MatchedN"]["text"] for p in prefixes if p + "MatchedN" in ctx.reg.records}
    return f"{what} use the same $n={ns.pop()}$ {filt_inline(ctx)}." if len(ns) == 1 else ""


def geowalk_vs_shell_gp_reading(pr) -> str:
    """RUN_RULES 2026-10-14 (3) and the analysis plan (2): the primary test (exact two-sided Wilcoxon on the per-task
    mean differences) decides; the leading arm is the subject; otherwise 'no detected difference at N = 30'."""
    if np.isfinite(pr.p_task) and pr.p_task < 0.05:
        lead = "\\geowalk{}" if pr.delta_taskmean > 0 else "the GP on the shell"
        won = pr.tasks_won if pr.delta_taskmean > 0 else pr.tasks_lost
        return (f"{lead} is ahead (higher on {won} of the {pr.n_tasks} tasks; exact task-level Wilcoxon "
                f"$p={N.psig(pr.p_task)}$)")
    return (f"no detected difference at $N=30$ (exact task-level Wilcoxon $p={N.psig(pr.p_task)}$; "
            f"\\geowalk{{}} higher on {pr.tasks_won} of the {pr.n_tasks} tasks)")


def partial_why(ctx: Ctx, arm: str) -> str:
    """Why a row covers only part of the matched set: 'k seeds' for an arm run on fewer seeds than the run, else the
    runs of the arm that did not complete ('one run crashed'), counted in the loaded files."""
    A = ctx.ds.arm(arm)
    if len(A.spec.seeds) < len(ctx.ds.config.seeds):
        return f"{len(A.spec.seeds)} seeds"
    bad = [c.status for c in A.cells.values() if not c.status_ok] + ["crashed"] * len(A.sidecar_crashes)
    k = len(bad)
    what = "crashed" if bad and all(b == "crashed" for b in bad) else "did not complete"
    return f"{_count_word(k)} run{'s' if k != 1 else ''} {what}"


# =========================================================================================
# Table 1
# =========================================================================================
def tab_main(ctx: Ctx) -> Table | None:
    table = "tab_main"
    v = ctx.var
    bl = BOX_LABEL[v.box]
    pm, pl = ctx.box("papenmeier_msr"), ctx.box("papenmeier_mle")
    pap_rows = [Row(pm, ctx.label(pm, LB.label(pm)), ctx.support(pm, bl), key="PapenmeierMsr"),
                Row(pl, ctx.label(pl, LB.label(pl)), ctx.support(pl, bl), key="PapenmeierMle")]
    if v.papenmeier_main == "mle_scaled":
        pap_rows = pap_rows[::-1]
    groups = [
        ("Box-supported, embedding and external baselines", [
            Row("lolbo", LB.label("lolbo"), "latent trust region", stored=True),
            Row("cmaes_full", LB.label("cmaes_full"), "unbounded $\\mathbb{R}^D$", stored=True),
            # the rerun of 2026-11-02 (root CRRD): the stored CMA-ES ported to the current harness, two populations
            Row("cmaes_full_rd", LB.label("cmaes_full_rd"), "unbounded $\\mathbb{R}^D$"),
            Row("cmaes_full_rd_pop5", LB.label("cmaes_full_rd_pop5"), "unbounded $\\mathbb{R}^D$"),
            Row("invbo", LB.label("invbo"), "latent trust region", stored=True),
            Row("turbo", LB.label("turbo"), "box of observed latents", stored=True),
            Row("turbo_box_faithful", LB.label("turbo_box_faithful"), "fixed box $[p_{01},p_{99}]$"),
            Row("baxus", LB.label("baxus"), "nested subspaces, box", stored=True),
            # the rerun of 2026-11-02: one of its 440 runs crashed (a terminal solver error), so it covers 439 cells; it
            # stays out of the intersection and is paired on the cells it covers, like a 10-seed row
            Row("baxus_cr", LB.label("baxus_cr"), "nested subspaces, box $[p_{01},p_{99}]$", partial=True),
            Row("saasbo", LB.label("saasbo"), "box $[-5,5]^D$", stored=True, subset=True),
            Row("rembo", LB.label("rembo"), "embedding, lifted to $\\bar r$"),
            Row("hesbo", LB.label("hesbo"), "embedding, lifted to $\\bar r$"),
        ] + ([Row("cowboys", LB.label("cowboys"), "Gaussian, $\\|z\\|\\approx 63$", stored=True, subset=True)]
             if v.cowboys else [])),
        ("GP recipes on the box (inputs rescaled to $[0,1]^D$)", [
            Row(ctx.box("hvarfner"), LB.label(ctx.box("hvarfner")), bl, key="VanillaBox"),
            *pap_rows,
            Row(ctx.box("xu"), LB.label(ctx.box("xu")), bl, key="XuBox"),
            Row("linear_bo", LB.label("linear_bo"), "$[p_{01},p_{99}]$", key="LinearBox"),
        ]),
    ]
    if ctx.mode == "cr":
        # published recipes (RUN_RULES 2026-10-12 (4)): named only if verified (crlib/published.py)
        from . import published
        pub = [Row(a, nm, published.support(a), key=k) for a, k, nm in published.guacamol_rows(ctx)]
        if pub:
            groups.append(("Published recipes (released-code settings)", pub))
    groups += [
        ("On the empirical shell or on the sphere manifold", [
            Row("hvarfner_scaled_gp", LB.label("hvarfner_scaled_gp"), "shell", key="VanillaSphere"),
            Row("papenmeier_msr", LB.label("papenmeier_msr"), "shell", key="PapenmeierSphere"),
            Row("xu_standard_gp", LB.label("xu_standard_gp"), "shell", key="XuSphere"),
            Row("linear_bo_sphere_calib", LB.label("linear_bo_sphere_calib"), "shell", key="LinearSphere"),
            Row("gabo", LB.label("gabo"), "$S^{D-1}$, trust region"),
            Row("matern_gabo", LB.label("matern_gabo"), "$S^{D-1}$, trust region", key="MaternGabo"),
        ]),
        ("This paper, on the empirical shell", [
            Row("random_search", "Random Search", "shell"),
            Row("cma_es_subspace", "\\geocma{}", "shell", key="Geocma"),
        ]),
    ]
    rows = [r for _, g in groups for r in g]
    # RUN_RULES 2026-10-29 (1): every statistic is computed over the FULL row set (one matched set, one BH family);
    # the printed table is a view of it, and the full table is a repository record (records/tab_main_full.tex,
    # records/TABLE1_FULL.md)
    joint, res = compare_block(ctx, table, "crMain", "geowalk", rows, "ref-row",
                               family="every comparator row of Table 1")
    if not res:
        return None
    byarm = {x["row"].arm: x for x in res}
    levels = [x["lev_row"] for x in res] + [ctx.eng.level("geowalk", joint)]
    bold = dict(zip([x["row"].arm for x in res] + ["geowalk"], _bold_best(levels)))
    hdr = (["Method", "Support", "$n$", "Mean", "$\\Delta$", "W/L", "$d$", stack("$p_{\\mathrm{BH}}$", "(cells)")]
           + ([stack("Tasks", "\\geowalk{}", "higher"), stack("$p_{\\mathrm{task}}$", "(uncorr.)")] if ctx.var.tasklevel
              else []))
    # legibility (RUN_RULES 2026-10-22): the method and support columns wrap, nothing is scaled
    t = Table("main", "tab:main", "", WRAP + wrapp("1.55cm") + "r" * (len(hdr) - 2), hdr, size="\\scriptsize",
              wide=True, tabcolsep="2.5pt")
    n_tasks = None
    for gname, g in groups:
        t.group(gname)
        for r in g:
            x = byarm.get(r.arm)
            if x is None:
                continue
            pr = x["pr"]
            n_tasks = n_tasks or pr.n_tasks
            lab = (no_stored_suffix(r.label) + "$^\\dagger$") if r.stored else r.label
            t.add([lab, r.support, N.intfmt(pr.n), mb(N.f4(x["lev_row"]), bold[r.arm]), m(N.s4(pr.delta)),
                   f"{pr.wins}/{pr.losses}", m(N.f2(pr.d)), m(N.pfmt(pr.p_bh))] + tl_vals(ctx, pr))
    t.rule()
    gw = ctx.eng.level("geowalk", joint)
    t.add(["\\geowalk{}", "shell", N.intfmt(len(joint)), mb(N.f4(gw), bold["geowalk"]), "--", "--", "--", "--"]
          + (["--", "--"] if ctx.var.tasklevel else []))
    sub = [x for x in res if x["row"].subset]
    stored_n = sorted({x['pr'].n for x in res if x['row'].stored and not x['row'].subset})
    sub_txt = "; ".join(f"{no_stored_suffix(x['row'].label)}: $n={x['pr'].n}$, {partial_why(ctx, x['row'].arm)}"
                        for x in sub)
    # Table 1 is a float page with a short caption; the column definitions and the notes on the rows go to a
    # generated text block that the kit sets in Appendix H (tab_main_notes.tex; RUN_RULES 2026-10-22, format only)
    t.placement = "p"
    t.caption = ("\\textbf{Cross-task means and paired statistics for \\geowalk{} against every comparator at $N=30$} "
                 f"on one matched set of $n={N.intfmt(len(joint))}$ (task, seed) cells. "
                 f"$^\\dagger$~{SUBMITTED_SHORT[0].upper() + SUBMITTED_SHORT[1:]}. "
                 "Column definitions, notes on the rows and abbreviations are in Appendix~\\ref{app:baselines} "
                 "(notes to Table~\\ref{tab:main}).")
    completed = ctx.eng.cell_filter == "completed"
    defs = (
        f"All rows are evaluated on one matched set of $n={len(joint)}$ (task, seed) cells. It is the intersection of "
        + ("the completed cells of every row except the rows that cover only part of it. A completed cell is a run that "
           "made all 25 proposals after the 5 cold starts; runs in which a surrogate fit failed and a step fell back to "
           "a uniform proposal are included. "
           if completed else f"the {filt_text(ctx)} of every row except the rows that cover only part of it. ")
        + f"The rows that cover only part of the set use the part they cover ({sub_txt or 'none'}). "
        f"Mean = {conv_text(ctx)}; bold marks the highest. $\\Delta$ = \\geowalk{{}} minus the row on the same cells, "
        f"as a {conv_text(ctx)} of the paired differences. W/L = cells won/lost by \\geowalk{{}}, ties excluded. "
        "$d$ = mean/sd of the cell differences. $p_{\\mathrm{BH}}$ (cells) = two-sided Wilcoxon signed-rank test "
        f"over the cells, Benjamini--Hochberg corrected over the {len(res)} rows."
        + (f" Tasks = tasks, of the {n_tasks}, on which \\geowalk{{}} has the higher mean. $p_{{\\mathrm{{task}}}}$ "
           "(uncorr.) = exact two-sided Wilcoxon test on the per-task mean differences. It is the primary test of the "
           "paper and is printed here without correction." if ctx.var.tasklevel else ""))
    over = [x["row"] for x in res if ctx.overridden(x["row"].arm)]
    exc = []
    if v.box == "minmax" and "linear_bo" in byarm:
        exc.append("the spherical linear-kernel row stays on $[p_{01},p_{99}]$ (no $[\\min,\\max]$ run exists)")
    # W4: the verified Linear BO recipe runs in the authors' own box (crlib/published.py GUACAMOL_SUPPORT)
    for x in (res if ctx.mode == "cr" else []):         # published rows exist in camera-ready mode only
        sup = published.support(x["row"].arm)
        if x["row"].key.startswith("Pub") and sup.startswith("box ") and sup != f"box {bl}":
            exc.append(f"{x['row'].label} runs in {published.support(x['row'].arm).replace('box ', 'the box ')}"
                       + (" of the authors' released code" if x["row"].arm == "linear_bo_box8" else ""))
    for r in over:
        exc.append(f"the row labelled ``{r.label}'' reads substitute files ({r.support}), not the arm's own method")
    pub_rows = [x for x in res if x["row"].key.startswith("Pub")]
    notes = ("$^\\dagger$ Run from the submitted version, not rerun for this version: it ran through a separate "
               "harness, or no current environment reproduces it. The $n$ of such a row is the number of its cells "
               "inside the matched set. "
               "Rows other than the published recipes are named by kernel, acquisition and maximizer, not by the "
               "method they adapt. Their deviations from the published methods are summarized in "
               "Table~\\ref{tab:baselines} and listed in full in the method cards released with the code "
               "(\\texttt{docs/baselines/}). "
               + ("The trust-region row on the fixed box was added for this version next to the trust-region row of "
                  "the submitted version. " if "turbo_box_faithful" in byarm else "")
               + ("Published recipes follow the authors' released code and carry the authors' names only if they "
                  "reproduced a published result (Table~\\ref{tab:verification}). Their comparison with \\geowalk{} "
                  "and the GP on the shell is Table~\\ref{tab:published}. " if pub_rows else "")
               + f"Box rows use the {bl} box." + (" Exception: " + "; ".join(exc) + "." if exc else "") + " "
               "A cell with $|\\mathrm{diff}|\\le10^{-6}$ is a tie: it is excluded from W/L and enters the cell-level "
               "Wilcoxon test as a zero difference. " + GLOSSARY)
    t.extra = {"tab_main_notes.tex": "\\paragraph{Notes to Table~\\ref{tab:main}.} " + defs + " " + notes}
    full = t
    # ---- RUN_RULES 2026-11-01: the printed Table 1 has the layout of the submitted version (authors' decision) ----
    # rows named and ordered as in the submitted version plus the rows the rebuttal promised or posted; columns
    # Cross-mean, Delta, W/L, p_task, d. Every value is the one of the full table above (same matched set, same
    # family); the other rows go to the full table in the records only
    bykey = {x["row"].key: x for x in res if x["row"].key}
    shown_spec = []
    for ref, name, cite in PRINTED_MAIN:
        x = bykey.get(ref[4:]) if ref.startswith("key:") else byarm.get(ref)
        if x is not None:
            shown_spec.append((x, name, cite))
    this_spec = [(byarm[a], name, "") for a, name in PRINTED_MAIN_THIS_PAPER if a in byarm]
    shown_arms = {x["row"].arm for x, _, _ in shown_spec + this_spec}
    drop = {x["row"].arm for x in res} - shown_arms
    dropped = [x for x in res if x["row"].arm in drop]
    shown = [x for x, _, _ in shown_spec + this_spec]
    # 2026-11-03 (authors' decision): Table 3 prints the best other method of each benchmark; its GuacaMol entry
    # is the highest cross-mean among the comparators printed here, on this matched set. Since 2026-11-06 the GP on
    # the shell is printed in Table 2, not here, and stays a candidate, so that the entry does not change.
    ctx.extra["main_printed"] = ([(name, x["row"].arm, x["lev_row"], x["cells"]) for x, name, _ in shown_spec]
                                 + [(SHELL_GP_NAME, x["row"].arm, x["lev_row"], x["cells"])
                                    for x in res if x["row"].arm == SHELL_GP and x["row"].arm not in shown_arms])
    pbold = dict(zip([x["row"].arm for x in shown] + ["geowalk"],
                     _bold_best([x["lev_row"] for x in shown] + [ctx.eng.level("geowalk", joint)])))
    # the p column is the task-level test (primary test, row 2026-10-14 (3)); the variant without task-level statistics
    # prints the cell-level BH-corrected p of the full table instead
    tl = ctx.var.tasklevel
    pcol = "$p_{\\mathrm{task}}$" if tl else "$p_{\\mathrm{BH}}$"
    phdr = ["Method", "Cross-mean", "$\\Delta$", "W/L", pcol, "$d$"]
    t = Table("main", "tab:main", "", WRAP + "r" * (len(phdr) - 1), phdr, size="\\small", wide=True)
    t.placement = "!t"
    sub_mark = "\\ddagger"

    def marks_of(x):
        r = x["row"]
        return (("\\dagger" if r.stored else "") + (sub_mark if r.subset else "")
                + ("\\S" if r.key.startswith("Pub") else ""))

    def prow(x, name, cite):
        pr, mk = x["pr"], marks_of(x)
        lab = name + (f"$^{{{mk}}}$" if mk else "") + (f"~\\citep{{{cite}}}" if cite else "")
        return [lab, mb(N.f4(x["lev_row"]), pbold[x["row"].arm]), m(N.s4(pr.delta)), f"{pr.wins}/{pr.losses}",
                m(N.psig(pr.p_task)) if tl else m(N.pfmt(pr.p_bh)), m(N.f2(pr.d))]

    for x, name, cite in shown_spec:
        t.add(prow(x, name, cite))
    t.rule()
    t.group("This paper, on the empirical shell (Eq.~\\ref{eq:on_sphere_scaffold}):")
    for x, name, cite in this_spec:
        t.add(prow(x, name, cite))
    gw = ctx.eng.level("geowalk", joint)
    t.add(["\\geowalk{}", mb(N.f4(gw), pbold["geowalk"]), "--", "--", "--", "--"])
    sub = [x for x in shown if x["row"].subset]   # RUN_RULES 2026-11-02 (1): only the rows that are printed
    printed_name = {x["row"].arm: name for x, name, _ in shown_spec + this_spec}
    psub_txt = "; ".join(f"{printed_name.get(x['row'].arm, no_stored_suffix(x['row'].label))}: $n={x['pr'].n}$, "
                         f"{partial_why(ctx, x['row'].arm)}" for x in sub)
    # authors' decision of 2026-11-02: the mark of a stored run and its sentences are printed only while a printed row
    # is a stored run
    shown_stored = any(x["row"].stored for x in shown)
    # the completed-cell sentence states 30 evaluations only if every completed cell of every printed row (and of
    # GeoWalk) recorded 30 of them (the stored runs of the full table did not evaluate a repeated molecule again)
    all30 = all(c.n_evaluated == 30 for a in [x["row"].arm for x in shown] + ["geowalk"]
                for c in ctx.ds.arm(a).cells.values() if c.status_ok)
    def _the(n):
        return "the " + n if n.startswith("GP on") else n

    def _and(ns):
        return ", ".join(ns[:-1]) + (" and " if len(ns) > 1 else "") + ns[-1] if ns else ""

    emb = [n for x, n, _ in shown_spec if x["row"].support.startswith("embedding")]
    shl = [_the(n) for x, n, _ in shown_spec if x["row"].support == "shell"]
    man = [n for x, n, _ in shown_spec if x["row"].support.startswith("$S^{D-1}$")]
    where = ((f"Candidates lie on the empirical shell for {_and(emb)} (embeddings lifted to the shell radius)"
              + (f" and for {_and(shl)}, which {'maximize' if len(shl) > 1 else 'maximizes'} the acquisition over a candidate set" if shl else "") + ". "
              if emb else (f"Candidates lie on the empirical shell for {_and(shl)}. " if shl else ""))
             + (f"{_and(man)} propose on the sphere manifold. " if man else "")
             + ("All other comparators propose off the shell. " if emb or shl or man else ""))
    t.caption = ("\\textbf{Cross-task means and paired statistics for \\geowalk{} versus all comparators at $N=30$} "
                 f"on one matched set of $n={N.intfmt(len(joint))}$ (task, seed) cells of the {n_tasks} GuacaMol tasks. "
                 "Rows marked $^\\S$ run the authors' released code. The other rows adapt the cited method to this "
                 "benchmark, and Appendix~\\ref{app:baselines} lists their deviations. "
                 + where
                 + ("$^\\dagger$~Stored run (Section~\\ref{sec:setup}). " if shown_stored else "")
                 + "$\\Delta$ is the improvement of \\geowalk{} over the comparator. W/L counts the matched cells where "
                 "\\geowalk{} wins or loses, excluding ties. "
                 + (f"$p_{{\\mathrm{{task}}}}$ is the exact two-sided Wilcoxon test on the {n_tasks} per-task mean "
                    "differences, not corrected. " if tl else "$p_{\\mathrm{BH}}$ is the two-sided Wilcoxon test over the "
                    "cells, Benjamini--Hochberg corrected over every comparator of the full table. ")
                 + "$d$ is Cohen's $d$ of the paired cell "
                 "differences. Bold marks the highest mean. Notes are in Appendix~\\ref{app:baselines}.")
    pdefs = (
        f"All rows are evaluated on one matched set of $n={len(joint)}$ (task, seed) cells. It is the intersection of "
        + ("the completed cells of the rows of the full version of the table, except the rows that cover only part of "
           "it. A completed cell is a run that made all 25 proposals after the 5 cold starts."
           + (" In every row printed here, such a run reached all 30 evaluations." if all30 else "")
           + " Runs in which a surrogate fit failed and a step fell back to a uniform proposal are included. "
           if completed else f"the {filt_text(ctx)} of the rows of the full version of the table, except the rows that "
                             "cover only part of it. ")
        + ((f"The rows marked $^{sub_mark}$ cover only part of that set and use the part they cover ({psub_txt}). "
            if len(sub) > 1 else
            f"The row marked $^{sub_mark}$ covers only part of that set and uses the part it covers ({psub_txt}). ")
           if sub else "")
        + f"Cross-mean = {conv_text(ctx)}, and bold marks the highest. The difference $\\Delta$ is \\geowalk{{}} minus "
        f"the row on the same cells, as a {conv_text(ctx)} of the paired differences. W/L counts the cells won and "
        "lost by \\geowalk{}. A cell with $|\\mathrm{diff}|\\le10^{-6}$ is a tie and is excluded from W/L. The "
        "effect size $d$ is the mean of the cell differences divided by their standard deviation. The column "
        "$p_{\\mathrm{task}}$ is the exact two-sided Wilcoxon test on the per-task mean differences. It is the "
        "primary test of the paper and is printed here without correction."
        + " The full version of this table is released with the code (\\texttt{docs/baselines/"
        + TABLE1_FULL_MD.replace("_", "\\_") + "} and \\texttt{docs/baselines/" + TABLE1_FULL_TEX.replace("_", "\\_")
        + f"}}). It has all {len(res)} comparator rows under the names of their implementation, and the cell-level "
        f"Wilcoxon test, Benjamini--Hochberg corrected over the {len(res)} rows."
        + (f" The {len(dropped)} rows not printed here are "
           + "; ".join(x["row"].label.replace(" (submitted version)", " (stored run)") for x in dropped)
           + ". Table~\\ref{tab:boxsphere} compares the GP recipes on the box and on the shell." if dropped else ""))
    # the rows rerun for this version and their box (RUN_RULES 2026-10-29 (4), the sentence the authors fixed)
    pexc = []
    for x in (res if ctx.mode == "cr" else []):
        sup = published.support(x["row"].arm)
        if x["row"].key.startswith("Pub") and sup.startswith("box ") and sup != f"box {bl}" and x["row"].arm not in drop:
            pexc.append(f"{printed_name.get(x['row'].arm, x['row'].label)}, which runs in {sup[len('box '):]}")
    if v.box == "minmax" and "linear_bo" in byarm and "linear_bo" not in drop:
        pexc.append("the spherical linear-kernel row, which stays on $[p_{01},p_{99}]$ (no $[\\min,\\max]$ run exists)")
    names = "; ".join(f"{printed_name[x['row'].arm]} is ``{x['row'].label.replace(' (submitted version)', ' (stored run)')}''"
                      for x, _, _ in shown_spec if not x["row"].key.startswith("Pub"))
    if SHELL_GP in drop and SHELL_GP in byarm:    # 2026-11-06: the row is not printed, its name is used elsewhere
        names += (". The GP on the shell of Section~\\ref{sec:main_results}, which Table~\\ref{tab:boxsphere} prints "
                  f"with its box version, is ``{byarm[SHELL_GP]['row'].label}''")
    pnotes = (("The mark $^\\dagger$ denotes a stored run (Section~\\ref{sec:setup}). " if shown_stored else "")
              + ("The mark $^\\S$ denotes a published recipe. Published recipes follow the authors' released code. "
                 "Their comparison with \\geowalk{} and the GP on the shell is Table~\\ref{tab:published}. "
                 if pub_rows else "")
              + "The other rows adapt the cited method to this benchmark. Their deviations from the published methods "
              "are summarized in Table~\\ref{tab:baselines} and listed in full in the method cards released with the "
              "code (\\texttt{docs/baselines/}). These tables name each row by its implementation"
              + (f": {names}." if names else ".")
              + (f" Box rows not marked $^\\dagger$ use the {bl} box" if shown_stored else f" Box rows use the {bl} box")
              + (", except " + " and ".join(pexc) if pexc else "")
              + ". Table~\\ref{tab:baselines} gives the support of every row."
              + "".join(f" The row labelled ``{r.label}'' reads substitute files ({r.support}), not the arm's own "
                        "method." for r in over if r.arm not in drop)
              + " " + PRINTED_GLOSSARY)
    t.extra = {"tab_main_notes.tex": "\\paragraph{Notes to Table~\\ref{tab:main}.} " + pdefs + " " + pnotes}
    # the full table: the bytes of the table printed until 2026-10-29 (records/tab_main_full.tex and its notes) and its
    # Markdown (records/TABLE1_FULL.md); written by make_tables.run_variant
    full.name = "main_full"
    full.extra = {"tab_main_full_notes.tex": full.extra["tab_main_notes.tex"]}
    t.record_tables = [full]
    from .records import table1_markdown
    t.record_files = {TABLE1_FULL_MD: table1_markdown(full, [x["row"].arm not in drop for x in res] + [True])}
    gp = byarm.get("hvarfner_scaled_gp")
    if gp is not None:
        # GeoWalk against the GP on the shell (RUN_RULES 2026-10-14 (3): primary test, no TOST, never "equivalent")
        pr = gp["pr"]
        fs = ctx.fs({"geowalk": gp["cells"], "hvarfner_scaled_gp": gp["cells"]})
        sent = geowalk_vs_shell_gp_reading(pr)
        ctx.num("crMainVanillaSphereReading", sent, sent, kind="text", table=table, fileset=fs,
                n=pr.n_tasks, rule="RUN_RULES 2026-10-14 (3) and analysis plan (2): GeoWalk vs the GP on the shell by "
                                   "the primary test (exact two-sided Wilcoxon on the per-task mean differences, "
                                   "p < 0.05); the leading arm is the subject; otherwise 'no detected difference at "
                                   "N = 30'; never 'equivalent'")
    fs_all = ctx.fs_union(*[ctx.fs({x["a"]: x["cells"], x["b"]: x["cells"]}) for x in res])
    ctx.num("crMainRows", len(res), str(len(res)), kind="count", table=table, fileset=fs_all,
            rule="comparator rows in the Table 1 BH family (file set: union of the rows' paired sets)")
    sig = sum(1 for x in res if x["pr"].p_bh < 0.05)
    ctx.num("crMainSignificantRows", sig, str(sig), kind="count", table=table, fileset=fs_all,
            rule="rows with p_BH < 0.05")
    if n_tasks:
        ctx.num("crNTasksGuacamol", n_tasks, str(n_tasks), kind="n_tasks", table=table, fileset=fs_all,
                rule="tasks of the GuacaMol protocol (the n of every task-level test on GuacaMol)")
    ds = [x["pr"].d for x in res if np.isfinite(x["pr"].d)]
    if ds:
        ctx.num("crMainDMin", min(ds), N.f2(min(ds)), kind="cohens_d", table=table, fileset=fs_all, rule="min d over rows")
        ctx.num("crMainDMax", max(ds), N.f2(max(ds)), kind="cohens_d", table=table, fileset=fs_all, rule="max d over rows")
    return t


# =========================================================================================
# Factorial helpers (Table 2 block B, Appendix J)
# =========================================================================================
FACT_CELLS = ("SA", "SC", "BA", "BC")   # shell/box x candidate-set argmax / continuous


def factorial_arms(ctx: Ctx, gen: str) -> dict | None:
    if gen == "cr":
        # RUN_RULES 2026-11-06 16:30: the shell x continuous cell is L-BFGS-B on the shell (control C4 of 2026-10-14),
        # so both supports use the same continuous maximiser; the projected-Adam arm is read by no printed table
        arms = {"SA": "hvarfner_scaled_gp", "SC": "hvarfner_scaled_gp_sphere_lbfgs",
                "BA": ctx.box("factorial_argmax"), "BC": ctx.box("hvarfner")}
    elif gen == "e1":
        arms = {"SA": "e1_hvarfner_scaled_gp", "SC": "e1_hvarfner_scaled_gp_sphere_continuous",
                "BA": "e1_hvarfner_scaled_gp_box_argmax", "BC": "e1_hvarfner_scaled_gp_box_faithful"}
    elif gen == "posted":
        arms = {"SA": "posted:hvarfner_scaled_gp", "SC": "posted:hvarfner_scaled_gp_sphere_continuous",
                "BA": "posted:hvarfner_scaled_gp_box_argmax", "BC": "posted:hvarfner_scaled_gp_box_faithful"}
    else:
        raise ValueError(gen)
    if not all(ctx.has(a) for a in arms.values()):
        return None
    return arms


CONTRASTS = [
    ("SupportCand", "support at candidate-set argmax (shell $-$ box)", ("SA", "BA")),
    ("SupportCont", "support at continuous maximisation (shell $-$ box)", ("SC", "BC")),
    ("AcqShell", "maximiser on the shell (argmax $-$ continuous)", ("SA", "SC")),
    ("AcqBox", "maximiser in the box (argmax $-$ continuous)", ("BA", "BC")),
    ("Diagonal", "submitted comparison (shell argmax $-$ box continuous)", ("SA", "BC")),
    ("Interaction", "interaction $(SA-BA)-(SC-BC)$", None),
]


def factorial_stats(ctx: Ctx, arms: dict) -> dict:
    cells = ctx.eng.joint(list(arms.values()))
    vals = {k: ctx.eng.vals(a, cells) for k, a in arms.items()} if cells else {k: {} for k in arms}
    out = {"cells": cells, "levels": {k: (stats.level(vals[k], cells, ctx.var.conv) if cells else float("nan"))
                                     for k in arms}, "contrasts": {}}
    for key, _, pair in CONTRASTS:
        if pair:
            x, y = vals[pair[0]], vals[pair[1]]
        else:
            x = {c: (vals["SA"][c] - vals["BA"][c]) - (vals["SC"][c] - vals["BC"][c]) for c in cells}
            y = {c: 0.0 for c in cells}
        out["contrasts"][key] = ctx.eng.paired_values(x, y, cells)
    return out


def _reg_factorial(ctx: Ctx, table: str, prefix: str, arms: dict, fsd: dict, keys) -> None:
    cells = fsd["cells"]
    fs = ctx.fs({a: cells for a in arms.values()})
    names = {"SA": "ShellCand", "SC": "ShellCont", "BA": "BoxCand", "BC": "BoxCont"}
    for k in FACT_CELLS:
        ctx.num(prefix + names[k], fsd["levels"][k], N.f4(fsd["levels"][k]), kind="level", table=table,
                fileset=ctx.fs({arms[k]: cells}), n=len(cells), rule=f"{conv_text(ctx)} of best@30, 4-way matched set")
    ctx.num(prefix + "N", len(cells), N.intfmt(len(cells)), kind="n", table=table, fileset=fs, n=len(cells),
            rule="intersection of the eligible cells of the four factorial arms")
    for key in keys:
        pr = fsd["contrasts"][key]
        p = prefix + key
        ctx.num(p + "Delta", pr.delta, N.s4(pr.delta), kind="delta", table=table, fileset=fs, n=pr.n,
                rule=f"{conv_text(ctx)} of the paired cell contrast")
        ctx.num(p + "D", pr.d, N.d2(pr.d), kind="cohens_d", table=table, fileset=fs, n=pr.n, rule="cells")
        ctx.num(p + "P", pr.p, N.pfmt(pr.p), kind="p_raw", table=table, fileset=fs, n=pr.n,
                rule="two-sided Wilcoxon over cells")
        ctx.num(p + "PBH", pr.p_bh, N.pfmt(pr.p_bh), kind="p_bh", table=table, fileset=fs, n=pr.n,
                rule="BH within the contrasts of this generation")
        ctx.num(p + "WL", f"{pr.wins}/{pr.losses}", f"{pr.wins}/{pr.losses}", kind="wins_losses", table=table,
                fileset=fs, n=pr.n, rule="cells")
        if ctx.var.tasklevel:
            ctx.num(p + "Tasks", pr.tasks_won, f"{pr.tasks_won}/{pr.n_tasks}", kind="tasks_won", table=table,
                    fileset=fs, n=pr.n_tasks, rule="tasks with a positive per-task mean contrast")
            ctx.num(p + "PTask", pr.p_task, N.psig(pr.p_task), kind="p_task", table=table, fileset=fs,
                    n=pr.n_tasks, rule="exact Wilcoxon on per-task mean contrasts")


# =========================================================================================
# Table 2: within-recipe box vs sphere + the 2x2 factorial block
# =========================================================================================
MSR_PUBLISHED = {"faithful": "papenmeier_msr_published", "minmax": "papenmeier_msr_published_norm_box"}


class MsrGuardError(RuntimeError):
    """The row labelled 'MSR (published maximiser)' would not read the published-MSR arm."""


def msr_guard(ctx: Ctx, arm: str) -> None:
    """Guard of the Table 2 row 'Papenmeier, MSR (published maximiser)' (AC condition; SYNTHESIS G16).

    In a camera-ready run the row must read the arm ``papenmeier_msr_published`` (``..._norm_box`` in the
    [min,max] variant): a camera-ready arm (kind 'cr'), whose files carry solver = that name and
    provenance.cr_arm = that name, with no ``display`` override. Anything else raises: the July RAASP
    trust-region files that stand in for it in --legacy runs must never be printed under the MSR label.
    In legacy mode a substitute is allowed only if it declares its own label (``display``), which the row prints."""
    want = MSR_PUBLISHED[ctx.var.box]
    if arm != want:
        raise MsrGuardError(f"Table 2 MSR row resolves to {arm!r}, expected {want!r}")
    if not ctx.has(arm):
        return                      # absent arm: the row is omitted and the skip is recorded
    spec = ctx.ds.arm(arm).spec
    if ctx.mode == "cr":
        bad = []
        if spec.kind != "cr":
            bad.append(f"kind {spec.kind!r} (not a camera-ready arm)")
        if spec.display:
            bad.append("a display override (substitute files)")
        if spec.solver_field != want:
            bad.append(f"solver field {spec.solver_field!r}")
        if spec.cr_arm != want:
            bad.append(f"cr_arm {spec.cr_arm!r}")
        if not spec.pattern.startswith(f"ours_{want}_"):
            bad.append(f"file pattern {spec.pattern!r}")
        if bad:
            raise MsrGuardError(f"Table 2 MSR row: arm {arm} has " + "; ".join(bad)
                                + f" -- the production row must read {want}")
    elif spec.solver_field != want and not spec.display:
        raise MsrGuardError(f"legacy Table 2 MSR row reads {spec.solver_field!r} files without a display label")
#: The posted 2x2 factorial printed in Table 2 (RUN_RULES 2026-10-14, scope item 2): five contrasts, the primary test
#: of the analysis plan (exact two-sided Wilcoxon on the 11 per-task mean differences, tasks won) and Benjamini--Hochberg
#: within these five. (key, row label, pair or None = interaction, lead phrase if > 0, lead phrase if < 0)
FACTORIAL_FIVE = [
    ("SupportCand", "support under candidate-set argmax (shell $-$ box)", ("SA", "BA"),
     "the shell is ahead of the box under candidate-set argmax", "the box is ahead of the shell under candidate-set argmax"),
    ("SupportCont", "support under continuous maximization (shell $-$ box)", ("SC", "BC"),
     "the shell is ahead of the box under continuous maximization",
     "the box is ahead of the shell under continuous maximization"),
    ("AcqShell", "maximizer on the shell (argmax $-$ L-BFGS-B)", ("SA", "SC"),
     "candidate-set argmax is ahead of L-BFGS-B on the shell",
     "L-BFGS-B is ahead of candidate-set argmax on the shell"),
    ("AcqBox", "maximizer in the box (argmax $-$ L-BFGS-B)", ("BA", "BC"),
     "candidate-set argmax is ahead of L-BFGS-B in the box", "L-BFGS-B is ahead of candidate-set argmax in the box"),
    ("Interaction", "support $\\times$ maximizer interaction $(SA-BA)-(SC-BC)$", None,
     "the shell's advantage is larger under candidate-set argmax than under continuous maximization",
     "the shell's advantage is smaller under candidate-set argmax than under continuous maximisation"),
]


def factorial_five(ctx: Ctx, arms: dict) -> dict:
    """The five contrasts on the 4-way matched set, with BH over their task-level p (cached per variant)."""
    key = ("factorial_five", ctx.var.name, tuple(sorted(arms.items())))
    store = ctx.cache.setdefault("_factorial_five", {})
    if key in store:
        return store[key]
    cells = ctx.eng.joint(list(arms.values()))
    vals = {k: ctx.eng.vals(a, cells) for k, a in arms.items()} if cells else {k: {} for k in arms}
    out = {"cells": cells, "levels": {k: (stats.level(vals[k], cells, ctx.var.conv) if cells else float("nan"))
                                     for k in arms}, "contrasts": {}}
    for k, _, pair, _, _ in FACTORIAL_FIVE:
        if pair:
            x, y = vals[pair[0]], vals[pair[1]]
        else:
            x = {c: (vals["SA"][c] - vals["BA"][c]) - (vals["SC"][c] - vals["BC"][c]) for c in cells}
            y = {c: 0.0 for c in cells}
        out["contrasts"][k] = stats.paired(x, y, cells, "taskmean")     # the primary test is task-level
    q = stats.bh([out["contrasts"][k].p_task for k, *_ in FACTORIAL_FIVE])
    for (k, *_), qq in zip(FACTORIAL_FIVE, q):
        out["contrasts"][k].p_task_bh = qq
    store[key] = out
    return out


#: RUN_RULES 2026-10-19 (c) ("by a smaller margin"; commitments C-27, C-46; R-4): the sentence, and the submitted
#: Table 2 values of the Papenmeier row it is compared with (reference_values.json, the entries the dossier attaches as
#: the 'submitted' old values of crBoxPapenmeierSphereMean / crBoxPapenmeierBoxMean; source main.tex:212-213)
SMALLER_MARGIN_SENTENCE = ("the sphere-supported version still outperforms the corrected box baseline, but by a "
                           "smaller margin")
SUBMITTED_T2 = {"Papenmeier": ("S-t2-crBoxPapenmeierSphereMean", "S-t2-crBoxPapenmeierBoxMean")}
REFERENCE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference_values.json")


def submitted_t2_delta(key: str, path: str = REFERENCE_FILE) -> tuple[Decimal, list[dict]] | None:
    """Submitted shell - box difference of a Table 2 row, from the printed submitted values (exact decimals)."""
    if key not in SUBMITTED_T2 or not os.path.exists(path):
        return None
    by = {e["id"]: e for e in json.load(open(path))["entries"]}
    ids = SUBMITTED_T2[key]
    if not all(i in by for i in ids):
        return None
    sph, box = (by[i] for i in ids)
    return Decimal(str(sph["printed"])) - Decimal(str(box["printed"])), [sph, box]


def smaller_margin(delta_shell_minus_box: float, p_task_bh: float, delta_submitted) -> bool:
    """Clause (c): (i) the shell arm leads with a BH-corrected exact task-level p < 0.05 within the recipe-row family,
    and (ii) the shell - box difference (mean of per-task means; higher is better) is smaller than the submitted one,
    compared at the printed precision (four decimals, as Table 2 prints it): a tie at that precision is not smaller
    (RUN_RULES 2026-10-19 second set (d))."""
    lead = np.isfinite(delta_shell_minus_box) and delta_shell_minus_box > 0
    sig = np.isfinite(p_task_bh) and p_task_bh < 0.05
    return bool(lead and sig and Decimal(N.s4(delta_shell_minus_box)) < Decimal(str(delta_submitted)))


def factorial_reading(pr, pos: str, neg: str) -> tuple[str, str]:
    """(lead phrase, sentence): 'significant' only if the BH-corrected task-level p < 0.05 (checklist B.4);
    otherwise 'on k of the 11 tasks' without the word."""
    up = pr.delta_taskmean > 0
    lead = pos if up else neg
    k = pr.tasks_won if up else pr.tasks_lost
    sent = f"{lead} on {k} of the {pr.n_tasks} tasks"
    if np.isfinite(pr.p_task_bh) and pr.p_task_bh < 0.05:
        sent += (f"; significant after Benjamini--Hochberg over the five factorial contrasts (task-level "
                 f"$p={N.psig(pr.p_task_bh)}$)")
    return lead, sent


def tab_boxsphere(ctx: Ctx) -> Table | None:
    table = "tab_boxsphere"
    bl = BOX_LABEL[ctx.var.box]
    pap_box = ctx.box("papenmeier_msr") if ctx.var.papenmeier_main == "published" else ctx.box("papenmeier_mle")
    pap_max = ("LogEI, L-BFGS-B with sample-around-best" if ctx.var.papenmeier_main == "published"
               else "qLogEI, L-BFGS-B")
    if ctx.var.papenmeier_main == "published":
        msr_guard(ctx, pap_box)
    # recipes are named by their GP (print rule, crlib/labels.py): kernel and lengthscale treatment
    recipes = [("Mat\\'ern-5/2 GP (DSP prior)", "VanillaBO", "hvarfner_scaled_gp", ctx.box("hvarfner"), "qLogEI, L-BFGS-B"),
               ("Mat\\'ern-5/2 GP ($\\ell_0=0.1\\sqrt{D}$)", "Papenmeier", "papenmeier_msr", pap_box, pap_max),
               ("Mat\\'ern-5/2 GP ($\\ell_0=\\sqrt{D}$)", "XuStandardGP", "xu_standard_gp", ctx.box("xu"), "qLogEI, L-BFGS-B")]
    keep = []
    for r in recipes:
        miss = [a for a in r[2:4] if not ctx.has(a)]
        for a in miss:
            ctx.skip(table, a, f"arm absent ({r[1]} row omitted)")
        if not miss:
            keep.append(r)
    recipes = keep
    if not recipes:
        return None
    need = [a for r in recipes for a in r[2:4]]
    joint = ctx.eng.joint(need)
    res = []
    for lab, key, sph, box, maxl in recipes:
        pr = ctx.eng.paired(sph, box, joint)
        res.append((lab, key, sph, box, maxl, pr))
    gen = ctx.var.factorial
    farms = factorial_arms(ctx, gen)
    fsd = factorial_five(ctx, farms) if farms else None
    if fsd is None:
        ctx.skip(table, f"factorial ({gen})", "factorial arms absent (block omitted)")
    # two families: the recipe rows (R-6: BH within the table's recipe tests) and the five factorial contrasts
    # (RUN_RULES 2026-10-14: BH within these five, on the task-level p)
    q = stats.bh([r[5].p for r in res])
    for r, qq in zip(res, q):
        r[5].p_bh = qq
    for r, qq in zip(res, stats.bh([r[5].p_task for r in res])):       # the same family on the task-level test
        r[5].p_task_bh = qq
    hdr = (["GP", "Box version", "Shell", "Box", "$\\Delta$", "$d$", stack("$p_{\\mathrm{BH}}$", "(cells)")]
           + tl_cols(ctx, "shell|higher") + (["$p_{\\mathrm{task}}^{\\mathrm{BH}}$"] if ctx.var.tasklevel else []))
    t = Table("boxsphere", "tab:boxsphere", "", WRAP + wrapp("2.3cm") + "r" * (len(hdr) - 2), hdr,
              size="\\footnotesize", wide=True, tabcolsep="3pt")
    view = []               # RUN_RULES 2026-11-01: the rows of the printed view, in the layout of the submitted Table 2
    for lab, key, sph, box, maxl, pr in res:
        p = "crBox" + key
        ls, _ = ctx.reg_level(p + "SphereMean", sph, joint, table)
        lb, _ = ctx.reg_level(p + "BoxMean", box, joint, table)
        out = ctx.reg_paired(p, pr, sph, box, joint, table)
        ctx.reg_pbh(p, pr, out["_fs"], table, f"the {len(res)} recipe rows of Table 2")
        # legacy substitute files (e.g. the July RAASP arm standing in for the published MSR) are named by
        # what they are, never by the arm they replace
        boxcell = ctx.label(box, "") if ctx.overridden(box) else f"{bl}, {maxl}"
        view.append((lab, ls, lb, pr, ctx.label(box, "") if ctx.overridden(box) else bl))
        ctx.num(p + "PTaskBH", pr.p_task_bh, N.psig(pr.p_task_bh), kind="p_task_bh", table=table, fileset=out["_fs"],
                n=pr.n_tasks, rule=f"Benjamini-Hochberg within the {len(res)} recipe rows of Table 2, on the exact "
                                   "task-level p (RUN_RULES 2026-10-19 (b), (c))")
        # W8: bold marks only what the caption defines (BH-corrected task-level p < 0.05), never the higher mean
        t.add([lab, boxcell, m(N.f4(ls)), m(N.f4(lb)), m(N.s4(pr.delta)),
               m(N.f2(pr.d)), m(N.pfmt(pr.p_bh))] + tl_vals(ctx, pr)
              + ([mb(N.psig(pr.p_task_bh), np.isfinite(pr.p_task_bh) and pr.p_task_bh < 0.05)] if ctx.var.tasklevel else []))
        sub = submitted_t2_delta(key)
        if sub is not None:
            dsub, ents = sub
            fsr = ctx.fs_paths([REFERENCE_FILE])
            ctx.num(p + "SubmittedDelta", float(dsub), N.s4(float(dsub)), kind="delta", table=table, fileset=fsr,
                    rule=f"submitted Table 2 row: {ents[0]['printed']} - {ents[1]['printed']} ({ents[0]['source']}, "
                         f"{ents[1]['source']}; reference_values.json {ents[0]['id']}, {ents[1]['id']})")
            holds = smaller_margin(pr.delta_taskmean, pr.p_task_bh, dsub)
            fsm = ctx.fs_union(out["_fs"], fsr)
            ctx.num(p + "SmallerMarginHolds", "yes" if holds else "no", "yes" if holds else "no",
                    kind="text", table=table, fileset=fsm, n=pr.n_tasks,
                    rule="RUN_RULES 2026-10-19 (c): shell leads with BH task-level p < 0.05 within the recipe rows AND "
                         "shell - box (mean of per-task means) < the submitted difference")
            ctx.num(p + "SmallerMargin", SMALLER_MARGIN_SENTENCE if holds else "",
                    SMALLER_MARGIN_SENTENCE if holds else "", kind="text", table=table,
                    fileset=fsm, n=pr.n_tasks,
                    rule="the sentence of RUN_RULES 2026-10-19 (c) if it holds, else empty (nothing replaces it)")
    fs_box = ctx.fs({a: joint for a in need})
    ctx.num("crBoxN", len(joint), N.intfmt(len(joint)), kind="n", table=table, fileset=fs_box,
            n=len(joint), rule="joint intersection of the six arms")
    deltas = [r[5].delta for r in res]
    ds = [r[5].d for r in res]
    ctx.num("crBoxDeltaMin", min(deltas), N.f4(min(deltas)), kind="delta", table=table, fileset=fs_box, rule="min over recipes")
    ctx.num("crBoxDeltaMax", max(deltas), N.f4(max(deltas)), kind="delta", table=table, fileset=fs_box, rule="max over recipes")
    ctx.num("crBoxDMin", min(ds), N.f2(min(ds)), kind="cohens_d", table=table, fileset=fs_box, rule="min over recipes")
    ctx.num("crBoxDMax", max(ds), N.f2(max(ds)), kind="cohens_d", table=table, fileset=fs_box, rule="max over recipes")
    over = [r for r in res if ctx.overridden(r[3])]
    cap = ("\\textbf{Box versus empirical shell within each GP recipe, and the support $\\times$ "
           "acquisition-maximization factorial.} Both versions of a recipe use the same GP constructor. Box inputs "
           "are rescaled per coordinate, shell inputs are raw. "
           f"Top: the shell version of each GP (qLogEI, candidate-set argmax on the shell) against its box version "
           f"({bl}, rescaled to $[0,1]^D$"
           + "".join(f"; the {r[1]} row reads substitute files: {ctx.label(r[3], '')}" for r in over)
           + f"). The rows use one matched set of $n={len(joint)}$ cells. The two versions differ in support "
           "\\emph{and} in the acquisition maximizer. "
           # RUN_RULES 2026-10-29 (4): the box row of the 0.1 sqrt(D) recipe is the published MSR call (default analysis)
           + ("For $\\ell_0=0.1\\sqrt{D}$, the box version uses analytic LogEI with sample-around-best, the "
              "acquisition call of MSR. This row therefore also differs in the acquisition. "
              if ctx.var.papenmeier_main == "published" and any(r[1] == "Papenmeier" for r in res) else "")
           + f"$p_{{\\mathrm{{BH}}}}$ (cells) is the cell-level Wilcoxon test"
           + (" and $p_{\\mathrm{task}}^{\\mathrm{BH}}$ the exact task-level Wilcoxon test" if ctx.var.tasklevel else "")
           + f". Both are corrected by Benjamini--Hochberg over these {len(res)} rows"
           + stats.bh_note([r[5].p_task for r in res] if ctx.var.tasklevel else [r[5].p for r in res], "rows")
           + (". Tasks = tasks, of the 11, on which the shell version has the higher mean" if ctx.var.tasklevel else "")
           + ". ")
    if fsd:
        fc = fsd["cells"]
        fs = ctx.fs({a: fc for a in farms.values()})
        names = {"SA": "ShellCand", "SC": "ShellCont", "BA": "BoxCand", "BC": "BoxCont"}
        L = fsd["levels"]
        for k in FACT_CELLS:
            ctx.num("crFactorial" + names[k], L[k], N.f4(L[k]), kind="level", table=table,
                    fileset=ctx.fs({farms[k]: fc}), n=len(fc), rule=f"{conv_text(ctx)} of best@30, 4-way matched set")
        ctx.num("crFactorialN", len(fc), N.intfmt(len(fc)), kind="n", table=table, fileset=fs, n=len(fc),
                rule="intersection of the eligible cells of the four factorial arms")
        # RUN_RULES 2026-11-02 (4): share of distinct scores per cell of the factorial, descriptive (the Mechanism
        # paragraph that read them left the paper on 2026-11-06; the macros stay as a record)
        dist = {k: 100.0 * float(np.mean([len(set(ctx.ds.arm(farms[k]).cells[c].scores[:30].tolist())) / 30.0 for c in fc]))
                for k in FACT_CELLS}
        rest = [dist[k] for k in FACT_CELLS if k != "SC"]
        for nm, v in (("ShellCont", dist["SC"]), ("OtherMin", min(rest)), ("OtherMax", max(rest))):
            ctx.num("crFactorialDistinct" + nm, v, f"{v:.1f}", kind="level", table=table, fileset=fs, n=len(fc),
                    rule="mean over the cells of the share of distinct values among the 30 scores, percent")
        crow = []
        crow_view = []
        for k, lab, _, pos, neg in FACTORIAL_FIVE:
            pr = fsd["contrasts"][k]
            p = "crFactorial" + k
            lo, hi = pr.ci(0.95)
            ctx.num(p + "Delta", pr.delta_taskmean, N.s4(pr.delta_taskmean), kind="delta", table=table, fileset=fs,
                    n=pr.n, rule="mean of per-task means of the paired cell contrast")
            ctx.num(p + "CiLo", lo, N.s4(lo), kind="ci", table=table, fileset=fs, n=pr.n_tasks,
                    rule=f"lower end of the 95 % task-bootstrap interval (10,000 resamples, seed {stats.BOOT_SEED})")
            ctx.num(p + "CiHi", hi, N.s4(hi), kind="ci", table=table, fileset=fs, n=pr.n_tasks,
                    rule="upper end of the 95 % task-bootstrap interval")
            ctx.num(p + "Tasks", pr.tasks_won, f"{pr.tasks_won}/{pr.n_tasks}", kind="tasks_won", table=table,
                    fileset=fs, n=pr.n_tasks, rule="tasks with a positive per-task mean contrast")
            ctx.num(p + "PTask", pr.p_task, N.psig(pr.p_task), kind="p_task", table=table, fileset=fs, n=pr.n_tasks,
                    rule="exact two-sided Wilcoxon signed-rank test on the per-task mean contrasts (primary)")
            ctx.num(p + "PTaskBH", pr.p_task_bh, N.psig(pr.p_task_bh), kind="p_task_bh", table=table, fileset=fs,
                    n=pr.n_tasks, rule="Benjamini-Hochberg within the five factorial contrasts, on the task-level p")
            ctx.num(p + "P", pr.p, N.pfmt(pr.p), kind="p_raw", table=table, fileset=fs, n=pr.n,
                    rule="two-sided Wilcoxon over cells (secondary, uncorrected; |diff| <= 1e-6 = zero)")
            ctx.num(p + "D", pr.d, N.d2(pr.d), kind="cohens_d", table=table, fileset=fs, n=pr.n,
                    rule="Cohen's d of the cell contrasts (secondary)")
            ctx.num(p + "WL", f"{pr.wins}/{pr.losses}", f"{pr.wins}/{pr.losses}", kind="wins_losses", table=table,
                    fileset=fs, n=pr.n, rule="cells with contrast > +1e-6 / < -1e-6")
            lead, sent = factorial_reading(pr, pos, neg)
            ctx.num(p + "Reading", sent, sent, kind="text", table=table, fileset=fs, n=pr.n_tasks,
                    rule="checklist B.4 / RUN_RULES 2026-10-14 (2): 'significant' only if the BH-corrected task-level "
                         "p < 0.05, otherwise 'on k of the 11 tasks' without that word")
            crow_view.append([lab, m(N.s4(pr.delta_taskmean)), f"{pr.tasks_won}/{pr.n_tasks}",
                              mb(N.psig(pr.p_task_bh), np.isfinite(pr.p_task_bh) and pr.p_task_bh < 0.05)])
            crow.append([lab, m(N.s4(pr.delta_taskmean)), m(f"[{N.s3(lo)}, {N.s3(hi)}]"), f"{pr.tasks_won}/{pr.n_tasks}",
                         m(N.psig(pr.p_task)), mb(N.psig(pr.p_task_bh), np.isfinite(pr.p_task_bh) and pr.p_task_bh < 0.05),
                         m(N.pfmt(pr.p)), m(N.f2(pr.d))])
        top = max(L.values())
        src = {"cr": "runs for this version" if ctx.mode == "cr" else "legacy generation (posted rebuttal files)",
               "e1": "E-1 rerun (abbd61e, batch runner)", "posted": "posted generation"}[gen]
        # W8: the cell means of the 2x2 block carry no bold (the caption defines bold only for p_task^BH < 0.05)
        t.add_block(f"Support $\\times$ acquisition maximization, {src}, $n={len(fc)}$ matched cells", "lrr",
                    ["Support", "Candidate-set argmax", "Continuous (L-BFGS-B)"],
                    [["Empirical shell", m(N.f4(L["SA"])), m(N.f4(L["SC"]))],
                     [f"Box {bl}", m(N.f4(L["BA"])), m(N.f4(L["BC"]))]])
        t.add_block("The five factorial contrasts (primary: exact task-level Wilcoxon, Benjamini--Hochberg within the five)",
                    WRAP + "rrrrrrr", ["Contrast", "$\\Delta$", stack("95\\,\\% CI", "(tasks)"), stack("Tasks", "$\\Delta>0$"),
                                       "$p_{\\mathrm{task}}$", "$p_{\\mathrm{task}}^{\\mathrm{BH}}$", stack("$p$", "(cells)"),
                                       "$d$"], crow)
        cap += (f"Bottom: the 2$\\times$2 factorial ({src}). Continuous maximization is L-BFGS-B on both supports "
                f"(on the shell over a direction mapped to the shell). The matched set of the four cells has $n={len(fc)}$. Each of the five "
                "factorial contrasts is tested by the exact two-sided Wilcoxon test on the 11 per-task mean differences. "
                "The tests are corrected by Benjamini--Hochberg within the five factorial contrasts"
                + stats.bh_note([fsd["contrasts"][k].p_task for k, *_ in FACTORIAL_FIVE], "contrasts")
                + ". Tasks = tasks with a positive contrast. The 95\\,\\% interval is a percentile bootstrap over tasks. "
                "The cell-level $p$ and $d$ are secondary. ")
    cap += (f"Means are {conv_plural(ctx)} over {filt_inline(ctx)}. $\\Delta$ of the factorial contrasts is the mean of "
            "per-task means. Bold, in both blocks: $p_{\\mathrm{task}}^{\\mathrm{BH}}<0.05$ within the block's family.")
    t.caption = cap
    # ---- RUN_RULES 2026-11-01: the printed Table 2 has the layout of the submitted version (authors' decision) -------
    # Variant, Candidate support, Cross-mean, plus the task count and the corrected task-level p of each recipe; the
    # posted 2x2 block; the five contrasts with their task count and corrected p. Every value is one of the full table
    # above, which stays in the records with the cell-level statistics and the intervals
    full = t
    full.name = "boxsphere_full"
    tl = ctx.var.tasklevel
    vhdr = (["Variant", "Candidate support", "Cross-mean"]
            + ([stack("Tasks", "shell", "higher"), "$p_{\\mathrm{task}}^{\\mathrm{BH}}$"] if tl else []))
    t = Table("boxsphere", "tab:boxsphere", "", WRAP + "l" + "r" * (len(vhdr) - 2), vhdr, size="\\small", wide=True)
    for i, (lab, ls, lb, pr, boxsup) in enumerate(view):
        if i:
            t.rule()
        t.add([f"{lab}, shell", "Empirical shell", m(N.f4(ls))] + tl_vals_bh(ctx, pr))
        t.add([f"{lab}, box", boxsup, m(N.f4(lb))] + (["", ""] if tl else []))
    pcap = ("\\textbf{Box versus empirical shell within each GP recipe, and the support $\\times$ "
            "acquisition-maximization factorial.} "
            f"Top: each GP on the empirical shell (qLogEI, candidate-set argmax) and in the box {bl} (L-BFGS-B)"
            + "".join(f"; the {r[1]} row reads substitute files: {ctx.label(r[3], '')}" for r in over)
            + f", on one matched set of $n={len(joint)}$ cells. Both versions of a recipe fit the GP in the same way, "
            "but box inputs are rescaled per coordinate to $[0,1]^D$ and shell inputs are raw. The two versions differ "
            "in support \\emph{and} in the acquisition maximizer. "
            + ("For $\\ell_0=0.1\\sqrt{D}$, the box version uses analytic LogEI with sample-around-best, the "
               "acquisition call of MSR. This row therefore also differs in the acquisition. "
               if ctx.var.papenmeier_main == "published" and any(r[1] == "Papenmeier" for r in res) else "")
            + ("Tasks = tasks, of the 11, on which the shell version has the higher mean. " if tl else ""))
    if fsd:
        fc = fsd["cells"]
        t.add_block(f"Support $\\times$ acquisition maximization, $n={len(fc)}$ matched cells", "lrr",
                    ["Support", "Candidate-set argmax", "Continuous (L-BFGS-B)"],
                    [["Empirical shell", m(N.f4(L["SA"])), m(N.f4(L["SC"]))],
                     [f"Box {bl}", m(N.f4(L["BA"])), m(N.f4(L["BC"]))]])
        t.add_block("The five factorial contrasts", WRAP + "rrr",
                    ["Contrast", "$\\Delta$", stack("Tasks", "$\\Delta>0$"), "$p_{\\mathrm{task}}^{\\mathrm{BH}}$"], crow_view)
        pcap += (f"Bottom: the 2$\\times$2 factorial. Continuous maximization is L-BFGS-B on both supports "
                 f"(on the shell over a direction mapped to the shell). The matched set of the four cells has $n={len(fc)}$. Tasks = tasks with a "
                 "positive contrast. ")
    pcap += ("In both blocks, $p_{\\mathrm{task}}^{\\mathrm{BH}}$ is the exact two-sided Wilcoxon test on the 11 "
             f"per-task mean differences, Benjamini--Hochberg corrected within the {len(res)} rows or within the five "
             "contrasts"
             + stats.bh_note([r[5].p_task for r in res] if tl else [r[5].p for r in res], "rows")
             + (stats.bh_note([fsd["contrasts"][k].p_task for k, *_ in FACTORIAL_FIVE], "contrasts") if fsd else "")
             + f". Means are {conv_plural(ctx)} over {filt_inline(ctx)}. Bold, in both blocks: "
             "$p_{\\mathrm{task}}^{\\mathrm{BH}}<0.05$ within the block's family. The cell-level statistics are in the "
             "full table in the repository (\\texttt{docs/baselines/tab\\_boxsphere\\_full.tex}).")
    t.caption = pcap
    t.record_tables = [full]
    return t


def tab_factorial_details(ctx: Ctx) -> Table | None:
    table = "tab_factorial_details"
    gens = [("cr", "Camera-ready runs" if ctx.mode == "cr" else "Legacy generation (the posted rebuttal files)", "crFactCr"),
            ("e1", "E-1 rerun (abbd61e, batch runner, R-7)", "crFactEone"),
            ("posted", "Posted generation (rebuttal files, recomputed under this table's conventions)", "crFactPosted")]
    hdr = ["Contrast", "$n$", "$\\Delta$", "$d$", "W/L", "$p$", "$p_{\\mathrm{BH}}$"] + tl_cols(ctx)
    t = Table("factorial_details", "tab:factorial_details", "", "l" + "r" * (len(hdr) - 1), hdr, size="\\footnotesize")
    any_gen = False
    for gen, title, prefix in gens:
        arms = factorial_arms(ctx, gen)
        if not arms:
            ctx.skip(table, gen, "factorial arms absent (generation omitted)")
            continue
        any_gen = True
        fsd = factorial_stats(ctx, arms)
        q = stats.bh([fsd["contrasts"][k].p for k, _, _ in CONTRASTS])
        for (k, _, _), qq in zip(CONTRASTS, q):
            fsd["contrasts"][k].p_bh = qq
        _reg_factorial(ctx, table, prefix, arms, fsd, [k for k, _, _ in CONTRASTS])
        L = fsd["levels"]
        t.group(f"{title}: cell means SA {N.f4(L['SA'])}, SC {N.f4(L['SC'])}, BA {N.f4(L['BA'])}, "
                f"BC {N.f4(L['BC'])} ($n={len(fsd['cells'])}$)")
        for k, lab, _ in CONTRASTS:
            pr = fsd["contrasts"][k]
            t.add([lab, N.intfmt(pr.n), m(N.s4(pr.delta)), m(N.f2(pr.d)), f"{pr.wins}/{pr.losses}",
                   m(N.pfmt(pr.p)), m(N.pfmt(pr.p_bh))] + tl_vals(ctx, pr))
    if not any_gen:
        return None
    t.caption = ("\\textbf{Support $\\times$ acquisition-maximisation factorial: every contrast in every generation.} "
                 "SA/SC = empirical shell with candidate-set argmax / continuous (Adam) maximisation; BA/BC = box "
                 f"{BOX_LABEL[ctx.var.box]} with candidate-set argmax / continuous (L-BFGS) maximisation; the same GP "
                 f"constructor in all four cells (box inputs rescaled per coordinate, shell inputs raw). Means and $\\Delta$: {conv_text(ctx)}; $d$, W/L (ties: $|\\mathrm{{diff}}|\\le10^{{-6}}$) and the Wilcoxon $p$ over cells; "
                 "$p_{\\mathrm{BH}}$ within each generation (six contrasts). The interaction is computed per cell. "
                 f"Cells: {filt_text(ctx)}, intersected over the four arms of each generation. The printed generation is "
                 "the camera-ready one (rule R-3 of the run rules released with the code). E-1: candidate-set cells reproduced bit for bit, "
                 "continuous cells redrawn (it is kept by R-7; it is not an independent replication of the "
                 "candidate-set cells). Ties ($|\\mathrm{diff}|\\le10^{-6}$) enter the Wilcoxon test as zero differences.")
    return t


# =========================================================================================
# Trajectories N = 10 / 20 / 30 on one matched set
# =========================================================================================
TRAJ_N = (10, 20, 30)


#: RUN_RULES 2026-10-29 (3), promise C-32: the verified Linear BO recipe of the published-recipe block (its own box
#: [-8, 8]^D), from the best-so-far trace its result files record; camera-ready mode only
TRAJ_PUBLISHED_LINEAR = "linear_bo_box8"
#: authors' decision of 2026-11-02: the stored TuRBO and BAxUS rows are replaced by the runs of this version that Table 1
#: prints under these names, the TuRBO-1 recipe of the published-recipe block and the BAxUS rerun of 2026-11-02 (root
#: CRRD); the BAxUS rerun covers 439 of the 440 cells (one run crashed), so it is a row that covers only part of the
#: matched set, as in Table 1
TRAJ_PUBLISHED_TURBO, TRAJ_BAXUS = "turbo_published", "baxus_cr"
#: rows added to the trajectory table after the final run of 2026-10-20: none may change the matched set of the others
TRAJ_ADDED = (TRAJ_PUBLISHED_LINEAR, TRAJ_PUBLISHED_TURBO)


class TrajectoryGuardError(RuntimeError):
    """An added trajectory row would change the matched set of the trajectory table."""


def traj_rows(ctx: Ctx) -> list[Row]:
    bl = BOX_LABEL[ctx.var.box]
    pub = ([Row(TRAJ_PUBLISHED_LINEAR, published.name_or_label(ctx, TRAJ_PUBLISHED_LINEAR),
                published.support(TRAJ_PUBLISHED_LINEAR), key="PubLinear")] if ctx.mode == "cr" else [])
    return [Row("geowalk", "\\geowalk{}", "shell"),
            Row("hvarfner_scaled_gp", LB.label("hvarfner_scaled_gp"), "shell", key="VanillaSphere"),
            Row("linear_bo_sphere_calib", LB.label("linear_bo_sphere_calib"), "shell", key="LinearSphere"),
            Row("random_search", "Random Search", "shell"),
            Row("linear_bo", LB.label("linear_bo"), "$[p_{01},p_{99}]$", key="LinearBox"),
            *pub,
            Row(ctx.box("hvarfner"), LB.label(ctx.box("hvarfner")), bl, key="VanillaBox"),
            # RUN_RULES 2026-11-06 21:30: the TuRBO-1 re-implementation is not a printed row (the full Table 1 keeps it)
            *([Row("turbo_box_faithful", LB.label("turbo_box_faithful"), "$[p_{01},p_{99}]$")] if ctx.mode != "cr" else []),
            *([Row(TRAJ_PUBLISHED_TURBO, LB.label(TRAJ_PUBLISHED_TURBO), published.support(TRAJ_PUBLISHED_TURBO),
                   key="PubTurbo"),
               Row(TRAJ_BAXUS, LB.label(TRAJ_BAXUS), "nested subspaces of $[p_{01},p_{99}]$", partial=True)]
              if ctx.mode == "cr" else
              [Row("turbo", LB.label("turbo"), "box of observed latents", stored=True),
               Row("baxus", LB.label("baxus"), "nested subspaces of the cold-start box", stored=True)])]


def tab_trajectories(ctx: Ctx) -> Table | None:
    table = "tab_trajectories"
    rows = [r for r in traj_rows(ctx) if ctx.has(r.arm) or ctx.skip(table, r.arm, "arm absent (row omitted)")]
    if not any(r.arm == "geowalk" for r in rows):
        return None
    # a row that covers only part of the matched set (the BAxUS rerun, one crashed run) stays out of the intersection
    # and is read on the cells of the set it covers, as in Table 1
    mark_partial(ctx, [], rows)
    joint = ctx.eng.joint([r.arm for r in rows if not r.subset])
    # RUN_RULES 2026-10-29 (3) and 2026-11-02: an added row must not change the matched set of the other rows
    added = [r.arm for r in rows if r.arm in TRAJ_ADDED]
    if added and joint != ctx.eng.joint([r.arm for r in rows if not r.subset and r.arm not in TRAJ_ADDED]):
        raise TrajectoryGuardError(f"adding {', '.join(added)} would shrink the matched set of the trajectory "
                                   "table and change the values of its other rows")
    cells_of = {r.arm: (joint & ctx.eng.elig(r.arm)) if r.subset else joint for r in rows}
    hdr = ["Method", "Support"] + [f"$N={n}$" for n in TRAJ_N]
    t = Table("trajectories", "tab:trajectories", "", WRAP + wrapp("2.2cm") + "r" * len(TRAJ_N), hdr,
              size="\\footnotesize", tabcolsep="4pt")
    levels = {r.arm: {n: ctx.eng.level(r.arm, cells_of[r.arm], n) for n in TRAJ_N} for r in rows}
    best = {n: max(levels[r.arm][n] for r in rows) for n in TRAJ_N}
    for r in rows:
        cells = []
        cs = cells_of[r.arm]
        for n in TRAJ_N:
            v = levels[r.arm][n]
            ctx.num(f"crTraj{r.k()}N{camel(str(n))}", v, N.f4(v), kind="level", table=table,
                    fileset=ctx.fs({r.arm: cs}), n=len(cs),
                    rule=f"{conv_text(ctx)} of best@{n}, " + ("the cells of the matched set that the row covers"
                                                              if r.subset else "one matched set"))
            cells.append(mb(N.f4(v), abs(v - best[n]) < 5e-5))
        t.add([(no_stored_suffix(r.label) + "$^\\dagger$") if r.stored else
               (r.label + "$^\\ddagger$") if r.subset else r.label, r.support] + cells)
    ctx.num("crTrajN", len(joint), N.intfmt(len(joint)), kind="n", table=table,
            fileset=ctx.fs({r.arm: joint for r in rows if not r.subset}), n=len(joint),
            rule="joint matched set of all rows that cover every cell (rows marked double dagger: their part of it)")
    part = [r for r in rows if r.subset]
    for r in part:
        ctx.num(f"crTraj{r.k()}N", len(cells_of[r.arm]), N.intfmt(len(cells_of[r.arm])), kind="n", table=table,
                fileset=ctx.fs({r.arm: cells_of[r.arm]}), n=len(cells_of[r.arm]),
                rule="cells of the matched set that a row marked double dagger covers")
    blk = []
    if ctx.has("hvarfner_scaled_gp"):
        prs = {n: ctx.eng.paired("geowalk", "hvarfner_scaled_gp", joint, n) for n in TRAJ_N}
        q = stats.bh([prs[n].p for n in TRAJ_N])
        for n, qq in zip(TRAJ_N, q):
            prs[n].p_bh = qq
            o = ctx.reg_paired(f"crTrajGwVsVanillaN{camel(str(n))}", prs[n], "geowalk", "hvarfner_scaled_gp", joint, table, n)
            ctx.reg_pbh(f"crTrajGwVsVanillaN{camel(str(n))}", prs[n], o["_fs"], table, "N = 10, 20, 30")
        blk = [["$\\Delta$"] + [m(N.s4(prs[n].delta)) for n in TRAJ_N],
               ["$d$"] + [m(N.f2(prs[n].d)) for n in TRAJ_N],
               ["$p_{\\mathrm{BH}}$"] + [m(N.pfmt(prs[n].p_bh)) for n in TRAJ_N]]
        if ctx.var.tasklevel:
            blk.append(["Tasks, \\geowalk{} higher"] + [f"{prs[n].tasks_won}/{prs[n].n_tasks}" for n in TRAJ_N])
        t.add_block("\\geowalk{} $-$ the GP on the shell (" + LB.label("hvarfner_scaled_gp") + "), paired on the same cells", WRAP + "r" * len(TRAJ_N),
                    [""] + [f"$N={n}$" for n in TRAJ_N], blk)
    stored_rows = any(r.stored for r in rows)
    t.caption = ("\\textbf{Best-so-far after $N=10$, $20$ and $30$ evaluations on one matched set.} "
                 + (f"All rows use the same $n={len(joint)}$ {filt_inline(ctx)}. " if not part else
                    f"All rows except the one marked $^\\ddagger$ use the same $n={len(joint)}$ {filt_inline(ctx)}. "
                    + " ".join(f"The row marked $^\\ddagger$ covers $n={N.intfmt(len(cells_of[r.arm]))}$ of them "
                               f"({partial_why(ctx, r.arm)})." for r in part) + " ")
                 + f"Values are {conv_plural(ctx)} of the "
                 "best score after $N$ evaluations: 5 shared cold-start molecules and $N-5$ proposals. Bold marks the "
                 "highest mean in the column."
                 + (" Rows marked $^\\dagger$ are stored runs. "
                    "Their 25 recorded best-so-far values follow the shared cold start." if stored_rows else "")
                 + "".join(f" {r.label} is the published recipe of Table~\\ref{{tab:published}}, in the box "
                           f"{r.support[len('box '):]} of the authors' released code."
                           for r in rows if r.arm == TRAJ_PUBLISHED_LINEAR)
                 + (" Bottom block: \\geowalk{} minus the GP on the shell, paired on the same cells. "
                    + stat_defs(None, f"over the {len(TRAJ_N)} budgets $N=10$, $20$ and $30$", None) if blk else ""))
    return t


# =========================================================================================
# Appendix I: radius, shells, estimators, convex combinations, PCA/directions, shell vs prior
# =========================================================================================
def _simple_compare(ctx: Ctx, table: str, name: str, label: str, prefix: str, ref: str, ref_label: str,
                    rows: list[Row], caption: str, extra_cols=None, direction: str = "row-ref") -> Table | None:
    joint, res = compare_block(ctx, table, prefix, ref, rows, direction, family=f"the {len(rows)} rows")
    if not res:
        return None
    who = "row|higher" if direction == "row-ref" else "reference|higher"
    hdr = ["Variant", "Mean", "$\\Delta$", "$d$", "$p_{\\mathrm{BH}}$"] + tl_cols(ctx, who)
    if extra_cols:
        hdr = hdr[:2] + [extra_cols[0]] + hdr[2:]
    t = Table(name, label, "", WRAP + "r" * (len(hdr) - 1), hdr, size="\\small", tabcolsep="4pt")
    refv = ctx.eng.level(ref, joint)
    levels = [x["lev_row"] for x in res] + [refv]
    bold = _bold_best(levels)
    for x, b in zip(res, bold):
        pr = x["pr"]
        cells = [x["row"].label, mb(N.f4(x["lev_row"]), b)]
        if extra_cols:
            cells.append(extra_cols[1](x["row"]))
        cells += [m(N.s4(pr.delta)), m(N.f2(pr.d)), m(N.pfmt(pr.p_bh))] + tl_vals(ctx, pr)
        t.add(cells)
    t.rule()
    cells = [ref_label + " (reference)", mb(N.f4(refv), bold[-1])]
    if extra_cols:
        cells.append(extra_cols[1](Row(ref, ref_label)))
    t.add(cells + ["--", "--", "--"] + (["--", "--"] if ctx.var.tasklevel else []))
    sign = "row minus reference" if direction == "row-ref" else "reference minus row"
    t.caption = (caption + f" The matched set has $n={len(joint)}$ {filt_text(ctx)}. Means are {conv_text(ctx)}; "
                 f"bold marks the highest. $\\Delta$ = {sign} on the same cells. $p_{{\\mathrm{{BH}}}}$ is corrected "
                 f"over the {len(res)} rows."
                 + (" Tasks = tasks on which the row has the higher mean." if ctx.var.tasklevel and direction == "row-ref"
                    else (" Tasks = tasks on which the reference has the higher mean." if ctx.var.tasklevel else "")))
    return t


def tab_radius(ctx: Ctx) -> Table | None:
    rows = [Row(f"geowalk_r{r}", f"$r={r}$", key=f"R{camel(str(r))}") for r in (4, 6, 10, 12, 16)]
    rows.append(Row("geowalk", "per-run $\\bar r$ (default)", key="Default"))
    return _simple_compare(ctx, "tab_radius", "radius", "tab:radius", "crRadius", "geowalk_r8", "$r=8$", rows,
                           "\\textbf{\\geowalk{} with a fixed proposal radius.} Every proposal is placed at radius $r$. "
                           "The default estimates $\\bar r$ per run from the five cold-start latents.")


def _block_n(ctx: Ctx, prefix: str) -> str:
    rec = ctx.reg.records.get(prefix + "MatchedN")
    return rec["text"] if rec else "--"


def merge_blocks(ctx: Ctx, host: str, t: Table, parts: list) -> Table:
    """RUN_RULES 2026-10-29 (2): tables of Appendix I with the same unit, printed as blocks of one table. Every block
    keeps its rows, its matched set and its Benjamini--Hochberg family exactly as its own builder computed and
    registered them; only the layout and the caption are new. The host keeps its label."""
    by = dict(parts)
    if host == "tab_radius":
        return _merge_radius(ctx, t, by.get("tab_estimators"), by.get("tab_shells"))
    if host == "tab_convex":
        return _merge_convex(ctx, t, by.get("tab_pca_directions"))
    raise ValueError(f"no merge rule for {host}")


def _n_rows(t: Table) -> int:
    return sum(1 for r in t.rows if isinstance(r, list)) - 1        # the last data row is the reference


def _merge_radius(ctx: Ctx, rad: Table, est: Table | None, sh: Table | None) -> Table:
    """One table 'Radius of GeoWalk proposals' (label tab:radius): fixed radius; radius estimator; thick shells. The
    block 'against r = 8' of the thick-shell table is not printed (its numbers stay registered)."""
    t = Table("radius", "tab:radius", "", rad.colspec, rad.header, size=rad.size, tabcolsep=rad.tabcolsep)
    prefixes = ["crRadius"] + [p for part, p in ((est, "crEst"), (sh, "crShell")) if part is not None]
    one_n = same_n_sentence(ctx, prefixes)

    def head(title: str, prefix: str) -> str:            # the block's n only when the blocks differ
        return f"{title})" if one_n else f"{title}, $n={_block_n(ctx, prefix)}$)"

    t.group(head("Fixed radius $r$ (reference $r=8$", "crRadius"))
    t.rows += rad.rows
    fams = [f"{_n_rows(rad)} rows for the fixed radius"]
    for part, prefix, title, fname in ((est, "crEst", "Radius estimator (reference: mean of the five norms", "estimator"),
                                       (sh, "crShell", "Thick shells (reference: single shell $\\bar r$", "thick shells")):
        if part is None:
            continue
        if part.header != rad.header:
            raise ValueError(f"merged block {part.name}: header {part.header} differs from {rad.header}")
        t.rule()
        t.group(head(title, prefix))
        t.rows += part.rows
        fams.append(f"{_n_rows(part)} for the {fname}")
    t.caption = ("\\textbf{Radius of \\geowalk{} proposals.} Three blocks, each against its own reference row. "
                 "Fixed radius: every proposal is placed at radius $r$. The default estimates $\\bar r$ per run from "
                 "the five cold-start latents. Radius estimator: the statistic of the five cold-start norms that sets "
                 "$\\bar r$. Thick shells: the radius of each proposal is drawn independently from the stated range. "
                 "The bounds $p_{05}$ and $p_{95}$ are the 5th and 95th percentiles of the encoder-mean norms of "
                 "ZINC-250k. "
                 + (one_n + " " if one_n else "Each block uses the " + filt_inline(ctx) + " of its rows. ")
                 + f"Means are {conv_plural(ctx)}. Bold marks the highest mean of each block. "
                 + stat_defs("the row minus the block's reference on the same cells",
                             f"within each block: {', '.join(fams)}",
                             "the row has the higher mean" if ctx.var.tasklevel else None))
    return t


def _merge_convex(ctx: Ctx, cvx: Table, pca: Table | None) -> Table:
    """The direction controls (former tab_pca_directions) as the last block of the convex-combination table (label
    tab:convex). The table as tab_convex built it, with its full Monte Carlo description, is a repository record
    (tab_convex_full.tex)."""
    if pca is None:
        return cvx
    import copy
    full = copy.copy(cvx)
    full.name, full.rows, full.blocks, full.extra = "convex_full", list(cvx.rows), list(cvx.blocks), dict(cvx.extra)
    k = sum(1 for r in pca.rows if isinstance(r, list))
    t = cvx
    t.add_block("Direction controls at the shell radius (reference: the same method with uniform directions on the "
                "shell)", pca.colspec, pca.header, pca.rows)
    ns = {ctx.reg.records[p]["text"] for p in ("crConvexMatchedN", "crDirPcaN", "crDirAngularN") if p in ctx.reg.records}
    se = ctx.cache.get("convex_se_max")
    t.caption = ("\\textbf{Samplers inside the latent cloud, and direction controls.} First block: surrogate-free "
                 "samplers against random search, which samples uniformly on the shell. The samplers draw convex "
                 "combinations of $k$ latents with Dirichlet(1) weights, or draw from a Gaussian fitted to the "
                 "latents. The latents are a cached sample of 3,000 encoded ZINC-250k molecules. "
                 "$\\overline{\\|z\\|}$ is the mean norm of the candidates. For random search it is $\\bar r$, "
                 "the radius of each run. "
                 + (f"The Monte Carlo standard error of $\\overline{{\\|z\\|}}$ is at most ${N.sci(se)}$, so every "
                    "printed digit is certain. " if se is not None and se == se else "")
                 + "The full table released with the code describes the estimation. "
                 "Second block: two direction controls. Random search on the sphere in the span of the top 64 principal "
                 "components is compared with random search on the full shell. The GP (DSP prior) with qLogEI argmax "
                 "over cube directions rescaled to $\\bar r$ is compared with the same GP on the uniform shell. "
                 + (f"All rows use the same $n={ns.pop()}$ {filt_inline(ctx)}. " if len(ns) == 1 else
                    f"Each block uses the {filt_inline(ctx)} of its rows. ")
                 + f"Means are {conv_plural(ctx)}. Bold marks the highest mean of the first block. "
                 + stat_defs("the row minus its reference on the same cells",
                             f"within each block, over {_n_rows(cvx)} rows and {k} comparisons",
                             "the row has the higher mean, of the 11" if ctx.var.tasklevel else None))
    t.record_tables = [full]
    return t


def tab_radius_pertask(ctx: Ctx) -> Table | None:
    table = "tab_radius_pertask"
    arms = [f"geowalk_r{r}" for r in (4, 6, 8, 10, 12, 16)] + ["geowalk"]
    if not all(ctx.has(a) for a in arms):
        ctx.skip(table, "radius arms", "absent")
        return None
    joint = ctx.eng.joint(arms)
    tm = {a: ctx.eng.task_means(a, joint) for a in arms}
    tasks = sorted(next(iter(tm.values())))
    hdr = ["Task"] + [f"$r={r}$" for r in (4, 6, 8, 10, 12, 16)] + ["default"]
    t = Table("radius_pertask", "tab:radius_pertask", "", "l" + "r" * (len(hdr) - 1), hdr, size="\\footnotesize")
    for task in tasks:
        vals = [tm[a][task] for a in arms]
        b = _bold_best(vals)
        for a, v in zip(arms, vals):
            ctx.num(f"crRadiusTask{camel(task)}{camel(a)}", v, N.f4(v), kind="level", table=table,
                    fileset=ctx.fs({a: {c for c in joint if c[0] == task}}), n=sum(1 for c in joint if c[0] == task),
                    rule="per-task mean of best@30")
        t.add([task] + [mb(N.f4(v), bb) for v, bb in zip(vals, b)])
    t.caption = (f"\\textbf{{Per-task means of the radius sweep.}} The matched set has $n={len(joint)}$ "
                 f"{filt_text(ctx)}. Bold marks the best radius on the task.")
    return t


def tab_shells(ctx: Ctx) -> Table | None:
    table = "tab_shells"
    rows = [Row("geowalk_shell10", "$r\\sim\\bar r\\,U[0.9,1.1]$", key="Ten"),
            Row("geowalk_shell20", "$r\\sim\\bar r\\,U[0.8,1.2]$", key="Twenty"),
            Row("geowalk_shellpct", "$r\\sim U[p_{05},p_{95}]$", key="Pct")]
    t = _simple_compare(ctx, table, "shells", "tab:shells", "crShell", "geowalk", "single shell $\\bar r$ (default)",
                        rows, "\\textbf{Thick shells.} Each proposal's radius is drawn independently from the stated range.")
    if t is None:
        return None
    if ctx.has("geowalk_r8"):
        joint = ctx.eng.joint(["geowalk", "geowalk_r8"] + [r.arm for r in rows if ctx.has(r.arm)])
        prs = [(r, ctx.eng.paired(r.arm, "geowalk_r8", joint)) for r in rows if ctx.has(r.arm)]
        q = stats.bh([pr.p for _, pr in prs])
        blk = []
        for (r, pr), qq in zip(prs, q):
            pr.p_bh = qq
            o = ctx.reg_paired(f"crShell{r.k()}VsREight", pr, r.arm, "geowalk_r8", joint, table)
            ctx.reg_pbh(f"crShell{r.k()}VsREight", pr, o["_fs"], table, "the 3 shells vs r = 8")
            blk.append([r.label, m(N.s4(pr.delta)), m(N.f2(pr.d)), m(N.pfmt(pr.p_bh))])
        ctx.reg_level("crShellREightMean", "geowalk_r8", joint, table)
        t.add_block(f"Against the fixed radius $r=8$ (mean {N.f4(ctx.eng.level('geowalk_r8', joint))}, $n={len(joint)}$)",
                    WRAP + "rrr", ["Variant", "$\\Delta$", "$d$", "$p_{\\mathrm{BH}}$"], blk)
    return t


def tab_estimators(ctx: Ctx) -> Table | None:
    rows = [Row("geowalk_rmedian", "median of the five cold-start norms", key="Median"),
            Row("geowalk_rpop", "fixed $7.98$ (ZINC mean norm)", key="Pop")]
    return _simple_compare(ctx, "tab_estimators", "estimators", "tab:estimators", "crEst", "geowalk",
                           "mean of the five cold-start norms", rows,
                           "\\textbf{Radius estimator of \\geowalk{}.}")


def convex_norms(ctx: Ctx) -> dict:
    """Mean candidate norm of the convex-combination and ellipsoid samplers (crlib.mc: control-variate Monte
    Carlo for k >= 2, exact for k = 1, quadrature for the Gaussian fit)."""
    from .mc import CONVEX_ARM
    from .supports import get_supports
    S = get_supports(ctx)
    out = {}
    for sid, arm in CONVEX_ARM.items():
        if sid in S:
            out[arm] = S[sid]
    if "gaussian_fit" in S:
        out["ellipsoid_rs"] = S["gaussian_fit"]
    return out


def tab_convex(ctx: Ctx) -> Table | None:
    table = "tab_convex"
    ks = [("convex_hull_k1", "1"), ("convex_hull_k2", "2"), ("convex_hull_rs", "3"), ("convex_hull_k4", "4"),
          ("convex_hull_k8", "8"), ("convex_hull_k16", "16"), ("convex_hull_k64", "64")]
    rows = [Row(a, f"convex, $k={k}$", key=f"K{camel(k)}") for a, k in ks] + \
        [Row("ellipsoid_rs", "Gaussian fit $N(\\hat\\mu,\\hat\\Sigma)$", key="Ellipsoid")]
    norms = convex_norms(ctx)
    from .supports import how_text, se_text
    for a, v in norms.items():
        fs = ctx.fs_paths(v["inputs"])
        ctx.num(f"crConvexNorm{camel(a)}", v["mean"], v["text_norm"], kind="norm", table=table, fileset=fs,
                rule="mean candidate norm, " + how_text(v))
        ctx.num(f"crConvexNorm{camel(a)}Se", v["se"], se_text(v["se"]), kind="mc_se", table=table, fileset=fs,
                rule="standard error of the mean norm (Monte Carlo) or quadrature error estimate (0 = exact)")

    def _norm_cell(r):
        if r.arm in norms:
            return m(norms[r.arm]["text_norm"])
        return m("\\bar r") if r.arm == "random_search" else "--"

    mc = [v for v in norms.values() if not v["exact"]]
    se_max = max((v["se"] for v in mc), default=float("nan"))
    ctx.cache["convex_se_max"] = se_max            # the caption of the merged table (RUN_RULES 2026-10-29 (4))
    dr = sorted({v["draws"] for v in mc})
    how = (f"Monte Carlo with $\\|z\\|^2$ as a control variate, a fixed seed and {N.intfmt(min(dr))}--"
           f"{N.intfmt(max(dr))} draws per sampler. The standard error is at most ${N.sci(se_max)}$, and a digit is "
           "printed only if it is established" if mc else "")
    t = _simple_compare(
        ctx, table, "convex", "tab:convex", "crConvex", "random_search", "uniform on the shell (random search)", rows,
        "\\textbf{Surrogate-free samplers inside the latent cloud versus uniform sampling on the shell.} "
        "The samplers are convex combinations of $k$ encoded ZINC latents (Dirichlet(1) weights) and a Gaussian "
        "fitted to them. $\\overline{\\|z\\|}$ is the mean candidate norm. "
        + (f"For $k\\ge2$ it is estimated by {how}. " if how else "")
        + "For $k=1$ it is the exact mean norm of the 3,000 cached latents, and the Gaussian fit is exact up to "
        "quadrature error.",
        extra_cols=("$\\overline{\\|z\\|}$", _norm_cell))
    present = [a for a, _ in ks[:1] + ks[1:] if ctx.has(a) and a != "convex_hull_rs"]
    if present and ctx.has("random_search"):
        joint = ctx.eng.joint(["random_search"] + [r.arm for r in rows if ctx.has(r.arm)])
        lv = [ctx.eng.level(a, joint) for a in present]
        fs = ctx.fs({a: joint for a in present})
        ctx.num("crConvexRangeMin", min(lv), N.f4(min(lv)), kind="level", table=table, fileset=fs, n=len(joint),
                rule="min over k in {1,2,4,8,16,64} (the posted set; k = 3 reported separately)")
        ctx.num("crConvexRangeMax", max(lv), N.f4(max(lv)), kind="level", table=table, fileset=fs, n=len(joint),
                rule="max over k in {1,2,4,8,16,64}")
    return t


def tab_pca_directions(ctx: Ctx) -> Table | None:
    table = "tab_pca_directions"
    pairs = [("pca_sphere_k64", "random_search", "Random search: top-64 PCA sphere vs full sphere", "Pca"),
             ("hvarfner_scaled_gp_angular_box", "hvarfner_scaled_gp",
              "GP (DSP prior), qLogEI argmax: cube directions rescaled to $\\bar r$ vs uniform shell", "Angular")]
    res = []
    for a, b, lab, key in pairs:
        if not (ctx.has(a) and ctx.has(b)):
            ctx.skip(table, a, "arm absent (row omitted)")
            continue
        cells = ctx.eng.joint([a, b])
        res.append((a, b, lab, key, cells, ctx.eng.paired(a, b, cells)))
    if not res:
        return None
    for r, q in zip(res, stats.bh([r[5].p for r in res])):
        r[5].p_bh = q
    hdr = ["Comparison", "$n$", "Variant", "Shell", "$\\Delta$", "$d$", "$p_{\\mathrm{BH}}$"] + tl_cols(ctx, "variant|higher")
    t = Table("pca_directions", "tab:pca_directions", "", WRAP + "r" * (len(hdr) - 1), hdr, size="\\footnotesize",
              tabcolsep="4pt")
    for a, b, lab, key, cells, pr in res:
        p = "crDir" + key
        la, _ = ctx.reg_level(p + "VariantMean", a, cells, table)
        lb, _ = ctx.reg_level(p + "ShellMean", b, cells, table)
        o = ctx.reg_paired(p, pr, a, b, cells, table)
        ctx.reg_pbh(p, pr, o["_fs"], table, "the 2 comparisons")
        t.add([lab, N.intfmt(pr.n), m(N.f4(la)), m(N.f4(lb)), m(N.s4(pr.delta)), m(N.f2(pr.d)), m(N.pfmt(pr.p_bh))]
              + tl_vals(ctx, pr))
    t.caption = ("\\textbf{Direction controls at the shell radius.} $\\Delta$ = variant minus uniform shell, paired on "
                 f"the pair's matched cells ({filt_text(ctx)}). Means are {conv_text(ctx)}."
                 + (" Tasks = tasks, of the 11, on which the variant has the higher mean." if ctx.var.tasklevel else ""))
    return t


def tab_shell_prior(ctx: Ctx) -> Table | None:
    table = "tab_shell_prior"
    pairs = [("hvarfner_scaled_gp_prior", "hvarfner_scaled_gp", "Vanilla BO: candidates from $N(0,I)$ vs shell", "Gp"),
             ("random_search_prior", "random_search", "Random search: $N(0,I)$ vs shell", "Rs"),
             ("random_box_faithful", "random_search", "Random search: uniform in the $[p_{01},p_{99}]$ box vs shell",
              "RsBox")]
    res = []
    for a, b, lab, key in pairs:
        if not (ctx.has(a) and ctx.has(b)):
            ctx.skip(table, a, "arm absent (row omitted)")
            continue
        cells = ctx.eng.joint([a, b])
        res.append((a, b, lab, key, cells, ctx.eng.paired(a, b, cells)))
    if not res:
        return None
    for r, q in zip(res, stats.bh([r[5].p for r in res])):
        r[5].p_bh = q
    hdr = ["Comparison", "$n$", "Alternative", "Shell", "$\\Delta$", "$d$", "$p_{\\mathrm{BH}}$"] + tl_cols(ctx)
    t = Table("shell_prior", "tab:shell_prior", "", "l" + "r" * (len(hdr) - 1), hdr, size="\\small")
    for a, b, lab, key, cells, pr in res:
        p = "crPrior" + key
        la, _ = ctx.reg_level(p + "AltMean", a, cells, table)
        lb, _ = ctx.reg_level(p + "ShellMean", b, cells, table)
        o = ctx.reg_paired(p, pr, a, b, cells, table)
        ctx.reg_pbh(p, pr, o["_fs"], table, f"the {len(res)} comparisons")
        t.add([lab, N.intfmt(pr.n), m(N.f4(la)), m(N.f4(lb)), m(N.s4(pr.delta)), m(N.f2(pr.d)), m(N.pfmt(pr.p_bh))]
              + tl_vals(ctx, pr))
    t.caption = ("\\textbf{Empirical shell versus the VAE prior $N(0,I_D)$ and the box as candidate support.} "
                 "$\\Delta$ = alternative support minus shell on the pair's matched cells "
                 f"({filt_text(ctx)}); means are {conv_text(ctx)}. The random-search rows have no surrogate.")
    return t


# =========================================================================================
# Appendix B: sensitivity of GeoWalk's schedule; GeoCMA subspace dimension
# =========================================================================================
def tab_sensitivity(ctx: Ctx) -> Table | None:
    table = "tab_sensitivity"
    rows = [Row("geowalk_const060", "constant angle $0.60$", key="ConstSixty"),
            Row("geowalk_const035", "constant angle $0.35$", key="ConstThirtyFive"),
            Row("geowalk_const015", "constant angle $0.15$", key="ConstFifteen"),
            Row("geowalk_explore8", "8 exploration steps ($\\rho\\approx0.32$)", key="ExploreEight"),
            Row("geowalk_explore24", "24 exploration steps ($\\rho\\approx0.96$)", key="ExploreTwentyFour"),
            # RUN_RULES 2026-10-12 clarification: not a 1/5 rule; the printed description gives the constants (S3)
            Row("geowalk_one_fifth", "success-based angle rule (start $0.30$, $\\times1.2$ / $\\times0.85$, "
                "clamp $[0.05,0.50]$)", key="OneFifth")]
    t = _simple_compare(ctx, table, "sensitivity", "tab:sensitivity", "crSens", "geowalk",
                        "default ($0.60\\to0.15$, 16 exploration steps)", rows,
                        "\\textbf{Sensitivity of \\geowalk{} to its step schedule.}")
    sd = [("cma_es_subspace_sd8", "$\\tilde D=8$", "Eight"), ("cma_es_subspace_sd64", "$\\tilde D=64$", "SixtyFour")]
    sd = [x for x in sd if ctx.has(x[0]) or ctx.skip(table, x[0], "arm absent (D~ row omitted)")]
    if t is not None and sd and ctx.has("cma_es_subspace"):
        joint = ctx.eng.joint(["cma_es_subspace"] + [a for a, _, _ in sd])
        prs = [(a, lab, key, ctx.eng.paired(a, "cma_es_subspace", joint)) for a, lab, key in sd]
        for x, q in zip(prs, stats.bh([x[3].p for x in prs])):
            x[3].p_bh = q
        ref = ctx.reg_level("crSensDtildeSixteenMean", "cma_es_subspace", joint, table)[0]
        blk = [["$\\tilde D=16$ (default)", m(N.f4(ref)), "--", "--", "--"]]
        for a, lab, key, pr in prs:
            lv = ctx.reg_level(f"crSensDtilde{key}Mean", a, joint, table)[0]
            o = ctx.reg_paired(f"crSensDtilde{key}", pr, a, "cma_es_subspace", joint, table)
            ctx.reg_pbh(f"crSensDtilde{key}", pr, o["_fs"], table, "the D~ rows")
            blk.append([lab, m(N.f4(lv)), m(N.s4(pr.delta)), m(N.f2(pr.d)), m(N.pfmt(pr.p_bh))])
        t.add_block(f"\\geocma{{}} subspace dimension, matched set $n={len(joint)}$", WRAP + "rrrr",
                    ["$\\tilde D$", "Mean", "$\\Delta$", "$d$", "$p_{\\mathrm{BH}}$"], blk)
    if t is not None:
        # RUN_RULES 2026-10-29 (4): the caption describes both blocks, each with its test and family
        k_top = _n_rows(t)
        k_bot = sum(1 for r in t.blocks[0][3] if isinstance(r, list)) - 1 if t.blocks else 0
        one_n = same_n_sentence(ctx, ["crSens"], "The top block") if not t.blocks else ""
        if t.blocks:
            ns = {ctx.reg.records["crSensMatchedN"]["text"]} | {str(len(joint))}
            one_n = (f"Both blocks use the same $n={ns.pop()}$ {filt_inline(ctx)}." if len(ns) == 1 else
                     f"The blocks use the {filt_inline(ctx)} of their rows (top $n="
                     f"{ctx.reg.records['crSensMatchedN']['text']}$, bottom $n={len(joint)}$).")
        t.caption = ("\\textbf{Sensitivity of \\geowalk{} and \\geocma{} to their settings.} Top: "
                     f"{_count_word(k_top)} variants of the step schedule and the exploration of \\geowalk{{}}, each "
                     "against the default (angle $0.60\\to0.15$, 16 exploration steps)."
                     + (f" Bottom: the subspace dimension $\\tilde D$ of \\geocma{{}}, 8 and 64 against the default "
                        "16." if t.blocks else "")
                     + f" {one_n} Means are {conv_plural(ctx)}. Bold marks the highest mean of the top block. "
                     + stat_defs("the variant minus the default, computed from unrounded means",
                                 f"within each block, over {k_top} and {k_bot} rows" if t.blocks else
                                 f"over the {k_top} rows",
                                 "the variant has a higher mean than the default" if ctx.var.tasklevel else None)
                     + (" The task-level test of the bottom block is given in the text." if t.blocks else ""))
    return t


# =========================================================================================
# Appendix C: per-task tables
# =========================================================================================
def tab_pertask(ctx: Ctx) -> Table | None:
    table = "tab_pertask"
    a, b = "geowalk", "hvarfner_scaled_gp"
    if not (ctx.has(a) and ctx.has(b)):
        ctx.skip(table, b, "absent")
        return None
    joint = ctx.eng.joint([a, b])
    va, vb = ctx.eng.vals(a, joint), ctx.eng.vals(b, joint)
    tasks = sorted({c[0] for c in joint})
    per = []
    for task in tasks:
        cs = sorted(c for c in joint if c[0] == task)
        pr = stats.paired({c: va[c] for c in cs}, {c: vb[c] for c in cs}, cs, "flat")
        per.append((task, cs, pr))
    holm = stats.holm([pr.p for _, _, pr in per])
    hdr = ["Task", "$n$", "\\geowalk{}", "GP (shell)", "$\\Delta$", "$d$", "$p$", "$p_{\\mathrm{Holm}}$"]
    t = Table("pertask", "tab:pertask", "", "lrrrrrrr", hdr, size="\\small")
    for (task, cs, pr), ph in zip(per, holm):
        p = f"crPertask{camel(task)}"
        fs = ctx.fs({a: set(cs), b: set(cs)})
        ctx.num(p + "Gw", pr.level_a, N.f4(pr.level_a), kind="level", table=table, fileset=fs, n=pr.n, rule="per-task mean")
        ctx.num(p + "Vanilla", pr.level_b, N.f4(pr.level_b), kind="level", table=table, fileset=fs, n=pr.n, rule="per-task mean")
        ctx.num(p + "Delta", pr.delta_flat, N.s4(pr.delta_flat), kind="delta", table=table, fileset=fs, n=pr.n, rule="per-task paired mean difference")
        ctx.num(p + "D", pr.d, N.d2(pr.d), kind="cohens_d", table=table, fileset=fs, n=pr.n, rule="per-task paired d")
        ctx.num(p + "P", pr.p, N.pfmt(pr.p), kind="p_raw", table=table, fileset=fs, n=pr.n, rule="per-task Wilcoxon, uncorrected")
        ctx.num(p + "PHolm", ph, N.pfmt(ph), kind="p_holm", table=table, fileset=fs, n=pr.n, rule="Holm over the tasks")
        t.add([task, N.intfmt(pr.n), m(N.f4(pr.level_a)), m(N.f4(pr.level_b)), m(N.s4(pr.delta_flat)),
               m(N.d2(pr.d)), m(N.pfmt(pr.p)), m(N.pfmt(ph))])
    pr = ctx.eng.paired(a, b, joint)
    t.rule()
    o = ctx.reg_paired("crPertaskAll", pr, a, b, joint, table)
    ctx.num("crPertaskAllGw", pr.level_a, N.f4(pr.level_a), kind="level", table=table, fileset=o["_fs"], n=pr.n,
            rule=conv_text(ctx))
    ctx.num("crPertaskAllVanilla", pr.level_b, N.f4(pr.level_b), kind="level", table=table, fileset=o["_fs"], n=pr.n,
            rule=conv_text(ctx))
    t.add(["all tasks", N.intfmt(pr.n), m(N.f4(pr.level_a)), m(N.f4(pr.level_b)), m(N.s4(pr.delta)),
           m(N.d2(pr.d)), m(N.pfmt(pr.p)), "--"])
    absd = [abs(x[2].delta_flat) for x in per]
    if absd:
        i = int(np.argmax(absd))
        ctx.num("crPertaskMaxAbsDelta", absd[i], N.f4(absd[i]), kind="delta", table=table, fileset=o["_fs"],
                rule=f"largest |per-task Delta| (task {per[i][0]})")
    ctx.num("crPertaskTasksGwHigher", pr.tasks_won, f"{pr.tasks_won}/{pr.n_tasks}", kind="tasks_won", table=table,
            fileset=o["_fs"], rule="tasks with GeoWalk mean > Vanilla BO (shell) mean")
    t.caption = ("\\textbf{\\geowalk{} versus the GP on the shell, per task.} GP (shell) = "
                 + LB.label("hvarfner_scaled_gp") + ". $\\Delta$ = \\geowalk{} minus GP (shell). Per task, "
                 "$\\Delta$, Cohen's $d$ and the two-sided Wilcoxon $p$ are paired over the seeds of the task's "
                 f"{filt_inline(ctx)}. $p_{{\\mathrm{{Holm}}}}$ is Holm-corrected over the {len(per)} tasks. Last row: "
                 f"all matched cells, as a {conv_text(ctx)}. The $p$ of the last row is the cell-level test over all "
                 "cells. That test is not one of the per-task tests, so it has no $p_{\\mathrm{Holm}}$ (--). Task "
                 "codes: " + ", ".join(f"{c} = {nm}" for c, nm in TASK_KEY[:6] if c in tasks) + ". Further codes: "
                 + ", ".join(f"{c} = {nm}" for c, nm in TASK_KEY[6:] if c in tasks)
                 + ". MPO = multi-property objective.")
    return t


def tab_pertask_full(ctx: Ctx) -> Table | None:
    table = "tab_pertask_full"
    cols = [("geowalk", "\\geowalk{}"), ("cma_es_subspace", "\\geocma{}"), ("random_search", stack("RS", "(shell)")),
            ("hvarfner_scaled_gp", stack("GP", "(shell)")), (ctx.box("hvarfner"), stack("GP", "(box)")),
            ("linear_bo_sphere_calib", stack("lin.\\ GP", "(shell)")), ("linear_bo", stack("lin.\\ GP", "(box)"))]
    role = {"geowalk": "Geowalk", "cma_es_subspace": "Geocma", "random_search": "RandomSearch",
            "hvarfner_scaled_gp": "VanillaSphere", ctx.box("hvarfner"): "VanillaBox",
            "linear_bo_sphere_calib": "LinearSphere", "linear_bo": "LinearBox"}
    cols = [c for c in cols if ctx.has(c[0]) or ctx.skip(table, c[0], "arm absent (column omitted)")]
    if not cols:
        return None
    joint = ctx.eng.joint([a for a, _ in cols])
    tm = {a: ctx.eng.task_means(a, joint) for a, _ in cols}
    tasks = sorted({c[0] for c in joint})
    hdr = ["Task", "$n$"] + [lab for _, lab in cols]
    t = Table("pertask_full", "tab:pertask_full", "", "lr" + "r" * len(cols), hdr, size="\\footnotesize",
              tabcolsep="4pt")
    for task in tasks:
        n_t = sum(1 for c in joint if c[0] == task)
        vals = [tm[a][task] for a, _ in cols]
        b = _bold_best(vals)
        for (a, _), v in zip(cols, vals):
            ctx.num(f"crFull{camel(task)}{role[a]}", v, N.f3(v), kind="level", table=table,
                    fileset=ctx.fs({a: {c for c in joint if c[0] == task}}), n=n_t, rule="per-task mean of best@30")
        t.add([task, N.intfmt(n_t)] + [mb(N.f3(v), bb) for v, bb in zip(vals, b)])
    t.rule()
    allv = [ctx.eng.level(a, joint) for a, _ in cols]
    for (a, _), v in zip(cols, allv):
        ctx.num(f"crFullAll{role[a]}", v, N.f3(v), kind="level", table=table, fileset=ctx.fs({a: joint}),
                n=len(joint), rule=conv_text(ctx))
    t.add(["mean", N.intfmt(len(joint))] + [mb(N.f3(v), bb) for v, bb in zip(allv, _bold_best(allv))])
    box = ctx.box("hvarfner")
    if ctx.has(box) and ctx.has("geowalk"):
        gaps = [tm[box][task] - tm["geowalk"][task] for task in tasks]
        med = float(np.median(gaps))
        ctx.num("crFullMedianBoxGap", med, N.s3(med), kind="delta", table=table,
                fileset=ctx.fs({box: joint, "geowalk": joint}), n=len(tasks),
                rule="median over tasks of (box mean - GeoWalk mean)")
    t.caption = ("\\textbf{Per-task means at $N=30$.} All columns use one matched set of "
                 f"$n={len(joint)}$ {filt_inline(ctx)}. Task codes as in Table~\\ref{{tab:pertask}}. Last row: "
                 f"{conv_text(ctx)}. Bold: best per task. RS = random search. Box = {BOX_LABEL[ctx.var.box]}. GP = "
                 + LB.label("hvarfner_scaled_gp").replace(", qLogEI, candidate-set argmax", "")
                 + " with qLogEI (candidate-set argmax on the shell, L-BFGS-B in the box). lin.\\ GP = "
                 + LB.label("linear_bo_sphere_calib").split(",")[0].replace(" (calibrated bounds)", "")
                 + " (calibrated bounds and candidate-set argmax on the shell, LogEI with "
                 "L-BFGS-B in the box).")
    return t


# =========================================================================================
# Box-definition sensitivity (plan R2) and TuRBO box
# =========================================================================================
def tab_box_sensitivity(ctx: Ctx) -> Table | None:
    table = "tab_box_sensitivity"
    recipes = [("Vanilla BO", "Vanilla", "hvarfner_scaled_gp",
                [("hvarfner_scaled_gp_norm_box", "$[\\min,\\max]$"), ("hvarfner_scaled_gp_box_faithful", "$[p_{01},p_{99}]$"),
                 ("hvarfner_scaled_gp_box", "$\\pm3\\sigma$")]),
               ("Papenmeier MLE (scaled)", "PapMle", "papenmeier_msr",
                [("papenmeier_msr_norm_box", "$[\\min,\\max]$"), ("papenmeier_msr_box_faithful", "$[p_{01},p_{99}]$"),
                 ("papenmeier_msr_box", "$\\pm3\\sigma$")]),
               ("Papenmeier MSR (published)", "PapMsr", "papenmeier_msr",
                [("papenmeier_msr_published_norm_box", "$[\\min,\\max]$"), ("papenmeier_msr_published", "$[p_{01},p_{99}]$")]),
               ("Xu Standard GP", "Xu", "xu_standard_gp",
                [("xu_standard_gp_norm_box", "$[\\min,\\max]$"), ("xu_standard_gp_box_faithful", "$[p_{01},p_{99}]$"),
                 ("xu_standard_gp_box", "$\\pm3\\sigma$")])]
    hdr = ["Recipe", "Box", "$n$", "Sphere", "Box mean", "$\\Delta$", "$d$", "$p_{\\mathrm{BH}}$"] + tl_cols(ctx)
    t = Table("box_sensitivity", "tab:box_sensitivity", "", "ll" + "r" * (len(hdr) - 2), hdr, size="\\footnotesize")
    allres = []
    for lab, key, sph, boxes in recipes:
        bx = [(a, bl) for a, bl in boxes if ctx.has(a) or ctx.skip(table, a, "arm absent (row omitted)")]
        if not bx or not ctx.has(sph):
            continue
        joint = ctx.eng.joint([sph] + [a for a, _ in bx])
        for a, bl in bx:
            allres.append((lab, key, sph, a, bl, joint, ctx.eng.paired(sph, a, joint)))
    tb = None
    if ctx.has("turbo") and ctx.has("turbo_box_faithful"):
        cells = ctx.eng.joint(["turbo", "turbo_box_faithful"])
        tb = (cells, ctx.eng.paired("turbo_box_faithful", "turbo", cells))
    ps = [r[6].p for r in allres] + ([tb[1].p] if tb else [])
    q = stats.bh(ps)
    for r, qq in zip(allres, q):
        r[6].p_bh = qq
    if tb:
        tb[1].p_bh = q[-1]
    cur = None
    for lab, key, sph, a, bl, joint, pr in allres:
        if cur is not None and lab != cur:
            t.rule()
        cur = lab
        p = f"crBoxsens{key}{camel(a)}"
        ls, _ = ctx.reg_level(p + "SphereMean", sph, joint, table)
        lb, _ = ctx.reg_level(p + "BoxMean", a, joint, table)
        o = ctx.reg_paired(p, pr, sph, a, joint, table)
        ctx.reg_pbh(p, pr, o["_fs"], table, "every row of the box-sensitivity table")
        t.add([lab, bl, N.intfmt(pr.n), m(N.f4(ls)), m(N.f4(lb)), m(N.s4(pr.delta)), m(N.f2(pr.d)),
               m(N.pfmt(pr.p_bh))] + tl_vals(ctx, pr))
    if tb:
        cells, pr = tb
        o = ctx.reg_paired("crBoxsensTurbo", pr, "turbo_box_faithful", "turbo", cells, table)
        ctx.reg_pbh("crBoxsensTurbo", pr, o["_fs"], table, "every row of the box-sensitivity table")
        t.add_block("TuRBO: fixed $[p_{01},p_{99}]$ box (TuRBO-1, camera-ready) minus the submitted harness "
                    "(box of the observed latents, stored)", "lrrrrr" + ("rr" if ctx.var.tasklevel else ""),
                    ["", "$n$", "fixed box", "submitted", "$\\Delta$", "$d$"] + tl_cols(ctx),
                    [["TuRBO", N.intfmt(pr.n), m(N.f4(pr.level_a)), m(N.f4(pr.level_b)), m(N.s4(pr.delta)),
                      m(N.f2(pr.d))] + tl_vals(ctx, pr)])
    if not allres and not tb:
        return None
    t.caption = ("\\textbf{Sensitivity of the sphere-versus-box contrast to the box definition.} Per-coordinate "
                 "$[\\min,\\max]$ (the submitted box) and $[p_{01},p_{99}]$ of 50,000 encoded ZINC latents rescaled to "
                 "$[0,1]^D$, and the isotropic $\\pm3\\sigma$ cube; $\\Delta$ = sphere minus box on each recipe's matched "
                 f"set ({filt_text(ctx)}); means are {conv_text(ctx)}; $p_{{\\mathrm{{BH}}}}$ over all rows.")
    return t


# =========================================================================================
# Appendix H: arm status (fallback and crash counts, provenance)
# =========================================================================================
def tab_arm_status(ctx: Ctx) -> Table | None:
    hdr = ["Method", "Run", "Files", stack("Com-", "pleted"), stack("Not com-", "pleted"), stack("Fallback", "cells"),
           stack("Fallback", "steps"), "Commit(s)"]
    t = Table("arm_status", "tab:arm_status", "", ">{\\raggedright\\arraybackslash}p{4.0cm}l" + "r" * 5
              + ">{\\raggedright\\arraybackslash}p{1.6cm}", hdr, size="\\scriptsize", long=True, tabcolsep="3pt")
    t.row_arms = []
    n = 0
    for name, a in sorted(ctx.ds.arms.items(), key=lambda kv: (kv[1].spec.namespace, kv[1].spec.kind, kv[0])):
        if a.absent or a.spec.namespace:
            continue
        r = a.report
        comp = sum(1 for c in a.cells.values() if c.status_ok)
        nonc = a.n_not_completed()
        nside = len(a.sidecar_crashes)
        commits = sorted({str(c.commit)[:7] for c in a.cells.values()})
        dirty = sorted({str(c.dirty) for c in a.cells.values()})
        fbrec = "" if r.get("fallback_recorded_cells", 0) else "$^{*}$"
        p = f"crStatus{camel(name)}"
        fs = ctx.eng.fileset_all(name)
        ctx.num(p + "Files", len(a.cells), str(len(a.cells)), kind="count", table="tab_arm_status", fileset=fs,
                rule="result files loaded")
        ctx.num(p + "NotCompleted", nonc, str(nonc), kind="count", table="tab_arm_status", fileset=fs,
                rule="result files with status != completed plus crash-sidecar cells without a result file")
        ctx.num(p + "FallbackCells", r["fallback_cells"], str(r["fallback_cells"]), kind="count", table="tab_arm_status",
                fileset=fs, rule="cells with >= 1 surrogate-fallback step")
        # RUN_RULES 2026-10-19 second set (c): the row carries the printed label, never the code name
        name_tex = published.printed_label(ctx, name) if ctx.mode == "cr" else name.replace("_", "\\_")
        com = (",".join(c if c != "None" else "not recorded" for c in commits[:3]) + ("..." if len(commits) > 3 else "")
               + ("" if dirty == ["False"] else " dirty"))
        run = {"cr": "this version", "stored": "submitted"}.get(a.spec.kind, a.spec.kind)
        t.row_arms.append(name)        # code names for the repository record (records/RUN_STATUS.md); not printed
        t.add([name_tex, run, str(len(a.cells)), str(comp), str(nonc) + (f" ({nside} crash)" if nside else ""),
               f"{r['fallback_cells']}{fbrec}", str(r["fallback_steps"]),
               com.replace("_", "\\_").replace(",", ",\\allowbreak{}").replace(" dirty", " (dirty)")])
        n += 1
    t.caption = ("\\textbf{Every loaded method: completed, non-completed and fallback cells, and the code version.} "
                 + ("Methods are named by their printed labels. Their code names are in the Appendix~H cards. "
                    if ctx.mode == "cr" else "")
                 + "Run = rerun for this version, or run of the submitted version (not rerun). Not completed = "
                 "result files whose status is not `completed', plus cells that left only a crash record (a hard crash "
                 "or time-out without a result file). A fallback step is a step whose proposal was replaced by a uniform "
                 "draw because the surrogate fit failed. $^{*}$: the files carry no fallback field (not recorded). "
                 "Commit(s) = the code version recorded in the result files (``dirty'': with uncommitted changes).")
    return t if n else None
