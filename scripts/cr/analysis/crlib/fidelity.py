"""Budget fidelity (Appendix H, plan P2-04 / P4-03): how many of the 25 proposals of a cell
produced a new molecule, by the score-equality proxy of cr/C_budget_audit.md section 3.

* in-repo arms ('ours' files): empty slots = proposals scored -1.0 (empty decode, no oracle call);
  repeats = post-cold-start slots whose nonzero score equals an earlier score of the cell exactly
* external harness: empty steps = steps whose own n_evaluated counter did not increase (the
  harness does not score a SMILES it has already scored); repeats = nonzero current_score of
  point-adding steps equal to an earlier added score
* distinct new (estimate) = 25 - (empty + repeats) per cell
The proxy misses repeated zero-score molecules and can count distinct molecules with equal scores.
Also: compute per arm (median elapsed seconds per cell and total process-hours), plus a few
inline numbers that rest on the same tables.
"""
from __future__ import annotations

import math

import numpy as np

from . import numbers as N
from .context import Ctx
from .latex import Table, m
from .numbers import camel
from .tables import conv_text, filt_text


def _repeats(values, start):
    seen, rnz = set(), 0
    for i, v in enumerate(values):
        if v is None or not math.isfinite(v):
            continue
        if i >= start and v in seen and v != 0.0:
            rnz += 1
        seen.add(v)
    return rnz


def cell_fidelity(arm, c) -> tuple[int, int]:
    """-> (empty slots/steps, nonzero repeats) for one cell."""
    if arm.spec.fmt == "ours":
        sc = [float(x) for x in (c.scores[:30] if c.scores is not None else [])]
        empty = sum(1 for v in sc if v == -1.0)
        return empty, _repeats(sc, 5)
    ex = c.extra or {}
    hn, hc = ex.get("h_neval") or [], ex.get("h_current") or []
    prev = ex.get("n_cold_start") or 5
    incs = []
    for v in hn:
        incs.append(v - prev)
        prev = v
    empty = sum(1 for d in incs if d == 0)
    added = [x for x, d in zip(hc, incs) if d > 0]
    return empty, _repeats([float(x) for x in added], 0)


FID_ARMS = [("geowalk", "\\geowalk{}"), ("cma_es_subspace", "\\geocma{}"), ("random_search", "RS (shell)"),
            ("hvarfner_scaled_gp", "Vanilla BO (shell, argmax)"),
            ("hvarfner_scaled_gp_sphere_continuous", "Vanilla BO (shell, Adam)"),
            ("BOX:factorial_argmax", "Vanilla BO (box, argmax)"), ("BOX:hvarfner", "Vanilla BO (box, L-BFGS)"),
            ("BOX:papenmeier_msr", "Papenmeier MSR (box)"), ("BOX:papenmeier_mle", "Papenmeier MLE (box)"),
            ("BOX:xu", "Xu (box)"), ("linear_bo", "Linear BO (box)"), ("linear_bo_sphere_calib", "Linear BO (shell)"),
            ("gabo", "GaBO"), ("matern_gabo", "Riemannian Mat\\'ern GP"), ("rembo", "REMBO"), ("hesbo", "HeSBO"),
            ("turbo_box_faithful", "TuRBO-1 (fixed box)"), ("turbo", "TuRBO$^\\dagger$"), ("baxus", "BAxUS$^\\dagger$"),
            ("lolbo", "LOL-BO$^\\dagger$"), ("invbo", "InvBO$^\\dagger$"), ("cmaes_full", "CMA-ES$^\\dagger$"),
            ("saasbo", "SAASBO$^\\dagger$"), ("cowboys", "COWBOYS$^\\dagger$")]
ROLE = {"BOX:factorial_argmax": "VanillaBoxArgmax", "BOX:hvarfner": "VanillaBox", "BOX:papenmeier_msr": "PapenmeierMsrBox",
        "BOX:papenmeier_mle": "PapenmeierMleBox", "BOX:xu": "XuBox"}


def tab_budget_fidelity(ctx: Ctx) -> Table | None:
    table = "tab_budget_fidelity"
    hdr = ["Arm", "Cells", "Empty / cell", "Repeats / cell", "Distinct new of 25", "Median s / cell", "Process-h"]
    t = Table("budget_fidelity", "tab:budget_fidelity", "", "lrrrrrr", hdr, size="\\footnotesize")
    n = 0
    for spec, lab in FID_ARMS:
        arm = ctx.box(spec.split(":")[1]) if spec.startswith("BOX:") else spec
        if not ctx.has(arm):
            ctx.skip(table, arm, "arm absent (row omitted)")
            continue
        a = ctx.ds.arm(arm)
        cells = sorted(ctx.eng.elig(arm))
        if not cells:
            continue
        emp = rep = 0
        for k in cells:
            e, r = cell_fidelity(a, a.cells[k])
            emp += e
            rep += r
        el = [float(a.cells[k].elapsed) for k in cells if isinstance(a.cells[k].elapsed, (int, float))]
        med = float(np.median(el)) if el else float("nan")
        tot = float(np.sum(el)) / 3600 if el else float("nan")
        distinct = 25 - (emp + rep) / len(cells)
        key = ROLE.get(spec, camel(arm))
        fs = ctx.fs({arm: set(cells)})
        for suf, v, txt, rule in (("Empty", emp / len(cells), N.f2(emp / len(cells)), "empty slots/steps per cell"),
                                  ("Repeats", rep / len(cells), N.f2(rep / len(cells)), "nonzero repeated scores per cell"),
                                  ("Distinct", distinct, N.f2(distinct), "25 - (empty + repeats)/cells"),
                                  ("MedianSeconds", med, f"{med:.1f}", "median elapsed seconds per cell"),
                                  ("ProcessHours", tot, f"{tot:.1f}", "sum of elapsed seconds / 3600")):
            ctx.num(f"crFid{key}{suf}", v, txt, kind="fidelity", table=table, fileset=fs, n=len(cells), rule=rule)
        t.add([ctx.label(arm, lab), str(len(cells)), m(N.f2(emp / len(cells))), m(N.f2(rep / len(cells))), m(N.f2(distinct)),
               m(f"{med:.1f}") if el else "--", m(f"{tot:.1f}") if el else "--"])
        n += 1
    if not n:
        return None
    t.caption = ("\\textbf{Budget fidelity per arm (5 shared cold-start molecules + 25 proposals).} Empty = in-repo "
                 "proposals that decoded to nothing (score $-1$) or external-harness steps whose own evaluation counter "
                 "did not increase; Repeats = proposals whose nonzero score equals an earlier score of the cell (proxy "
                 "for a re-proposed molecule); Distinct new = $25-$(Empty+Repeats). Median s and process-hours from the "
                 f"files' elapsed times (concurrency differs between generations). Cells: {filt_text(ctx)}.")
    return t


def extra_numbers(ctx: Ctx) -> None:
    """Inline numbers that do not belong to one table."""
    T = "numbers_inline"
    if ctx.has("geowalk"):
        cells = ctx.eng.elig("geowalk")
        ctx.reg_level("crGeoWalkMean", "geowalk", cells, T)
        ctx.num("crGeoWalkN", len(cells), N.intfmt(len(cells)), kind="n", table=T, fileset=ctx.fs({"geowalk": cells}),
                n=len(cells), rule="GeoWalk's own eligible cells")
    rec = ctx.reg.records
    for alias, src in (("crRbar", "crGeomRbar"), ("crRbarSd", "crGeomRbarSd")):
        if src in rec:
            r = rec[src]
            ctx.num(alias, r["value"], r["text"], kind=r["kind"], table=T, fileset=ctx.reg.filesets[src], n=r["n"],
                    rule=f"alias of {src}")
    # P7-02: surrogate on the shell vs uniform shell sampling
    for a, key in (("geowalk", "Geowalk"), ("hvarfner_scaled_gp", "VanillaSphere")):
        if ctx.has(a) and ctx.has("random_search"):
            cells = ctx.eng.joint([a, "random_search"])
            pr = ctx.eng.paired(a, "random_search", cells)
            ctx.reg_paired(f"crShellGain{key}", pr, a, "random_search", cells, T)
    # P2-08: shell / candidate-set argmax (Vanilla BO GP) against GaBO (continuous on the sphere, trust region)
    if ctx.has("hvarfner_scaled_gp") and ctx.has("gabo"):
        cells = ctx.eng.joint(["hvarfner_scaled_gp", "gabo"])
        pr = ctx.eng.paired("hvarfner_scaled_gp", "gabo", cells)
        ctx.reg_paired("crShellArgmaxVsGabo", pr, "hvarfner_scaled_gp", "gabo", cells, T)
        # P7-03: Median 1 is a box win (per-task paired d_z on med1)
    box = ctx.box("hvarfner")
    if ctx.has("hvarfner_scaled_gp") and ctx.has(box):
        cells = {c for c in ctx.eng.joint(["hvarfner_scaled_gp", box]) if c[0] == "med1"}
        if len(cells) > 1:
            pr = ctx.eng.paired(box, "hvarfner_scaled_gp", cells, conv="flat")
            fs = ctx.fs({box: cells, "hvarfner_scaled_gp": cells})
            ctx.num("crMedOneBoxMinusShellDelta", pr.delta_flat, N.s4(pr.delta_flat), kind="delta", table=T, fileset=fs,
                    n=pr.n, rule="Median 1: box minus shell (Vanilla BO), paired over seeds")
            ctx.num("crMedOneBoxMinusShellDz", pr.d, N.d2(pr.d), kind="cohens_d", table=T, fileset=fs, n=pr.n,
                    rule="Median 1: d_z of the paired differences")
    # SU-5: range of per-task means across the printed methods of tab_pertask_full
    arms = [x for x in ("geowalk", "cma_es_subspace", "random_search", "hvarfner_scaled_gp", box) if ctx.has(x)]
    if arms:
        cells = ctx.eng.joint(arms)
        tm = [v for a in arms for v in ctx.eng.task_means(a, cells).values()]
        if tm:
            fs = ctx.fs({a: cells for a in arms})
            ctx.num("crPertaskRangeMin", min(tm), N.f2(min(tm)), kind="level", table=T, fileset=fs, n=len(cells),
                    rule="smallest per-task mean over the per-task table's arms")
            ctx.num("crPertaskRangeMax", max(tm), N.f2(max(tm)), kind="level", table=T, fileset=fs, n=len(cells),
                    rule="largest per-task mean over the per-task table's arms")
    # CK-4: result files read by this run (main namespace)
    main = [n for n, a in ctx.ds.arms.items() if not a.absent and not a.spec.namespace]
    nfiles = sum(len(ctx.ds.arms[n].cells) for n in main)
    ctx.num("crFilesLoaded", nfiles, N.intfmt(nfiles), kind="count", table=T,
            fileset=ctx.fs_union(*[ctx.eng.fileset_all(n) for n in main]),
            rule="per-cell result files loaded by this run (main namespace, all arms)")
    crs = [n for n in main if ctx.ds.arms[n].spec.kind == "cr"]
    ncr = sum(len(ctx.ds.arms[n].cells) for n in crs)
    if crs:
        ctx.num("crFilesCameraReady", ncr, N.intfmt(ncr), kind="count", table=T,
                fileset=ctx.fs_union(*[ctx.eng.fileset_all(n) for n in crs]),
                rule="camera-ready result files (provenance-checked)")
    else:
        ctx.skip(T, "crFilesCameraReady", "no camera-ready arm in this run (legacy mode)")
