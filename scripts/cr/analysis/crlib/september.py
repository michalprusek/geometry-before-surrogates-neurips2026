"""Stored September runs read by side loaders (never the camera-ready generation; RUN_RULES R-3, R-7).

* tab_e1_extra_tasks   the E-1 rerun (``em1_premise``, commit abbd61e, batch runner) on the tasks OUTSIDE the
                       11-task protocol: per-task cell means of the support x maximiser 2 x 2, the sign of the
                       support effect under candidate-set argmax and of the interaction on the 8 extra eligible
                       tasks, and the interaction / support effect pooled over T18 = T11 + the 7 bounded extra
                       tasks with an exact task-level test. logP (unbounded score) enters sign counts only.
* tab_selection_rule   selection-rule robustness from the stored p1 blocks (commit 4780475, batch runner):
                       support and surrogate contrasts under exact Thompson sampling and under the Doumont
                       linear kernel (calibrated and as shipped), next to the no-surrogate and qLogEI-argmax
                       rows. Label: "September, descriptive, cross-commit".

Both sets hold tasks the arm machinery (config.py / loader.py) does not know, so they are read here directly:
one anchored file name per (solver, task, seed) cell, the JSON's solver/task/seed fields checked against the
name, the provenance commit checked against the stored set's commit (clean tree), every file hashed. A
missing, duplicate, unreadable or contradictory file raises SeptemberError (nothing is dropped silently).

Cell conventions (stated in every caption):
  clean      status completed, >= 30 best-so-far values, no surrogate-fallback step, in every arm of the
             contrast (the generator's E-1 convention, as the E-1 row of tab_factorial_by_box: n = 426 on T11)
  completed  status completed and >= 30 best-so-far values in every arm (cells with fallback steps included;
             RUN_RULES R-5)
  itt        intention to treat, the convention of the September reports: all 40 seeds; an unfinished cell
             keeps the best score it reached inside the budget (a cell without any score would take the best
             of its 5 shared cold-start molecules)
Levels are per-task means; a pooled contrast is the mean of the per-task mean contrasts; the task-level test is
the exact two-sided Wilcoxon signed-rank test on the per-task contrasts, computed by full enumeration of the
sign patterns with midranks (zeros dropped), so it stays exact with ties and for 18 tasks.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from . import TASKS_T11
from . import numbers as N
from . import stats
from .context import Ctx
from .engine import fileset_from_rows
from .latex import Table, m
from .numbers import camel

# ---- task sets ------------------------------------------------------------------------------------------
EXTRA_BOUNDED = {"albu": "albuterol_similarity", "cele": "celecoxib_rediscovery", "isoa": "isomers_c7h8n2o2",
                 "isob": "isomers_c9h10n2o2pf2cl", "mest": "mestranol_similarity", "thio": "thiothixene_rediscovery",
                 "trog": "troglitazone_rediscovery"}
EXTRA_UNBOUNDED = {"logp": "logp"}          # unbounded score: signs only, never pooled
EXCLUDED = {"valt": "valsartan_smarts", "qed": "qed"}   # excluded arm-free by the September plan (section 4.5)
ALL_TASKS = {**TASKS_T11, **EXTRA_BOUNDED, **EXTRA_UNBOUNDED, **EXCLUDED}
T11 = list(TASKS_T11)
X7 = list(EXTRA_BOUNDED)
X8 = X7 + list(EXTRA_UNBOUNDED)
T18 = T11 + X7
T19 = T18 + list(EXTRA_UNBOUNDED)
TASK_LABEL = {"albu": "Albuterol similarity", "cele": "Celecoxib rediscovery",
              "isoa": "Isomers C$_7$H$_8$N$_2$O$_2$", "isob": "Isomers C$_9$H$_{10}$N$_2$O$_2$PF$_2$Cl",
              "mest": "Mestranol similarity", "thio": "Thiothixene rediscovery", "trog": "Troglitazone rediscovery",
              "logp": "logP", "valt": "Valsartan SMARTS", "qed": "QED"}
SEEDS = list(range(42, 82))

E1_SOLVERS = {"SA": "hvarfner_scaled_gp", "BA": "hvarfner_scaled_gp_box_argmax",
              "SC": "hvarfner_scaled_gp_sphere_continuous", "BC": "hvarfner_scaled_gp_box_faithful"}
E1_ROLE = {"SA": "ShellArgmax", "BA": "BoxArgmax", "SC": "ShellCont", "BC": "BoxCont"}
E1_COMMIT = "abbd61e"
E1_DEFAULT_DIR = "/home/prusek/NeurIPS/results/guacamol/em1_premise"
P1_SOLVERS = {"US": "random_search", "UB": "random_box_faithful", "TS": "hvarfner_scaled_gp_ts",
              "TB": "hvarfner_scaled_gp_box_argmax_ts", "LS": "linear_bo_sphere_calib", "LB": "linear_bo_box_argmax",
              "LR": "linear_bo_sphere"}
P1_COMMIT = "4780475"
CONVENTIONS = ("clean", "completed", "itt")
LABEL = "September, descriptive, cross-commit"


class SeptemberError(Exception):
    """A stored September set is incomplete or contradicts its file names."""


# ---- exact task-level test -----------------------------------------------------------------------------
def exact_signed_rank_p(diffs) -> float:
    """Exact two-sided Wilcoxon signed-rank p: zeros dropped, midranks of |d|, the null distribution of W+
    from all 2^n sign patterns (dynamic programme over doubled ranks); p = min(1, 2 min(P(W+ <= w), P(W+ >= w))).
    Equals scipy.stats.wilcoxon(method='exact') on tie-free data and stays exact with ties."""
    d = np.asarray([x for x in diffs if np.isfinite(x) and x != 0], dtype=float)
    if d.size < 2:
        return float("nan")
    r2 = np.rint(2.0 * rankdata(np.abs(d))).astype(int)          # doubled midranks are integers
    w2 = int(r2[d > 0].sum())
    dist = np.zeros(int(r2.sum()) + 1)
    dist[0] = 1.0
    for r in r2:
        new = 0.5 * dist
        new[r:] += 0.5 * dist[:-r]
        dist = new
    return float(min(1.0, 2.0 * min(dist[:w2 + 1].sum(), dist[w2:].sum())))


# ---- side loader -----------------------------------------------------------------------------------------
@dataclass
class SCell:
    task: str
    seed: int
    file: str
    sha256: str
    status_ok: bool
    fallback: int
    trace: np.ndarray        # best-so-far per evaluation
    cold5: float             # best of the 5 shared cold-start molecules (nan when the trace is shorter)

    @property
    def complete(self) -> bool:
        return self.status_ok and self.trace.size >= 30

    @property
    def clean(self) -> bool:
        return self.complete and self.fallback == 0


def _prov(d: dict) -> dict:
    p = d.get("provenance")
    if isinstance(p, dict) and isinstance(p.get("provenance"), dict):
        p = p["provenance"]
    return p if isinstance(p, dict) else {}


def load_block(directory: str, solvers: dict, tasks: list[str], commit: str, seeds=None, suffix: str = "256d") -> dict:
    """{role: {(task short id, seed): SCell}} for every role of ``solvers`` (role -> solver name).

    Raises SeptemberError listing every problem: missing cell, two files for one cell, unreadable JSON, a
    solver/task/seed field that contradicts the file name, a commit other than ``commit`` or a dirty tree."""
    if not os.path.isdir(directory):
        raise SeptemberError(f"directory does not exist: {directory}")
    seeds = list(SEEDS if seeds is None else seeds)
    poli = {ALL_TASKS[t]: t for t in tasks}
    by_solver = {s: r for r, s in solvers.items()}
    rx = re.compile("^ours_(?P<s>" + "|".join(re.escape(s) for s in sorted(by_solver, key=len, reverse=True)) + ")_(?P<t>"
                    + "|".join(re.escape(t) for t in sorted(ALL_TASKS.values(), key=len, reverse=True))
                    + r")_s(?P<seed>\d+)_" + re.escape(suffix) + r"\.json$")
    want = set(seeds)
    found: dict = {}
    problems = []
    for name in sorted(os.listdir(directory)):
        mm = rx.match(name)
        if not mm or mm.group("t") not in poli or int(mm.group("seed")) not in want:
            continue
        key = (by_solver[mm.group("s")], poli[mm.group("t")], int(mm.group("seed")))
        if key in found:
            problems.append(f"duplicate cell {key}: {found[key]} and {name}")
            continue
        found[key] = name
    out = {r: {} for r in solvers}
    for role, solver in solvers.items():
        for t in tasks:
            for sd in seeds:
                name = found.get((role, t, sd))
                if name is None:
                    problems.append(f"missing cell {solver} {t}:{sd}")
                    continue
                path = os.path.join(directory, name)
                try:
                    raw = open(path, "rb").read()
                    d = json.loads(raw)
                except (OSError, json.JSONDecodeError) as e:
                    problems.append(f"unreadable {path}: {e}")
                    continue
                if d.get("solver") != solver or d.get("task") != ALL_TASKS[t] or d.get("seed") is None \
                        or int(d["seed"]) != sd:
                    problems.append(f"{path}: fields solver={d.get('solver')!r} task={d.get('task')!r} "
                                    f"seed={d.get('seed')!r} contradict the file name")
                    continue
                pv = _prov(d)
                if not str(pv.get("git_commit", "")).startswith(commit) or pv.get("git_dirty") is not False:
                    problems.append(f"{path}: provenance commit {pv.get('git_commit')!r} dirty={pv.get('git_dirty')!r}, "
                                    f"expected {commit} and a clean tree")
                    continue
                tr = np.asarray([np.nan if v is None else v for v in (d.get("best_trace") or [])], dtype=float)
                status = str(d.get("status", "absent")).split(":")[0].strip()
                out[role][(t, sd)] = SCell(task=t, seed=sd, file=path, sha256=hashlib.sha256(raw).hexdigest(),
                                           status_ok=(status == "completed"),
                                           fallback=int(d.get("surrogate_fallback_count") or 0), trace=tr,
                                           cold5=float(tr[4]) if tr.size >= 5 else float("nan"))
    if problems:
        raise SeptemberError(f"{len(problems)} problem(s) in {directory}:\n  " + "\n  ".join(problems[:40])
                             + ("\n  ..." if len(problems) > 40 else ""))
    return out


def cell_value(block: dict, role: str, key, conv: str) -> float:
    """best@30 of one cell under a convention ('itt': an unfinished cell keeps its best so far; a cell without
    any score takes the best of the 5 cold-start molecules it shares with the other arms)."""
    c = block[role][key]
    if c.trace.size >= 30:
        return float(c.trace[29])
    if conv != "itt":
        return float("nan")
    fin = c.trace[np.isfinite(c.trace)]
    if fin.size:
        return float(fin[-1])
    for other in block.values():
        o = other.get(key)
        if o is not None and np.isfinite(o.cold5):
            return o.cold5
    raise SeptemberError(f"{role} {key}: no score and no cold-start value to impute")


def eligible(block: dict, roles, tasks, conv: str) -> set:
    """Cells of ``tasks`` usable in every role under the convention."""
    out = set()
    for key in block[roles[0]]:
        if key[0] not in tasks:
            continue
        cs = [block[r].get(key) for r in roles]
        if any(c is None for c in cs):
            continue
        if conv == "itt" or (conv == "completed" and all(c.complete for c in cs)) or \
                (conv == "clean" and all(c.clean for c in cs)):
            out.add(key)
    return out


def rows_of(block: dict, roles, cells) -> list:
    return [(block[r][k].file, block[r][k].sha256) for r in roles for k in cells]


def contrast(vals: dict, terms, cells) -> stats.Paired:
    """Per-cell linear contrast sum(sign * value(role)) tested against zero; p_task replaced by the exact
    enumeration. ``vals``: role -> {cell: value}; ``terms``: [(role, +1 | -1), ...]."""
    cells = sorted(cells)
    x = {k: float(sum(sg * vals[r][k] for r, sg in terms)) for k in cells}
    pr = stats.paired(x, {k: 0.0 for k in cells}, cells, "taskmean")
    pr.p_task = exact_signed_rank_p(list(pr.task_diffs.values()))
    return pr


def _vals(block: dict, roles, cells, conv: str) -> dict:
    return {r: {k: cell_value(block, r, k, conv) for k in cells} for r in roles}


def _subset(pr_tasks: dict, tasks) -> dict:
    return {t: pr_tasks[t] for t in tasks if t in pr_tasks}


def pooled(task_diffs: dict) -> dict:
    """Pooled statistics of a set of per-task contrasts."""
    v = np.asarray([task_diffs[t] for t in sorted(task_diffs)], dtype=float)
    lo, hi = stats.task_bootstrap_ci(task_diffs)
    return {"delta": float(v.mean()) if v.size else float("nan"), "pos": int((v > 0).sum()), "neg": int((v < 0).sum()),
            "n_tasks": int(v.size), "p_task": exact_signed_rank_p(v), "lo": lo, "hi": hi}


def pfmt_task(p) -> str:
    """Exact task-level p: two significant digits; below 1e-4 as m x 10^e (18 tasks reach 7.6e-6)."""
    if p is not None and np.isfinite(p) and p < 1e-4:
        return N.pexp(p)
    return N.psig(p)


def _full_protocol(ctx: Ctx) -> bool:
    cfg = ctx.ds.config
    return set(cfg.tasks) == set(TASKS_T11) and set(cfg.seeds) == set(SEEDS)


def _loud(ctx: Ctx, table: str, what: str, msg: str) -> None:
    """Record a skip and print one banner per run (the tables are rebuilt for every variant)."""
    ctx.skip(table, what, msg)
    if not ctx.cache.get("september_warned_" + table):
        ctx.cache["september_warned_" + table] = True
        print("!" * 100 + f"\n!!! WARNING: {msg} -> {table} omitted\n" + "!" * 100, flush=True)
        ctx.cache.setdefault("loud_warnings", []).append(f"{table}: {msg}")


# =========================================================================================================
# E-1 outside the 11-task protocol
# =========================================================================================================
E1_INT = [("SA", 1), ("BA", -1), ("SC", -1), ("BC", 1)]
E1_SUP = [("SA", 1), ("BA", -1)]


def e1_dir(ctx: Ctx) -> str:
    if ctx.extra.get("e1_dir"):
        return ctx.extra["e1_dir"]
    a = ctx.ds.arms.get("e1_hvarfner_scaled_gp")
    return a.spec.dir if a is not None else E1_DEFAULT_DIR


def e1_analysis(ctx: Ctx) -> dict | None:
    """Per-task means and contrasts of the E-1 2 x 2 on all 21 tasks under the three conventions (cached for
    the whole run: the result does not depend on the variant)."""
    c = ctx.cache
    if "september_e1" in c:
        return c["september_e1"]
    d = e1_dir(ctx)
    if not os.path.isdir(d):
        c["september_e1"] = None
        return None
    block = load_block(d, E1_SOLVERS, list(ALL_TASKS), E1_COMMIT)
    roles = list(E1_SOLVERS)
    out = {"dir": d, "block": block, "conv": {}}
    for conv in CONVENTIONS:
        cells = eligible(block, roles, set(ALL_TASKS), conv)
        vals = _vals(block, roles, cells, conv)
        per = {}
        for t in ALL_TASKS:
            tc = sorted(k for k in cells if k[0] == t)
            if not tc:
                per[t] = {"n": 0, "cells": []}
                continue
            mean = {r: float(np.mean([vals[r][k] for k in tc])) for r in roles}
            per[t] = {"n": len(tc), "cells": tc, "mean": mean, "sup": mean["SA"] - mean["BA"],
                      "int": (mean["SA"] - mean["BA"]) - (mean["SC"] - mean["BC"])}
        prs = {}
        for name, terms in (("sup", E1_SUP), ("int", E1_INT)):
            for sname, ts in (("T11", T11), ("X7", X7), ("T18", T18)):
                cs = {k for k in cells if k[0] in ts}
                prs[(name, sname)] = contrast(vals, terms, cs) if cs else None
        out["conv"][conv] = {"cells": cells, "per": per, "pr": prs}
    out["unfinished"] = {r: sorted(k for k, cl in block[r].items() if k[0] in T19 and not cl.complete) for r in roles}
    c["september_e1"] = out
    return out


def _check_against_arms(ctx: Ctx, A: dict) -> str:
    """The T11 per-task interactions of the side loader must equal what the generator computes from the e1_* arms
    of the sources config (same files, same clean rule). -> 'checked' | 'not checked (...)'. Raises on a mismatch."""
    arms = {r: f"e1_{s}" for r, s in E1_SOLVERS.items()}
    if not all(ctx.has(a) for a in arms.values()):
        return "not checked (the e1_* arms are not loaded in this run)"
    cells = ctx.eng.joint(list(arms.values()), "clean")
    v = {r: ctx.eng.vals(a, cells) for r, a in arms.items()}
    inter = {k: (v["SA"][k] - v["BA"][k]) - (v["SC"][k] - v["BC"][k]) for k in cells}
    ref = stats.task_means(inter, cells)
    mine = A["conv"]["clean"]["pr"][("int", "T11")]
    if len(cells) != mine.n or set(ref) != set(mine.task_diffs) or \
            max(abs(ref[t] - mine.task_diffs[t]) for t in ref) > 1e-12:
        raise SeptemberError("E-1 side loader disagrees with the e1_* arms of the sources config on T11 "
                             f"(n {mine.n} vs {len(cells)})")
    return "checked"


def tab_e1_extra_tasks(ctx: Ctx) -> Table | None:
    table = "tab_e1_extra_tasks"
    if not _full_protocol(ctx):
        ctx.skip(table, "E-1 extra tasks", "run on a task or seed subset (stored 21-task block needs the full protocol)")
        return None
    A = e1_analysis(ctx)
    if A is None:
        _loud(ctx, table, "em1_premise", f"E-1 directory absent: {e1_dir(ctx)}")
        return None
    check = _check_against_arms(ctx, A)
    if "september_e1_arm_check" not in ctx.cache:          # once per run (the table is rebuilt for every variant)
        ctx.cache["september_e1_arm_check"] = check
        print(f"    [september] E-1 side loader against the e1_* arms on T11 (clean cells, per-task interactions): {check}",
              flush=True)
    block = A["block"]
    roles = list(E1_SOLVERS)
    CL, CO = A["conv"]["clean"], A["conv"]["completed"]

    def fs(cells):
        return fileset_from_rows(rows_of(block, roles, cells), [f"e1:{s}" for s in E1_SOLVERS.values()])

    hdr = ["Task", "$n$", "SA", "BA", "SC", "BC", "SA$-$BA", "Interaction"]
    t = Table("e1_extra_tasks", "tab:e1_extra_tasks", "", "lrrrrrrr", hdr, size="\\footnotesize", wide=True)
    for tk in X8 + list(EXCLUDED):
        unb, exc = tk in EXTRA_UNBOUNDED, tk in EXCLUDED
        P = (CO if exc else CL)["per"][tk]              # excluded tasks: completed cells (valsartan has no clean cell)
        if tk == "valt":
            t.rule()
        key = camel(tk)
        ncomp = CO["per"][tk]["n"]
        lab = TASK_LABEL[tk] + ("$^\\dagger$" if unb else "$^\\ddagger$" if exc else "")
        if not P["n"]:
            t.add([lab, f"0 ({ncomp})"] + ["--"] * 6)
            continue
        f = fs(P["cells"])
        fmt, sfmt = (N.f2, N.d2) if unb else (N.f4, N.s4)
        ctx.num(f"crEoneExtra{key}N", P["n"], str(P["n"]), kind="n", table=table, fileset=f, n=P["n"],
                rule=f"E-1, {tk}: seeds {'completed' if exc else 'clean'} in all four arms")
        ctx.num(f"crEoneExtra{key}NCompleted", ncomp, str(ncomp), kind="n", table=table,
                fileset=fs(CO["per"][tk]["cells"]) if ncomp else f, n=ncomp,
                rule=f"E-1, {tk}: seeds completed in all four arms (fallback cells included)")
        for r in roles:
            ctx.num(f"crEoneExtra{key}{E1_ROLE[r]}", P["mean"][r], fmt(P["mean"][r]), kind="level", table=table,
                    fileset=fileset_from_rows(rows_of(block, [r], P["cells"]), [f"e1:{E1_SOLVERS[r]}"]), n=P["n"],
                    rule=f"E-1 per-task mean of best@30, {tk}, cell {r}")
        for nm, k2, what in (("Support", "sup", "support under argmax (SA - BA)"),
                             ("Interaction", "int", "interaction (SA - BA) - (SC - BC)")):
            ctx.num(f"crEoneExtra{key}{nm}", P[k2], sfmt(P[k2]), kind="delta", table=table, fileset=f, n=P["n"],
                    rule=f"E-1 per-task {what}, {tk}" + (" (unbounded score: sign only enters counts)" if unb else ""))
            sg = "+" if P[k2] > 0 else ("-" if P[k2] < 0 else "0")
            ctx.num(f"crEoneExtra{key}{nm}Sign", float(np.sign(P[k2])), sg, kind="sign", table=table, fileset=f,
                    n=P["n"], rule=f"sign of the E-1 per-task {what}, {tk}")
        t.add([lab, f"{P['n']} ({ncomp})" if not exc else str(P["n"])] + [m(fmt(P["mean"][r])) for r in roles]
              + [m(sfmt(P["sup"])), m(sfmt(P["int"]))])

    # ---- pooled block ---------------------------------------------------------------------------------
    blk = []
    conv_lab = {"clean": "clean cells", "completed": "completed cells", "itt": "intention to treat"}
    for sname, ts, slab, convs in (("TEleven", T11, "T11 (the protocol)", ("clean",)),
                                   ("ExtraSeven", X7, "7 extra bounded tasks", ("clean",)),
                                   ("TEighteen", T18, "T18 = T11 + 7", CONVENTIONS)):
        for conv in convs:
            R = A["conv"][conv]
            cs = {k for k in R["cells"] if k[0] in ts}
            f = fs(cs)
            suf = "" if conv == "clean" else camel(conv)
            row = [f"{slab}, {conv_lab[conv]}", N.intfmt(len(cs))]
            for nm, k2, what in (("Support", "sup", "support under argmax (SA - BA)"),
                                 ("Int", "int", "interaction (SA - BA) - (SC - BC)")):
                td = {tk: R["per"][tk][k2] for tk in ts if R["per"][tk]["n"]}
                Pl = pooled(td)
                p = f"crEone{sname}{nm}"
                rule = f"E-1, {slab}, {conv_lab[conv]}: {what}"
                ctx.num(p + "Delta" + suf, Pl["delta"], N.s4(Pl["delta"]), kind="delta", table=table, fileset=f,
                        n=len(cs), rule="mean of the per-task mean contrasts; " + rule)
                ctx.num(p + "CiLo" + suf, Pl["lo"], N.s4(Pl["lo"]), kind="ci", table=table, fileset=f, n=Pl["n_tasks"],
                        rule=f"95 % task-bootstrap interval, lower end (10,000 resamples, seed {stats.BOOT_SEED}); " + rule)
                ctx.num(p + "CiHi" + suf, Pl["hi"], N.s4(Pl["hi"]), kind="ci", table=table, fileset=f, n=Pl["n_tasks"],
                        rule="95 % task-bootstrap interval, upper end; " + rule)
                ctx.num(p + "Tasks" + suf, Pl["pos"], f"{Pl['pos']}/{Pl['n_tasks']}", kind="tasks_won", table=table,
                        fileset=f, n=Pl["n_tasks"], rule="tasks with a positive per-task contrast; " + rule)
                ctx.num(p + "PTask" + suf, Pl["p_task"], pfmt_task(Pl["p_task"]), kind="p_task", table=table, fileset=f,
                        n=Pl["n_tasks"], rule="exact two-sided Wilcoxon signed-rank test on the per-task contrasts "
                                              "(full enumeration, midranks); " + rule)
                row += [m(N.s4(Pl["delta"])), m(f"[{N.s3(Pl['lo'])}, {N.s3(Pl['hi'])}]"), f"{Pl['pos']}/{Pl['n_tasks']}",
                        m(pfmt_task(Pl["p_task"]))]
            ctx.num(f"crEone{sname}N{suf}", len(cs), N.intfmt(len(cs)), kind="n", table=table, fileset=f, n=len(cs),
                    rule=f"E-1 cells, {slab}, {conv_lab[conv]}")
            blk.append(row)
    t.add_block("Pooled over tasks: mean of the per-task contrasts, 95\\,\\% task-bootstrap interval, tasks with a "
                "positive contrast, exact two-sided task-level Wilcoxon $p$", "lrrrrrrrrr",
                ["Tasks, cells", "$n$", "SA$-$BA", "95\\,\\% CI", "Tasks", "$p_{\\mathrm{task}}$", "Interaction",
                 "95\\,\\% CI", "Tasks", "$p_{\\mathrm{task}}$"], blk)

    # ---- sign counts (logP enters here and nowhere else) --------------------------------------------------
    counts = {}
    for conv in CONVENTIONS:
        R = A["conv"][conv]
        suf = "" if conv == "clean" else camel(conv)
        for sname, ts in (("Extra", X8), ("TNineteen", T19)):
            cs = {k for k in R["cells"] if k[0] in ts}
            f = fs(cs)
            for nm, k2 in (("Support", "sup"), ("Interaction", "int")):
                have = [tk for tk in ts if R["per"][tk]["n"]]
                pos = sum(1 for tk in have if R["per"][tk][k2] > 0)
                counts[(conv, sname, k2)] = (pos, len(have))
                ctx.num(f"crEone{sname}{nm}Positive{suf}", pos, f"{pos}/{len(have)}", kind="tasks_won", table=table,
                        fileset=f, n=len(have),
                        rule=f"E-1, {conv_lab[conv]}: tasks with a positive per-task "
                             f"{'support effect under argmax (SA - BA)' if k2 == 'sup' else 'interaction'} among "
                             f"{'the 8 extra eligible tasks (7 bounded + logP)' if sname == 'Extra' else 'T19 = T11 + the 8 extra tasks'}"
                             " (logP by sign only)")
    iso = [CL["per"][tk]["n"] for tk in ("isoa", "isob")], [CO["per"][tk]["n"] for tk in ("isoa", "isob")]
    unf = A["unfinished"]
    n_unf = sum(len(v) for v in unf.values())
    n_runs = len(T19) * len(SEEDS) * len(roles)
    ctx.num("crEoneUnfinishedRuns", n_unf, str(n_unf), kind="count", table=table,
            fileset=fs({k for k in CL["cells"] | A["conv"]["itt"]["cells"] if k[0] in T19}), n=n_runs,
            rule="E-1 runs on T19 (19 tasks x 40 seeds x 4 arms) whose status is not completed or that hold fewer than "
                 "30 best-so-far values")
    unf_txt = (f"{n_unf} of the {N.intfmt(n_runs)} runs on these 19 tasks did not finish ("
               + ", ".join(f"{len(v)} in {r}" for r, v in unf.items() if v) + "). ") if n_unf else ""
    ex, t19 = counts[("clean", "Extra", "sup")], counts[("clean", "TNineteen", "int")]
    exi, t19s = counts[("clean", "Extra", "int")], counts[("clean", "TNineteen", "sup")]
    t.caption = (
        "\\textbf{E-1 outside the 11-task protocol (stored September rerun, commit abbd61e, batch runner; not the "
        "camera-ready generation).} E-1 ran the support $\\times$ maximiser 2$\\times$2 (SA/SC = empirical shell with "
        "candidate-set argmax / projected Adam; BA/BC = $[p_{01},p_{99}]$ box with candidate-set argmax / L-BFGS-B) on "
        "21 GuacaMol tasks. Top: per-task means of best@30 on the 8 extra tasks that the September plan kept as "
        "eligible; $n$ = seeds that are clean (completed, no surrogate-fallback step) in all four arms, in "
        "parentheses the seeds completed in all four. On the two isomer tasks many runs contain surrogate-fallback "
        f"steps (a proposal replaced by a uniform draw): only {iso[0][0]} of {iso[1][0]} and {iso[0][1]} of "
        f"{iso[1][1]} completed seeds are clean. "
        "$^\\dagger$ logP is unbounded, so only its signs enter any count and it is never pooled. "
        "$^\\ddagger$ Excluded arm-free by the range criteria of the September plan (its section 4.5), information "
        "only, means over completed cells; every valsartan score is 0. "
        "Bottom: contrasts pooled over tasks (mean of per-task contrasts; T18 = the 11 protocol tasks + the 7 bounded "
        "extra tasks), under the clean-cell convention and, for T18, also over completed cells (fallback cells "
        "included) and by intention to treat (all 40 seeds; an unfinished run keeps its best score so far; the "
        "convention of the September reports). " + unf_txt +
        f"Signs (clean cells): SA$-$BA is positive on {ex[0]}/{ex[1]} extra tasks and {t19s[0]}/{t19s[1]} of T19 "
        f"(T18 + logP); the interaction on {exi[0]}/{exi[1]} extra tasks and {t19[0]}/{t19[1]} of T19.")
    ctx.extra.setdefault("september_notes", {})["e1_arm_check"] = check
    return t


# =========================================================================================================
# selection-rule robustness (p1 blocks)
# =========================================================================================================
# (group key, group title, [(contrast key, label, a, b)])  with roles: US/UB uniform shell/box, TS/TB Thompson
# sampling shell/box, LS/LR/LB Doumont kernel calibrated shell / as-shipped shell / unit box, SA/BA E-1 qLogEI argmax
SEL_GROUPS = [
    ("None", "No surrogate (one uniform draw per step)",
     [("Support", "support (shell $-$ box)", "US", "UB")]),
    ("Logei", "Vanilla BO GP, qLogEI argmax over 2,048 candidates (E-1 files, commit abbd61e)",
     [("Support", "support (shell $-$ box)", "SA", "BA"),
      ("SurrShell", "surrogate on the shell (GP $-$ uniform)", "SA", "US"),
      ("SurrBox", "surrogate in the box (GP $-$ uniform)", "BA", "UB")]),
    ("Ts", "Vanilla BO GP, exact joint Thompson sample over 2,048 candidates",
     [("Support", "support (shell $-$ box)", "TS", "TB"),
      ("SurrShell", "surrogate on the shell (GP $-$ uniform)", "TS", "US"),
      ("SurrBox", "surrogate in the box (GP $-$ uniform)", "TB", "UB")]),
    ("Lin", "Doumont linear kernel, qLogEI argmax; shell arm calibrated",
     [("Support", "support (shell $-$ box)", "LS", "LB"),
      ("SurrShell", "surrogate on the shell (GP $-$ uniform)", "LS", "US"),
      ("SurrBox", "surrogate in the box (GP $-$ uniform)", "LB", "UB")]),
    ("LinShipped", "Doumont linear kernel, qLogEI argmax; shell arm as shipped",
     [("Support", "support (shell $-$ box)", "LR", "LB"),
      ("SurrShell", "surrogate on the shell (GP $-$ uniform)", "LR", "US")]),
]
SEL_LEVELS = [("US", "Uniform, shell", "UniformShell"), ("UB", "Uniform, box", "UniformBox"),
              ("SA", "qLogEI argmax, shell (E-1)", "LogeiShell"), ("BA", "qLogEI argmax, box (E-1)", "LogeiBox"),
              ("TS", "Thompson, shell", "TsShell"), ("TB", "Thompson, box", "TsBox"),
              ("LS", "Doumont, shell (calibrated)", "LinShell"), ("LR", "Doumont, shell (as shipped)", "LinShellShipped"),
              ("LB", "Doumont, unit box", "LinBox")]


def p1_dir(ctx: Ctx) -> str:
    return ctx.extra.get("p1_sel_dir") or os.path.join(ctx.extra.get("inputs_root", ""), "p1_blocks_sel")


def sel_analysis(ctx: Ctx, conv: str) -> dict | None:
    """Contrasts of the p1 blocks (plus the E-1 argmax cells) on T11 and T18 under one convention (cached)."""
    c = ctx.cache
    key = f"september_sel_{conv}"
    if key in c:
        return c[key]
    d = p1_dir(ctx)
    if not os.path.isdir(d):
        c[key] = None
        return None
    if "september_p1_block" not in c:
        blk = load_block(d, P1_SOLVERS, T18, P1_COMMIT)
        E = e1_analysis(ctx)
        if E is not None:                     # the two E-1 argmax cells join as roles SA and BA
            for r in ("SA", "BA"):
                blk[r] = {k: v for k, v in E["block"][r].items() if k[0] in T18}
        c["september_p1_block"] = blk
    block = c["september_p1_block"]
    roles = [r for r, _, _ in SEL_LEVELS if r in block]
    out = {"dir": d, "block": block, "roles": roles, "sets": {}}
    for sname, ts in (("T11", T11), ("T18", T18)):
        cells = eligible(block, roles, set(ts), conv)
        vals = _vals(block, roles, cells, conv)
        res = {"cells": cells, "pr": {}, "levels": {}}
        for r in roles:
            res["levels"][r] = stats.level(vals[r], cells, "taskmean")
        for gk, _, cons in SEL_GROUPS:
            for ck, _, a, b in cons:
                if a in block and b in block:
                    res["pr"][(gk, ck)] = contrast(vals, [(a, 1), (b, -1)], cells)
        res["fallback_cells"] = {r: sum(1 for k in cells if block[r][k].fallback > 0) for r in roles}
        # the arms of one (task, seed) cell must start from the same five molecules, also across the two commits
        res["coldstart_mismatch"] = sum(1 for r in roles[1:] for k in cells
                                        if not abs(block[r][k].cold5 - block[roles[0]][k].cold5) <= 1e-9)
        out["sets"][sname] = res
    c[key] = out
    return out


def tab_selection_rule(ctx: Ctx) -> Table | None:
    table = "tab_selection_rule"
    if not _full_protocol(ctx):
        ctx.skip(table, "p1 blocks", "run on a task or seed subset (stored September blocks need the full protocol)")
        return None
    conv = "clean" if ctx.var.cell_filter == "clean" else "completed"
    A = sel_analysis(ctx, conv)
    if A is None:
        _loud(ctx, table, "p1_blocks_sel", f"p1 selection-rule set absent: {p1_dir(ctx)}")
        return None
    block = A["block"]
    solver_of = {**P1_SOLVERS, "SA": "e1:" + E1_SOLVERS["SA"], "BA": "e1:" + E1_SOLVERS["BA"]}
    if "SA" not in block:
        ctx.skip(table, "qLogEI rows", "E-1 directory absent (rows omitted)")

    def fs(rs, cells):
        return fileset_from_rows(rows_of(block, rs, cells), [solver_of[r] if r in ("SA", "BA") else f"p1:{solver_of[r]}"
                                                             for r in rs])

    R11, R18 = A["sets"]["T11"], A["sets"]["T18"]
    hdr = ["Contrast", "$\\Delta$", "95\\,\\% CI (tasks)", "Tasks", "$p_{\\mathrm{task}}$", "$d$",
           "$\\Delta^{\\mathrm{T18}}$", "Tasks$^{\\mathrm{T18}}$", "$p_{\\mathrm{task}}^{\\mathrm{T18}}$"]
    t = Table("selection_rule", "tab:selection_rule", "", "lrrrrrrrr", hdr, size="\\footnotesize", wide=True)
    for gk, gtitle, cons in SEL_GROUPS:
        cons = [x for x in cons if (gk, x[0]) in R11["pr"]]
        if not cons:
            continue
        t.group(gtitle)
        for ck, lab, a, b in cons:
            row = [lab]
            for sname, R, pre in (("T11", R11, f"crSel{gk}{ck}"), ("T18", R18, f"crSelTEighteen{gk}{ck}")):
                pr = R["pr"][(gk, ck)]
                lo, hi = pr.ci()
                f = fs([a, b], R["cells"])
                what = f"{solver_of[a]} - {solver_of[b]} on {sname} ({LABEL}; {conv} cells)"
                ctx.num(pre + "Delta", pr.delta, N.s4(pr.delta), kind="delta", table=table, fileset=f, n=pr.n,
                        rule="mean of the per-task mean differences of best@30: " + what)
                ctx.num(pre + "CiLo", lo, N.s4(lo), kind="ci", table=table, fileset=f, n=pr.n_tasks,
                        rule=f"95 % task-bootstrap interval, lower end (10,000 resamples, seed {stats.BOOT_SEED}): " + what)
                ctx.num(pre + "CiHi", hi, N.s4(hi), kind="ci", table=table, fileset=f, n=pr.n_tasks,
                        rule="95 % task-bootstrap interval, upper end: " + what)
                ctx.num(pre + "Tasks", pr.tasks_won, f"{pr.tasks_won}/{pr.n_tasks}", kind="tasks_won", table=table,
                        fileset=f, n=pr.n_tasks, rule="tasks with a positive per-task mean difference: " + what)
                ctx.num(pre + "PTask", pr.p_task, pfmt_task(pr.p_task), kind="p_task", table=table, fileset=f,
                        n=pr.n_tasks, rule="exact two-sided Wilcoxon signed-rank test on the per-task mean differences, "
                                           "native scale (full enumeration, midranks): " + what)
                ctx.num(pre + "D", pr.d, N.d2(pr.d), kind="cohens_d", table=table, fileset=f, n=pr.n,
                        rule="Cohen's d = mean/sd of the cell differences (secondary): " + what)
                if sname == "T11":
                    row += [m(N.s4(pr.delta)), m(f"[{N.s3(lo)}, {N.s3(hi)}]"), f"{pr.tasks_won}/{pr.n_tasks}",
                            m(pfmt_task(pr.p_task)), m(N.d2(pr.d))]
                else:
                    row += [m(N.s4(pr.delta)), f"{pr.tasks_won}/{pr.n_tasks}", m(pfmt_task(pr.p_task))]
            t.add(row)
    lev = []
    for r, lab, key in SEL_LEVELS:
        if r not in block:
            continue
        cells_row = []
        for sname, R, pre in (("T11", R11, "crSelLevel"), ("T18", R18, "crSelTEighteenLevel")):
            v = R["levels"][r]
            ctx.num(pre + key, v, N.f4(v), kind="level", table=table, fileset=fs([r], R["cells"]), n=len(R["cells"]),
                    rule=f"mean of per-task means of best@30, {solver_of[r]}, {sname} matched set ({LABEL}; {conv} cells)")
            nfb = R["fallback_cells"][r]
            ctx.num(pre + key + "FallbackCells", nfb, str(nfb), kind="count", table=table, fileset=fs([r], R["cells"]),
                    n=len(R["cells"]), rule=f"cells of the {sname} matched set with >= 1 surrogate-fallback step, {solver_of[r]}")
            cells_row += [m(N.f4(v)), str(nfb)]
        lev.append([lab] + cells_row)
    all_roles = A["roles"]
    for sname, R, nm in (("T11", R11, "crSelN"), ("T18", R18, "crSelTEighteenN")):
        ctx.num(nm, len(R["cells"]), N.intfmt(len(R["cells"])), kind="n", table=table, fileset=fs(all_roles, R["cells"]),
                n=len(R["cells"]), rule=f"joint matched set of every arm of the table on {sname} ({conv} cells)")
    mism = R18["coldstart_mismatch"]
    ctx.num("crSelColdStartMismatch", mism, str(mism), kind="count", table=table, fileset=fs(all_roles, R18["cells"]),
            n=len(R18["cells"]) * (len(all_roles) - 1),
            rule="(arm, cell) pairs of the T18 matched set whose best@5 differs from the uniform-shell arm's by more "
                 "than 1e-9 (0 = every arm of a cell starts from the same five molecules)")
    cs_txt = ("In every cell all arms start from the same five cold-start molecules (best@5 identical). " if mism == 0
              else f"WARNING: best@5 differs between arms in {mism} (arm, cell) pairs. ")
    t.add_block("Levels on the two matched sets (mean of per-task means of best@30) and the number of cells with a "
                "surrogate-fallback step", "lrrrr",
                ["Arm", "T11", "fallback cells", "T18", "fallback cells"], lev)
    filt = ("completed cells (cells with surrogate-fallback steps included)" if conv == "completed"
            else "clean cells (completed, no surrogate-fallback step)")
    t.caption = (
        f"\\textbf{{Selection-rule robustness of the support and surrogate contrasts ({LABEL}).}} These arms ran in "
        "September at commit 4780475 with the batch runner and are not part of the camera-ready generation; the "
        "qLogEI rows pair the E-1 cells (commit abbd61e) with the uniform arms of commit 4780475, so they cross two "
        "commits. " + cs_txt + "Thompson sampling: one exact joint "
        "posterior sample over the 2,048 candidates per step instead of the qLogEI argmax, same GP; Doumont kernel: "
        "the spherically projected linear kernel with qLogEI argmax, on the shell with a calibrated bound and as "
        "shipped, and on the unit box. One matched set of all arms per task set: "
        f"$n={len(R11['cells'])}$ on the 11 protocol tasks and $n={len(R18['cells'])}$ on T18 (the protocol tasks + "
        f"the 7 bounded extra tasks of E-1), {filt}. $\\Delta$ = mean of the per-task mean differences; 95\\,\\% CI "
        "from 10,000 bootstrap resamples of the tasks; Tasks = tasks with a positive difference; "
        "$p_{\\mathrm{task}}$ = exact two-sided Wilcoxon signed-rank test on the per-task differences in native "
        "units (the September reports gave one-sided tests on a rescaled score); $d$ = mean/sd over cells "
        "(secondary). Uncorrected, descriptive.")
    return t
