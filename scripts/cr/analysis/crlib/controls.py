"""The control block (RUN_RULES amendment 2026-10-12, "Control block"; tag cr-controls-2026-10-12 = fd8a03c).

Arms (output root ``cr2026ctl`` of the control clone, config root ${CRCTL}, their own expected commit):

* C1  ``hvarfner_scaled_gp_box_argmax_raw``   the factorial's box x candidate-set cell with the GP fitted on raw
      latents instead of per-coordinate rescaled inputs (input-transform control);
* C2a ``hvarfner_scaled_gp_boxdir_argmax``    uniform [p01,p99]-box draws renormalised to the run's shell radius,
      GP on raw inputs, candidate-set argmax; C2b ``random_boxdir`` the same draws without a surrogate
      (anisotropy control);
* C3  ``<arm>__s82`` for the six decomposition arms: the same solvers on seeds 82-121 (new cold starts), an
      independent replication. The files carry the BASE solver name; they are grouped by provenance.cr_arm and
      live in their own arms with their own seed range, so they can never be pooled with seeds 42-81 (the
      matched set of an ``__s82`` arm and its base arm is empty by construction, and ``assert_not_pooled``
      refuses a table that tries).

tab_controls     levels of the seven arms of C1/C2 on ONE matched set, the contrasts against the shell and the
                 box arms, tasks won, exact task-level p, task-bootstrap interval, fallback counts per arm, and
                 the pre-committed readings of C1 and C2.
tab_replication  the nine pre-registered decomposition contrasts on seeds 82-121 next to seeds 42-81, the 2 x 3
                 levels of both seed blocks, fallback counts per arm, and the pre-committed reading of C3.

The pre-committed readings are computed MECHANICALLY from the rule text of the amendment (quoted in RULE_TEXT):
every rule is a predicate on (tasks with the required sign, number of tasks, exact two-sided task-level
Wilcoxon p), and each outcome prints the verdict string the rule prescribes. Nothing here looks at effect sizes.
A rule is evaluated only when all N_TASKS_RULE = 11 tasks are present ("on >= 8/11 tasks"); otherwise its
verdict is "n/a" (smoke runs). One case is not covered by the rule text and is printed as such: C2 names two
contrasts ("for the GP and for no surrogate") and prescribes a reading when the criterion fails ("is not
positive ...") and when it holds ("if it is"); when it holds for one of the two only, the verdict is
"mixed: the pre-committed rule does not cover this case" together with both sub-results (open question for the
authors, GEN_v3 report).
"""
from __future__ import annotations

import numpy as np

from . import numbers as N
from . import stats
from .context import Ctx
from .decomp import CELL_ROLE, DEC_ARMS, DEC_CONTRASTS, ci_text, dec_compute, decomposition_arms, reg_contrast, stack
from .latex import Table, m, mb
from .tables import conv_text, filt_text

N_TASKS_RULE = 11
MIN_TASKS = 8
ALPHA = 0.05
S82 = "__s82"

RULE_TEXT = {
    "C1": ("if shell $-$ box\\_raw under argmax is positive on $\\ge 8/11$ tasks with $p_{\\mathrm{task}}<0.05$, the "
           "paper says the shell advantage under candidate-set argmax does not depend on the box arm's input "
           "rescaling; otherwise it says that part of the posted support effect reflects the input rescaling, and "
           "the support effect proper is reported as the raw-input contrast"),
    "C2": ("if shell $-$ boxdir is not positive on $\\ge 8/11$ tasks with $p_{\\mathrm{task}}<0.05$ (for the GP and "
           "for no surrogate), ``geometry'' is glossed as the radius; if it is, ``geometry'' is glossed as isotropic "
           "directions at the encoder-mean radius, and the radius alone is not called sufficient"),
    "C3": ("the decomposition is called replicated only if support under argmax and the support $\\times$ maximiser "
           "interaction have the same sign on $\\ge 8/11$ tasks with $p_{\\mathrm{task}}<0.05$ in seeds 82--121; "
           "otherwise ``not replicated on new seeds'' is printed next to the seed-42--81 result"),
}
# verdict strings: (macro text, sentence)
VERDICTS = {
    ("C1", True): ("independent of the input rescaling",
                   "the shell advantage under candidate-set argmax does not depend on the box arm's input rescaling"),
    ("C1", False): ("partly the input rescaling",
                    "part of the posted support effect reflects the input rescaling; the support effect proper is "
                    "reported as the raw-input contrast"),
    ("C2", "radius"): ("the radius", "``geometry'' is glossed as the radius"),
    ("C2", "isotropic"): ("isotropic directions at the encoder-mean radius",
                          "``geometry'' is glossed as isotropic directions at the encoder-mean radius; the radius "
                          "alone is not called sufficient"),
    ("C2", "mixed"): ("mixed (rule does not cover this case)",
                      "mixed: the pre-committed rule does not cover this case (the criterion holds for one of the "
                      "two contrasts only)"),
    ("C3", True): ("replicated on new seeds", "replicated on seeds 82--121"),
    ("C3", False): ("not replicated on new seeds", "not replicated on new seeds"),
    "na": ("n/a", "n/a (the rule needs all 11 tasks)"),
}


# =========================================================================================
# mechanical readings (pure functions of the task-level results)
# =========================================================================================
def criterion(task_diffs: dict, p_task: float, sign: int = +1, min_tasks: int = MIN_TASKS, alpha: float = ALPHA,
              n_rule: int = N_TASKS_RULE) -> dict:
    """'has sign ``sign`` on >= min_tasks of n_rule tasks with p_task < alpha' (p_task = the exact two-sided
    task-level Wilcoxon p of the contrast). ``evaluable`` is False unless exactly n_rule tasks are present."""
    v = np.asarray([task_diffs[k] for k in sorted(task_diffs)], dtype=float)
    k = int((np.sign(v) == sign).sum()) if sign else 0
    evaluable = bool(v.size == n_rule and np.isfinite(p_task))
    met = bool(evaluable and k >= min_tasks and p_task < alpha)
    return {"tasks": k, "n_tasks": int(v.size), "p_task": float(p_task), "sign": int(sign), "evaluable": evaluable,
            "met": met, "min_tasks": min_tasks, "alpha": alpha}


def reading_c1(shell_minus_boxraw) -> dict:
    """C1: shell - box_raw under argmax positive on >= 8/11 tasks with p_task < 0.05."""
    c = criterion(shell_minus_boxraw.task_diffs, shell_minus_boxraw.p_task, +1)
    key = ("C1", c["met"]) if c["evaluable"] else "na"
    return {"rule": "C1", "criteria": {"shell_minus_boxraw": c}, "evaluable": c["evaluable"], "met": c["met"],
            "macro": VERDICTS[key][0], "verdict": VERDICTS[key][1]}


def reading_c2(shell_minus_boxdir_gp, shell_minus_boxdir_none) -> dict:
    """C2: shell - boxdir positive on >= 8/11 tasks with p_task < 0.05, for the GP and for no surrogate.
    Neither -> 'radius'; both -> 'isotropic directions ...'; exactly one -> 'mixed' (not covered by the rule)."""
    cg = criterion(shell_minus_boxdir_gp.task_diffs, shell_minus_boxdir_gp.p_task, +1)
    cn = criterion(shell_minus_boxdir_none.task_diffs, shell_minus_boxdir_none.p_task, +1)
    ev = cg["evaluable"] and cn["evaluable"]
    if not ev:
        key, outcome = "na", "na"
    elif cg["met"] and cn["met"]:
        key, outcome = ("C2", "isotropic"), "isotropic"
    elif not cg["met"] and not cn["met"]:
        key, outcome = ("C2", "radius"), "radius"
    else:
        key, outcome = ("C2", "mixed"), "mixed"
    return {"rule": "C2", "criteria": {"gp": cg, "none": cn}, "evaluable": ev, "outcome": outcome,
            "macro": VERDICTS[key][0], "verdict": VERDICTS[key][1]}


def reading_c3(ref: dict, rep: dict) -> dict:
    """C3: support under argmax and the support x maximiser interaction have, in seeds 82-121 (``rep``), the same
    sign as in seeds 42-81 (``ref``) on >= 8/11 tasks, each with p_task < 0.05 in seeds 82-121. ``ref`` and ``rep``
    map 'SupportArgmax' and 'IntMax' to their stats.Paired results. 'Same sign' is the sign of the seed-42-81
    contrast as printed (Delta of the run's level convention: the mean of per-task means by default); a zero
    reference contrast has no sign and fails."""
    crit, ok, ev = {}, True, True
    for k in ("SupportArgmax", "IntMax"):
        ref_sign = int(np.sign(ref[k].delta)) if np.isfinite(ref[k].delta) else 0
        c = criterion(rep[k].task_diffs, rep[k].p_task, ref_sign)
        c["ref_sign"] = ref_sign
        c["ref_evaluable"] = bool(ref[k].n_tasks == N_TASKS_RULE)
        crit[k] = c
        ev = ev and c["evaluable"] and c["ref_evaluable"]
        ok = ok and c["met"]
    key = ("C3", bool(ok)) if ev else "na"
    return {"rule": "C3", "criteria": crit, "evaluable": ev, "met": bool(ok and ev), "macro": VERDICTS[key][0],
            "verdict": VERDICTS[key][1]}


def _crit_text(c: dict, what: str) -> str:
    sgn = {1: "positive", -1: "negative", 0: "of no sign"}[c["sign"]]
    if not c["evaluable"]:
        return f"{what}: {c['n_tasks']} of {N_TASKS_RULE} tasks present, not evaluated"
    return (f"{what}: {sgn} on {c['tasks']}/{c['n_tasks']} tasks, $p_{{\\mathrm{{task}}}}={N.psig(c['p_task'])}$ "
            f"(criterion {'met' if c['met'] else 'not met'})")


def assert_not_pooled(arms) -> None:
    """A table may not mix an '<arm>__s82' arm with its base arm in one matched set or one level."""
    arms = list(arms)
    for a in arms:
        if a.endswith(S82) and a[:-len(S82)] in arms:
            raise RuntimeError(f"refusing to pool {a} with {a[:-len(S82)]} (seeds 82-121 vs 42-81)")


def _fallback_cells(ctx: Ctx, arm: str) -> tuple[str, int, int, int]:
    """(text 'cells (steps)', fallback cells, fallback steps, not completed) over every loaded file of the arm."""
    A = ctx.ds.arm(arm)
    r = A.report
    txt = f"{r['fallback_cells']} ({r['fallback_steps']})" if r.get("fallback_recorded_cells", 0) else "n.r."
    return txt, int(r["fallback_cells"]), int(r["fallback_steps"]), A.n_not_completed()


def _reg_arm_counts(ctx: Ctx, table: str, prefix: str, arm: str) -> tuple[str, int]:
    txt, fc, fsn, nonc = _fallback_cells(ctx, arm)
    fs = ctx.eng.fileset_all(arm)
    ctx.num(prefix + "FallbackCells", fc, str(fc), kind="count", table=table, fileset=fs,
            rule=f"cells of {arm} with >= 1 surrogate-fallback step (every loaded file)")
    ctx.num(prefix + "FallbackSteps", fsn, str(fsn), kind="count", table=table, fileset=fs,
            rule=f"surrogate-fallback steps of {arm} (every loaded file)")
    ctx.num(prefix + "NotCompleted", nonc, str(nonc), kind="count", table=table, fileset=fs,
            rule=f"result files of {arm} not completed plus crash-sidecar cells")
    return txt, nonc


def _commits(ctx: Ctx, arms) -> str:
    """Short commits recorded in the loaded files of the arms, e.g. 'fd8a03c'."""
    return "/".join(sorted({str(c.commit)[:7] for a in arms for c in ctx.ds.arm(a).cells.values()}))


def _text_macro(ctx: Ctx, name: str, text: str, *, table: str, fileset: dict, rule: str, n=None) -> None:
    ctx.num(name, text, "\\text{" + text + "}", kind="verdict", table=table, fileset=fileset, n=n, rule=rule)


# =========================================================================================
# tab_controls (C1, C2)
# =========================================================================================
CTL_ARMS = [  # key, arm, label, support, GP inputs, macro role
    ("SA", "hvarfner_scaled_gp", "Shell, GP + argmax", "shell at $r$", "raw", "ShellArgmax"),
    ("BA", "hvarfner_scaled_gp_box_argmax", "Box, GP + argmax", "box $[p_{01},p_{99}]$", "rescaled to $[0,1]^D$",
     "BoxArgmax"),
    ("BR", "hvarfner_scaled_gp_box_argmax_raw", "C1: box, GP + argmax, raw inputs", "box $[p_{01},p_{99}]$", "raw",
     "BoxArgmaxRaw"),
    ("DA", "hvarfner_scaled_gp_boxdir_argmax", "C2a: box directions at $r$, GP + argmax", "box draws renormalised to $r$",
     "raw", "BoxdirArgmax"),
    ("S0", "random_search", "Shell, no surrogate", "shell at $r$", "--", "ShellNone"),
    ("B0", "random_box_faithful", "Box, no surrogate", "box $[p_{01},p_{99}]$", "--", "BoxNone"),
    ("D0", "random_boxdir", "C2b: box directions at $r$, no surrogate", "box draws renormalised to $r$", "--",
     "BoxdirNone"),
]
CTL_CONTRASTS = [  # macro key, block, label, (a, b), reading role
    ("COneShellMinusBoxRaw", "C1", "shell $-$ box, GP on raw inputs in both (pre-committed)", ("SA", "BR"), "c1"),
    ("COneBoxRawMinusBox", "C1", "input transform in the box (raw $-$ rescaled inputs)", ("BR", "BA"), None),
    ("COneShellMinusBox", "C1", "shell $-$ box, box inputs rescaled (the factorial's contrast)", ("SA", "BA"), None),
    ("CTwoShellMinusBoxdirGp", "C2", "shell $-$ box directions at $r$, GP + argmax (pre-committed)", ("SA", "DA"), "c2gp"),
    ("CTwoShellMinusBoxdirNone", "C2", "shell $-$ box directions at $r$, no surrogate (pre-committed)", ("S0", "D0"),
     "c2none"),
    ("CTwoBoxdirMinusBoxRawGp", "C2", "box directions at $r$ $-$ box, GP + argmax, raw inputs (radius at box directions)",
     ("DA", "BR"), None),
    ("CTwoBoxdirMinusBoxNone", "C2", "box directions at $r$ $-$ box, no surrogate (radius at box directions)",
     ("D0", "B0"), None),
]


def tab_controls(ctx: Ctx) -> Table | None:
    table = "tab_controls"
    arms = {k: a for k, a, *_ in CTL_ARMS}
    new = [arms[k] for k in ("BR", "DA", "D0")]
    if not any(ctx.has(a) for a in new):
        for a in new:
            ctx.skip(table, a, "control-block arm absent (table omitted)")
        return None
    missing = [a for a in arms.values() if not ctx.has(a)]
    if missing:
        # the readings are defined on ONE matched set of all seven arms; a partial block is not tabulated
        for a in missing:
            ctx.skip(table, a, "arm of the control table absent (table omitted: the contrasts share one matched set)")
        return None
    assert_not_pooled(arms.values())
    cells = ctx.eng.joint(list(arms.values()))
    if not cells:
        ctx.skip(table, "matched set", "empty")
        return None
    vals = {k: ctx.eng.vals(a, cells) for k, a in arms.items()}
    fs_all = ctx.fs({a: cells for a in arms.values()})
    ctx.num("crCtlN", len(cells), N.intfmt(len(cells)), kind="n", table=table, fileset=fs_all, n=len(cells),
            rule="joint matched set of the seven arms of the control table (C1, C2)")
    hdr = ["Arm", "Candidates", "GP inputs", "Best@30", stack("Fallback", "cells (steps)"), stack("Not", "compl.")]
    t = Table("controls", "tab:controls", "", "p{4.3cm}p{3.2cm}p{2.0cm}rrr", hdr, size="\\footnotesize", wide=True)
    lev = {}
    for k, a, lab, sup, gpin, role in CTL_ARMS:
        lev[k] = stats.level(vals[k], cells, ctx.var.conv)
        ctx.num(f"crCtlLevel{role}", lev[k], N.f4(lev[k]), kind="level", table=table, fileset=ctx.fs({a: cells}),
                n=len(cells), rule=f"{conv_text(ctx)} of best@30 of {a}, joint matched set of the control table")
        fb, nonc = _reg_arm_counts(ctx, table, f"crCtlArm{role}", a)
        t.add([lab, sup, gpin, m(N.f4(lev[k])), fb, str(nonc)])
    res = {}
    blk = []
    for key, block, lab, (a, b), role in CTL_CONTRASTS:
        pr = stats.paired(vals[a], vals[b], cells, ctx.var.conv)
        res[key] = pr
        fs = ctx.fs({arms[a]: cells, arms[b]: cells})
        lo, hi = reg_contrast(ctx, f"crCtl{key}", pr, fs, table, f"{arms[a]} - {arms[b]}")
        blk.append([lab, m(N.s4(pr.delta)), m(ci_text(lo, hi)), f"{pr.tasks_won}/{pr.n_tasks}", m(N.psig(pr.p_task)),
                    m(N.d2(pr.d)), f"{pr.wins}/{pr.losses}"])
    t.add_block(f"Contrasts on the same matched set ($n={len(cells)}$ cells): $\\Delta$, 95\\,\\% task-bootstrap interval, "
                "tasks with a positive per-task contrast, exact two-sided task-level Wilcoxon $p$ (uncorrected, as "
                "pre-committed); $d$ and W/L over cells are secondary",
                "p{7.0cm}rrrrrr", ["Contrast", "$\\Delta$", "95\\,\\% CI", "Tasks", "$p_{\\mathrm{task}}$", "$d$", "W/L"],
                blk)
    r1 = reading_c1(res["COneShellMinusBoxRaw"])
    r2 = reading_c2(res["CTwoShellMinusBoxdirGp"], res["CTwoShellMinusBoxdirNone"])
    fs1 = ctx.fs({arms["SA"]: cells, arms["BR"]: cells})
    fs2 = ctx.fs({arms[k]: cells for k in ("SA", "DA", "S0", "D0")})
    for rd, fs, tag in ((r1, fs1, "COne"), (r2, fs2, "CTwo")):
        _text_macro(ctx, f"crCtl{tag}Verdict", rd["macro"], table=table, fileset=fs, n=N_TASKS_RULE,
                    rule=f"pre-committed reading {rd['rule']} (RUN_RULES amendment 2026-10-12), computed mechanically: "
                         + rd["verdict"].replace("``", "'").replace("''", "'"))
        for ck, c in rd["criteria"].items():
            nm = f"crCtl{tag}Criterion" + {"shell_minus_boxraw": "", "gp": "Gp", "none": "None"}[ck]
            _text_macro(ctx, nm, ("n/a" if not c["evaluable"] else ("met" if c["met"] else "not met")), table=table,
                        fileset=fs, n=c["n_tasks"],
                        rule=f"criterion 'positive on >= {MIN_TASKS}/{N_TASKS_RULE} tasks with p_task < {ALPHA}': "
                             f"{c['tasks']}/{c['n_tasks']} tasks, p_task {c['p_task']:.4g}")
    t.notes = ("\\textbf{Pre-committed readings (amendment of 2026-10-12 to the run rules released with the code, fixed before any production result "
               "existed), evaluated mechanically.} "
               f"C1 rule: {RULE_TEXT['C1']}. Result: "
               + _crit_text(r1["criteria"]["shell_minus_boxraw"], "shell $-$ box\\_raw") + ". "
               f"\\textbf{{Reading C1: {r1['verdict']}.}} "
               f"C2 rule: {RULE_TEXT['C2']}. Result: " + _crit_text(r2["criteria"]["gp"], "GP")
               + "; " + _crit_text(r2["criteria"]["none"], "no surrogate") + ". "
               f"\\textbf{{Reading C2: {r2['verdict']}.}}")
    t.caption = ("\\textbf{Control block: input-transform control (C1) and anisotropy control (C2).} Best@30, "
                 f"{conv_text(ctx)}, on one matched set of the seven arms, $n={len(cells)}$ cells ({filt_text(ctx)}), "
                 "seeds 42--81. C1 repeats the box $\\times$ candidate-set cell with the GP fitted on raw latents (as "
                 "the shell arm fits it) instead of inputs rescaled per coordinate to $[0,1]^D$; both box arms score "
                 "the same 2,048 uniform box draws per step. C2 renormalises uniform $[p_{01},p_{99}]$-box draws to the "
                 "run's shell radius $r$, with the GP (raw inputs, argmax) and without a surrogate, so its candidates "
                 "share the shell's radius and keep the box's directions. The three control arms were run from the "
                 f"control tag (commit {_commits(ctx, new)}), the other four from the main campaign's commit "
                 f"({_commits(ctx, [a for a in arms.values() if a not in new])}); the shared code was shown "
                 "bit-identical at both commits before the block started. Fallback = cells with at least one step whose "
                 "proposal was replaced by a uniform draw from the arm's own support (steps in parentheses; n.r. = not "
                 "recorded), over every loaded file of the arm.")
    return t


# =========================================================================================
# tab_replication (C3)
# =========================================================================================
def replication_arms(ctx: Ctx) -> tuple[dict | None, list]:
    arms = {k: a + S82 for k, a in DEC_ARMS.items()}
    missing = [a for a in arms.values() if not ctx.has(a)]
    return (None if missing else arms), missing


def tab_replication(ctx: Ctx) -> Table | None:
    table = "tab_replication"
    rep_arms, missing = replication_arms(ctx)
    if rep_arms is None:
        for a in missing:
            ctx.skip(table, a, "replication arm (seeds 82-121) absent (table omitted)")
        return None
    base_arms, legacy_b0, miss0 = decomposition_arms(ctx)
    if base_arms is None or legacy_b0:
        ctx.skip(table, ",".join(miss0) or "legacy B0", "decomposition arms of seeds 42-81 absent (table omitted)")
        return None
    assert_not_pooled(list(base_arms.values()))
    assert_not_pooled(list(rep_arms.values()))
    R1 = dec_compute(ctx, base_arms)             # seeds 42-81, its own joint matched set
    R2 = dec_compute(ctx, rep_arms)              # seeds 82-121, its own joint matched set
    c1, c2 = R1["cells"], R2["cells"]
    if not c1 or not c2:
        ctx.skip(table, "matched set", "empty")
        return None
    if {s for _, s in c1} & {s for _, s in c2}:
        raise RuntimeError("tab_replication: the two seed blocks share a seed (they must be disjoint)")
    seeds2 = sorted({s for _, s in c2})
    fs2 = ctx.fs({a: c2 for a in rep_arms.values()})
    ctx.num("crRepN", len(c2), N.intfmt(len(c2)), kind="n", table=table, fileset=fs2, n=len(c2),
            rule="joint matched set of the six replication arms (seeds 82-121)")
    R25 = dec_compute(ctx, rep_arms, 5)
    hdr = ["Contrast", stack("$\\Delta$", "42--81"), "Tasks", "$p_{\\mathrm{task}}$", stack("$\\Delta$", "82--121"),
           "95\\,\\% CI", "Tasks", "$p_{\\mathrm{task}}$", stack("Same sign", "as 42--81")]
    t = Table("replication", "tab:replication", "", "p{4.4cm}rrrrrrrr", hdr, size="\\footnotesize", wide=True)
    for k, lab, _, _ in DEC_CONTRASTS:
        p1, p2 = R1["pr"][k], R2["pr"][k]
        lo, hi = reg_contrast(ctx, f"crRep{k}", p2, fs2, table, lab + " (seeds 82-121)")
        sgn = int(np.sign(p1.delta)) if np.isfinite(p1.delta) else 0
        same = int((np.sign(np.asarray(list(p2.task_diffs.values()))) == sgn).sum()) if sgn else 0
        ctx.num(f"crRep{k}SameSign", same, f"{same}/{p2.n_tasks}", kind="tasks_won", table=table, fileset=fs2,
                n=p2.n_tasks, rule="tasks whose per-task contrast in seeds 82-121 has the sign of the seed-42-81 "
                                   "task-mean contrast")
        pre = k in ("SupportArgmax", "IntMax")
        t.add([lab + ("$^\\star$" if pre else ""), m(N.s4(p1.delta)), f"{p1.tasks_won}/{p1.n_tasks}", m(N.psig(p1.p_task)),
               m(N.s4(p2.delta)), m(ci_text(lo, hi)), f"{p2.tasks_won}/{p2.n_tasks}",
               mb(N.psig(p2.p_task), bool(p2.p_task < ALPHA)), f"{same}/{p2.n_tasks}"])
    L1, L2, L25 = R1["levels"], R2["levels"], R25["levels"]
    rows = []
    for blk_lab, L, key in (("seeds 42--81", L1, None), ("seeds 82--121", L2, "crRepLevel")):
        for sup, ks in (("shell", ("S0", "SA", "SC")), ("box $[p_{01},p_{99}]$", ("B0", "BA", "BC"))):
            rows.append([blk_lab, sup] + [m(N.f4(L[x])) for x in ks])
        if key:
            for x in DEC_ARMS:
                ctx.num(f"{key}{CELL_ROLE[x]}", L[x], N.f4(L[x]), kind="level", table=table,
                        fileset=ctx.fs({rep_arms[x]: c2}), n=len(c2),
                        rule=f"{conv_text(ctx)} of best@30, seeds 82-121, joint matched set of the six replication arms")
    ctx.num("crRepBestFive", L25["S0"], N.f4(L25["S0"]), kind="level", table=table, fileset=ctx.fs({rep_arms["S0"]: c2}),
            n=len(c2), rule="best of the 5 shared cold-start molecules, seeds 82-121 (new cold starts)")
    t.add_block(f"Levels (best@30, {conv_text(ctx)}): seeds 42--81 on their matched set ($n={len(c1)}$), seeds 82--121 "
                f"on theirs ($n={len(c2)}$; best@5 of the new cold starts ${N.f4(L25['S0'])}$)", "llrrr",
                ["Seeds", "Support", "No surrogate", "GP + argmax", "GP + continuous"], rows)
    frows = []
    for x in DEC_ARMS:
        fb1, n1 = _fallback_cells(ctx, base_arms[x])[0], ctx.ds.arm(base_arms[x]).n_not_completed()
        fb2, n2 = _reg_arm_counts(ctx, table, f"crRepArm{CELL_ROLE[x]}", rep_arms[x])
        frows.append([base_arms[x].replace("_", "\\_"), fb1, str(n1), fb2, str(n2)])
    t.add_block("Surrogate-fallback cells (steps) and non-completed cells per arm, over every loaded file",
                "lrrrr", ["Arm", "Fallback, 42--81", "Not compl.", "Fallback, 82--121", "Not compl."], frows)
    rd = reading_c3(R1["pr"], R2["pr"])
    _text_macro(ctx, "crRepVerdict", rd["macro"], table=table, fileset=ctx.fs_union(fs2, ctx.fs({a: c1 for a in base_arms.values()})),
                n=N_TASKS_RULE, rule="pre-committed reading C3 (RUN_RULES amendment 2026-10-12), computed mechanically: "
                                     + rd["verdict"])
    for k, c in rd["criteria"].items():
        _text_macro(ctx, f"crRepCriterion{k}", ("n/a" if not (c["evaluable"] and c["ref_evaluable"]) else
                                                ("met" if c["met"] else "not met")), table=table, fileset=fs2,
                    n=c["n_tasks"],
                    rule=f"criterion 'same sign as seeds 42-81 on >= {MIN_TASKS}/{N_TASKS_RULE} tasks with p_task < "
                         f"{ALPHA} in seeds 82-121': {c['tasks']}/{c['n_tasks']} tasks, p_task {c['p_task']:.4g}, "
                         f"reference sign {c['ref_sign']:+d}")
    lab_of = {k: lab for k, lab, _, _ in DEC_CONTRASTS}
    t.notes = ("\\textbf{Pre-committed reading C3 (amendment of 2026-10-12 to the run rules released with the code), evaluated mechanically.} Rule: "
               f"{RULE_TEXT['C3']}. Result ($^\\star$ rows, seeds 82--121): "
               + "; ".join(_crit_text(dict(c, sign=c["ref_sign"]), lab_of[k]) for k, c in rd["criteria"].items())
               + f". \\textbf{{Reading C3: {rd['verdict']}.}} ``Same sign'' is the sign of the seed-42--81 task-mean "
               "contrast; the other seven rows are shown for information.")
    t.caption = ("\\textbf{Replication of the decomposition on new seeds (control C3).} The nine pre-registered "
                 "contrasts of Table~\\ref{tab:decomposition_contrasts} on seeds 42--81 and, independently, on seeds "
                 f"{seeds2[0]}--{seeds2[-1]} (new cold-start molecules; the same six solvers, run from the control tag, "
                 f"commit {_commits(ctx, rep_arms.values())}, whose shared code was shown bit-identical to the main "
                 f"campaign's commit {_commits(ctx, base_arms.values())}). The two "
                 "seed blocks are never pooled: each has its own joint matched set of the six arms "
                 f"($n={len(c1)}$ and $n={len(c2)}$ cells, {filt_text(ctx)}). $\\Delta$ = {conv_text(ctx)} of the paired "
                 "cell contrast; Tasks = tasks with a positive per-task contrast; $p_{\\mathrm{task}}$ = exact two-sided "
                 "Wilcoxon signed-rank test over the per-task contrasts (uncorrected; bold: $<0.05$); 95\\,\\% CI from "
                 "10,000 task-bootstrap resamples.")
    return t
