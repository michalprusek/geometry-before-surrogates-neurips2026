"""Published-recipe block: verdicts, printed names, the baseline family and the verification table.

Rules (RUN_RULES):
  * 2026-10-12, published-recipe block: (1) a row carries a published method's name only if its implementation
    follows the released code AND the recipe passed the verification; every other arm is printed under a
    descriptive label (crlib/labels.py). (4) Published-recipe rows enter Table 1, Appendix H and the LassoBench
    table as baseline comparisons: GeoWalk and the GP on the shell against each published row, task-level exact
    Wilcoxon primary, Benjamini--Hochberg within this baseline family; a recipe that is not significantly worse
    is described as not worse, one that is better as better, in a sentence of its own.
  * 2026-10-13 revision 1 (c): Hvarfner passes if the 30-point or the 41-point design passes, both verdicts are
    reported; MSR / MLE (scaled): exactly one noise variant passes -> it is printed; both -> learned. (g): the arm
    printed under the Hvarfner name is the one of ``hvarfner_published`` / ``hvarfner_published_allobs`` with the
    higher mean, over the 11 tasks, of the per-task mean best score after 30 evaluations; both appear in Appendix H.
  * 2026-10-19: the verification ends with the recipes whose runs are complete (Hvarfner, Linear BO, Xu, MSR and
    MLE (scaled) with learned noise); BAxUS, TuRBO and CMA-ES are "not verified" and keep descriptive labels; the
    frozen-noise runs were stopped unread after the learned variants passed, so the printed variant is learned.

The verdicts come from ONE file, ``verdicts.json`` of the summarize pass of 2026-10-19 (kajman
``/data2/prusek/cr_verify/summarize_2026-10-19/verdicts.json``), pinned by sha256: any other file stops the run.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from . import labels
from . import numbers as N
from . import stats
from .context import Ctx
from .latex import WRAP, Table, m, mb, stack, wrapp
from .numbers import camel

VERDICTS_SHA256 = "cd22e4006d0617b58fd880ea257ada54025cbf63bfb1410af632dd1c190bef33"
VERDICTS_TAG_COMMIT = "963ef040cae648d4132e1d3330a2a9c06456e97d"     # tag cr-verify-2026-10-13
TARGETS_SHA256 = "8b0507645cf5141613190ec502a2d4e7a3f4587ef39c6782e44e8b48da47bfc1"
ALPHA = 0.05
NOT_VERIFIED_RULE = "not verified: its verification runs were not complete when the verification ended"

#: recipe -> verdict entries (recipe, n_init) of verdicts.json, the name printed when it passed, and its arms.
RECIPES = {
    "hvarfner": {"verdicts": [("hvarfner", 30), ("hvarfner", 41)], "rule": "either design passes",
                 "name": "Vanilla BO (Hvarfner et al.)", "plain": "Vanilla BO (Hvarfner et al.)",
                 "target": "Hvarfner et al., Fig.~7, Lasso-DNA", "guacamol": ["hvarfner_published", "hvarfner_published_allobs"],
                 "lasso": ["hvarfner_published"]},
    "msr": {"verdicts": [("msr_noise_learned", 40)], "rule": "learned noise",
            "name": "MSR (Papenmeier et al.)", "plain": "MSR (Papenmeier et al.)",
            "target": "Papenmeier et al., Fig.~8, Lasso-DNA", "guacamol": ["papenmeier_msr_reference_learned"],
            "lasso": ["papenmeier_published_learned"]},
    "mle_scaled": {"verdicts": [("mle_scaled_noise_learned", 40)], "rule": "learned noise",
                   "name": "MLE (scaled) (Papenmeier et al.)", "plain": "MLE (scaled) (Papenmeier et al.)",
                   "target": "Papenmeier et al., Fig.~8, Lasso-DNA",
                   "guacamol": ["papenmeier_mle_scaled_reference_learned"],
                   "lasso": ["papenmeier_mle_scaled_published_learned"]},
    "xu": {"verdicts": [("xu", 20)], "rule": "", "name": "Standard GP (Xu et al.)", "plain": "Standard GP (Xu et al.)",
           "target": "Xu et al., Fig.~6, Lasso-DNA", "guacamol": ["xu_published"], "lasso": ["xu_published"]},
    "doumont": {"verdicts": [("doumont", 30)], "rule": "", "name": "Linear BO (Doumont et al.)",
                "plain": "Linear BO (Doumont et al.)", "target": "Doumont et al., Fig.~2, Lasso-DNA",
                "guacamol": ["linear_bo_box8"], "lasso": ["linear_bo_published"]},
    "turbo": {"verdicts": [], "rule": NOT_VERIFIED_RULE, "name": None, "plain": None,
              "target": "LassoBench result files, synthetic simple and medium", "guacamol": ["turbo_published"],
              "lasso": ["turbo_published"]},
    "baxus": {"verdicts": [], "rule": NOT_VERIFIED_RULE, "name": None, "plain": None,
              "target": "Papenmeier et al. (2022), Fig.~3, Lasso-Hard and Lasso-High", "guacamol": [],
              "lasso": ["baxus_official"]},
    "cmaes": {"verdicts": [], "rule": NOT_VERIFIED_RULE, "name": None, "plain": None,
              "target": "LassoBench result files, four synthetic tasks", "guacamol": [], "lasso": ["cmaes_full_published"]},
}
#: frozen-noise variants: production arms that are not printed (the printed variant is the learned one)
FROZEN = {"msr": ("msr", 40), "mle_scaled": ("mle_scaled", 40)}
GUACAMOL_ARMS = [a for r in RECIPES.values() for a in r["guacamol"]]
LASSO_METHODS = [a for r in RECIPES.values() for a in r["lasso"]]
#: the order of the printed GuacaMol rows (Table 1 group and tab_published); "hvarfner" resolves to the chosen arm
GUACAMOL_ROWS = [("hvarfner", "PubHvarfner"), ("msr", "PubMsr"), ("mle_scaled", "PubMle"), ("xu", "PubXu"),
                 ("doumont", "PubLinear"), ("turbo", "PubTurbo")]
GUACAMOL_SUPPORT = {"linear_bo_box8": "box $[-8,8]^D$", "turbo_published": "trust region in $[p_{01},p_{99}]$"}
REFS = [("geowalk", "Geowalk", "\\geowalk{}"), ("hvarfner_scaled_gp", "ShellGp", "the GP on the shell")]


class PublishedInputError(RuntimeError):
    """The verdict file is absent, has another sha256, or contradicts the protocol."""


def recipe_of(arm: str, bench: str = "guacamol") -> str | None:
    key = "guacamol" if bench == "guacamol" else "lasso"
    for r, d in RECIPES.items():
        if arm in d[key]:
            return r
    return None


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_verdicts(path: str) -> dict:
    """{recipe: {"verified": bool, "entries": [verdict dicts], "path": path}} from the pinned verdict file."""
    if not path or not os.path.exists(path):
        raise PublishedInputError(f"verdict file {path!r} is absent (copy it with tools/sync_verification.sh)")
    sha = _sha256(path)
    if sha != VERDICTS_SHA256:
        raise PublishedInputError(f"verdict file {path} has sha256 {sha}, expected {VERDICTS_SHA256} "
                                  "(the summarize pass of 2026-10-19)")
    d = json.load(open(path))
    if d.get("targets_sha256") != TARGETS_SHA256 or d.get("targets_sha256_expected") != TARGETS_SHA256:
        raise PublishedInputError(f"verdict file {path}: targets sha256 {d.get('targets_sha256')} is not the one of "
                                  "the verification protocol")
    prov = d.get("summarizer_provenance") or {}
    if prov.get("git_commit_full") != VERDICTS_TAG_COMMIT or prov.get("git_dirty") is not False:
        raise PublishedInputError(f"verdict file {path}: summarize ran at {prov.get('git_commit_full')!r} "
                                  f"(dirty {prov.get('git_dirty')!r}), not at the clean tag commit")
    by = {(v["recipe"], v["n_init"]): v for v in d["verdicts"]}
    out = {}
    for r, spec in RECIPES.items():
        ents = []
        for key in spec["verdicts"]:
            if key not in by:
                raise PublishedInputError(f"verdict file {path}: no verdict for {key}")
            v = by[key]
            if v.get("refusals"):
                raise PublishedInputError(f"verdict {key} carries refusals {v['refusals']}")
            if v.get("commits") != [VERDICTS_TAG_COMMIT]:
                raise PublishedInputError(f"verdict {key}: runs from commits {v.get('commits')}")
            ents.append(v)
        verified = bool(ents) and any(e["pass"] is True for e in ents)
        out[r] = {"verified": verified, "entries": ents, "path": path}
    for fk in FROZEN.values():         # the frozen variants must NOT have a verdict (their runs were stopped unread)
        if fk in by:
            raise PublishedInputError(f"verdict file {path} carries a verdict for {fk}, which RUN_RULES 2026-10-19 "
                                      "stopped unread")
    return out


def verdicts(ctx: Ctx) -> dict | None:
    """The verdicts (cached); None in legacy mode. A camera-ready run with a published arm needs the file."""
    c = ctx.cache
    if "verdicts" not in c:
        if ctx.mode != "cr":
            c["verdicts"] = None
        else:
            c["verdicts"] = load_verdicts(ctx.extra.get("verdicts_path"))
    return c["verdicts"]


def verified(ctx: Ctx, recipe: str) -> bool:
    V = verdicts(ctx)
    return bool(V and V[recipe]["verified"])


#: the GuacaMol arm of a recipe whose settings the verification ran (Hvarfner: the in-box qLogNEI baseline, as Ax);
#: another arm of the same recipe is a variant that was not verified itself
VERIFIED_SETTING_ARMS = {"hvarfner": ("hvarfner_published",)}
#: printed verification status (Tables 13, 14 and 15 say the same; RUN_RULES 2026-10-29 (4))
STATUS_YES, STATUS_ALLOWANCE, STATUS_NO = "yes", "yes, with allowance", "no"


def _and(names: list) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def recipe_status(ctx: Ctx, recipe: str) -> str:
    """'yes' if a design of the recipe passed also in the strict reading, 'yes, with allowance' if it passed only with
    the allowances of the protocol, 'no' if it was not verified."""
    V = verdicts(ctx)
    if not V or not V[recipe]["verified"]:
        return STATUS_NO
    ents = V[recipe]["entries"]
    return STATUS_YES if any(e["pass"] and e["pass_strict"] for e in ents) else STATUS_ALLOWANCE


def verification_status(ctx: Ctx, arm: str, bench: str = "guacamol") -> str:
    """The status of a published-recipe arm: the recipe's status, or 'no' for a GuacaMol variant of a recipe whose
    verification ran other settings (the all-observations Hvarfner run)."""
    r = recipe_of(arm, bench)
    if r is None:
        return STATUS_NO
    if bench == "guacamol" and r in VERIFIED_SETTING_ARMS and arm not in VERIFIED_SETTING_ARMS[r]:
        return STATUS_NO
    return recipe_status(ctx, r)


def name_or_label(ctx: Ctx, arm: str, bench: str = "guacamol", plain: bool = False) -> str:
    """The printed name of a published-recipe arm: the authors' method name if its recipe was verified, else the
    descriptive label followed by 'not verified'."""
    r = recipe_of(arm, bench)
    if r is not None and verified(ctx, r):
        return RECIPES[r]["plain" if plain else "name"]
    if bench == "guacamol":
        lab = labels.plain(arm) if plain else labels.label(arm)
    else:
        lab = labels.lasso_plain(arm, arm) if plain else labels.lasso_label(arm, arm)
    return lab + (" (not verified)" if r is not None else "")


def printed_label(ctx: Ctx, arm: str) -> str:
    """The label a GuacaMol arm carries where it stands alone -- an Appendix H card title, a row of the arm-status
    table (RUN_RULES 2026-10-19 second set (c)): the authors' name only for the arm that carries a verified recipe's
    name, otherwise the descriptive label (crlib/labels.py) with the arm's support; a stored budget or codec arm is
    named by its prefix and the arm it repeats; any other arm (GeoWalk variants, controls) by its config label.
    Never the code name."""
    r = recipe_of(arm)
    if r is not None:
        if r == "hvarfner" and arm != hvarfner_arm(ctx):
            return labels.label(arm) + " (the other run of the same recipe)"
        return name_or_label(ctx, arm)
    if arm in labels.DESCRIPTIVE:
        sup = labels.SUPPORT.get(arm)
        lab = labels.label(arm) + (f", {sup}" if sup else "")
        for pre, txt in labels.STORED_PREFIX:
            if arm.startswith(pre):
                return f"{txt}: {lab}"
        return lab
    for pre, txt in labels.STORED_PREFIX:
        if arm.startswith(pre):
            return f"{txt}: {printed_label(ctx, arm[len(pre):])}"
    if ctx.ds is not None and arm in ctx.ds.arms:
        from .baselines import tex_escape            # a config label is plain text (it may hold %, ~, _)
        lab = ctx.label(arm, ctx.ds.arms[arm].spec.label)
        return lab if "$" in lab or "\\" in lab else tex_escape(lab)
    raise KeyError(f"no printed label for arm {arm!r}")


def other_row_label(ctx: Ctx, row_arm: str) -> str:
    """Card title of a facts row of another benchmark (``lassobench_pub:<method>``, ``lassobench:<method>``,
    ``nasbench:<method>``): the printed name or descriptive label, never the code name."""
    bench, _, mth = row_arm.partition(":")
    if bench == "lassobench_pub":
        return name_or_label(ctx, mth, "lasso")
    return labels.OTHER_STORED[bench][mth] + " (stored runs)"


def verdict_fileset(ctx: Ctx) -> dict:
    V = verdicts(ctx)
    return ctx.fs_paths([next(iter(V.values()))["path"]])


# =========================================================================================
# Hvarfner: which of the two production arms carries the name (revision 1 (g))
# =========================================================================================
def hvarfner_arm(ctx: Ctx) -> str | None:
    """``hvarfner_published`` or ``hvarfner_published_allobs``: the higher mean over the 11 tasks of the per-task
    mean best@30, on the matched cells of the two arms (R-5). Chosen ONCE, on the default analysis, and the same arm
    is used in every variant (RUN_RULES 2026-10-19 second set (e); review S6); each variant's dossier records it."""
    a, b = "hvarfner_published", "hvarfner_published_allobs"
    if not (ctx.has(a) and ctx.has(b)):
        return a if ctx.has(a) else (b if ctx.has(b) else None)
    key = ("hvarfner_arm",)                         # run-wide (ctx.cache is shared by every variant)
    if key in ctx.cache:
        chosen = ctx.cache[key]
        if "crPubHvarfnerChosen" not in ctx.reg.records:
            cells = ctx.eng.joint([a, b])
            ctx.num("crPubHvarfnerChosen", chosen, labels.label(chosen), kind="text", table="published_rules",
                    fileset=ctx.fs({a: cells, b: cells}), n=len(cells),
                    rule="RUN_RULES 2026-10-13 rev. 1 (g), 2026-10-19 second set (e): chosen once on the default "
                         "analysis and used in every variant")
        return chosen
    if ctx.var.name != "default":
        raise PublishedInputError("the Hvarfner arm is chosen on the default analysis (RUN_RULES 2026-10-19 second "
                                  f"set (e)), but variant {ctx.var.name} asks for it first")
    cells = ctx.eng.joint([a, b])
    la = stats.level(ctx.eng.vals(a, cells), cells, "taskmean")
    lb = stats.level(ctx.eng.vals(b, cells), cells, "taskmean")
    chosen = a if la >= lb else b
    fs = ctx.fs({a: cells, b: cells})
    # the value is the code name (dossier); the printed text is the arm's descriptive label (second set (c))
    ctx.num("crPubHvarfnerChosen", chosen, labels.label(chosen), kind="text",
            table="published_rules", fileset=fs, n=len(cells),
            rule="RUN_RULES 2026-10-13 rev. 1 (g): the Hvarfner arm with the higher mean of per-task means of best@30 "
                 "on the matched cells of the two arms (ties: the in-box baseline arm)")
    ctx.num("crPubHvarfnerInboxMean", la, N.f4(la), kind="level", table="published_rules", fileset=ctx.fs({a: cells}),
            n=len(cells), rule="mean of per-task means of best@30, hvarfner_published (in-box qLogNEI baseline)")
    ctx.num("crPubHvarfnerAllobsMean", lb, N.f4(lb), kind="level", table="published_rules", fileset=ctx.fs({b: cells}),
            n=len(cells), rule="mean of per-task means of best@30, hvarfner_published_allobs (all observations)")
    ctx.cache[key] = chosen
    return chosen


def guacamol_rows(ctx: Ctx) -> list[tuple[str, str, str]]:
    """[(arm, key, printed name)] of the printed GuacaMol published rows (present arms only)."""
    out = []
    for r, key in GUACAMOL_ROWS:
        arm = hvarfner_arm(ctx) if r == "hvarfner" else RECIPES[r]["guacamol"][0]
        if arm is None or not ctx.has(arm):
            continue
        out.append((arm, key, name_or_label(ctx, arm)))
    return out


def support(arm: str) -> str:
    return GUACAMOL_SUPPORT.get(arm, "box $[p_{01},p_{99}]$")


# =========================================================================================
# the reading of rule (4), as a pre-written sentence chosen by the numbers
# =========================================================================================
#: comparatives of the third case (RUN_RULES 2026-10-22): (better mean, better on a task) per direction of improvement
COMPARATIVE = {"higher": ("a higher mean", "higher on"), "lower loss": ("a lower mean loss", "lower on")}
#: printed text of the verdict categories; the category "not worse" of rule (4) is kept as the value, but the words
#: "not worse" are not printed (RUN_RULES 2026-10-22)
VERDICT_TEXT = {"better": "better", "worse": "worse", "not worse": "no significant difference"}


def reading(delta_ref_minus_row: float, p_task_bh: float, tasks_ref: int, tasks_row: int, n_tasks: int,
            ref_text: str, recipe_text: str, ahead: str = "higher") -> tuple[str, str]:
    """(verdict category, sentence) of rule (4). delta = reference minus the published row in the direction of
    improvement (positive: the reference is ahead). Significant = BH-corrected task-level p < 0.05. ``ahead`` is
    the comparative of that direction ("higher" on GuacaMol, "lower loss" on LassoBench, review B1).

    The categories are those of rule (4) and the second set of 2026-10-19 (a); the sentences follow the wording row of
    RUN_RULES 2026-10-22: a significant lead has the leading arm as subject ("{reference} is ahead of {recipe}",
    "{recipe} is ahead of {reference}"); without a significant difference the arm with the better mean of per-task
    means is the subject, then the tasks on which it is better, then the corrected p and "not significant after
    correction". The words "not worse" are never printed."""
    sig = np.isfinite(p_task_bh) and p_task_bh < ALPHA
    p = N.psig(p_task_bh)
    pbh = f"task-level $p_{{\\mathrm{{BH}}}}={p}$"
    if sig and delta_ref_minus_row < 0:
        return "better", f"{recipe_text} is ahead of {ref_text} ({ahead} on {tasks_row} of the {n_tasks} tasks; {pbh})"
    if sig and delta_ref_minus_row > 0:
        return "worse", f"{ref_text} is ahead of {recipe_text} ({ahead} on {tasks_ref} of the {n_tasks} tasks; {pbh})"
    mean_cmp, per_task = COMPARATIVE[ahead]
    if np.isfinite(delta_ref_minus_row) and delta_ref_minus_row > 0:
        lead, other, k = ref_text, recipe_text, tasks_ref
    elif np.isfinite(delta_ref_minus_row) and delta_ref_minus_row < 0:
        lead, other, k = recipe_text, ref_text, tasks_row
    else:
        return "not worse", (f"{ref_text} and {recipe_text} have the same mean; the difference is not significant "
                             f"after correction ({pbh})")
    return "not worse", (f"{lead} has {mean_cmp} than {other} and is {per_task} {k} of the {n_tasks} tasks; the "
                         f"difference is not significant after correction ({pbh})")


# =========================================================================================
# tab_published: the baseline family on GuacaMol
# =========================================================================================
def tab_published(ctx: Ctx) -> Table | None:
    table = "tab_published"
    if ctx.mode != "cr":
        ctx.skip(table, "published recipes", "no camera-ready published-recipe run in legacy mode")
        return None
    rows = guacamol_rows(ctx)
    if not rows:
        ctx.skip(table, "published recipes", "no published-recipe arm present (table omitted)")
        return None
    refs = [r for r in REFS if ctx.has(r[0])]
    for r in REFS:
        if r not in refs:
            ctx.skip(table, r[0], "reference arm absent (column block omitted)")
    if not refs:
        return None
    joint = ctx.eng.joint([a for a, _, _ in rows] + [r[0] for r in refs])
    if not joint:
        ctx.skip(table, "matched set", "empty")
        return None
    res = {}
    for arm, key, _ in rows:
        for ref, rkey, _ in refs:
            res[(arm, ref)] = ctx.eng.paired(ref, arm, joint, conv="taskmean")
    fam = [(arm, ref) for arm, _, _ in rows for ref, _, _ in refs]
    for k, q in zip(fam, stats.bh([res[k].p_task for k in fam])):
        res[k].p_task_bh = q
    fs_all = ctx.fs({a: joint for a in [x[0] for x in rows] + [r[0] for r in refs]})
    ctx.num("crPubN", len(joint), N.intfmt(len(joint)), kind="n", table=table, fileset=fs_all, n=len(joint),
            rule="joint matched set of the published rows, GeoWalk and the GP on the shell")
    ctx.num("crPubFamilySize", len(fam), str(len(fam)), kind="count", table=table, fileset=fs_all,
            rule="tests of the baseline family (published rows x references), BH within this family")
    # RUN_RULES 2026-10-29: one table, the two references side by side (the values of the two former blocks); a grouped
    # header names the reference of each group of four columns
    short = {"Geowalk": "\\geowalk{}", "ShellGp": "GP on the shell"}
    per_ref = ["$\\Delta$", "Tasks", "$p_{\\mathrm{task}}$", "$p_{\\mathrm{task}}^{\\mathrm{BH}}$"]
    t = Table("published", "tab:published", "", WRAP + wrapp("1.3cm") + "r" + "rrrr" * len(refs),
              ["Recipe", "Verified", "Mean"] + per_ref * len(refs), size="\\footnotesize", wide=True, tabcolsep="3pt")
    t.header_pre = ["\\multicolumn{3}{c}{} & " + " & ".join(
        f"\\multicolumn{{4}}{{c}}{{Reference: {short[rkey]}}}" for _, rkey, _ in refs) + " \\\\",
        "".join(f"\\cmidrule(lr){{{4 + 4 * i}-{7 + 4 * i}}}" for i in range(len(refs)))]
    for arm, key, name in rows:
        r = recipe_of(arm)
        lv, _ = ctx.reg_level(f"cr{key}Mean", arm, joint, table)
        row = [name, verification_status(ctx, arm), m(N.f4(lv))]
        for ref, rkey, rtxt in refs:
            pr = res[(arm, ref)]
            p = f"cr{key}Vs{rkey}"
            fs = ctx.fs({arm: joint, ref: joint})
            ctx.num(p + "Delta", pr.delta, N.s4(pr.delta), kind="delta", table=table, fileset=fs, n=pr.n,
                    rule=f"mean of per-task means of {ref} - {arm}, best@30 (positive: the reference is ahead)")
            ctx.num(p + "Tasks", pr.tasks_won, f"{pr.tasks_won}/{pr.n_tasks}", kind="tasks_won", table=table,
                    fileset=fs, n=pr.n_tasks, rule=f"tasks on which {ref} has the higher per-task mean")
            ctx.num(p + "PTask", pr.p_task, N.psig(pr.p_task), kind="p_task", table=table, fileset=fs,
                    n=pr.n_tasks, rule="exact two-sided Wilcoxon on the per-task mean differences (primary)")
            ctx.num(p + "PTaskBH", pr.p_task_bh, N.psig(pr.p_task_bh), kind="p_task_bh", table=table, fileset=fs,
                    n=pr.n_tasks, rule=f"Benjamini-Hochberg within the baseline family ({len(fam)} task-level tests)")
            word, sent = reading(pr.delta, pr.p_task_bh, pr.tasks_won, pr.tasks_lost, pr.n_tasks, rtxt, name)
            ctx.num(p + "Verdict", word, VERDICT_TEXT[word], kind="text", table=table, fileset=fs, n=pr.n_tasks,
                    rule="RUN_RULES 2026-10-12 (4): 'better' if BH task-level p < 0.05 and the recipe is ahead, "
                         "'worse' if p < 0.05 and the reference is ahead, else 'not worse' (printed as 'no significant "
                         "difference', RUN_RULES 2026-10-22)")
            ctx.num(p + "Sentence", sent, sent, kind="text", table=table, fileset=fs, n=pr.n_tasks,
                    rule="sentence of rule (4) in the wording of RUN_RULES 2026-10-22, chosen by the verdict above")
            row += [m(N.s4(pr.delta)), f"{pr.tasks_won}/{pr.n_tasks}", m(N.psig(pr.p_task)),
                    mb(N.psig(pr.p_task_bh), np.isfinite(pr.p_task_bh) and pr.p_task_bh < ALPHA)]
        t.add(row)
    hv = hvarfner_arm(ctx)
    other = [a for a in RECIPES["hvarfner"]["guacamol"] if a != hv and ctx.has(a)]
    note_h = ""
    if hv and other:
        # revision 1 (g): both Hvarfner arms appear in Appendix H; the other one is shown below the family, outside it
        o = other[0]
        oc = joint & ctx.eng.elig(o)
        t.rule()
        lv, _ = ctx.reg_level("crPubHvarfnerOtherMean", o, oc, table)
        olab = labels.label(o) + " (the other run of the same recipe)"
        row = [olab, verification_status(ctx, o), m(N.f4(lv))]
        for ref, rkey, _ in refs:
            pr = ctx.eng.paired(ref, o, oc, conv="taskmean")
            fs = ctx.fs({o: oc, ref: oc})
            ctx.num(f"crPubHvarfnerOtherVs{rkey}Delta", pr.delta, N.s4(pr.delta), kind="delta", table=table, fileset=fs,
                    n=pr.n, rule=f"mean of per-task means of {ref} - {o} (outside the baseline family)")
            ctx.num(f"crPubHvarfnerOtherVs{rkey}PTask", pr.p_task, N.psig(pr.p_task), kind="p_task", table=table,
                    fileset=fs, n=pr.n_tasks, rule="exact two-sided Wilcoxon on the per-task differences (no correction)")
            row += [m(N.s4(pr.delta)), f"{pr.tasks_won}/{pr.n_tasks}", m(N.psig(pr.p_task)), "--"]
        t.add(row)
    if hv and other:
        base = {"hvarfner_published": "the in-box qLogNEI baseline, as Ax",
                "hvarfner_published_allobs": "all observations as the qLogNEI baseline"}
        where = {"hvarfner_published": "points inside the box, as Ax does",
                 "hvarfner_published_allobs": "all observations"}
        note_h = (" The Hvarfner recipe has two runs. They differ in the points that qLogNEI uses as its baseline. "
                  f"The Hvarfner row uses {where[hv]}, and it has the higher mean of the two. The other run uses "
                  f"{where[other[0]]}. It is the last row, outside the corrected family, and "
                  "Table~\\ref{tab:baselines} describes it.")
    t.caption = ("\\textbf{Published recipes against \\geowalk{} and the GP on the shell at $N=30$ (baseline family).} "
                 "Each recipe follows the authors' released code in every tabulated setting. In this table, a recipe carries the "
                 "authors' name only if it reproduced a published result (Table~\\ref{tab:verification}). Otherwise "
                 "it is named by kernel, acquisition and maximizer and marked as not verified. ``Yes, with allowance'' "
                 "means that the recipe passed only with the allowances of the protocol. All rows use one "
                 f"matched set of $n={len(joint)}$ cells. Mean and $\\Delta$ are means of per-task means of best@30, "
                 "and $\\Delta$ = the reference minus the recipe. The reference is \\geowalk{} in the left group of "
                 "columns and the GP on the shell in the right group. Tasks = tasks, of the 11, on which the reference "
                 "has the higher mean. $p_{\\mathrm{task}}$ = exact two-sided Wilcoxon test on the 11 per-task differences, the "
                 f"primary test. It is corrected by Benjamini--Hochberg within the {len(fam)} tests of this table"
                 + stats.bh_note([res[k].p_task for k in fam]) + ". Bold marks "
                 "$p_{\\mathrm{task}}^{\\mathrm{BH}}<0.05$." + note_h)
    return t


# =========================================================================================
# tab_verification: the reproduction of published results (Appendix H)
# =========================================================================================
def tab_verification(ctx: Ctx) -> Table | None:
    table = "tab_verification"
    if ctx.mode != "cr":
        ctx.skip(table, "verification", "camera-ready mode only")
        return None
    V = verdicts(ctx)
    fs = verdict_fileset(ctx)
    hdr = ["Recipe", "Target", "Design", "Seeds", "Verdict", "Strict reading", "Checkpoints", "Fit skipped"]
    hdr = ["Recipe", "Target", "Design", "Seeds", "Verdict", stack("Strict", "reading"), "Checkpoints",
           stack("Uniform", "steps")]
    t = Table("verification", "tab:verification", "", WRAP + ">{\\raggedright\\arraybackslash}p{2.6cm}"
              ">{\\raggedright\\arraybackslash}p{1.2cm}rllrr", hdr, size="\\footnotesize", wide=True, tabcolsep="3pt")
    n_ver = 0
    for r, spec in RECIPES.items():
        ents = V[r]["entries"]
        name = spec["name"] or labels.lasso_label(spec["lasso"][0], spec["lasso"][0])
        if not ents:
            t.add([name, spec["target"], "--", "--", "not verified", "--", "--", "--"])
            continue
        n_ver += V[r]["verified"]
        for e in ents:
            k = f"crVer{camel(r)}N{camel(str(e['n_init']))}"
            gate = [x for x in e["rows"] if x["gate"]]
            ok = sum(1 for x in gate if x["pass"])
            oks = sum(1 for x in gate if x["pass_strict"])
            ctx.num(k + "Pass", bool(e["pass"]), "passed" if e["pass"] else "failed", kind="text", table=table,
                    fileset=fs, n=len(e["seeds_used"]), rule="protocol verdict (RUN_RULES 2026-10-13 revision 1 (b))")
            ctx.num(k + "PassStrict", bool(e["pass_strict"]), "passed" if e["pass_strict"] else "failed",
                    kind="text", table=table, fileset=fs, n=len(e["seeds_used"]),
                    rule="strict reading of 2026-10-12 (delta 0, SE_pub 0 where unstated), reported beside the verdict")
            ctx.num(k + "Seeds", len(e["seeds_used"]), str(len(e["seeds_used"])), kind="n", table=table, fileset=fs,
                    n=len(e["seeds_used"]), rule="verification runs (seeds 0 ... n-1)")
            ctx.num(k + "FitSkipped", e["fallbacks"], N.intfmt(e["fallbacks"]), kind="count", table=table, fileset=fs,
                    n=len(e["seeds_used"]), rule="steps whose GP fit failed and was skipped, summed over the runs")
            design = f"{e['n_init']} initial points"
            t.add([name, spec["target"], design, str(len(e["seeds_used"])), "passed" if e["pass"] else "failed",
                   "passed" if e["pass_strict"] else "failed", f"{ok}/{len(gate)} ({oks}/{len(gate)})",
                   N.intfmt(e["fallbacks"])])
    # RUN_RULES 2026-10-29 (tables cut to what the text needs): the frozen-noise variants have no row; the caption
    # says that their runs were stopped without being read
    ctx.num("crVerRecipesVerified", n_ver, str(n_ver), kind="count", table=table, fileset=fs,
            rule="recipes that passed the verification (their rows carry the authors' names)")
    allow = [RECIPES[r]["name"] for r in RECIPES if RECIPES[r]["name"] and recipe_status(ctx, r) == STATUS_ALLOWANCE]
    t.caption = ("\\textbf{Verification of the published recipes: our implementation in the authors' setting against "
                 "a published result.} A recipe passes if our mean lies within the pass margin of the published value "
                 "at every listed evaluation count of its target. The pass margin is two combined standard errors, "
                 "$2\\sqrt{SE_{\\mathrm{ours}}^2+SE_{\\mathrm{pub}}^2}$. It is widened by the reading uncertainty "
                 "of a value read from a figure and by an allowance for the measured difference between our objective "
                 "and the released one. Where the paper states no standard error, $SE_{\\mathrm{pub}}$ is taken equal "
                 "to ours. The strict reading drops the allowance and takes an unstated $SE_{\\mathrm{pub}}$ as zero. "
                 "Checkpoints = listed evaluation counts at which our mean is within the margin (strict reading in "
                 "parentheses). Uniform steps = steps in which a numerical failure replaced the proposal by a uniform "
                 "point in the box, as the authors' code does, summed over the runs. The main runs handle a failed "
                 "step in the same way (notes to Table~\\ref{tab:main}). Hvarfner passes if either initial design "
                 "passes: 30 points as in the paper, or 41 as in the released configuration. The Hvarfner "
                 "verification uses the in-box qLogNEI baseline, as Ax does. The all-observations run of "
                 "Tables~\\ref{tab:baselines} and~\\ref{tab:published} is a variant that was not verified itself."
                 + (f" {_and(allow)} passed only with the allowances: the strict reading fails. Tables~\\ref{{tab:baselines}} "
                    f"and~\\ref{{tab:published}} therefore mark {'it' if len(allow) == 1 else 'them'} ``yes, with "
                    "allowance''." if allow else "")
                 + " MSR and MLE (scaled) were run with frozen noise, the noise variance fixed at $10^{-4}$ as in the "
                 "authors' default, and with learned noise, fitted with the model. The learned-noise variant passed, so "
                 "it is the printed variant. The frozen-noise runs were stopped without being read. BAxUS, TuRBO and "
                 "CMA-ES are not verified. Their runs were not complete when the verification ended, so they keep "
                 "descriptive names.")
    return t
