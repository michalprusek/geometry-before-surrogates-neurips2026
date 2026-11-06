"""Run context: dataset + engine + number registry + decision variant."""
from __future__ import annotations

import re

from dataclasses import dataclass, field, replace

from . import numbers as N
from .engine import Engine, fileset_from_paths, fileset_union
from .loader import Dataset
from .numbers import Registry


@dataclass(frozen=True)
class Variant:
    """Analysis decisions (plan section 3). The default follows RUN_RULES R-6 and the plan's
    recommendations; every other entry of VARIANTS changes exactly one decision."""
    name: str = "default"
    conv: str = "taskmean"            # R1: 'taskmean' (mean of per-task means) | 'flat'
    box: str = "faithful"             # R2: 'faithful' = per-coordinate [p01,p99] | 'minmax'
    papenmeier_main: str = "published"  # R3: box row of Table 2 = published MSR | 'mle_scaled'
    factorial: str = "cr"             # R4: printed factorial = camera-ready runs | 'e1'
    tasklevel: bool = True            # R7: tasks-won column and exact task-level Wilcoxon
    cowboys: bool = False             # COWBOYS row in Table 1 (RUN_RULES 3: drop recommended)
    cell_filter: str | None = None    # None = the config's (R-5: completed); 'clean' drops fallback cells
    description: str = "RUN_RULES R-6 defaults and the plan's recommended decisions"


VARIANTS = {
    "default": Variant(),
    "R1_flat": Variant(name="R1_flat", conv="flat", description="R1: flat mean over cells instead of the mean of per-task means"),
    "R2_minmax": Variant(name="R2_minmax", box="minmax", description="R2: per-coordinate [min,max] box (the submitted box) instead of [p01,p99]"),
    "R3_mle_scaled_main": Variant(name="R3_mle_scaled_main", papenmeier_main="mle_scaled",
                                  description="R3: Papenmeier box row of Table 2 = MLE (scaled) (L-BFGS, no RAASP) instead of the published MSR maximiser"),
    "R4_E1": Variant(name="R4_E1", factorial="e1", description="R4: factorial block printed from the E-1 rerun (abbd61e, batch runner) instead of the camera-ready runs"),
    "R7_no_tasklevel": Variant(name="R7_no_tasklevel", tasklevel=False, description="R7: cell-level statistics only (no tasks-won / task-level test columns)"),
    "cowboys_kept": Variant(name="cowboys_kept", cowboys=True, description="COWBOYS row kept in Table 1 (stored 10-seed files, in the BH family)"),
    "clean_nofallback": Variant(name="clean_nofallback", cell_filter="clean",
                                description="cells with any surrogate-fallback step excluded (the convention of the posted rebuttal tables)"),
}

BOX = {  # recipe -> (faithful [p01,p99], minmax [min,max])
    "hvarfner": ("hvarfner_scaled_gp_box_faithful", "hvarfner_scaled_gp_norm_box"),
    "papenmeier_mle": ("papenmeier_msr_box_faithful", "papenmeier_msr_norm_box"),
    "papenmeier_msr": ("papenmeier_msr_published", "papenmeier_msr_published_norm_box"),
    "xu": ("xu_standard_gp_box_faithful", "xu_standard_gp_norm_box"),
    "factorial_argmax": ("hvarfner_scaled_gp_box_argmax", "hvarfner_scaled_gp_norm_box_argmax"),
}
BOX_LABEL = {"faithful": "$[p_{01},p_{99}]$", "minmax": "$[\\min,\\max]$"}


@dataclass
class Ctx:
    ds: Dataset
    eng: Engine
    reg: Registry
    var: Variant
    outdir: str
    script_commit: str
    mode: str                         # 'cr' | 'legacy'
    skipped: list = field(default_factory=list)
    written: list = field(default_factory=list)
    not_built: list = field(default_factory=list)   # builders not called (camera-ready scope)
    extra: dict = field(default_factory=dict)

    def has(self, arm: str) -> bool:
        return self.ds.present(arm)

    def box(self, recipe: str) -> str:
        f, mm = BOX[recipe]
        return f if self.var.box == "faithful" else mm

    def label(self, arm: str, default: str) -> str:
        """Row label of an arm; a config 'display.label' overrides it (legacy substitutes)."""
        if arm in self.ds.arms:
            return self.ds.arms[arm].spec.display.get("label", default)
        return default

    def support(self, arm: str, default: str) -> str:
        if arm in self.ds.arms:
            return self.ds.arms[arm].spec.display.get("support", default)
        return default

    def overridden(self, arm: str) -> bool:
        return arm in self.ds.arms and bool(self.ds.arms[arm].spec.display)

    def skip(self, table: str, what: str, why: str) -> None:
        self.skipped.append({"table": table, "what": what, "why": why})

    def prov_comment(self, table: str) -> str:
        return (f"{table}: scripts/cr/analysis/make_tables.py, commit {self.script_commit}, mode {self.mode}, "
                f"variant {self.var.name}, cell filter {self.eng.cell_filter}, levels {self.var.conv}")

    @property
    def cache(self) -> dict:
        """Run-wide cache shared by all variants (Monte Carlo norms, geometry arrays)."""
        return self.extra.setdefault("_shared_cache", {})

    # ---- number registration helpers ---------------------------------------------------
    def fs(self, arm_cells: dict) -> dict:
        return self.eng.fileset(arm_cells)

    def fs_paths(self, paths, arms=None) -> dict:
        return fileset_from_paths(paths, arms)

    def fs_union(self, *filesets) -> dict:
        return fileset_union(*filesets)

    def fs_of(self, *names) -> dict:
        """Union of the file sets of numbers already registered (for derived numbers)."""
        return fileset_union(*[self.reg.filesets[n] for n in names])

    def num(self, name: str, value, text: str, *, kind: str, table: str, fileset: dict | None,
            n=None, rule: str = "", note: str = "") -> str:
        self.reg.add(name, value, text, kind=kind, table=table, fileset=fileset, n=n, rule=rule, note=note)
        self._count_only(name, text, table=table, fileset=fileset, rule=rule)
        return text

    #: task totals of the benchmarks (count-only macros of RUN_RULES 2026-10-22: \crNTasksGuacamol, \crNTasksLasso)
    N_TASKS = {"guacamol": 11, "lasso": 6}

    def _count_only(self, name: str, text: str, *, table: str, fileset: dict | None, rule: str) -> None:
        """Count-only reprint of a task count (RUN_RULES 2026-10-22, generator edits after the final run): a macro
        \\crXTasks... that prints "k/n" gets \\crXTasks...K printing k alone, and \\crXTasks...N printing n where n is
        not the benchmark's task total (11 on GuacaMol, 6 on LassoBench), so that prose can say "on k of the n tasks"
        without typing a number. Value and file set are those of the original macro."""
        if "Tasks" not in name or not isinstance(text, str):
            return
        mo = re.fullmatch(r"(\d+)/(\d+)", text)
        if not mo:
            return
        k, n_tot = int(mo.group(1)), int(mo.group(2))
        self.reg.add(name + "K", k, str(k), kind="tasks_won_k", table=table, fileset=fileset, n=n_tot,
                     rule=f"count only of \\{name} ({rule})")
        default = self.N_TASKS["lasso" if name.startswith("crLasso") else "guacamol"]
        if n_tot != default:
            self.reg.add(name + "N", n_tot, str(n_tot), kind="n_tasks", table=table, fileset=fileset, n=n_tot,
                         rule=f"task total of \\{name} (differs from the benchmark's {default})")

    def reg_level(self, name: str, arm: str, cells, table: str, N_eval: int = 30, fmt=N.f4) -> tuple:
        v = self.eng.level(arm, cells, N_eval)
        rule = f"{'mean of per-task means' if self.var.conv == 'taskmean' else 'flat mean'} of best@{N_eval}"
        t = self.num(name, v, fmt(v), kind="level", table=table, fileset=self.fs({arm: cells}),
                     n=len(cells), rule=rule)
        return v, t

    def reg_paired(self, prefix: str, pr, a: str, b: str, cells, table: str, N_eval: int = 30,
                   what: str = "") -> dict:
        """Registers Delta, d, p, (p_BH later), W/L, n, tasks won, p_task of a - b."""
        fs = self.fs({a: cells, b: cells})
        conv = "task-mean" if self.var.conv == "taskmean" else "flat"
        rule = f"paired {a} - {b} on matched cells, best@{N_eval}"
        out = {}
        out["delta"] = self.num(prefix + "Delta", pr.delta, N.s4(pr.delta), kind="delta", table=table, fileset=fs,
                                n=pr.n, rule=f"{conv} difference; " + rule)
        out["d"] = self.num(prefix + "D", pr.d, N.d2(pr.d), kind="cohens_d", table=table, fileset=fs, n=pr.n,
                            rule="Cohen's d = mean(diff)/sd(diff, ddof 1), cells; " + rule)
        out["p"] = self.num(prefix + "P", pr.p, N.pfmt(pr.p), kind="p_raw", table=table, fileset=fs, n=pr.n,
                            rule="two-sided Wilcoxon signed-rank on cell differences (scipy defaults; |diff| <= 1e-6 "
                                 "is a tie = zero difference); " + rule)
        out["px"] = self.num(prefix + "PExact", pr.p_exact, N.pfmt(pr.p_exact), kind="p_raw_exact", table=table,
                             fileset=fs, n=pr.n,
                             rule="old rule: Wilcoxon on the exact float differences (kept for the legacy checks "
                                  "and the correspondence tables); " + rule)
        out["wl"] = self.num(prefix + "WL", f"{pr.wins}/{pr.losses}", f"{pr.wins}/{pr.losses}", kind="wins_losses",
                             table=table, fileset=fs, n=pr.n,
                             rule="cells with difference > +1e-6 / < -1e-6 (|diff| <= 1e-6 is a tie); " + rule)
        out["wlx"] = self.num(prefix + "WLExact", f"{pr.wins_exact}/{pr.losses_exact}",
                              f"{pr.wins_exact}/{pr.losses_exact}", kind="wins_losses_exact", table=table,
                              fileset=fs, n=pr.n,
                              rule="old W/L rule (exact float differences), kept for the correspondence tables; " + rule)
        out["n"] = self.num(prefix + "N", pr.n, N.intfmt(pr.n), kind="n", table=table, fileset=fs, n=pr.n,
                            rule="matched cells")
        if self.var.tasklevel:
            out["tasks"] = self.num(prefix + "Tasks", pr.tasks_won, f"{pr.tasks_won}/{pr.n_tasks}", kind="tasks_won",
                                    table=table, fileset=fs, n=pr.n_tasks,
                                    rule="tasks with positive per-task mean difference; " + rule)
            out["ptask"] = self.num(prefix + "PTask", pr.p_task, N.psig(pr.p_task), kind="p_task", table=table,
                                    fileset=fs, n=pr.n_tasks,
                                    rule="exact two-sided Wilcoxon on the per-task mean differences; " + rule)
        out["_fs"] = fs
        return out

    def reg_pbh(self, prefix: str, pr, fs: dict, table: str, family: str) -> str:
        if pr.p_bh_exact == pr.p_bh_exact:      # not NaN: the family also has the exact-difference version
            self.num(prefix + "PBHExact", pr.p_bh_exact, N.pfmt(pr.p_bh_exact), kind="p_bh_exact", table=table,
                     fileset=fs, n=pr.n, rule=f"old rule: Benjamini-Hochberg over {family}, exact float differences")
        return self.num(prefix + "PBH", pr.p_bh, N.pfmt(pr.p_bh), kind="p_bh", table=table, fileset=fs, n=pr.n,
                        rule=f"Benjamini-Hochberg over {family}")
