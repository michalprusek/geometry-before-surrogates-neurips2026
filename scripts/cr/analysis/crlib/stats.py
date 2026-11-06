"""Statistics (RUN_RULES R-6).

* level     = mean of per-task means over the matched cells (``conv='taskmean'``, default)
              or the flat mean over cells (``conv='flat'``, variant R1)
* paired    = cell-level two-sided Wilcoxon signed-rank test (scipy defaults:
              zero_method='wilcox', correction=False, method='auto'), Cohen's d =
              mean(diff) / sd(diff, ddof=1), wins/losses (ties excluded); Delta is reported
              both as the flat mean difference and as the task-mean of per-task mean
              differences (= difference of the two printed task-mean levels on the same cells)
* ties      = a cell difference with |diff| <= TIE_TOL = 1e-6 is a TIE, in W/L and in the
              cell-level Wilcoxon test alike (GEN v3): a cell is a win (loss) only if its
              difference exceeds +TIE_TOL (is below -TIE_TOL), and the Wilcoxon test sees such
              a difference as an exact zero (dropped by zero_method='wilcox'). Reason
              (GEN_verify_1 6.1): when neither arm improves on the shared cold start both end
              at the same molecule, but the stored external harness kept float32 scores
              (|diff| up to 2.6e-8) and the in-repo oracle differs in the last bit between
              code states (|diff| <= 1.1e-16); exact float comparison ranked those true ties
              as the smallest wins or losses. The exact-difference versions (the rule of
              every earlier printed W/L and p) are kept as ``wins_exact`` / ``losses_exact``
              / ``p_exact`` for the legacy checks and the correspondence tables. Delta and d
              use the exact differences (a tie contributes at most 1e-6); the task-level test
              acts on per-task means and has no tie rule.
* task level = number of tasks with a positive per-task mean difference, and the exact
              two-sided Wilcoxon signed-rank test on the per-task mean differences
* task bootstrap = percentile interval of Delta over B = 10,000 resamples of the tasks
              (with replacement), fixed seed BOOT_SEED (RUN_RULES amendment 2026-10-12)
* TOST      = equivalence at margin delta on the per-task mean differences: two one-sided
              t-tests (df = n_tasks - 1) and two one-sided exact Wilcoxon tests; the printed
              verdict is 'equivalent' only if the 90 % task-bootstrap interval lies inside
              (-delta, +delta) (pre-registered rule)
* BH        = Benjamini-Hochberg step-up within a family (one table)
* seed-block permutation = robustness of the task-level test to the dependence that the shared
              cold start induces between tasks (a seed fixes the 5 cold-start molecules of
              every task): one random sign per SEED is applied to that seed's cell contrasts in
              every task (= permuting the arm labels within the seed jointly across tasks),
              statistic |W+ - n(n+1)/4| of the Wilcoxon signed-rank test on the resulting
              per-task means, p = (1 + #{perm >= observed}) / (n_perm + 1), 10,000
              permutations, default_rng(0). Ported from scripts/analysis/gate0_premise.py at
              4780475 (plan 2.5) with two stated differences: native scale (no division by the
              task range R_t) and, because matched sets may lack cells, per-task means over
              the cells present (the September tool refused incomplete matrices)
* shared-cold-start correlation = r-bar, the mean over task pairs of the correlation, across
              the seeds both tasks have, of the two tasks' cell contrasts (plan 2.5)
* recovered fraction = mean_t(a_t - c_t) / mean_t(b_t - c_t) with a task-bootstrap interval
              (the same resampled tasks in numerator and denominator)
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sps

TIE_TOL = 1e-6          # |diff| <= TIE_TOL is a tie in W/L (see the module docstring)
BOOT_B = 10_000         # task-bootstrap resamples (pre-registered, RUN_RULES amendment 2026-10-12)
BOOT_SEED = 20261012    # fixed seed of every task bootstrap
EQUIV_MARGIN = 0.01     # TOST margin of the two pre-registered equivalence claims
PERM_N = 10_000         # seed-block permutations (gate0_premise.py N_PERM)
PERM_SEED = 0           # default_rng seed of the seed-block permutation (gate0_premise.py RNG_SEED)


def group_by_task(cells) -> dict:
    g = defaultdict(list)
    for t, s in cells:
        g[t].append((t, s))
    return g


def task_means(values: dict, cells) -> dict:
    g = group_by_task(cells)
    return {t: float(np.mean([values[k] for k in ks])) for t, ks in sorted(g.items())}


def level(values: dict, cells, conv: str = "taskmean") -> float:
    cells = list(cells)
    if not cells:
        return float("nan")
    if conv == "flat":
        return float(np.mean([values[k] for k in cells]))
    if conv == "taskmean":
        return float(np.mean(list(task_means(values, cells).values())))
    raise ValueError(conv)


def level_se(values: dict, cells) -> float:
    """SE of the mean of per-task means, tasks treated as independent strata."""
    g = group_by_task(cells)
    var = 0.0
    for t, ks in g.items():
        x = np.asarray([values[k] for k in ks], dtype=float)
        if x.size > 1:
            var += x.var(ddof=1) / x.size
    return math.sqrt(var) / len(g) if g else float("nan")


def wilcoxon_p(diffs: np.ndarray) -> float:
    d = np.asarray(diffs, dtype=float)
    if d.size == 0:
        return float("nan")
    if not np.any(d != 0):
        return 1.0          # every pair tied: no evidence (scipy would raise)
    return float(sps.wilcoxon(d).pvalue)


def zero_ties(diffs) -> np.ndarray:
    """Differences with |diff| <= TIE_TOL set to exactly zero (the tie rule of W/L and of the cell-level test)."""
    d = np.asarray(diffs, dtype=float).copy()
    d[np.abs(d) <= TIE_TOL] = 0.0
    return d


def wilcoxon_p_ties(diffs) -> float:
    """Cell-level two-sided Wilcoxon signed-rank p with |diff| <= TIE_TOL treated as zero."""
    return wilcoxon_p(zero_ties(diffs))


def wilcoxon_exact_p(diffs) -> float:
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0]
    if d.size < 2:
        return float("nan")
    return float(sps.wilcoxon(d, method="exact").pvalue)


def cohens_d(diffs) -> float:
    d = np.asarray(diffs, dtype=float)
    if d.size < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(d.mean() / sd)


@dataclass
class Paired:
    n: int
    n_tasks: int
    level_a: float
    level_b: float
    delta_taskmean: float
    delta_flat: float
    sd_diff: float
    d: float
    wins: int
    losses: int
    ties: int
    p: float
    task_diffs: dict = field(default_factory=dict)
    tasks_won: int = 0
    tasks_lost: int = 0
    p_task: float = float("nan")
    p_bh: float = float("nan")
    conv: str = "taskmean"
    wins_exact: int = 0          # W/L under exact float differences (the rule of earlier printed numbers)
    losses_exact: int = 0
    ties_exact: int = 0
    task_n: dict = field(default_factory=dict)     # cells per task (for the flat-convention bootstrap)
    p_task_bh: float = float("nan")
    p_exact: float = float("nan")                  # cell-level p on the exact float differences (earlier rule)
    p_bh_exact: float = float("nan")               # BH over the family's exact-difference p (legacy checks)
    cell_diffs: dict = field(default_factory=dict, repr=False)   # cell -> a - b (for the seed-block statistics)

    @property
    def delta(self) -> float:
        return self.delta_taskmean if self.conv == "taskmean" else self.delta_flat

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        """Task-bootstrap percentile interval of Delta (B = 10,000, fixed seed)."""
        return task_bootstrap_ci(self.task_diffs, self.task_n if self.conv == "flat" else None, level=level)

    def as_dict(self) -> dict:
        return {"n": self.n, "n_tasks": self.n_tasks, "level_a": self.level_a, "level_b": self.level_b,
                "delta": self.delta, "delta_taskmean": self.delta_taskmean, "delta_flat": self.delta_flat,
                "sd_diff": self.sd_diff, "d": self.d, "wins": self.wins, "losses": self.losses,
                "ties": self.ties, "wins_exact": self.wins_exact, "losses_exact": self.losses_exact,
                "ties_exact": self.ties_exact, "tie_tol": TIE_TOL, "p": self.p, "p_exact": self.p_exact,
                "p_bh": self.p_bh,
                "tasks_won": self.tasks_won, "tasks_lost": self.tasks_lost, "p_task": self.p_task,
                "p_task_bh": self.p_task_bh, "conv": self.conv, "task_diffs": self.task_diffs}


def paired(a: dict, b: dict, cells, conv: str = "taskmean") -> Paired:
    """a, b: cell -> value. Differences are a - b."""
    cells = sorted(cells)
    if not cells:
        return Paired(0, 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                      float("nan"), 0, 0, 0, float("nan"), conv=conv)
    x = np.asarray([a[k] for k in cells], dtype=float)
    y = np.asarray([b[k] for k in cells], dtype=float)
    d = x - y
    diff = dict(zip(cells, d))
    td = task_means(diff, cells)
    tdv = np.asarray(list(td.values()))
    tn = {t: len(ks) for t, ks in group_by_task(cells).items()}
    return Paired(
        n=len(cells), n_tasks=len(td), level_a=level(a, cells, conv), level_b=level(b, cells, conv),
        delta_taskmean=float(tdv.mean()), delta_flat=float(d.mean()),
        sd_diff=float(d.std(ddof=1)) if d.size > 1 else float("nan"), d=cohens_d(d),
        wins=int((d > TIE_TOL).sum()), losses=int((d < -TIE_TOL).sum()),
        ties=int((np.abs(d) <= TIE_TOL).sum()),
        # tasks won / lost with the same 1e-6 tie rule as the cell counts (RUN_RULES 2026-10-12 (e); review M1);
        # the exact task-level test keeps the exact differences
        p=wilcoxon_p_ties(d), task_diffs=td, tasks_won=int((tdv > TIE_TOL).sum()), tasks_lost=int((tdv < -TIE_TOL).sum()),
        p_task=wilcoxon_exact_p(tdv), conv=conv,
        wins_exact=int((d > 0).sum()), losses_exact=int((d < 0).sum()), ties_exact=int((d == 0).sum()),
        task_n=tn, p_exact=wilcoxon_p(d), cell_diffs=diff,
    )


def task_bootstrap_ci(task_diffs: dict, task_n: dict | None = None, *, level: float = 0.95,
                      B: int = BOOT_B, seed: int = BOOT_SEED) -> tuple[float, float]:
    """Percentile interval of the mean of per-task differences over B resamples of the tasks.

    ``task_n`` given: the resampled statistic is the cell-weighted mean (flat convention);
    otherwise the plain mean of the per-task differences (task-mean convention, the default).
    The resampling stream depends only on (seed, number of tasks), so every contrast of a table
    with the same tasks is resampled with the same task indices."""
    keys = sorted(task_diffs)
    v = np.asarray([task_diffs[k] for k in keys], dtype=float)
    if v.size == 0 or not np.isfinite(v).all():
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(B, v.size))
    if task_n:
        w = np.asarray([task_n[k] for k in keys], dtype=float)
        stat = (v[idx] * w[idx]).sum(1) / w[idx].sum(1)
    else:
        stat = v[idx].mean(1)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(stat, [a, 1.0 - a])
    return float(lo), float(hi)


def _task_seed_matrix(cell_diffs: dict) -> tuple[np.ndarray, list, list]:
    """(tasks x seeds) array of cell contrasts, NaN where the matched set has no cell."""
    tasks = sorted({t for t, _ in cell_diffs})
    seeds = sorted({s for _, s in cell_diffs})
    D = np.full((len(tasks), len(seeds)), np.nan)
    ti, si = {t: i for i, t in enumerate(tasks)}, {s: i for i, s in enumerate(seeds)}
    for (t, s), v in cell_diffs.items():
        D[ti[t], si[s]] = v
    return D, tasks, seeds


def shared_coldstart_rbar(cell_diffs: dict) -> dict:
    """r-bar = mean over task pairs t < t' of corr_s(d_{t,s}, d_{t',s}) over the seeds present in both tasks
    (pairs with fewer than 3 common seeds or a constant column are dropped and counted)."""
    D, tasks, _ = _task_seed_matrix(cell_diffs)
    vals, dropped = [], 0
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            ok = np.isfinite(D[i]) & np.isfinite(D[j])
            if ok.sum() < 3 or np.std(D[i, ok]) == 0 or np.std(D[j, ok]) == 0:
                dropped += 1
                continue
            vals.append(float(np.corrcoef(D[i, ok], D[j, ok])[0, 1]))
    return {"rbar": float(np.mean(vals)) if vals else float("nan"), "n_pairs": len(vals),
            "n_pairs_dropped": dropped, "r_min": float(min(vals)) if vals else float("nan"),
            "r_max": float(max(vals)) if vals else float("nan")}


def _perm_stats_rows(E: np.ndarray) -> np.ndarray:
    """|W+ - n(n+1)/4| of the signed-rank statistic for every row of E (midranks of |e| among the non-zero
    entries of the row; zeros discarded). Row-wise identical to gate0_premise._perm_stat(e, 'two-sided')."""
    E = np.asarray(E, dtype=float)
    absE = np.where(E != 0, np.abs(E), np.nan)
    ranks = sps.rankdata(absE, axis=1, nan_policy="omit")
    wplus = np.nansum(np.where(E > 0, ranks, 0.0), axis=1)
    n = (E != 0).sum(axis=1)
    return np.abs(wplus - n * (n + 1) / 4.0)


def seed_block_permutation(cell_diffs: dict, n_perm: int = PERM_N, seed: int = PERM_SEED) -> dict:
    """Two-sided seed-block permutation p of the task-level signed-rank statistic (see the module docstring).
    With a complete tasks x seeds matrix this is gate0_premise.seed_block_permutation with R = 1."""
    D, tasks, seeds = _task_seed_matrix(cell_diffs)
    T, S = D.shape
    if T < 2 or S < 2:
        return {"p": float("nan"), "n_perm": n_perm, "rng": seed, "n_tasks": T, "n_seeds": S, "complete": False}
    M = np.isfinite(D)
    D0 = np.where(M, D, 0.0)
    cnt = M.sum(axis=1).astype(float)
    obs = _perm_stats_rows((D0.sum(axis=1) / cnt)[None, :])[0]
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, S))
    E = (signs @ D0.T) / cnt[None, :]
    st = _perm_stats_rows(E)
    count = int((st >= obs - 1e-12).sum())
    return {"p": (1 + count) / (n_perm + 1), "statistic_observed": float(obs), "n_perm": n_perm, "rng": seed,
            "n_ge_observed": count, "n_tasks": int(T), "n_seeds": int(S), "complete": bool(M.all())}


def ratio_bootstrap(num: dict, den: dict, *, level: float = 0.95, B: int = BOOT_B, seed: int = BOOT_SEED) -> dict:
    """Recovered fraction mean_t(num_t) / mean_t(den_t) with a percentile task-bootstrap interval: every
    resample draws the tasks once and uses the same tasks in numerator and denominator. ``den_nonpositive``
    counts resamples whose denominator is <= 0 (their ratio is not meaningful; reported, not dropped)."""
    keys = sorted(set(num) & set(den))
    a = np.asarray([num[k] for k in keys], dtype=float)
    b = np.asarray([den[k] for k in keys], dtype=float)
    if a.size == 0 or b.mean() == 0:
        return {"ratio": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_tasks": int(a.size),
                "den_nonpositive": 0}
    out = {"ratio": float(a.mean() / b.mean()), "n_tasks": int(a.size)}
    if a.size == 1:
        out.update(lo=out["ratio"], hi=out["ratio"], den_nonpositive=0)
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(B, a.size))
    den_b = b[idx].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a[idx].mean(1) / den_b
    q = (1.0 - level) / 2.0
    lo, hi = np.nanquantile(r, [q, 1.0 - q])
    out.update(lo=float(lo), hi=float(hi), den_nonpositive=int((den_b <= 0).sum()))
    return out


def tost(task_diffs: dict, margin: float = EQUIV_MARGIN, task_n: dict | None = None) -> dict:
    """Equivalence of the mean per-task difference to zero at +-margin.

    p_t: max of the two one-sided t-tests on the per-task differences (df = n - 1),
         H01: mean <= -margin, H02: mean >= +margin;
    p_w: max of the two one-sided exact Wilcoxon signed-rank tests on (d + margin) > 0 and
         (d - margin) < 0;
    ci90: 90 % task-bootstrap interval (B = 10,000, fixed seed);
    equivalent: ci90 inside (-margin, +margin) -- the pre-registered decision rule."""
    v = np.asarray([task_diffs[k] for k in sorted(task_diffs)], dtype=float)
    out = {"margin": margin, "n_tasks": int(v.size), "mean": float(v.mean()) if v.size else float("nan")}
    if v.size >= 2 and v.std(ddof=1) > 0:
        se = v.std(ddof=1) / math.sqrt(v.size)
        t_lo = (v.mean() + margin) / se
        t_hi = (v.mean() - margin) / se
        p_lo = float(sps.t.sf(t_lo, v.size - 1))
        p_hi = float(sps.t.cdf(t_hi, v.size - 1))
        out.update(p_t=max(p_lo, p_hi), p_t_lower=p_lo, p_t_upper=p_hi)
    else:
        out.update(p_t=float("nan"), p_t_lower=float("nan"), p_t_upper=float("nan"))
    if v.size >= 2:
        lo_, hi_ = v + margin, v - margin
        pw_lo = float(sps.wilcoxon(lo_[lo_ != 0], alternative="greater", method="exact").pvalue) if (lo_ != 0).sum() >= 1 else float("nan")
        pw_hi = float(sps.wilcoxon(hi_[hi_ != 0], alternative="less", method="exact").pvalue) if (hi_ != 0).sum() >= 1 else float("nan")
        out.update(p_w=max(pw_lo, pw_hi), p_w_lower=pw_lo, p_w_upper=pw_hi)
    else:
        out.update(p_w=float("nan"), p_w_lower=float("nan"), p_w_upper=float("nan"))
    lo, hi = task_bootstrap_ci(task_diffs, task_n, level=0.90)
    out["ci90"] = (lo, hi)
    # no verdict from fewer than two tasks (the bootstrap interval of a single task is a point)
    out["equivalent"] = bool(v.size >= 2 and np.isfinite(lo) and np.isfinite(hi) and lo > -margin and hi < margin)
    out["verdict"] = "n/a" if v.size < 2 else ("equivalent" if out["equivalent"] else "not shown")
    return out


def bh(pvals) -> list[float]:
    """Benjamini-Hochberg adjusted p-values; NaN entries pass through and are not counted."""
    p = np.asarray([np.nan if v is None else v for v in pvals], dtype=float)
    ok = ~np.isnan(p)
    out = np.full(p.shape, np.nan)
    if ok.any():
        out[ok] = sps.false_discovery_control(p[ok], method="bh")
    return [float(v) for v in out]


def bh_note(pvals, noun: str = "tests") -> str:
    """'' when every test of a BH family has a p-value; otherwise the sentence a caption adds, because ``bh`` drops
    NaN p-values from the family (a test whose per-task differences are all zero; review M6)."""
    p = [v for v in pvals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n, k = len(list(pvals)), len(p)
    why = "all per-task differences zero, or too few tasks"
    if k == n:
        return ""
    if k == 0:
        return f" (none of the {n} {noun} has a $p$-value: {why}; nothing is corrected)"
    return (f" ({n - k} of the {n} {noun} {'has' if n - k == 1 else 'have'} no $p$-value: {why}; the correction "
            f"runs over the other {k})")


def bh_reference(pvals) -> list[float]:
    """Plain step-up implementation (used by the tests to cross-check scipy)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return [float(v) for v in out]


def holm(pvals) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    out = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return [float(v) for v in out]
