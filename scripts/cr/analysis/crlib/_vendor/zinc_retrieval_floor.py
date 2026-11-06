#!/usr/bin/env python
"""ZINC retrieval floor for the E0 admissibility screen (ICML 2027 plan).

Scores every molecule of the SELFIES-VAE training set (``data/zinc_250k.txt``,
one SMILES per line) on each benchmark task and computes, per task, the
*retrieval floor*: the expected maximum score of ``N`` molecules drawn uniformly
at random from that set, for ``N`` in {5, 10, 20, 30, 50, 100, 300, 1000}.

A Bayesian-optimisation method that does not beat ``E[max of 30 random ZINC]``
at ``N = 30`` is doing no better than enumerating its own training data. The
5-molecule cold start of every benchmark run is itself drawn (without
replacement, ``np.random.RandomState(seed).choice``) from ``data/zinc_10k.txt``,
which is exactly the first 10,000 lines of ``zinc_250k.txt``; ``E[max_5]`` is
therefore the *cold-start floor* (reported both over the full set and over the
10k prefix the benchmark actually samples from — the prefix column is filled
only when the prefix identity has been verified for the scored file).

Exact ``E[max_N]`` with ties (draws WITH replacement, i.i.d.)::

    sort scores, unique values v_1 < ... < v_m, F(v_j) = #{scores <= v_j} / n
    E[max_N] = sum_j v_j * (F(v_j)^N - F(v_{j-1})^N),   F(v_0) = 0
    P(max_N <= v) = F(v)^N   ->  median and 5 % / 95 % quantiles of max_N

A second, hypergeometric variant (draws WITHOUT replacement, ``N`` distinct
molecules) is reported alongside: ``P(max_N <= v_j) = C(c_j, N) / C(n, N)`` with
``c_j = #{scores <= v_j}``. At ``n = 249,455`` the two agree to ~1e-4 relative;
they differ visibly only in small-n smoke runs.

Null hypothesis of the admissibility screen (``compare`` subcommand)
-------------------------------------------------------------------
H0: for every seed ``s``, ``best@B_s`` is distributed as the maximum of ``B``
molecules drawn uniformly from the scored set, i.i.d. across seeds. The CDF of
that maximum is known exactly, ``G(v) = F(v)^B`` (``--null replace``, the
default) or ``G(v) = C(c_j, B) / C(n, B)`` (``--null norepl``), so the null
distribution of the test statistic ``mean_s best@B_s`` is obtained by seeded
Monte Carlo *from the exact null CDF* (inverse-CDF sampling, no re-scoring),
and the two one-sided p-values are

    p_above = P_H0(mean >= observed),   p_below = P_H0(mean <= observed)

each as ``(1 + #{null >= obs}) / (reps + 1)``. A second, fully exact test is
reported alongside: the binomial sign test of ``K = #{s : best@B_s > m}``
against ``K ~ Bin(S, p0)`` with ``m`` the null median of ``max_B`` and
``p0 = 1 - G(m)`` (NOT 0.5 — on atomic tasks such as ``valt``, ``p0`` is ~0.9 %).

No paired reference sample is used. An earlier version calibrated the test
against one fixed 40-draw reference realisation shared by every solver on the
task; that reference sits at a random, task-specific offset from ``E[max_B]``
that every solver inherits, which made the screen anti-conservative on some
tasks and conservative on others (measured: 23.5 % "admissible" under H0 at
alpha = 0.05). The exact null above removes both the offset and the doubled
variance.

Verdicts are symmetric and each is carried by a test: ``admissible`` iff
``p_above < alpha``, ``below_floor`` iff ``p_below < alpha``, else
``inconclusive``; the label ``equivalent`` is used only when a TOST against an
explicit ``--equiv-margin`` passes (never inferred from a non-significant p).
Across the (task, solver) rows of one call the p-values are Holm-adjusted
within each solver family (``--multiplicity holm``, the default; ``none``
prints the uncorrected screen and says so in the table header).

Analysis set: by default only ``status == 'completed'`` cells with at least
``B`` scores enter a row (per protocol; the rest are listed as excluded).
``--itt`` (intention to treat) uses every requested cell, imputing a cell with
no finite score inside the budget by its exact cold-start best@5 (a lower
bound), and reports per row the non-finite, missing and imputed counts and the
status counts. ``--require-commit HASH`` refuses to run unless every result JSON
in the directory was produced at that clean commit; the output payload records
the dataset sha256 and n, the compare code's own commit/dirty state and argv.

Scoring and corrupt scores
--------------------------
``src.oracle.guacamol.score_smiles`` returns 0.0 for invalid SMILES, 0.0 on a
swallowed exception, and 0.0 whenever the GuacaMol objective returns its
``corrupt_score`` (-1.0). Only the first of those is loud. This script therefore

* counts invalid SMILES separately (``is_valid_smiles``, task independent),
* calls ``benchmark.objective.score`` itself for GuacaMol/PMO tasks (inside the
  same invalid-SMILES guard and the same try/except that ``score_smiles`` uses,
  so the stored value is bit-identical to what the benchmark oracle returns)
  and flags ``corrupt`` when that call raises or returns ``None``/``< 0``,
* counts records on BOTH the ``src.oracle.guacamol`` logger (the logp/qed
  failure path) and the ``guacamol`` logger, whose ``NullHandler`` otherwise
  hides "Unknown exception thrown during scoring". That second channel is the
  only way to see a component failure inside an ``ArithmeticMeanScoringFunction``,
  which averages a component's -1.0 into a *positive-looking* total that no
  ``raw < 0`` test can catch. Of the 19 GuacaMol/PMO objectives exactly two are
  such a mean, ``dhop`` and ``shop`` (measured: patching one leaf of ``dhop``
  turns 0.58-0.62 into a clean-looking 0.25 with no negative anywhere); every
  other task is a Geometric/Isomer/Tanimoto objective, which propagates
  ``corrupt_score`` and is therefore caught by the hard channel.

A component-corrupt value is reported (``n_component_warnings``) but NOT
rewritten to 0.0: the score vector must stay bit-identical to the oracle the
benchmark arms were run against, which is what makes the cold-start check in
``compare`` exact. Hard-corrupt values (``None``/``< 0``/exception) are stored
as 0.0, exactly as ``score_smiles`` would.

Paths and idempotency
---------------------
Relative paths (``--smiles``, ``--zinc10k``, ``--outdir``, ``--floor-dir``) are
resolved against the *repository root*, never the current working directory, so
the same command from any cwd reads and writes the same files; ``--outdir``
outside the repository is refused unless ``--allow-outside-repo``. Every
``meta_<task>.json`` and a ``dataset.json`` sentinel record the SMILES sha256
and ``n``; a task is skipped only when its vector matches that sha and n, so a
run with a different ``--smiles`` file rescores instead of republishing stale
numbers under the new sha.

Usage (from anywhere, CPU only)::

    python scripts/analysis/zinc_retrieval_floor.py score \\
        --tasks med1,logp,valt --n-molecules 3000 --workers 8 \\
        --outdir results/e0_floor_smoke

    python scripts/analysis/zinc_retrieval_floor.py score          # all 21 tasks

    python scripts/analysis/zinc_retrieval_floor.py compare \\
        --results-dir results/guacamol/em1_premise --tasks med1,valt \\
        --seeds 42,43 --solvers hvarfner_scaled_gp --budget 30 \\
        --floor-dir results/e0_floor

Outputs in ``--outdir``: ``scores_<task>.npy`` (float64, aligned to the
non-empty lines of the SMILES file), ``corrupt_<task>.npy``,
``meta_<task>.json`` (per-task sidecar: dataset sha, counts, timing),
``invalid_mask.npy`` + ``invalid_mask_meta.json``, ``dataset.json``,
``zinc_floor_summary.json`` and ``zinc_floor_summary.md``.
"""
from __future__ import annotations

import os

# Must precede every import: spawned workers inherit the environment. This
# script is CPU-only by design; guacamol pulls TensorFlow lazily at the first
# benchmark build and TF must neither see a GPU nor spam stderr. The thread
# caps are set unconditionally (not setdefault): a preset OMP_NUM_THREADS in
# the calling shell would otherwise be inherited by every worker and
# oversubscribe the machine by workers x that value.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

N_LIST: tuple[int, ...] = (5, 10, 20, 30, 50, 100, 300, 1000)
MAX_QUANTILES: tuple[float, ...] = (0.05, 0.50, 0.95)
SCORE_QUANTILES: tuple[float, ...] = (0.5, 0.9, 0.99, 0.999, 0.9999)
MC_DRAWS = 20_000
MC_N = 30
NULL_REPS = 100_000
DEFAULT_SMILES = "data/zinc_250k.txt"        # resolved against REPO_ROOT
DEFAULT_ZINC10K = "data/zinc_10k.txt"        # resolved against REPO_ROOT
DEFAULT_OUTDIR = "results/e0_floor"          # resolved against REPO_ROOT
COLD_START_POOL = 10_000  # generate_cold_start_ours: load_zinc_smiles(n_samples=10000)


def resolve_path(p: str | Path) -> Path:
    """Relative paths are anchored at the repository root, not at os.getcwd().

    Provenance (git commit, repo_root) is REPO_ROOT-anchored, so cwd-relative
    data/output paths could silently pair one tree's data with another tree's
    commit stamp.
    """
    q = Path(p).expanduser()
    return q if q.is_absolute() else (REPO_ROOT / q)


def all_task_ids() -> list[str]:
    from src.oracle.guacamol import EXTRA_TASKS, GUACAMOL_TASKS, PMO_EXTRA_TASKS

    return list(GUACAMOL_TASKS) + list(PMO_EXTRA_TASKS) + list(EXTRA_TASKS)


# ── Worker side ───────────────────────────────────────────────────────────────

_W: dict = {"oracle": None, "n_exc": 0, "n_guac": 0}


class _LogCounter(logging.Handler):
    """Counts records on one logger into ``_W[key]``.

    ``src.oracle.guacamol`` (key ``n_exc``): ``score_smiles`` logs every
    swallowed exception ("Scoring failed ..." at INFO, "Penalized logP
    calculation failed" at DEBUG) and returns 0.0.

    ``guacamol`` (key ``n_guac``): ``MoleculewiseScoringFunction.score`` logs
    "Unknown exception thrown during scoring of ..." at WARNING immediately
    before returning ``corrupt_score`` (-1.0). That record is the only visible
    trace of a *component* failure inside a composite objective, because
    ``ArithmeticMeanScoringFunction`` averages the -1.0 into the total instead
    of propagating it.
    """

    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        _W[self.key] += 1


def _worker_init(warm_tasks: tuple[str, ...] = ()) -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import src.oracle.guacamol as oracle  # per-process benchmark cache lives here

    for name, key in (("src.oracle.guacamol", "n_exc"), ("guacamol", "n_guac")):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)  # the logp failure path logs at DEBUG
        lg.propagate = False
        lg.handlers.clear()
        lg.addHandler(_LogCounter(key))
    _W["oracle"] = oracle
    # Warm the per-process cache now (TensorFlow import + benchmark objects,
    # sascorer/networkx for logp) so the per-task timings measure scoring only.
    for t in warm_tasks:
        if t in oracle.GUACAMOL_TASKS or t in oracle.PMO_EXTRA_TASKS:
            oracle._get_benchmark(t)
        elif t in oracle.EXTRA_TASKS:
            oracle.score_smiles("CCO", t)
    _W["n_exc"] = 0
    _W["n_guac"] = 0


def _pid_after_pause(_: int) -> int:
    """Warm-up probe: a short pause stops one early worker from draining the queue."""
    time.sleep(0.05)
    return os.getpid()


def _validity_chunk(item: tuple[int, list[str]]) -> tuple[int, np.ndarray, int]:
    start, smiles = item
    oracle = _W["oracle"]
    invalid = np.fromiter(
        (not oracle.is_valid_smiles(s) for s in smiles), dtype=bool, count=len(smiles)
    )
    return start, invalid, os.getpid()


def _score_chunk(
    item: tuple[str, int, list[str]]
) -> tuple[str, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Score one chunk; return (task, start, scores, exc, corrupt, warn, pid).

    For GuacaMol/PMO tasks the benchmark objective is called directly, inside
    exactly the guards ``score_smiles`` applies (empty string -> 0.0, RDKit
    parse failure -> 0.0, exception -> 0.0, ``None``/``< 0`` -> 0.0), so the
    stored vector is bit-identical to ``score_smiles`` while the corrupt cases
    become visible instead of being indistinguishable from a genuine 0.0.
    """
    task, start, smiles = item
    oracle = _W["oracle"]
    n = len(smiles)
    out = np.empty(n, dtype=np.float64)
    exc = np.zeros(n, dtype=bool)      # src.oracle.guacamol log records
    corrupt = np.zeros(n, dtype=bool)  # objective returned None/<0, or raised
    warn = np.zeros(n, dtype=bool)     # guacamol logger records (component corrupt)

    if task in oracle.GUACAMOL_TASKS or task in oracle.PMO_EXTRA_TASKS:
        from rdkit import Chem

        objective = oracle._get_benchmark(task).objective
        for i, s in enumerate(smiles):
            e0, g0 = _W["n_exc"], _W["n_guac"]
            if s is None or len(s) == 0:
                out[i] = 0.0
            elif Chem.MolFromSmiles(s) is None:
                out[i] = 0.0
            else:
                try:
                    raw = objective.score(s)
                except Exception:  # BatchScoringFunction.score_list does not catch
                    out[i] = 0.0
                    corrupt[i] = True
                else:
                    bad = raw is None or raw < 0
                    out[i] = 0.0 if bad else float(raw)
                    corrupt[i] = bool(bad)
            exc[i] = _W["n_exc"] > e0
            warn[i] = _W["n_guac"] > g0
    else:  # logp, qed: score_smiles owns the try/except and logs every failure
        for i, s in enumerate(smiles):
            e0, g0 = _W["n_exc"], _W["n_guac"]
            out[i] = float(oracle.score_smiles(s, task))
            exc[i] = _W["n_exc"] > e0
            warn[i] = _W["n_guac"] > g0
            corrupt[i] = exc[i]  # the swallowed exception IS the corrupt channel here
    return task, start, out, exc, corrupt, warn, os.getpid()


# ── Exact order statistics of the empirical distribution ─────────────────────

def _unique_cdf(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Unique sorted values, cumulative counts c_j = #{x <= v_j}, and n."""
    values, counts = np.unique(scores, return_counts=True)
    return values, np.cumsum(counts), int(scores.size)


def exact_emax(values: np.ndarray, cum: np.ndarray, n: int, N: int) -> float:
    """E[max of N i.i.d. draws WITH replacement] from the empirical distribution."""
    F = cum / n
    P = F ** N
    P_prev = np.concatenate(([0.0], P[:-1]))
    return float(np.sum(values * (P - P_prev)))


def exact_max_quantile(
    values: np.ndarray, cum: np.ndarray, n: int, N: int, q: float
) -> float:
    """Smallest v with P(max_N <= v) = F(v)^N >= q."""
    P = (cum / n) ** N
    idx = int(np.searchsorted(P, q, side="left"))
    return float(values[min(idx, values.size - 1)])


def _log_comb(a: np.ndarray, k: int) -> np.ndarray:
    from scipy.special import gammaln

    return gammaln(a + 1.0) - gammaln(k + 1.0) - gammaln(a - k + 1.0)


def max_cdf(values: np.ndarray, cum: np.ndarray, n: int, N: int, scheme: str) -> np.ndarray:
    """G_j = P(max_N <= v_j) under H0, for scheme 'replace' or 'norepl'."""
    if scheme == "replace":
        G = (cum / n) ** N
    elif scheme == "norepl":
        if N > n:
            raise ValueError(f"norepl null needs N={N} <= n={n}")
        cum_f = cum.astype(np.float64)
        G = np.zeros_like(cum_f)
        ok = cum >= N
        G[ok] = np.exp(_log_comb(cum_f[ok], N) - _log_comb(np.float64(n), N))
    else:
        raise ValueError(f"unknown null scheme {scheme!r}")
    G = np.clip(G, 0.0, 1.0)
    G = np.maximum.accumulate(G)  # monotone despite floating-point dust
    G[-1] = 1.0
    return G


def exact_emax_no_replacement(
    values: np.ndarray, cum: np.ndarray, n: int, N: int
) -> float | None:
    """E[max of N distinct molecules] (hypergeometric): P(max<=v_j)=C(c_j,N)/C(n,N)."""
    if N > n:
        return None
    G = max_cdf(values, cum, n, N, "norepl")
    G_prev = np.concatenate(([0.0], G[:-1]))
    return float(np.sum(values * (G - G_prev)))


def emax_from_cdf(values: np.ndarray, G: np.ndarray) -> float:
    G_prev = np.concatenate(([0.0], G[:-1]))
    return float(np.sum(values * (G - G_prev)))


def sample_max_from_cdf(
    values: np.ndarray, G: np.ndarray, size: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    """Inverse-CDF draws of max_N: u ~ U(0,1), take the smallest v with G(v) >= u.

    Exact for the discrete null (atoms and ties included) and independent of n,
    so no index sampling over the 250k vector is needed.
    """
    u = rng.random(size)
    idx = np.searchsorted(G, u.ravel(), side="left")
    np.minimum(idx, values.size - 1, out=idx)
    return values[idx].reshape(size)


def null_mean_draws(
    values: np.ndarray,
    G: np.ndarray,
    n_seeds: int,
    reps: int = NULL_REPS,
    seed: int = 0,
    max_block_cells: int = 4_000_000,
) -> np.ndarray:
    """Seeded MC sample of the exact null distribution of mean_s(max_B), n_seeds seeds.

    Depends only on (task CDF, n_seeds, reps, seed), never on the observed arm,
    so one draw serves every solver evaluated on the same task and seed count.
    """
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    block = max(1, int(max_block_cells // max(1, n_seeds)))
    i = 0
    while i < reps:
        r = min(block, reps - i)
        means[i:i + r] = sample_max_from_cdf(values, G, (r, n_seeds), rng).mean(axis=1)
        i += r
    return means


def null_mean_test(means: np.ndarray, obs_mean: float, n_seeds: int, seed: int) -> dict:
    """Both one-sided p-values as (1 + #{null on that side}) / (reps + 1).

    The +1 keeps them strictly positive and the test valid for a discrete
    statistic; ties are counted on both sides (inclusive), which is conservative.
    """
    reps = int(means.size)
    tol = 1e-12 * max(1.0, abs(obs_mean))
    ge = int(np.count_nonzero(means >= obs_mean - tol))
    le = int(np.count_nonzero(means <= obs_mean + tol))
    p_above = (1.0 + ge) / (reps + 1.0)
    p_below = (1.0 + le) / (reps + 1.0)
    return {
        "reps": reps, "seed": int(seed), "n_seeds": int(n_seeds),
        "p_above": p_above, "p_below": p_below,
        "p_mc_se": float(math.sqrt(max(p_above * (1 - p_above), p_below * (1 - p_below)) / reps)),
        "null_mean": float(means.mean()), "null_sd": float(means.std(ddof=1)) if reps > 1 else float("nan"),
        "null_q025": float(np.quantile(means, 0.025)), "null_q975": float(np.quantile(means, 0.975)),
    }


def null_sign_test(values: np.ndarray, G: np.ndarray, best: np.ndarray) -> dict:
    """Exact binomial sign test of K = #{best_s > m} against Bin(S, p0).

    ``m`` is the null median of max_B and ``p0 = 1 - G(m)``; p0 is NOT 0.5 for a
    discrete/atomic null (on ``valt``, P(max_30 > 0) is under 1 %), which is why
    a 0.5 sign test would be badly miscalibrated there.
    """
    from scipy.stats import binom

    j = int(np.searchsorted(G, 0.5, side="left"))
    j = min(j, values.size - 1)
    m = float(values[j])
    p0 = float(1.0 - G[j])
    S = int(best.size)
    k = int(np.count_nonzero(best > m))
    if p0 <= 0.0:
        p_above = 1.0 if k == 0 else 0.0
        p_below = 1.0
    else:
        p_above = float(binom.sf(k - 1, S, p0))
        p_below = float(binom.cdf(k, S, p0))
    return {"median_null": m, "p0": p0, "n_above_median": k, "n_seeds": S,
            "p_above": p_above, "p_below": p_below}


def tost_equivalence(best: np.ndarray, floor: float, margin: float) -> dict:
    """Two one-sided t-tests of mean(best) - floor against +/- margin.

    The floor is a known constant (computed exactly from the score vector), so a
    one-sample TOST around it is the right form. Equivalence is claimed only
    when this passes; a non-significant one-sided p is never read as a tie.
    """
    from scipy.stats import t as student_t

    S = int(best.size)
    d = float(best.mean() - floor)
    if S < 2:
        return {"margin": margin, "diff": d, "se": None, "p_tost": None, "equivalent": None,
                "note": "need >= 2 seeds"}
    se = float(best.std(ddof=1) / math.sqrt(S))
    if se == 0.0:
        eq = bool(abs(d) < margin)
        return {"margin": margin, "diff": d, "se": 0.0, "p_tost": 0.0 if eq else 1.0,
                "equivalent": eq, "note": "zero variance"}
    df = S - 1
    p_lower = float(student_t.sf((d + margin) / se, df))   # H0: d <= -margin
    p_upper = float(student_t.cdf((d - margin) / se, df))  # H0: d >= +margin
    p_tost = max(p_lower, p_upper)
    return {"margin": margin, "diff": d, "se": se, "p_tost": p_tost,
            "p_lower": p_lower, "p_upper": p_upper, "equivalent": None}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (NaNs pass through)."""
    idx = [i for i, p in enumerate(pvals) if p is not None and not math.isnan(p)]
    adj: list[float] = [float("nan")] * len(pvals)
    m = len(idx)
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = float(min(1.0, running))
    return adj


def monte_carlo_emax(
    scores: np.ndarray, N: int, draws: int, seed: int
) -> dict:
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, scores.size, size=(draws, N))  # with replacement
    mx = scores[idx].max(axis=1)
    mean = float(mx.mean())
    se = float(mx.std(ddof=1) / math.sqrt(draws)) if draws > 1 else float("nan")
    return {"N": N, "draws": draws, "seed": seed, "mean": mean, "se": se}


def floor_stats(
    scores: np.ndarray,
    n_list: tuple[int, ...] = N_LIST,
    seed: int = 0,
    mc_draws: int = MC_DRAWS,
    mc_n: int = MC_N,
    prefix_n: int = COLD_START_POOL,
    prefix_ok: bool | None = None,
) -> dict:
    """All distributional numbers for one task from its score vector.

    ``prefix_ok`` gates the ``prefix10k`` block: the 10k prefix is the benchmark's
    cold-start pool only if the scored file's first 10k SMILES have been verified
    to equal ``data/zinc_10k.txt``. Otherwise the block stays ``None`` rather than
    being labelled as the cold-start pool.
    """
    finite_mask = np.isfinite(scores)
    finite = scores[finite_mask]
    n = int(finite.size)
    if n == 0:
        raise ValueError("no finite scores")
    values, cum, _ = _unique_cdf(finite)

    emax = {str(N): exact_emax(values, cum, n, N) for N in n_list}
    emax_norepl = {str(N): exact_emax_no_replacement(values, cum, n, N) for N in n_list}
    max_q = {
        str(N): {f"q{int(round(q * 100)):02d}": exact_max_quantile(values, cum, n, N, q)
                 for q in MAX_QUANTILES}
        for N in n_list
    }

    # Internal consistency: E[max_1] is the mean; E[max_N] is nondecreasing in N.
    e1 = exact_emax(values, cum, n, 1)
    mean = float(finite.mean())
    if abs(e1 - mean) > 1e-9 * max(1.0, abs(mean)):
        raise AssertionError(f"E[max_1]={e1!r} != mean={mean!r}")
    seq = [emax[str(N)] for N in n_list]
    if any(b < a - 1e-12 for a, b in zip(seq, seq[1:])):
        raise AssertionError(f"E[max_N] not nondecreasing: {seq}")

    mc = monte_carlo_emax(finite, mc_n, mc_draws, seed)
    exact_mc = exact_emax(values, cum, n, mc_n)
    disc = exact_mc - mc["mean"]
    if mc["se"] > 0:
        z = disc / mc["se"]
        within = bool(abs(z) <= 3.0)
    else:  # degenerate sample (e.g. every drawn maximum is 0.0): z undefined
        z = None
        within = bool(abs(disc) <= 1e-12)
    mc_check = {
        **mc,
        "exact": exact_mc,
        "discrepancy_exact_minus_mc": disc,
        "z": z,
        "within_3se": within,
    }

    # Cold-start pool: the benchmark draws its 5 start molecules from the first
    # 10k lines only. Report the same floors over that prefix when the scored
    # file's prefix has been verified to be exactly data/zinc_10k.txt.
    prefix = None
    if prefix_ok is True and scores.size >= prefix_n:
        pf = scores[:prefix_n]
        pf = pf[np.isfinite(pf)]
        if pf.size:
            pv, pc, pn = _unique_cdf(pf)
            prefix = {
                "n": int(pn),
                "emax": {str(N): exact_emax(pv, pc, pn, N) for N in n_list},
                "emax_norepl": {
                    str(N): exact_emax_no_replacement(pv, pc, pn, N) for N in n_list
                },
            }

    argmax = int(np.nanargmax(np.where(finite_mask, scores, -np.inf)))
    return {
        "n_finite": n,
        "n_nonfinite": int(scores.size - n),
        "mean": mean,
        "std": float(finite.std(ddof=0)),
        "frac_zero": float(np.mean(finite == 0.0)),
        "frac_ge1": float(np.mean(finite >= 1.0)),
        "quantiles": {f"q{q * 100:g}": float(np.quantile(finite, q)) for q in SCORE_QUANTILES},
        "max": float(finite.max()),
        "argmax_index": argmax,
        "emax": emax,
        "emax_norepl": emax_norepl,
        "max_quantiles": max_q,
        "mc_check": mc_check,
        "prefix10k": prefix,
        "prefix10k_gate": prefix_ok,
    }


# ── Provenance ────────────────────────────────────────────────────────────────

def _git(*args: str) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=10
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _pkg_version(name: str) -> str | None:
    import importlib.metadata as md

    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def collect_provenance() -> dict:
    return {
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_commit_full": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": _git("status", "--porcelain") != "",
        "repo_root": str(REPO_ROOT),
        "cwd": os.getcwd(),
        "script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "cli_args": sys.argv,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "packages": {p: _pkg_version(p) for p in ("numpy", "scipy", "rdkit", "guacamol", "networkx")},
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


# ── Data ──────────────────────────────────────────────────────────────────────

def load_smiles(path: Path) -> tuple[list[str], dict]:
    raw = path.read_bytes()
    lines = [ln.strip() for ln in raw.decode("utf-8").split("\n")]
    smiles = [ln for ln in lines if ln]
    n_dupl = len(smiles) - len(set(smiles))
    info = {
        "path": str(path),
        "path_resolved": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "n_lines_nonempty": len(smiles),
        "n_duplicate_smiles": int(n_dupl),
        "alignment": "scores_<task>.npy[i] is the i-th non-empty, whitespace-stripped line",
    }
    return smiles, info


def zinc10k_prefix_identity(smiles: list[str], path: Path) -> bool | None:
    """True iff data/zinc_10k.txt equals the first 10k SMILES of the scored file."""
    if not path.exists() or len(smiles) < COLD_START_POOL:
        return None
    ten_k = [ln.strip() for ln in path.read_text().split("\n") if ln.strip()]
    return ten_k == smiles[:COLD_START_POOL]


# ── Atomic writes ────────────────────────────────────────────────────────────

def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """np.save appends '.npy' to a name that lacks it, so the temp name keeps it."""
    tmp = path.with_name(path.stem + ".tmp.npy")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ── Pool driving ─────────────────────────────────────────────────────────────

def _chunks(n: int, chunk: int) -> list[tuple[int, int]]:
    return [(s, min(s + chunk, n)) for s in range(0, n, chunk)]


def _auto_chunk(n: int, requested: int, workers: int) -> int:
    """Cap the chunk so every worker sees at least ~4 chunks."""
    return max(1, min(requested, math.ceil(n / max(1, 4 * workers))))


def _fmt_z(z: float | None) -> str:
    return "undef (se=0)" if z is None else f"{z:+.2f}"


def _fmt_s(sec: float) -> str:
    sec = max(0.0, sec)
    if sec < 90:
        return f"{sec:.0f}s"
    if sec < 5400:
        return f"{sec / 60:.1f}min"
    return f"{sec / 3600:.2f}h"


def _drive(
    pool,
    func,
    items,
    n_total: int,
    label: str,
    progress_every: float = 3.0,
    item_timeout: float = 600.0,
    heartbeat: float = 30.0,
):
    """Run ``func`` over ``items`` with imap_unordered; yield results; print ETA.

    ``it.next(timeout=...)`` instead of a bare ``for`` loop: a worker killed by
    the OOM killer is respawned by ``multiprocessing.Pool`` but its in-flight
    item is lost, so a plain iteration would block forever with no output. Here
    a stall prints a heartbeat and raises after ``item_timeout`` seconds.
    """
    t0 = time.time()
    last = t0
    last_result = t0
    done_mols = 0
    n_items = len(items)
    pids: set[int] = set()
    _drive.last_pids = pids  # type: ignore[attr-defined]
    it = pool.imap_unordered(func, items, chunksize=1)
    k = 0
    while True:
        try:
            res = it.next(timeout=max(1.0, min(heartbeat, item_timeout)))
        except StopIteration:
            break
        except mp.TimeoutError:
            stalled = time.time() - last_result
            print(
                f"  [{label}] HEARTBEAT: no result for {_fmt_s(stalled)} "
                f"({k}/{n_items} chunks done, {done_mols}/{n_total} mol); "
                f"failing at {_fmt_s(item_timeout)}",
                flush=True,
            )
            if stalled > item_timeout:
                raise RuntimeError(
                    f"[{label}] no chunk completed for {stalled:.0f}s "
                    f"(> --item-timeout {item_timeout:.0f}s) after {k}/{n_items} chunks; "
                    "a worker was probably killed (OOM) and its work item lost"
                )
            continue
        k += 1
        last_result = time.time()
        pids.add(res[-1])
        done_mols += res[2].size if func is _score_chunk else res[1].size
        now = time.time()
        if now - last >= progress_every or k == n_items:
            el = now - t0
            rate = done_mols / el if el > 0 else float("nan")
            eta = (n_total - done_mols) / rate if rate > 0 else float("nan")
            print(
                f"  [{label}] {k}/{n_items} chunks  {done_mols}/{n_total} mol  "
                f"{rate:,.0f} mol/s  elapsed {_fmt_s(el)}  ETA {_fmt_s(eta)}  "
                f"workers seen {len(pids)}",
                flush=True,
            )
            last = now
        yield res


# ── Score command ────────────────────────────────────────────────────────────

def _npy_shape_ok(path: Path, n: int, dtype=np.float64) -> bool:
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception:
        return False
    return arr.ndim == 1 and arr.shape[0] == n and arr.dtype == dtype


def _vector_ok(outdir: Path, task: str, n: int, sha: str) -> tuple[bool, str]:
    """A cached vector is reusable only if its meta sidecar carries this dataset.

    Length alone is not identity: a different --smiles file with the same n
    would otherwise be silently republished under the new file's sha256.
    Matching is on (sha256, n) and never on the path, because data/zinc_250k.txt
    is a symlink whose str() and resolve() differ between trees.
    """
    npy = outdir / f"scores_{task}.npy"
    if not npy.exists():
        return False, "no vector"
    if not _npy_shape_ok(npy, n):
        return False, f"vector is not float64[{n}]"
    meta_path = outdir / f"meta_{task}.json"
    if not meta_path.exists():
        return False, "no meta sidecar (interrupted run?)"
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False, "unreadable meta sidecar"
    if meta.get("smiles_sha256") != sha:
        return False, f"meta sha256 {str(meta.get('smiles_sha256'))[:12]}... != {sha[:12]}..."
    if int(meta.get("n", -1)) != n:
        return False, f"meta n={meta.get('n')} != {n}"
    return True, "ok"


def _invalid_ok(outdir: Path, n: int, sha: str) -> bool:
    if not _npy_shape_ok(outdir / "invalid_mask.npy", n, dtype=np.bool_):
        return False
    side = outdir / "invalid_mask_meta.json"
    if not side.exists():
        return False
    try:
        meta = json.loads(side.read_text())
    except json.JSONDecodeError:
        return False
    return meta.get("smiles_sha256") == sha and int(meta.get("n", -1)) == n


def _check_dataset_sentinel(outdir: Path, ds_info: dict, force: bool) -> None:
    """Refuse to mix two datasets in one outdir (unless --force rescores it)."""
    path = outdir / "dataset.json"
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except json.JSONDecodeError:
            old = {}
        same = (old.get("sha256") == ds_info["sha256"]
                and int(old.get("n_molecules_used", -1)) == int(ds_info["n_molecules_used"]))
        if not same and not force:
            raise SystemExit(
                f"refusing to write into {outdir}: it holds sha256 "
                f"{str(old.get('sha256'))[:16]}... n={old.get('n_molecules_used')}, "
                f"this run is sha256 {ds_info['sha256'][:16]}... n={ds_info['n_molecules_used']}. "
                "Use a different --outdir, or --force to rescore the directory."
            )
    _atomic_write_text(path, json.dumps(
        {k: ds_info[k] for k in ("path", "path_resolved", "sha256", "bytes",
                                 "n_lines_nonempty", "n_molecules_used",
                                 "zinc10k_prefix_identity")}
        | {"written_at": datetime.now().astimezone().isoformat(timespec="seconds")},
        indent=2))


def run_score(args: argparse.Namespace) -> int:
    t_run0 = time.time()
    tasks_all = all_task_ids()
    tasks = tasks_all if args.tasks in (None, "", "all") else [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in tasks_all]
    if unknown:
        raise SystemExit(f"unknown tasks {unknown}; known: {tasks_all}")

    outdir = resolve_path(args.outdir)
    if not args.allow_outside_repo:
        try:
            outdir.resolve().relative_to(REPO_ROOT.resolve())
        except ValueError:
            raise SystemExit(
                f"--outdir {outdir} resolves outside the repository {REPO_ROOT}; "
                "pass --allow-outside-repo if that is intended"
            )
    outdir.mkdir(parents=True, exist_ok=True)
    smiles_path = resolve_path(args.smiles)
    smiles, ds_info = load_smiles(smiles_path)
    if args.n_molecules is not None:
        smiles = smiles[: args.n_molecules]
    n = len(smiles)
    ds_info["n_molecules_used"] = n
    ds_info["zinc10k_prefix_identity"] = zinc10k_prefix_identity(smiles, resolve_path(args.zinc10k))
    ds_info["zinc10k_path"] = str(resolve_path(args.zinc10k))
    ds_info["outdir"] = str(outdir)
    ds_info["cwd"] = os.getcwd()
    print(
        f"SMILES file {smiles_path}  sha256 {ds_info['sha256'][:16]}...  "
        f"{ds_info['n_lines_nonempty']} lines, {ds_info['n_duplicate_smiles']} duplicates, "
        f"using n={n}; zinc_10k is the 10k prefix: {ds_info['zinc10k_prefix_identity']}\n"
        f"outdir {outdir}  (cwd {os.getcwd()}, repo root {REPO_ROOT})",
        flush=True,
    )
    _check_dataset_sentinel(outdir, ds_info, args.force)

    if args.nice:
        try:
            os.nice(args.nice)
        except OSError:
            pass

    pending: list[str] = []
    skipped: list[str] = []
    for t in tasks:
        ok, why = _vector_ok(outdir, t, n, ds_info["sha256"])
        if args.force or not ok:
            pending.append(t)
            if not args.force and (outdir / f"scores_{t}.npy").exists():
                print(f"[{t}] rescoring: cached vector rejected ({why})")
        else:
            skipped.append(t)
            print(f"[{t}] skip: {outdir / f'scores_{t}.npy'} matches sha256 "
                  f"{ds_info['sha256'][:12]}... n={n} (use --force to rescore)")

    invalid_path = outdir / "invalid_mask.npy"
    need_validity = args.force or not _invalid_ok(outdir, n, ds_info["sha256"])

    chunk = _auto_chunk(n, args.chunk_size, args.workers)
    per_task_meta: dict[str, dict] = {}
    invalid: np.ndarray | None = None
    warmup_seconds: float | None = None

    if pending or need_validity:
        ctx = mp.get_context("spawn")
        print(f"starting spawn pool: {args.workers} workers, chunk={chunk} "
              f"({len(_chunks(n, chunk))} chunks/task)", flush=True)
        t_pool0 = time.time()
        with ctx.Pool(args.workers, initializer=_worker_init, initargs=(tuple(pending),)) as pool:
            warm_pids = set(pool.map(_pid_after_pause, range(4 * args.workers), chunksize=1))
            warmup_seconds = time.time() - t_pool0
            print(
                f"pool warm-up (TF import + {len(pending)} benchmark build(s) per worker): "
                f"{_fmt_s(warmup_seconds)}; {len(warm_pids)}/{args.workers} distinct worker pids answered",
                flush=True,
            )
            if need_validity:
                t0 = time.time()
                invalid = np.zeros(n, dtype=bool)
                filled = np.zeros(n, dtype=bool)
                items = [(s, smiles[s:e]) for s, e in _chunks(n, chunk)]
                for start, inv, _pid in _drive(pool, _validity_chunk, items, n, "validity",
                                               item_timeout=args.item_timeout):
                    invalid[start:start + inv.size] = inv
                    filled[start:start + inv.size] = True
                assert filled.all(), "validity pass left gaps"
                _atomic_save_npy(invalid_path, invalid)
                _atomic_write_text(outdir / "invalid_mask_meta.json", json.dumps({
                    "n": n, "smiles_sha256": ds_info["sha256"], "smiles_path": str(smiles_path),
                    "n_invalid": int(invalid.sum()),
                    "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                }, indent=2))
                print(f"  [validity] n_invalid={int(invalid.sum())} in {_fmt_s(time.time() - t0)}", flush=True)
            else:
                invalid = np.load(invalid_path)

            task_times: list[float] = []
            for ti, task in enumerate(pending, 1):
                t0 = time.time()
                scores = np.full(n, np.nan, dtype=np.float64)
                exc = np.zeros(n, dtype=bool)
                corrupt = np.zeros(n, dtype=bool)
                warn = np.zeros(n, dtype=bool)
                filled = np.zeros(n, dtype=bool)
                items = [(task, s, smiles[s:e]) for s, e in _chunks(n, chunk)]
                for _task, start, sc, ex, co, wa, _pid in _drive(
                        pool, _score_chunk, items, n, task, item_timeout=args.item_timeout):
                    scores[start:start + sc.size] = sc
                    exc[start:start + sc.size] = ex
                    corrupt[start:start + sc.size] = co
                    warn[start:start + sc.size] = wa
                    filled[start:start + sc.size] = True
                assert filled.all(), f"[{task}] scoring left gaps"
                wall = time.time() - t0
                n_pids = len(getattr(_drive, "last_pids", set()))
                _atomic_save_npy(outdir / f"scores_{task}.npy", scores)
                _atomic_save_npy(outdir / f"corrupt_{task}.npy", corrupt)
                exc_idx = np.flatnonzero(exc)
                corrupt_idx = np.flatnonzero(corrupt)
                comp_only_idx = np.flatnonzero(warn & ~corrupt)
                meta = {
                    "task": task,
                    "n": n,
                    "smiles_sha256": ds_info["sha256"],
                    "smiles_path": str(smiles_path),
                    "smiles_path_resolved": ds_info["path_resolved"],
                    "n_molecules_used": n,
                    "n_invalid": int(invalid.sum()),
                    "n_exceptions": int(exc.sum()),
                    "n_corrupt": int(corrupt.sum()),
                    "n_component_warnings": int(warn.sum()),
                    "n_component_only": int((warn & ~corrupt).sum()),
                    "exception_indices_head": exc_idx[:50].tolist(),
                    "exception_smiles_head": [smiles[i] for i in exc_idx[:50]],
                    "corrupt_indices_head": corrupt_idx[:50].tolist(),
                    "corrupt_smiles_head": [smiles[i] for i in corrupt_idx[:50]],
                    "component_only_indices_head": comp_only_idx[:50].tolist(),
                    "component_only_smiles_head": [smiles[i] for i in comp_only_idx[:50]],
                    "n_nonfinite": int((~np.isfinite(scores)).sum()),
                    "wall_seconds": wall,
                    "mol_per_s": n / wall if wall > 0 else None,
                    "workers": args.workers,
                    "distinct_worker_pids": n_pids,
                    "chunk_size": chunk,
                    "scored_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "git_commit": _git("rev-parse", "--short", "HEAD"),
                    "git_dirty": _git("status", "--porcelain") != "",
                }
                _atomic_write_text(outdir / f"meta_{task}.json", json.dumps(meta, indent=2))
                per_task_meta[task] = meta
                task_times.append(wall)
                mean_t = float(np.mean(task_times))
                remaining = len(pending) - ti
                print(
                    f"[{task}] done: {n} mol in {_fmt_s(wall)} ({n / wall:,.0f} mol/s, "
                    f"{n_pids} distinct worker pids)  n_exc={meta['n_exceptions']}  "
                    f"n_corrupt={meta['n_corrupt']}  n_component_only={meta['n_component_only']}  "
                    f"n_nonfinite={meta['n_nonfinite']}  | tasks {ti}/{len(pending)}, "
                    f"run elapsed {_fmt_s(time.time() - t_run0)}, rough ETA {_fmt_s(mean_t * remaining)} "
                    f"(mean per task; per-task cost varies ~50x)",
                    flush=True,
                )
        print(f"pool closed after {_fmt_s(time.time() - t_pool0)}", flush=True)
    else:
        invalid = np.load(invalid_path)

    # Stats for every requested task (freshly scored or skipped) from the vectors.
    results: dict[str, dict] = {}
    for task in tasks:
        scores = np.load(outdir / f"scores_{task}.npy")
        meta_path = outdir / f"meta_{task}.json"
        if task in per_task_meta:
            meta = per_task_meta[task]
        elif meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:  # unreachable via the skip gate, kept as a defensive default
            print(f"[{task}] warning: no meta sidecar; exception count unknown")
            meta = {"n_exceptions": None, "n_corrupt": None, "wall_seconds": None,
                    "mol_per_s": None, "n_nonfinite": int((~np.isfinite(scores)).sum())}
        st = floor_stats(scores, N_LIST, seed=args.seed, mc_draws=args.mc_draws, mc_n=MC_N,
                         prefix_ok=ds_info["zinc10k_prefix_identity"])
        results[task] = {
            "task": task,
            "n": int(scores.size),
            "n_invalid": int(invalid.sum()),
            "n_exceptions": meta.get("n_exceptions"),
            "n_corrupt": meta.get("n_corrupt"),
            "n_component_only": meta.get("n_component_only"),
            "n_nonfinite": st["n_nonfinite"],
            **{k: v for k, v in st.items() if k not in ("n_nonfinite",)},
            "argmax_smiles": smiles[st["argmax_index"]],
            "wall_seconds": meta.get("wall_seconds"),
            "mol_per_s": meta.get("mol_per_s"),
            "distinct_worker_pids": meta.get("distinct_worker_pids"),
            "scored_in_this_run": task in per_task_meta,
            "smiles_sha256": meta.get("smiles_sha256", ds_info["sha256"]),
            "npy": str(outdir / f"scores_{task}.npy"),
        }
        mc = st["mc_check"]
        print(
            f"[{task}] n={scores.size} n_invalid={invalid.sum()} n_exc={meta.get('n_exceptions')} "
            f"n_corrupt={meta.get('n_corrupt')} n_component_only={meta.get('n_component_only')} "
            f"mean={st['mean']:.4f} frac0={st['frac_zero']:.4f} max={st['max']:.4f} | "
            f"E[max5]={st['emax']['5']:.4f} E[max30]={st['emax']['30']:.4f} "
            f"E[max100]={st['emax']['100']:.4f} E[max1000]={st['emax']['1000']:.4f} | "
            f"MC30 {mc['mean']:.5f}±{mc['se']:.5f} vs exact {mc['exact']:.5f} "
            f"(discrepancy {mc['discrepancy_exact_minus_mc']:+.2e}, z={_fmt_z(mc['z'])}, "
            f"within 3se: {mc['within_3se']})",
            flush=True,
        )

    write_summary(outdir, results, ds_info, {
        "workers": args.workers, "chunk_size": chunk, "seed": args.seed,
        "mc_draws": args.mc_draws, "mc_n": MC_N, "n_list": list(N_LIST),
        "tasks_requested": tasks, "tasks_scored": pending, "tasks_skipped": skipped,
        "pool_warmup_seconds": warmup_seconds,
        "item_timeout": args.item_timeout,
        "run_wall_seconds": time.time() - t_run0,
        "timing_note": "per-task wall_seconds/mol_per_s measure scoring only; worker start-up "
                       "(TensorFlow import, benchmark construction) is pool_warmup_seconds",
    }, force=args.force)
    print(f"summary -> {outdir / 'zinc_floor_summary.json'} and .md  (run wall {_fmt_s(time.time() - t_run0)})")
    return 0


def write_summary(outdir: Path, results: dict[str, dict], ds_info: dict, run_info: dict,
                  force: bool = False) -> None:
    """Merge per-task results into the summary JSON and regenerate the .md table."""
    js_path = outdir / "zinc_floor_summary.json"
    existing: dict = {}
    if js_path.exists():
        try:
            existing = json.loads(js_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    tasks_merged = dict(existing.get("tasks", {}))
    old_ds = existing.get("dataset", {})
    if old_ds.get("sha256") not in (None, ds_info["sha256"]) or \
            old_ds.get("n_molecules_used") not in (None, ds_info["n_molecules_used"]):
        if not force:
            raise SystemExit(
                f"{js_path} was written for sha256 {str(old_ds.get('sha256'))[:16]}... "
                f"n={old_ds.get('n_molecules_used')}, this run is "
                f"{ds_info['sha256'][:16]}... n={ds_info['n_molecules_used']}. "
                "Refusing to republish the old tasks under the new dataset; "
                "use a fresh --outdir or --force."
            )
        print("warning: --force: existing summary was for a different SMILES file / n; discarding its tasks")
        tasks_merged = {}
    # A merged (not rescored) task must itself belong to this dataset.
    stale = [t for t, r in tasks_merged.items()
             if t not in results and r.get("smiles_sha256") not in (None, ds_info["sha256"])]
    for t in stale:
        del tasks_merged[t]
    if stale:
        print(f"dropped {len(stale)} task(s) from the summary carrying a different sha256: {stale}")
    tasks_merged.update(results)
    ordered = [t for t in all_task_ids() if t in tasks_merged]
    summary = {
        "purpose": "E0 retrieval floor: E[max of N uniform random ZINC molecules] per task",
        "definitions": {
            "emax": "E[max_N], N i.i.d. draws WITH replacement from the empirical score distribution: "
                    "sum_j v_j (F_j^N - F_{j-1}^N)",
            "emax_norepl": "E[max_N] for N DISTINCT molecules (hypergeometric): P(max<=v_j)=C(c_j,N)/C(n,N)",
            "max_quantiles": "quantiles of max_N from P(max_N <= v) = F(v)^N",
            "mc_check": "seeded Monte Carlo of `draws` maxima of N=mc_n draws with replacement, RandomState(seed)",
            "prefix10k": "same floors over the first 10,000 SMILES = data/zinc_10k.txt, the benchmark's "
                         "cold-start pool; filled only when zinc10k_prefix_identity is True "
                         "(prefix10k_gate records the flag that was applied)",
            "n_invalid": "SMILES rejected by RDKit (is_valid_smiles); score_smiles returns 0.0 for them",
            "n_exceptions": "records on the src.oracle.guacamol logger (the logp/qed failure path)",
            "n_corrupt": "GuacaMol/PMO objective returned None or a negative corrupt_score, or raised: "
                         "stored as 0.0 exactly as score_smiles would, but counted here instead of "
                         "being indistinguishable from a genuine 0.0",
            "n_component_only": "the 'guacamol' logger warned (a component of a composite objective "
                                "returned corrupt_score) while the total stayed >= 0: the value is KEPT "
                                "so the vector remains bit-identical to the benchmark oracle, and only "
                                "flagged here",
            "frac_zero": "fraction of finite scores exactly 0.0 (includes genuine zeros, invalid and corrupt zeros)",
            "null_hypothesis": "compare subcommand H0: best@B ~ max of B molecules drawn uniformly from this "
                               "scored set, i.i.d. across seeds; CDF G(v)=F(v)^B (--null replace, default) or "
                               "C(c_j,B)/C(n,B) (--null norepl). p-values from seeded MC of that exact null "
                               "plus an exact binomial sign test; no paired reference realisation is used",
        },
        "dataset": ds_info,
        "run": run_info,
        "provenance": collect_provenance(),
        "tasks": {t: tasks_merged[t] for t in ordered},
    }
    _atomic_write_text(js_path, json.dumps(summary, indent=2))

    def f4(x: float | None) -> str:
        return "—" if x is None else f"{x:.4f}"

    def i_or_q(x) -> str:
        return "?" if x is None else str(x)

    lines = [
        "# ZINC retrieval floor (E0 admissibility screen)",
        "",
        f"Scored file: `{ds_info['path']}` · sha256 `{ds_info['sha256']}` · "
        f"{ds_info['n_lines_nonempty']} SMILES ({ds_info['n_duplicate_smiles']} duplicates) · "
        f"n used = **{ds_info['n_molecules_used']}** · zinc_10k is the 10k prefix: {ds_info['zinc10k_prefix_identity']}",
        "",
        f"Provenance: commit `{summary['provenance']['git_commit']}` "
        f"(dirty={summary['provenance']['git_dirty']}), {summary['provenance']['timestamp']}, "
        f"numpy {summary['provenance']['packages'].get('numpy')}, rdkit {summary['provenance']['packages'].get('rdkit')}, "
        f"guacamol {summary['provenance']['packages'].get('guacamol')}.",
        "",
        "E[max_N] is the exact expectation of the maximum of N i.i.d. draws **with replacement** from the "
        "empirical score distribution (ties handled through the CDF); q05/q50/q95 are quantiles of max_30. "
        "`E[max5] 10k` is the cold-start floor over the 10k prefix the benchmark actually samples from "
        "(blank unless the prefix identity was verified). MC30 = seeded Monte Carlo (20,000 draws of 30) "
        "mean ± s.e. and z = (exact − MC)/s.e. `n_corrupt` = objective returned None/<0 or raised "
        "(stored as 0.0, as the oracle does); `n_comp` = a component of a composite objective returned "
        "corrupt_score while the total stayed ≥ 0 (value kept, flagged only).",
        "",
        "| task | n | E[max5] | E[max5] 10k | E[max30] | q05 / q50 / q95 of max30 | E[max100] | E[max1000] | frac0 | n_invalid | n_exc | n_corrupt | n_comp | MC30 check |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for t in ordered:
        r = tasks_merged[t]
        mq = r["max_quantiles"]["30"]
        mc = r["mc_check"]
        pre = r.get("prefix10k")
        pre5 = f4(pre["emax"]["5"]) if pre else "—"
        lines.append(
            f"| {t} | {r['n']} | {f4(r['emax']['5'])} | {pre5} | "
            f"{f4(r['emax']['30'])} | {f4(mq['q05'])} / {f4(mq['q50'])} / {f4(mq['q95'])} | "
            f"{f4(r['emax']['100'])} | {f4(r['emax']['1000'])} | {r['frac_zero']:.4f} | "
            f"{r['n_invalid']} | {i_or_q(r['n_exceptions'])} | {i_or_q(r.get('n_corrupt'))} | "
            f"{i_or_q(r.get('n_component_only'))} | "
            f"MC {mc['mean']:.5f}±{mc['se']:.5f}, z={_fmt_z(mc['z'])} |"
        )
    lines += ["", "Per-task score distribution:", "",
              "| task | mean | std | frac≥1 | q50 | q90 | q99 | q99.9 | q99.99 | max | E[max30] no-repl. | wall s | mol/s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in ordered:
        r = tasks_merged[t]
        q = r["quantiles"]
        wall_s = "—" if r.get("wall_seconds") is None else f"{r['wall_seconds']:.1f}"
        rate_s = "—" if r.get("mol_per_s") is None else f"{r['mol_per_s']:,.0f}"
        lines.append(
            f"| {t} | {r['mean']:.4f} | {r['std']:.4f} | {r['frac_ge1']:.5f} | {q['q50']:.4f} | {q['q90']:.4f} | "
            f"{q['q99']:.4f} | {q['q99.9']:.4f} | {q['q99.99']:.4f} | {r['max']:.4f} | "
            f"{f4(r['emax_norepl']['30'])} | {wall_s} | {rate_s} |"
        )
    _atomic_write_text(outdir / "zinc_floor_summary.md", "\n".join(lines) + "\n")


# ── Compare command ──────────────────────────────────────────────────────────

class MissingCellsError(RuntimeError):
    pass


class ProvenanceError(RuntimeError):
    """A loaded result JSON was not produced at the required clean commit."""


def _stamped_provenance(d: dict) -> tuple[str | None, bool | None]:
    """(git_commit, git_dirty) as stamped by the batch runner.

    ``scripts/run_geowalk_family_batch.py`` writes ``d['provenance']['provenance']
    [...]`` (the runner wraps the collector's dict under a second ``provenance``
    key; verified on the ``_smoke_em1_*`` files). That nested path is the ONLY
    one accepted (plan section 3.1, item 2): a flat ``d['provenance'][...]`` or a
    missing stamp returns (None, None) so the caller reports the file as an
    offender instead of raising KeyError.
    """
    p = d.get("provenance")
    if not (isinstance(p, dict) and isinstance(p.get("provenance"), dict)):
        return None, None
    p = p["provenance"]
    return p.get("git_commit"), p.get("git_dirty")


def check_result_provenance(index: dict[tuple[str, int, str], dict], require_commit: str) -> int:
    """Assert every loaded JSON was produced at ``require_commit`` with a clean tree.

    Returns the number of files checked; raises ProvenanceError listing every
    offending file (wrong commit, dirty tree, or provenance missing).
    """
    want = require_commit.strip()
    if not want:
        raise ValueError("--require-commit needs a non-empty hash")
    bad: list[str] = []
    for d in index.values():
        commit, dirty = _stamped_provenance(d)
        why = []
        if commit != want:
            why.append(f"git_commit={commit!r} != {want!r}")
        if dirty is not False:
            why.append(f"git_dirty={dirty!r} (must be False)")
        if why:
            bad.append(f"{d['_file']}: " + "; ".join(why))
    if bad:
        raise ProvenanceError(
            f"{len(bad)} of {len(index)} result file(s) fail --require-commit {want}:\n  "
            + "\n  ".join(bad)
        )
    return len(index)


def _load_result_dir(
    results_dir: Path, require_commit: str | None = None
) -> dict[tuple[str, int, str], dict]:
    """Index every benchmark JSON by (our_task_id, seed, solver); raise on duplicates.

    With ``require_commit`` every file in the directory (not only the requested
    cells) must carry that clean commit stamp (see ``check_result_provenance``).
    """
    index: dict[tuple[str, int, str], dict] = {}
    files = sorted(results_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no *.json in {results_dir}")
    for f in files:
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"unreadable JSON {f}: {e}") from e
        key = (str(d.get("our_task_id", d.get("task"))), int(d["seed"]), str(d["solver"]))
        if key in index:
            raise RuntimeError(f"duplicate cell {key}: {index[key]['_file']} and {f}")
        d["_file"] = str(f)
        index[key] = d
    if require_commit is not None:
        check_result_provenance(index, require_commit)
    return index


def _status_class(status) -> str:
    """Bucket a result's status: 'completed', 'crashed' (any 'crashed...' string), else the raw value."""
    if status == "completed":
        return "completed"
    if isinstance(status, str) and status.startswith("crashed"):
        return "crashed"
    return f"other:{status!r}"


def coldstart_best(scores: np.ndarray, seed: int, k: int = 5) -> float:
    """best@k of the benchmark's cold start, recomputed from the floor vector.

    ``generate_cold_start_ours`` draws ``k`` molecules from the first
    ``COLD_START_POOL`` lines with ``RandomState(seed).choice(POOL, k,
    replace=False)``; the floor vector is aligned to the same file, so this is
    exactly the best score the arm held after its cold start, and hence an exact
    lower bound for best@B at any B >= k. Used to impute a cell that has no
    finite score inside the budget under --itt.
    """
    if scores.size < COLD_START_POOL:
        raise ValueError(f"floor vector has {scores.size} < {COLD_START_POOL} entries; cannot recompute the cold start")
    idx = np.random.RandomState(seed).choice(COLD_START_POOL, k, replace=False)
    cs = scores[idx]
    if not np.isfinite(cs).all():
        raise ValueError(f"cold start of seed {seed} hits a non-finite floor score at indices {idx.tolist()}")
    return float(cs.max())


def compare_arms(
    results_dir: str | Path,
    tasks: list[str],
    seeds: list[int],
    solvers: list[str],
    budget: int = 30,
    floor_dir: str | Path = DEFAULT_OUTDIR,
    alpha: float = 0.05,
    null: str = "replace",
    null_reps: int = NULL_REPS,
    null_seed: int = 0,
    multiplicity: str = "holm",
    equiv_margin: float | None = None,
    verbose: bool = True,
    itt: bool = False,
    require_commit: str | None = None,
    out_meta: dict | None = None,
) -> list[dict]:
    """Per (task, solver): mean best@budget vs the retrieval floor E[max_budget].

    best@budget = nanmax(all_scores[:budget]) — never the file's best_score.

    Cell inclusion (plan section 4.1):

    * default (per protocol): a cell enters the arm only if ``status ==
      'completed'``, it holds at least ``budget`` scores and at least one of the
      first ``budget`` is finite; anything else is listed under ``excluded``.
    * ``itt=True`` (intention to treat): EVERY requested cell enters the arm.
      best@B = nanmax over the first B entries that exist; a cell with no finite
      score inside the budget is imputed with its exact cold-start best@k,
      k = the file's ``n_initial_points`` (``coldstart_best``, a lower bound since
      best@B >= best@k for k <= B) and listed in ``imputed_seeds``. Imputation
      REFUSES (RuntimeError) when the 10k prefix identity of the floor vector is
      not established, when ``n_initial_points`` is missing or <= 0 (no default is
      assumed), or when ``n_initial_points > budget`` (best@k would then be an
      upper bound). Per row the payload carries ``n_nonfinite_in_budget`` (NaN
      entries among the first B), ``n_missing_in_budget`` (sum over seeds of
      max(0, B - len(all_scores))), ``n_imputed`` and ``status_counts``.

    ``require_commit`` makes every JSON in ``results_dir`` prove it was produced
    at that clean commit (``ProvenanceError`` otherwise). ``out_meta``, if given,
    receives ``n_files_checked``, ``dataset_sha256``, ``floor_n``, ``prefix_ok``.

    The cold-start check (``coldstart_check``) compares the first
    ``n_initial_points`` scores of each used cell with the floor vector at the
    benchmark's own cold-start indices; cells it cannot check are listed in
    ``coldstart_unchecked_seeds`` (with reasons) instead of being skipped, so
    ``coldstart_check == 'ok'`` is a statement about ``coldstart_checked_seeds``.
    Under ``itt=True`` an imputed cell, or one whose first ``n_initial_points``
    entries are not all finite, is COUNTED in ``coldstart_unchecked_seeds`` and
    not checked (plan section 3.1: "imputed cells are counted, not checked");
    such seeds are also listed in ``coldstart_nonfinite_seeds`` for information.
    Per protocol (the default) a non-finite cold-start prefix on a used cell is a
    MISMATCH, as at c33f51d.

    Inference is against the EXACT null (see the module docstring): under H0 the
    per-seed best is the maximum of `budget` uniform draws from the task's score
    vector, whose CDF G is known in closed form, so the null distribution of the
    mean over seeds is obtained by seeded inverse-CDF Monte Carlo. Two one-sided
    p-values (p_above, p_below) are reported, Holm-adjusted within each solver
    across the tasks of this call, plus an exact binomial sign test against the
    null median. Verdict: admissible iff p_above < alpha, below_floor iff
    p_below < alpha, else inconclusive — and `equivalent` only when a TOST
    against `equiv_margin` passes.

    Raises MissingCellsError if any requested (task, seed, solver) is absent.
    """
    from src.analysis.config import POLI_TO_GUACAMOL

    results_dir = Path(results_dir)
    floor_dir = resolve_path(floor_dir)
    tasks = [POLI_TO_GUACAMOL.get(t, t) for t in tasks]
    index = _load_result_dir(results_dir, require_commit=require_commit)

    missing = [(t, s, sv) for t in tasks for s in seeds for sv in solvers if (t, s, sv) not in index]
    if missing:
        have = sorted({k[0] for k in index}), sorted({k[1] for k in index}), sorted({k[2] for k in index})
        raise MissingCellsError(
            f"{len(missing)} requested cell(s) missing from {results_dir} "
            f"(first 10: {missing[:10]}); directory holds tasks={have[0]} seeds={have[1]} solvers={have[2]}"
        )

    dataset_path = floor_dir / "dataset.json"
    dataset_info: dict = {}
    if dataset_path.exists():
        try:
            dataset_info = json.loads(dataset_path.read_text())
        except json.JSONDecodeError:
            dataset_info = {}

    summary_path = floor_dir / "zinc_floor_summary.json"
    prefix_ok = None
    if summary_path.exists():
        prefix_ok = json.loads(summary_path.read_text()).get("dataset", {}).get("zinc10k_prefix_identity")
    elif dataset_info:  # sentinel only (no summary yet): same field, same producer
        prefix_ok = dataset_info.get("zinc10k_prefix_identity")

    if out_meta is not None:
        out_meta.update({
            "n_files_checked": len(index) if require_commit is not None else None,
            "require_commit": require_commit,
            "dataset_sha256": dataset_info.get("sha256"),
            "floor_n": dataset_info.get("n_molecules_used"),
            "dataset_json": str(dataset_path) if dataset_info else None,
            "prefix_ok": prefix_ok,
            "itt": bool(itt),
        })

    rows: list[dict] = []
    null_cache: dict[tuple[str, int], np.ndarray] = {}
    for task in tasks:
        npy = floor_dir / f"scores_{task}.npy"
        if not npy.exists():
            raise FileNotFoundError(f"floor vector {npy} missing — run the score command first")
        scores = np.load(npy)
        finite = scores[np.isfinite(scores)]
        values, cum, n = _unique_cdf(finite)
        G = max_cdf(values, cum, n, budget, null)
        floor = emax_from_cdf(values, G)
        floor_replace = exact_emax(values, cum, n, budget)
        floor_norepl = exact_emax_no_replacement(values, cum, n, budget)
        can_check_cs = bool(prefix_ok) and scores.size >= COLD_START_POOL

        for solver in solvers:
            arm, used_seeds, excluded, cs_diffs = [], [], [], []
            nonfinite_in_budget: dict[str, int] = {}
            missing_in_budget: dict[str, int] = {}
            imputed: dict[str, float] = {}
            status_counts: dict[str, int] = {}
            cs_nonfinite_seeds: list[int] = []
            cs_checked_seeds: list[int] = []
            cs_unchecked: dict[str, str] = {}
            for s in seeds:
                d = index[(task, s, solver)]
                a = np.asarray(d.get("all_scores") or [], dtype=float)
                head = a[:budget]
                sc = _status_class(d.get("status"))
                status_counts[sc] = status_counts.get(sc, 0) + 1
                if itt:
                    n_miss = max(0, budget - int(a.size))
                    if n_miss:
                        missing_in_budget[str(s)] = n_miss
                    if head.size and np.isfinite(head).any():
                        best = float(np.nanmax(head))
                    else:  # nothing finite inside the budget: exact cold-start lower bound
                        cell = f"({task}, seed {s}, {solver}) from {d['_file']}"
                        if not can_check_cs:
                            raise RuntimeError(
                                f"cannot impute {cell}: the 10k prefix identity of the floor vector is not "
                                f"established (zinc10k_prefix_identity={prefix_ok!r}, floor size {scores.size}), so "
                                f"the cold-start lower bound cannot be recomputed from it")
                        k_raw = d.get("n_initial_points")
                        k_imp = int(k_raw) if k_raw is not None else 0
                        if k_imp <= 0:
                            raise RuntimeError(
                                f"cannot impute {cell}: n_initial_points={k_raw!r} is missing or not positive; "
                                f"no default is assumed")
                        if k_imp > budget:
                            raise RuntimeError(
                                f"cannot impute {cell}: n_initial_points={k_imp} > budget={budget}; best@{k_imp} "
                                f"would be an upper bound for best@{budget}, not a lower bound")
                        best = coldstart_best(scores, s, k=k_imp)
                        imputed[str(s)] = best
                    nf = int((~np.isfinite(head)).sum())
                else:
                    usable = d.get("status") == "completed" and a.size >= budget and np.isfinite(head).any()
                    if not usable:
                        excluded.append({"seed": s, "status": d.get("status"), "n_scores": int(a.size),
                                         "n_finite_in_budget": int(np.isfinite(head).sum()) if a.size else 0,
                                         "file": d["_file"]})
                        continue
                    best = float(np.nanmax(head))
                    nf = int((~np.isfinite(head)).sum())
                arm.append(best)
                if nf:
                    nonfinite_in_budget[str(s)] = nf
                used_seeds.append(s)
                k = int(d.get("n_initial_points") or 0)
                if can_check_cs:
                    if d.get("mode") != "ours":
                        cs_unchecked[str(s)] = f"mode={d.get('mode')!r} != 'ours'"
                    elif k <= 0:
                        cs_unchecked[str(s)] = f"n_initial_points={d.get('n_initial_points')!r}"
                    elif a.size < k:
                        cs_unchecked[str(s)] = f"n_scores={a.size} < n_initial_points={k}"
                    elif itt and str(s) in imputed:
                        # plan 3.1: imputed cells are counted, not checked
                        cs_unchecked[str(s)] = "imputed: no finite score in budget (counted, not checked)"
                        cs_nonfinite_seeds.append(s)   # informational under --itt; never a MISMATCH
                    elif itt and not np.isfinite(a[:k]).all():
                        cs_unchecked[str(s)] = f"non-finite in cold-start prefix (first {k}; counted, not checked)"
                        cs_nonfinite_seeds.append(s)   # informational under --itt; never a MISMATCH
                    else:
                        cs_idx = np.random.RandomState(s).choice(COLD_START_POOL, k, replace=False)
                        cs_checked_seeds.append(s)
                        if not np.isfinite(a[:k]).all():
                            cs_nonfinite_seeds.append(s)   # a NaN would silently pass a nanmax check
                        else:
                            cs_diffs.append(float(np.max(np.abs(a[:k] - scores[cs_idx]))))
            arm_a = np.asarray(arm, dtype=float)
            n_used = arm_a.size
            if n_used:
                mean_best = float(arm_a.mean())
                diff_floor = mean_best - floor
                key = (task, n_used)
                if key not in null_cache:  # the null draw is arm-independent: reuse across solvers
                    null_cache[key] = null_mean_draws(values, G, n_used, reps=null_reps, seed=null_seed)
                nt = null_mean_test(null_cache[key], mean_best, n_used, null_seed)
                p_above, p_below = nt["p_above"], nt["p_below"]
                sign = null_sign_test(values, G, arm_a)
                tost = tost_equivalence(arm_a, floor, equiv_margin) if equiv_margin else None
            else:
                mean_best = diff_floor = float("nan")
                p_above = p_below = float("nan")
                nt = {"reps": null_reps, "seed": null_seed, "p_above": float("nan"),
                      "p_below": float("nan"), "null_mean": float("nan"), "null_sd": float("nan"),
                      "null_q025": float("nan"), "null_q975": float("nan"), "p_mc_se": float("nan")}
                sign = None
                tost = None
            cs_max = float(max(cs_diffs)) if cs_diffs else None
            if cs_nonfinite_seeds and not itt:   # per protocol only; under --itt these seeds are unchecked
                cs_check = "MISMATCH (non-finite cold-start scores)"
            elif cs_max is None:
                cs_check = None
            else:
                cs_check = "ok" if cs_max <= 1e-9 else "MISMATCH"
            rows.append({
                "task": task, "solver": solver, "budget": budget,
                "analysis_set": "itt" if itt else "per_protocol",
                "n_seeds_requested": len(seeds), "n_used": n_used, "n_excluded": len(excluded),
                "excluded": excluded, "seeds_used": used_seeds,
                "status_counts": status_counts,
                "n_missing_in_budget": int(sum(missing_in_budget.values())),
                "missing_in_budget_per_seed": missing_in_budget,
                "n_imputed": len(imputed), "imputed_seeds": sorted(int(s) for s in imputed),
                "imputed_best_per_seed": imputed,
                "coldstart_checked_seeds": cs_checked_seeds,
                "coldstart_unchecked_seeds": sorted(int(s) for s in cs_unchecked),
                "coldstart_unchecked_reasons": cs_unchecked,
                "coldstart_checkable": bool(can_check_cs),
                "mean_best_at_budget": mean_best,
                "floor_emax": floor, "null_scheme": null,
                "floor_emax_replace": floor_replace, "floor_emax_norepl": floor_norepl,
                "diff_vs_floor": diff_floor,
                "p_above_raw": p_above, "p_below_raw": p_below,
                "p_above": p_above, "p_below": p_below,  # replaced below if multiplicity == holm
                "multiplicity": multiplicity,
                "null_mc": {k: v for k, v in nt.items() if not k.startswith("_")},
                "sign_test": sign,
                "tost": tost,
                "alpha": alpha, "verdict": None,
                "n_nonfinite_in_budget": int(sum(nonfinite_in_budget.values())),
                "nonfinite_in_budget_per_seed": nonfinite_in_budget,
                "coldstart_max_abs_diff": cs_max,
                "coldstart_nonfinite_seeds": cs_nonfinite_seeds,
                "coldstart_check": cs_check,
                "floor_n": int(n), "floor_npy": str(npy),
                "best_per_seed": dict(zip(map(str, used_seeds), arm)),
            })

    # Multiplicity: Holm within each solver family, across the tasks of this call.
    if multiplicity == "holm":
        for solver in solvers:
            fam = [r for r in rows if r["solver"] == solver]
            for field in ("p_above", "p_below"):
                adj = holm([r[f"{field}_raw"] for r in fam])
                for r, a in zip(fam, adj):
                    r[field] = a
                    r["family_size"] = len(fam)
    elif multiplicity != "none":
        raise ValueError(f"unknown multiplicity {multiplicity!r} (holm|none)")

    for r in rows:
        if r["n_used"] == 0:
            r["verdict"] = "no_usable_cells"
        elif r["p_above"] < r["alpha"]:
            r["verdict"] = "admissible"
        elif r["p_below"] < r["alpha"]:
            r["verdict"] = "below_floor"
        elif r["tost"] and r["tost"]["p_tost"] is not None and r["tost"]["p_tost"] < r["alpha"]:
            r["tost"]["equivalent"] = True
            r["verdict"] = "equivalent"
        else:
            if r["tost"] and r["tost"]["p_tost"] is not None:
                r["tost"]["equivalent"] = False
            r["verdict"] = "inconclusive"

    if verbose:
        small = [r for r in rows if 0 < r["n_used"] < 5]
        if small:
            print(f"note: {len(small)} row(s) use fewer than 5 seeds; the exact-null test is still "
                  f"calibrated, but power is minimal and 'inconclusive' is the expected verdict "
                  f"unless the observed best is extreme under H0.")
        print(compare_table(rows))
    return rows


def compare_table(rows: list[dict]) -> str:
    mult = rows[0]["multiplicity"] if rows else "holm"
    fam = rows[0].get("family_size") if rows else None
    head = (
        f"H0: best@B is the max of B molecules drawn uniformly from the task's ZINC score vector "
        f"(scheme `{rows[0]['null_scheme'] if rows else 'replace'}`); p_above/p_below are the two "
        f"one-sided p-values from seeded Monte Carlo of that exact null "
        f"({rows[0]['null_mc']['reps'] if rows else '?'} reps), sign p from the exact binomial test "
        f"against the null median. "
        + (f"p-values are Holm-adjusted within each solver across the {fam} task(s) of this call — "
           f"the adjustment therefore depends on which tasks were requested."
           if mult == "holm" else
           "p-values are UNCORRECTED per (task, solver): with 21 tasks x 4 solvers that is 84 one-sided "
           "tests and ~4 spurious 'admissible' labels are expected at alpha=0.05.")
    )
    aset = rows[0].get("analysis_set", "per_protocol") if rows else "per_protocol"
    head += (" Analysis set: INTENTION TO TREAT — every requested cell is used; a cell with no finite "
             "score inside the budget is imputed with its exact cold-start best@5 (a lower bound)."
             if aset == "itt" else
             " Analysis set: PER PROTOCOL — only completed cells with >= B scores are used; the rest are excluded.")
    hdr = ("| task | solver | B | n used / n req (excl.) | status | missing in B | imputed | "
           "mean best@B | E[max_B] (n) | diff | "
           "p_above | p_below | sign p_above | non-finite in B | verdict | cold-start check | cs unchecked |")
    sep = "|---|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|"
    out = [head, "", hdr, sep]
    for r in rows:
        def pf(x) -> str:
            return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4g}"
        cs = r["coldstart_check"]
        cs_s = ("n/a" if cs is None else
                (f"{cs} (max|Δ|={r['coldstart_max_abs_diff']:.1e}, {len(r.get('coldstart_checked_seeds', []))} seeds)"
                 if r["coldstart_max_abs_diff"] is not None else cs))
        unc = r.get("coldstart_unchecked_seeds", [])
        unc_s = "—" if not unc else f"{len(unc)}: {unc[:8]}{'…' if len(unc) > 8 else ''}"
        st = r.get("status_counts", {})
        st_s = " ".join(f"{k}={v}" for k, v in sorted(st.items())) or "—"
        sign_p = r["sign_test"]["p_above"] if r["sign_test"] else None
        mb = "—" if math.isnan(r["mean_best_at_budget"]) else f"{r['mean_best_at_budget']:.4f}"
        df = "—" if math.isnan(r["diff_vs_floor"]) else f"{r['diff_vs_floor']:+.4f}"
        out.append(
            f"| {r['task']} | {r['solver']} | {r['budget']} | {r['n_used']} / {r['n_seeds_requested']} ({r['n_excluded']}) | "
            f"{st_s} | {r.get('n_missing_in_budget', 0)} | {r.get('n_imputed', 0)} | "
            f"{mb} | {r['floor_emax']:.4f} (n={r['floor_n']}) | {df} | "
            f"{pf(r['p_above'])} | {pf(r['p_below'])} | {pf(sign_p)} | {r['n_nonfinite_in_budget']} | "
            f"{r['verdict']} | {cs_s} | {unc_s} |"
        )
    return "\n".join(out)


def run_compare(args: argparse.Namespace) -> int:
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    meta: dict = {}
    rows = compare_arms(args.results_dir, tasks, seeds, solvers, budget=args.budget,
                        floor_dir=args.floor_dir, alpha=args.alpha, null=args.null,
                        null_reps=args.null_reps, null_seed=args.null_seed,
                        multiplicity=args.multiplicity, equiv_margin=args.equiv_margin,
                        itt=args.itt, require_commit=args.require_commit, out_meta=meta)
    if args.require_commit:
        print(f"--require-commit {args.require_commit}: {meta['n_files_checked']} result file(s) verified clean")
    if args.out:
        out = resolve_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        prov = collect_provenance()
        payload = {"results_dir": str(args.results_dir), "floor_dir": str(resolve_path(args.floor_dir)),
                   "budget": args.budget, "alpha": args.alpha, "null_scheme": args.null,
                   "null_reps": args.null_reps, "null_seed": args.null_seed,
                   "multiplicity": args.multiplicity, "equiv_margin": args.equiv_margin,
                   "analysis_set": "itt" if args.itt else "per_protocol", "itt": bool(args.itt),
                   "require_commit": args.require_commit, "n_files_checked": meta.get("n_files_checked"),
                   "dataset_sha256": meta.get("dataset_sha256"), "floor_n": meta.get("floor_n"),
                   "dataset_json": meta.get("dataset_json"), "zinc10k_prefix_identity": meta.get("prefix_ok"),
                   "code_git_commit": prov["git_commit"], "code_git_commit_full": prov["git_commit_full"],
                   "code_git_dirty": prov["git_dirty"], "argv": list(sys.argv),
                   "tasks": tasks, "seeds": seeds,
                   "solvers": solvers, "provenance": prov, "rows": rows}
        _atomic_write_text(out.with_suffix(".json"), json.dumps(payload, indent=2))
        _atomic_write_text(out.with_suffix(".md"),
            f"# compare_arms — {args.results_dir} vs floor {resolve_path(args.floor_dir)}, "
            f"budget {args.budget}, {payload['analysis_set']}"
            + (f", results at commit {args.require_commit} ({meta.get('n_files_checked')} files checked)"
               if args.require_commit else "")
            + f"; dataset sha256 {str(meta.get('dataset_sha256'))[:16]} n={meta.get('floor_n')}; "
            f"code {prov['git_commit']}{' (dirty)' if prov['git_dirty'] else ''}\n\n"
            + compare_table(rows) + "\n")
        print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="zinc_retrieval_floor.py",
        description=__doc__.split("\n\n")[0] + "  (subcommands: score, compare; 'score' is assumed "
        "when the first argument is an option)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score the ZINC set on each task and compute the floors")
    s.add_argument("--tasks", default="all", help="comma list of task ids (default: all 21)")
    s.add_argument("--n-molecules", type=int, default=None, help="score only the first n SMILES (smoke)")
    s.add_argument("--workers", type=int, default=32, help="spawn-context worker processes (default 32)")
    s.add_argument("--outdir", default=DEFAULT_OUTDIR,
                   help=f"output directory, relative paths anchored at the repo root (default {DEFAULT_OUTDIR})")
    s.add_argument("--seed", type=int, default=0, help="seed of the Monte-Carlo check (default 0)")
    s.add_argument("--chunk-size", type=int, default=500,
                   help="molecules per work item (auto-capped so each worker sees >=4 chunks)")
    s.add_argument("--mc-draws", type=int, default=MC_DRAWS, help=f"Monte-Carlo draws of 30 (default {MC_DRAWS})")
    s.add_argument("--force", action="store_true",
                   help="rescore every task and overwrite a summary written for another dataset")
    s.add_argument("--smiles", default=DEFAULT_SMILES,
                   help=f"SMILES file, one per line, repo-root-relative (default {DEFAULT_SMILES})")
    s.add_argument("--zinc10k", default=DEFAULT_ZINC10K, help="cold-start pool file, checked to be the 10k prefix")
    s.add_argument("--nice", type=int, default=10, help="os.nice increment applied to the pool (0 = off)")
    s.add_argument("--item-timeout", type=float, default=600.0,
                   help="fail if no work item completes for this many seconds (dead worker; default 600)")
    s.add_argument("--allow-outside-repo", action="store_true",
                   help="permit an --outdir that resolves outside the repository")
    s.set_defaults(func=run_score)

    c = sub.add_parser("compare", help="compare benchmark arms against the floor (raises on missing cells)")
    c.add_argument("--results-dir", required=True, help="directory of benchmark result JSONs")
    c.add_argument("--tasks", required=True, help="comma list of task ids (short ids or poli names)")
    c.add_argument("--seeds", required=True, help="comma list of seeds; every (task, seed, solver) must exist")
    c.add_argument("--solvers", required=True, help="comma list of solver names as stamped in the JSONs")
    c.add_argument("--budget", type=int, default=30, help="best@budget = max(all_scores[:budget]) (default 30)")
    c.add_argument("--floor-dir", default=DEFAULT_OUTDIR, help="outdir of the score command")
    c.add_argument("--alpha", type=float, default=0.05)
    c.add_argument("--null", choices=("replace", "norepl"), default="replace",
                   help="H0 sampling scheme: B i.i.d. draws with replacement (default) or B distinct molecules")
    c.add_argument("--null-reps", type=int, default=NULL_REPS,
                   help=f"Monte-Carlo replications of the exact null (default {NULL_REPS})")
    c.add_argument("--null-seed", type=int, default=0, help="seed of the null Monte Carlo (default 0)")
    c.add_argument("--multiplicity", choices=("holm", "none"), default="holm",
                   help="Holm-adjust p within each solver across the tasks of this call (default), or none")
    c.add_argument("--equiv-margin", type=float, default=None,
                   help="equivalence margin on the score scale; only with it can a row be labelled "
                        "'equivalent' (TOST). Without it, a non-rejection is 'inconclusive', never a tie")
    c.add_argument("--itt", action="store_true",
                   help="intention to treat: use EVERY requested cell (crashed/short/all-NaN included); a cell "
                        "with no finite score inside the budget is imputed with its exact cold-start best@5 "
                        "(lower bound) and counted under n_imputed. Default: per protocol (completed cells "
                        "with >= B scores only; the rest are listed as excluded)")
    c.add_argument("--require-commit", default=None, metavar="HASH",
                   help="assert every result JSON in --results-dir carries provenance git_commit == HASH "
                        "and git_dirty == false; abort listing the offending files otherwise")
    c.add_argument("--out", default=None, help="write <out>.json and <out>.md")
    c.set_defaults(func=run_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("score", "compare", "-h", "--help"):
        argv.insert(0, "score")
    if not argv:
        argv = ["score"]
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
