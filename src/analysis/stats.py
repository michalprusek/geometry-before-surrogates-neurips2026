"""Standardized statistical testing utilities.

Provides Wilcoxon signed-rank tests, Benjamini-Hochberg FDR correction,
Cohen's d, Friedman+Nemenyi, TOST equivalence and bootstrap CI.
Prefer this module for all statistical tests to ensure consistent
methodology across analysis scripts.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def paired_wilcoxon(
    baseline_means: list[float] | np.ndarray,
    method_means: list[float] | np.ndarray,
    alternative: str = "two-sided",
) -> dict:
    """Paired Wilcoxon signed-rank test on task-level means.

    Requires at least 6 paired observations (tasks).

    Args:
        baseline_means: Per-task mean scores for baseline method.
        method_means: Per-task mean scores for comparison method.
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        Dict with keys: statistic, p_value, mean_diff, wins, losses, ties, n_pairs.
    """
    baseline_means = np.asarray(baseline_means, dtype=float)
    method_means = np.asarray(method_means, dtype=float)

    if len(baseline_means) != len(method_means):
        raise ValueError(
            f"Mismatched lengths: {len(baseline_means)} vs {len(method_means)}"
        )

    diff = method_means - baseline_means
    n = len(diff)
    wins = int(np.sum(diff > 0))
    losses = int(np.sum(diff < 0))
    ties = int(np.sum(diff == 0))
    mean_diff = float(np.mean(diff))

    result = {
        "mean_diff": mean_diff,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n_pairs": n,
    }

    if n >= 6 and not np.allclose(diff, 0):
        try:
            stat, p = stats.wilcoxon(diff, alternative=alternative)
            result["statistic"] = float(stat)
            result["p_value"] = float(p)
        except ValueError as e:
            warnings.warn(
                f"Wilcoxon test failed (n={n}, alternative={alternative!r}): {e}"
            )
            result["statistic"] = float("nan")
            result["p_value"] = float("nan")
    else:
        result["statistic"] = float("nan")
        result["p_value"] = float("nan")

    return result


def cross_task_comparison(
    baseline_results: dict[str, float],
    method_results: dict[str, float],
    tasks: list[str] | None = None,
) -> dict:
    """Compare two methods using paired Wilcoxon on shared task-level means.

    Args:
        baseline_results: Dict mapping task -> mean score for baseline.
        method_results: Dict mapping task -> mean score for method.
        tasks: Optional explicit task list (default: use shared keys).

    Returns:
        Dict with test results including p_value, mean_diff, wins/losses.
    """
    if tasks is None:
        tasks = sorted(set(baseline_results.keys()) & set(method_results.keys()))

    baseline_means = [baseline_results[t] for t in tasks if t in baseline_results and t in method_results]
    method_means = [method_results[t] for t in tasks if t in baseline_results and t in method_results]

    if not baseline_means:
        return {"p_value": float("nan"), "mean_diff": float("nan"), "n_pairs": 0}

    return paired_wilcoxon(baseline_means, method_means)


def format_significance(p_value: float, thresholds: tuple[float, ...] = (0.001, 0.01, 0.05)) -> str:
    """Format p-value with significance stars.

    Args:
        p_value: The p-value to format.
        thresholds: Tuple of thresholds for ***, **, * (default: 0.001, 0.01, 0.05).

    Returns:
        String like "0.008**" or "0.461".
    """
    if np.isnan(p_value):
        return "N/A"

    stars = ""
    for thresh in sorted(thresholds):
        if p_value < thresh:
            stars += "*"

    return f"{p_value:.3f}{stars}"


def benjamini_hochberg(
    p_values: list[float], alpha: float = 0.05
) -> list[tuple[int, float, float, bool]]:
    """Benjamini-Hochberg step-up FDR correction.

    Args:
        p_values: List of raw p-values.
        alpha: False discovery rate level.

    Returns:
        List of (original_index, raw_p, adjusted_p, significant) tuples,
        sorted by original index.
    """
    m = len(p_values)
    indexed = [(i, p) for i, p in enumerate(p_values)]
    # Sort ascending by p-value
    indexed.sort(key=lambda x: x[1])

    # Compute adjusted p-values: p * m / rank, then reverse cumulative min
    adjusted = [0.0] * m
    for rank_0, (orig_idx, raw_p) in enumerate(indexed):
        rank = rank_0 + 1  # 1-based rank
        adjusted[rank_0] = min(raw_p * m / rank, 1.0)

    # Reverse cumulative minimum (enforce monotonicity from bottom)
    for j in range(m - 2, -1, -1):
        adjusted[j] = min(adjusted[j], adjusted[j + 1])

    results = []
    for rank_0, (orig_idx, raw_p) in enumerate(indexed):
        adj_p = adjusted[rank_0]
        results.append((orig_idx, raw_p, adj_p, adj_p < alpha))

    # Sort back to original order
    results.sort(key=lambda x: x[0])
    return results


def cohens_d(x, y) -> float:
    """Paired Cohen's d = mean(x-y) / std(x-y, ddof=1)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    sd = np.std(diff, ddof=1)
    if sd < 1e-15:
        return 0.0
    return float(np.mean(diff) / sd)


# ---------------------------------------------------------------------------
# Friedman + Nemenyi post-hoc (Papenmeier/Hvarfner style primary test)
# ---------------------------------------------------------------------------


def friedman_test(
    data: np.ndarray | list[list[float]],
    higher_is_better: bool = True,
) -> dict:
    """Friedman non-parametric test for k paired samples.

    Args:
        data: (n_blocks, n_methods) array. Each row is one "block" (e.g., task)
            with method scores.
        higher_is_better: If True (default), higher score → better → rank 1.
            If False, lower score → better → rank 1. Ensures mean_ranks has
            consistent semantics across callers.

    Returns:
        Dict with keys: statistic, p_value, n_blocks, n_methods, mean_ranks,
        higher_is_better. Returns NaN for degenerate inputs and logs an error;
        callers should check n_blocks/n_methods before trusting the result.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (n_blocks, n_methods), got shape {arr.shape}")

    n_blocks, n_methods = arr.shape
    if n_methods < 2 or n_blocks < 2:
        logger.error(
            f"friedman_test: degenerate input (n_blocks={n_blocks}, "
            f"n_methods={n_methods}); returning NaN. Callers must not "
            f"propagate this NaN into paper tables without explicit handling."
        )
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_blocks": n_blocks,
            "n_methods": n_methods,
            "mean_ranks": [float("nan")] * n_methods,
            "higher_is_better": higher_is_better,
        }

    # scipy.stats.friedmanchisquare expects separate columns as positional args
    columns = [arr[:, j] for j in range(n_methods)]
    stat, p = stats.friedmanchisquare(*columns)

    # Per-method mean ranks. scipy.stats.rankdata assigns rank 1 to smallest.
    # For "higher is better" we rank on negated scores so that higher score → lower rank.
    rank_input = -arr if higher_is_better else arr
    ranks = np.apply_along_axis(lambda row: stats.rankdata(row), axis=1, arr=rank_input)
    mean_ranks = ranks.mean(axis=0).tolist()

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_blocks": int(n_blocks),
        "n_methods": int(n_methods),
        "mean_ranks": [float(r) for r in mean_ranks],
        "higher_is_better": higher_is_better,
    }


def nemenyi_posthoc(data: np.ndarray | list[list[float]]) -> np.ndarray:
    """Nemenyi post-hoc pairwise test (after Friedman).

    Returns an (n_methods x n_methods) matrix of pairwise p-values.
    Requires scikit-posthocs.

    Args:
        data: (n_blocks, n_methods) array.

    Returns:
        (n_methods, n_methods) ndarray of pairwise p-values; diagonal = 1.
    """
    try:
        import scikit_posthocs as sp
    except ImportError as e:
        raise ImportError(
            "nemenyi_posthoc requires scikit-posthocs: pip install scikit-posthocs"
        ) from e

    arr = np.asarray(data, dtype=float)
    p_matrix = sp.posthoc_nemenyi_friedman(arr)
    return p_matrix.values if hasattr(p_matrix, "values") else np.asarray(p_matrix)


# ---------------------------------------------------------------------------
# TOST equivalence test (pre-register margin delta = 0.02 by default)
# ---------------------------------------------------------------------------


def tost_equivalence(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    delta: float = 0.02,
    alpha: float = 0.05,
) -> dict:
    """Two One-Sided Tests (TOST) for equivalence between paired samples.

    Tests whether |mean(x) - mean(y)| < delta at level alpha. Returns equivalent
    iff both one-sided t-tests reject in favor of equivalence.

    Args:
        x, y: paired samples (e.g., per-seed scores for two methods on one task).
        delta: equivalence margin (pre-registered; 0.02 = ~half reviewer Cohen-d).
        alpha: significance level (default 0.05; each one-sided test uses alpha).

    Returns:
        Dict with keys: mean_diff, p_lower, p_upper, equivalent (bool).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Mismatched shapes: {x.shape} vs {y.shape}")

    diff = x - y
    n = len(diff)
    if n < 2:
        logger.warning(
            f"tost_equivalence: n={n} < 2, returning NaN; 'equivalent'=False "
            f"but this reflects insufficient data, not evidence of non-equivalence."
        )
        return {
            "mean_diff": float("nan"),
            "p_lower": float("nan"),
            "p_upper": float("nan"),
            "equivalent": False,
            "n": n,
        }

    mean_d = float(np.mean(diff))
    # Upper bound test: H0: mean_d >= delta vs H1: mean_d < delta
    t_upper, p_upper = stats.ttest_1samp(diff, delta, alternative="less")
    # Lower bound test: H0: mean_d <= -delta vs H1: mean_d > -delta
    t_lower, p_lower = stats.ttest_1samp(diff, -delta, alternative="greater")

    return {
        "mean_diff": mean_d,
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "equivalent": bool(p_upper < alpha and p_lower < alpha),
        "delta": delta,
        "alpha": alpha,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (per-task 95%)
# ---------------------------------------------------------------------------


def bootstrap_ci(
    scores: list[float] | np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> dict:
    """Bootstrap CI for mean(scores) at confidence level (1 - alpha).

    Args:
        scores: 1D array of per-seed scores for one method+task combo.
        n_resamples: Number of bootstrap resamples.
        alpha: Two-sided confidence = 1 - alpha.
        rng_seed: Seed for reproducibility.

    Returns:
        Dict with keys: mean, std_err, lower, upper, n.
    """
    arr = np.asarray(scores, dtype=float)
    n = len(arr)
    if n < 2:
        logger.warning(
            f"bootstrap_ci: n={n} < 2, returning NaN CI. Downstream tables "
            f"should mark this cell explicitly rather than format NaN brackets."
        )
        return {
            "mean": float(arr.mean()) if n == 1 else float("nan"),
            "std_err": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "n": n,
        }

    rng = np.random.default_rng(rng_seed)
    # Vectorized resampling
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = arr[idx].mean(axis=1)

    lower = float(np.quantile(boot_means, alpha / 2))
    upper = float(np.quantile(boot_means, 1 - alpha / 2))
    mean = float(arr.mean())
    std_err = float(boot_means.std(ddof=1))

    return {
        "mean": mean,
        "std_err": std_err,
        "lower": lower,
        "upper": upper,
        "n": int(n),
        "n_resamples": int(n_resamples),
    }
