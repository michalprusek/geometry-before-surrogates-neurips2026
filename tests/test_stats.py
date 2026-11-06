"""Tests for statistical functions.

Tests Cohen's d, Benjamini-Hochberg FDR correction, paired Wilcoxon,
cross-task comparison, and significance formatting. All CPU-only.
"""

import math

import numpy as np
import pytest

from src.analysis.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    cohens_d,
    cross_task_comparison,
    format_significance,
    friedman_test,
    paired_wilcoxon,
    tost_equivalence,
)
from src.analysis.config import (
    ABLATION_SEEDS,
    BENCHMARK_SEEDS,
    BENCHMARK_TASKS,
    SEED_REGEX,
    SMALL_N_COLD_START,
    SMALL_N_ITERATIONS,
    SMALL_N_EVALUATED,
    extract_seed,
)


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------


class TestCohensD:
    """Tests for paired Cohen's d = mean(x-y) / std(x-y, ddof=1)."""

    def test_identical_arrays(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert cohens_d(x, y) == 0.0

    def test_known_value(self):
        # x - y = [1, 1, 1, 1], mean=1, std=0 -> returns 0 (guarded)
        x = [2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0]
        # diff = [1, 1, 1, 1], sd = 0 -> returns 0.0
        assert cohens_d(x, y) == 0.0

    def test_known_nonzero_value(self):
        x = [10.0, 20.0, 30.0]
        y = [8.0, 22.0, 28.0]
        # diff = [2, -2, 2], mean = 2/3, std(ddof=1) = sqrt(16/3)/sqrt(3) = 2*sqrt(3)/3... let me compute
        diff = np.array([2.0, -2.0, 2.0])
        expected = float(np.mean(diff) / np.std(diff, ddof=1))
        result = cohens_d(x, y)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_symmetric_with_sign_flip(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 1.0, 4.0, 3.0, 6.0]
        d_xy = cohens_d(x, y)
        d_yx = cohens_d(y, x)
        np.testing.assert_allclose(d_xy, -d_yx, atol=1e-10)

    def test_large_effect(self):
        """Large constant difference -> large d (but guarded by zero std)."""
        x = [100.0, 100.0, 100.0]
        y = [0.0, 0.0, 0.0]
        # diff = [100, 100, 100], std = 0 -> guarded, returns 0.0
        assert cohens_d(x, y) == 0.0

    def test_numpy_input(self):
        x = np.array([1.0, 3.0, 5.0])
        y = np.array([2.0, 2.0, 4.0])
        result = cohens_d(x, y)
        assert isinstance(result, float)

    def test_varied_differences(self):
        """Non-constant differences give non-zero d."""
        x = [5.0, 10.0, 15.0, 20.0, 25.0]
        y = [4.0, 8.0, 16.0, 18.0, 26.0]
        d = cohens_d(x, y)
        assert d != 0.0
        # diff = [1, 2, -1, 2, -1], mean=0.6, std>0
        diff = np.array(x) - np.array(y)
        expected = float(np.mean(diff) / np.std(diff, ddof=1))
        np.testing.assert_allclose(d, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------


class TestBenjaminiHochberg:
    """Tests for BH FDR correction."""

    def test_all_significant(self):
        p_values = [0.001, 0.002, 0.003]
        results = benjamini_hochberg(p_values, alpha=0.05)
        # All should be significant
        for orig_idx, raw_p, adj_p, sig in results:
            assert sig, f"p={raw_p} should be significant after BH"

    def test_none_significant(self):
        p_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        results = benjamini_hochberg(p_values, alpha=0.05)
        for orig_idx, raw_p, adj_p, sig in results:
            assert not sig, f"p={raw_p} should NOT be significant after BH"

    def test_adjusted_p_monotonic(self):
        """Adjusted p-values should be monotonically non-decreasing when sorted by raw p."""
        p_values = [0.01, 0.04, 0.03, 0.002, 0.5]
        results = benjamini_hochberg(p_values, alpha=0.05)
        # Sort by raw_p
        by_raw = sorted(results, key=lambda x: x[1])
        adj_ps = [r[2] for r in by_raw]
        for i in range(len(adj_ps) - 1):
            assert adj_ps[i] <= adj_ps[i + 1] + 1e-12, (
                f"Adjusted p-values not monotonic: {adj_ps}"
            )

    def test_adjusted_p_bounded_by_1(self):
        p_values = [0.5, 0.8, 0.9]
        results = benjamini_hochberg(p_values, alpha=0.05)
        for _, _, adj_p, _ in results:
            assert adj_p <= 1.0 + 1e-12

    def test_preserves_original_order(self):
        p_values = [0.05, 0.01, 0.10]
        results = benjamini_hochberg(p_values, alpha=0.05)
        # Results should be sorted by original index
        indices = [r[0] for r in results]
        assert indices == [0, 1, 2]

    def test_single_p_value(self):
        results = benjamini_hochberg([0.03], alpha=0.05)
        assert len(results) == 1
        orig_idx, raw_p, adj_p, sig = results[0]
        assert orig_idx == 0
        assert raw_p == 0.03
        assert adj_p == 0.03  # m=1, rank=1 -> adj = raw
        assert sig is True

    def test_known_correction(self):
        """Manual BH on 3 p-values."""
        # p = [0.01, 0.04, 0.03], sorted: [0.01, 0.03, 0.04]
        # ranks: 1, 2, 3; m=3
        # adj: 0.01*3/1=0.03, 0.03*3/2=0.045, 0.04*3/3=0.04
        # reverse cummin: min(0.03, 0.04) = 0.03 (idx 0), min(0.045, 0.04) = 0.04 (idx 1), 0.04 (idx 2)
        # Wait, reverse cummin from bottom: start at idx 2: 0.04, then idx 1: min(0.045, 0.04)=0.04, idx 0: min(0.03, 0.04)=0.03
        p_values = [0.01, 0.04, 0.03]
        results = benjamini_hochberg(p_values, alpha=0.05)
        # Original indices: 0->0.01, 1->0.04, 2->0.03
        # Sorted: (0, 0.01), (2, 0.03), (1, 0.04)
        # adj: 0.01*3/1=0.03, 0.03*3/2=0.045, 0.04*3/3=0.04
        # rev cummin: 0.04, min(0.045,0.04)=0.04, min(0.03,0.04)=0.03
        # So: (0, 0.01, 0.03, True), (2, 0.03, 0.04, True), (1, 0.04, 0.04, True)
        # Re-sorted by original index: (0, 0.01, 0.03, True), (1, 0.04, 0.04, True), (2, 0.03, 0.04, True)
        assert results[0] == (0, 0.01, 0.03, True)
        assert results[1] == (1, 0.04, 0.04, True)
        assert results[2] == (2, 0.03, 0.04, True)


# ---------------------------------------------------------------------------
# Paired Wilcoxon
# ---------------------------------------------------------------------------


class TestPairedWilcoxon:
    """Tests for paired_wilcoxon."""

    def test_returns_correct_keys(self):
        baseline = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        method = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
        result = paired_wilcoxon(baseline, method)
        assert "mean_diff" in result
        assert "wins" in result
        assert "losses" in result
        assert "ties" in result
        assert "n_pairs" in result
        assert "statistic" in result
        assert "p_value" in result

    def test_identical_inputs(self):
        x = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        result = paired_wilcoxon(x, x)
        assert result["mean_diff"] == 0.0
        assert result["wins"] == 0
        assert result["losses"] == 0
        assert result["ties"] == 6
        assert math.isnan(result["p_value"])  # all diffs zero

    def test_method_clearly_better(self):
        baseline = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        method = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        result = paired_wilcoxon(baseline, method)
        assert result["mean_diff"] > 0
        assert result["wins"] == 6
        assert result["losses"] == 0
        assert result["p_value"] < 0.05

    def test_n_pairs(self):
        baseline = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        method = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
        result = paired_wilcoxon(baseline, method)
        assert result["n_pairs"] == 8

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="Mismatched"):
            paired_wilcoxon([1, 2, 3], [1, 2])

    def test_too_few_pairs_returns_nan(self):
        result = paired_wilcoxon([0.5, 0.6], [0.55, 0.65])
        assert math.isnan(result["p_value"])
        assert result["n_pairs"] == 2


# ---------------------------------------------------------------------------
# Cross-task comparison
# ---------------------------------------------------------------------------


class TestCrossTaskComparison:
    """Tests for cross_task_comparison."""

    def test_shared_tasks(self):
        baseline = {"adip": 0.5, "dhop": 0.4, "fexo": 0.3, "med1": 0.6, "med2": 0.5, "osmb": 0.4}
        method = {"adip": 0.6, "dhop": 0.5, "fexo": 0.4, "med1": 0.7, "med2": 0.6, "osmb": 0.5}
        result = cross_task_comparison(baseline, method)
        assert result["n_pairs"] == 6
        assert result["mean_diff"] > 0

    def test_disjoint_tasks(self):
        baseline = {"adip": 0.5}
        method = {"dhop": 0.6}
        result = cross_task_comparison(baseline, method)
        assert result["n_pairs"] == 0
        assert math.isnan(result["p_value"])

    def test_explicit_task_list(self):
        baseline = {"adip": 0.5, "dhop": 0.4, "fexo": 0.3, "med1": 0.6, "med2": 0.5, "osmb": 0.4}
        method = {"adip": 0.6, "dhop": 0.5, "fexo": 0.4, "med1": 0.7, "med2": 0.6, "osmb": 0.5}
        result = cross_task_comparison(baseline, method, tasks=["adip", "dhop"])
        assert result["n_pairs"] == 2


# ---------------------------------------------------------------------------
# Format significance
# ---------------------------------------------------------------------------


class TestFormatSignificance:
    """Tests for p-value formatting with stars."""

    def test_highly_significant(self):
        s = format_significance(0.0005)
        assert "***" in s
        assert "0.001" in s or "0.000" in s

    def test_significant_two_stars(self):
        s = format_significance(0.008)
        assert "**" in s
        assert "***" not in s

    def test_marginally_significant(self):
        s = format_significance(0.04)
        assert "*" in s
        assert "**" not in s

    def test_not_significant(self):
        s = format_significance(0.15)
        assert "*" not in s
        assert "0.150" in s

    def test_nan(self):
        assert format_significance(float("nan")) == "N/A"


# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------


class TestConfig:
    """Tests for benchmark configuration constants."""

    def test_12_tasks(self):
        assert len(BENCHMARK_TASKS) == 12

    def test_40_seeds(self):
        # Main paper uses 40 seeds (42-81); ABLATION_SEEDS uses 10 (42-51)
        assert len(BENCHMARK_SEEDS) == 40
        assert BENCHMARK_SEEDS[0] == 42
        assert BENCHMARK_SEEDS[-1] == 81

    def test_ablation_seeds(self):
        assert len(ABLATION_SEEDS) == 10
        assert ABLATION_SEEDS[0] == 42
        assert ABLATION_SEEDS[-1] == 51

    def test_budget_arithmetic(self):
        assert SMALL_N_EVALUATED == SMALL_N_COLD_START + SMALL_N_ITERATIONS

    def test_extract_seed_from_filename(self):
        assert extract_seed("ours_cma_es_subspace_amlodipine_mpo_s42_16d.json") == 42
        assert extract_seed("hdbo_cma_es_deco_hop_s51_128d.json") == 51

    def test_extract_seed_no_match(self):
        assert extract_seed("no_seed_here.json") is None

    def test_seed_regex_pattern(self):
        m = SEED_REGEX.search("method_task_s99_dim.json")
        assert m is not None
        assert m.group(1) == "99"

    def test_all_task_codes_are_4_chars(self):
        for task in BENCHMARK_TASKS:
            assert len(task) == 4, f"Task code {task!r} is not 4 characters"


# ---------------------------------------------------------------------------
# Friedman test (load-bearing for Table 1 significance in paper)
# ---------------------------------------------------------------------------


class TestFriedmanTest:
    """Friedman non-parametric test — checked against hand-derived cases."""

    def test_matches_scipy_friedman(self):
        # Cross-check our wrapper against scipy.stats.friedmanchisquare directly.
        from scipy.stats import friedmanchisquare
        rng = np.random.default_rng(0)
        data = rng.standard_normal((10, 4))
        expected_stat, expected_p = friedmanchisquare(*[data[:, j] for j in range(4)])
        result = friedman_test(data, higher_is_better=True)
        assert result["statistic"] == pytest.approx(expected_stat, rel=1e-10)
        assert result["p_value"] == pytest.approx(expected_p, rel=1e-10)

    def test_clear_winner_rank_1(self):
        # Method 0 always best (highest). With higher_is_better=True, mean_rank≈1.
        rng = np.random.default_rng(42)
        n_blocks, n_methods = 10, 3
        data = rng.standard_normal((n_blocks, n_methods))
        data[:, 0] += 5.0  # method 0 consistently highest
        result = friedman_test(data, higher_is_better=True)
        assert result["mean_ranks"][0] == pytest.approx(1.0, abs=0.1)
        assert result["mean_ranks"][0] < result["mean_ranks"][1]
        assert result["mean_ranks"][0] < result["mean_ranks"][2]

    def test_higher_is_better_inverts_ranks(self):
        # Same data, two calls: higher_is_better=True vs False should swap.
        rng = np.random.default_rng(7)
        data = rng.standard_normal((8, 4))
        res_high = friedman_test(data, higher_is_better=True)
        res_low = friedman_test(data, higher_is_better=False)
        # Statistic is invariant to direction; only ranks flip.
        assert res_high["statistic"] == pytest.approx(res_low["statistic"])
        for i in range(4):
            # rank + rank_inv = n_methods+1 = 5
            assert res_high["mean_ranks"][i] + res_low["mean_ranks"][i] == pytest.approx(5.0, abs=0.01)

    def test_degenerate_input_returns_nan(self, caplog):
        import logging
        # 1 block × 3 methods → friedman needs ≥2 blocks
        data = np.array([[1.0, 2.0, 3.0]])
        with caplog.at_level(logging.ERROR, logger="src.analysis.stats"):
            result = friedman_test(data, higher_is_better=True)
        assert math.isnan(result["statistic"])
        assert math.isnan(result["p_value"])
        assert any("degenerate" in rec.message.lower() for rec in caplog.records)

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D array"):
            friedman_test(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# TOST equivalence (load-bearing for paper's "GeoWalk ≡ top 8" claim)
# ---------------------------------------------------------------------------


class TestTostEquivalence:
    """Two One-Sided Tests for equivalence — paper-critical correctness."""

    def test_identical_distributions_equivalent(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.5, 0.01, 50)
        y = rng.normal(0.5, 0.01, 50)  # essentially identical
        result = tost_equivalence(x, y, delta=0.02, alpha=0.05)
        assert result["equivalent"] is True
        assert abs(result["mean_diff"]) < 0.02

    def test_large_difference_not_equivalent(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.5, 0.01, 50)
        y = rng.normal(0.3, 0.01, 50)  # |diff|=0.2 >> delta=0.02
        result = tost_equivalence(x, y, delta=0.02, alpha=0.05)
        assert result["equivalent"] is False
        assert result["mean_diff"] == pytest.approx(0.2, abs=0.01)

    def test_sign_flip_invariant(self):
        # tost(x, y) should report same equivalent-bool as tost(y, x).
        rng = np.random.default_rng(1)
        x = rng.normal(0.5, 0.02, 30)
        y = rng.normal(0.5 + 0.005, 0.02, 30)  # tiny offset, within margin
        r_xy = tost_equivalence(x, y, delta=0.02)
        r_yx = tost_equivalence(y, x, delta=0.02)
        assert r_xy["equivalent"] == r_yx["equivalent"]

    def test_n_less_than_2_returns_nan(self):
        result = tost_equivalence([1.0], [1.0], delta=0.02)
        assert math.isnan(result["mean_diff"])
        assert result["equivalent"] is False

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError, match="Mismatched shapes"):
            tost_equivalence([1.0, 2.0], [1.0], delta=0.02)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_covers_true_mean_at_95pct(self):
        # With 50 samples from N(mu, 1), 95% CI should contain mu ~95% of the time.
        rng = np.random.default_rng(0)
        true_mu = 0.5
        n_trials = 200
        n_covered = 0
        for _ in range(n_trials):
            scores = rng.normal(true_mu, 1.0, 50)
            ci = bootstrap_ci(scores, n_resamples=2000, alpha=0.05)
            if ci["lower"] <= true_mu <= ci["upper"]:
                n_covered += 1
        # Expect ~95%, allow ±4% slack for n_trials=200
        coverage = n_covered / n_trials
        assert 0.91 <= coverage <= 0.99, f"coverage {coverage} outside [0.91, 0.99]"

    def test_ci_shrinks_with_larger_n(self):
        rng = np.random.default_rng(0)
        small = bootstrap_ci(rng.normal(0.5, 1.0, 10), n_resamples=2000)
        rng = np.random.default_rng(0)
        large = bootstrap_ci(rng.normal(0.5, 1.0, 200), n_resamples=2000)
        assert (large["upper"] - large["lower"]) < (small["upper"] - small["lower"])

    def test_n_less_than_2_returns_nan(self):
        ci = bootstrap_ci([1.0], n_resamples=100)
        assert math.isnan(ci["std_err"])
        assert math.isnan(ci["lower"])

    def test_deterministic_with_seed(self):
        scores = np.random.default_rng(0).normal(0.5, 1.0, 50)
        ci1 = bootstrap_ci(scores, n_resamples=1000, rng_seed=42)
        ci2 = bootstrap_ci(scores, n_resamples=1000, rng_seed=42)
        assert ci1["lower"] == ci2["lower"]
        assert ci1["upper"] == ci2["upper"]
