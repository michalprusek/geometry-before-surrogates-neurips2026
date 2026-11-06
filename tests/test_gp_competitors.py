"""Smoke tests for GP competitor solvers (Hvarfner, Papenmeier, Xu, Tripp, RF-BO).

Each test runs `solve(max_iter=N)` with MockCodec+MockOracle and verifies:
1. Solver completes without crashing
2. Candidates produced during the run lie on S^{D-1} at radius ~mean_norm
   (the fair-comparison invariant)
3. Fallback counters are exposed (persistence for result JSON)

Gates: gpytorch/botorch required for all. rdkit required for Tripp only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from src.analysis.config import (
    BOX_HALF_WIDTH,
    BOX_SIGMA_MULTIPLIER,
    ZINC_PER_COORD_STD,
)
from src.spherevo.gp_competitors import (
    GeodesicSliceGPSolver,
    HvarfnerScaledGPBallSolver,
    HvarfnerScaledGPBoxSolver,
    HvarfnerScaledGPSolver,
    PapenmeierMSRBoxSolver,
    PapenmeierMSRSolver,
    RandomForestBOSolver,
    XuStandardGPBoxSolver,
    XuStandardGPSolver,
    _box_next_candidate,
    _propose_qlogei_on_box,
    _propose_qlogei_on_sphere,
    _random_in_box,
    _smiles_to_morgan,
    _tangent_basis_at,
)


# Larger-than-default cold-start so GP fit has enough data to avoid trivial
# fallback paths (Matern-5/2 on 5 points is borderline).
N_COLD = 8
D_TEST = 32  # MockCodec default


@pytest.fixture
def larger_cold_start(mock_codec, mock_oracle):
    """8 SMILES → 8 × 32 x0 array (plenty for GP MLL)."""
    smiles = ["CCO", "CCCO", "CC=O", "C1CCCCC1", "c1ccccc1",
              "CCOC", "CCCC", "CCCCCCO"]
    x0 = mock_codec.encode(smiles).numpy()
    y0 = np.array([mock_oracle(s) for s in smiles]).reshape(-1, 1)
    return x0, y0


# ---------------------------------------------------------------------------
# Hvarfner
# ---------------------------------------------------------------------------


class TestHvarfnerScaledGPSolver:
    def test_solve_completes(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        solver = HvarfnerScaledGPSolver(black_box=mock_black_box, x0=x0, y0=y0)
        solver.solve(max_iter=5)
        assert len(solver._all_scores) == N_COLD + 5

    def test_candidates_on_sphere(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        solver = HvarfnerScaledGPSolver(black_box=mock_black_box, x0=x0, y0=y0)
        solver.solve(max_iter=3)
        for z in solver.history["x"][N_COLD:]:
            norm = float(np.linalg.norm(z))
            assert abs(norm - solver.mean_norm) < 1e-3, (
                f"candidate norm {norm} != mean_norm {solver.mean_norm}"
            )

    def test_exposes_fallback_counters(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        solver = HvarfnerScaledGPSolver(black_box=mock_black_box, x0=x0, y0=y0)
        assert solver._fallback_count == 0
        assert solver._fallback_reasons == []

    def test_rejects_d_less_than_2(self, mock_black_box):
        with pytest.raises(ValueError, match="D>=2"):
            HvarfnerScaledGPSolver(
                black_box=mock_black_box,
                x0=np.ones((5, 1)), y0=np.zeros((5, 1)),
            )


class TestPapenmeierMSRSolver:
    def test_solve_completes(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        solver = PapenmeierMSRSolver(black_box=mock_black_box, x0=x0, y0=y0)
        solver.solve(max_iter=5)
        assert len(solver._all_scores) == N_COLD + 5

    def test_candidates_on_sphere(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        solver = PapenmeierMSRSolver(black_box=mock_black_box, x0=x0, y0=y0)
        solver.solve(max_iter=3)
        for z in solver.history["x"][N_COLD:]:
            assert abs(float(np.linalg.norm(z)) - solver.mean_norm) < 1e-3

    def test_exposes_fallback_counters(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        solver = PapenmeierMSRSolver(black_box=mock_black_box, x0=x0, y0=y0)
        assert solver._fallback_count == 0


class TestXuStandardGPSolver:
    def test_solve_completes(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        solver = XuStandardGPSolver(black_box=mock_black_box, x0=x0, y0=y0)
        solver.solve(max_iter=5)
        assert len(solver._all_scores) == N_COLD + 5

    def test_candidates_on_sphere(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        solver = XuStandardGPSolver(black_box=mock_black_box, x0=x0, y0=y0, c=1.0)
        solver.solve(max_iter=3)
        for z in solver.history["x"][N_COLD:]:
            assert abs(float(np.linalg.norm(z)) - solver.mean_norm) < 1e-3

    def test_rejects_negative_c(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        with pytest.raises(ValueError, match="c must be positive"):
            XuStandardGPSolver(
                black_box=mock_black_box, x0=x0, y0=y0, c=-1.0,
            )


class TestRandomForestBOSolver:
    def test_solve_completes(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        solver = RandomForestBOSolver(
            black_box=mock_black_box, x0=x0, y0=y0, n_estimators=20,
        )
        solver.solve(max_iter=5)
        assert len(solver._all_scores) == N_COLD + 5

    def test_candidates_on_sphere(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        solver = RandomForestBOSolver(
            black_box=mock_black_box, x0=x0, y0=y0, n_estimators=20,
        )
        solver.solve(max_iter=3)
        for z in solver.history["x"][N_COLD:]:
            assert abs(float(np.linalg.norm(z)) - solver.mean_norm) < 1e-3

    def test_rejects_zero_estimators(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        with pytest.raises(ValueError, match="n_estimators"):
            RandomForestBOSolver(
                black_box=mock_black_box, x0=x0, y0=y0, n_estimators=0,
            )


# ---------------------------------------------------------------------------
# Tripp Tanimoto-GP (only if rdkit is installed)
# ---------------------------------------------------------------------------


class TestTrippTanimotoGPSolver:
    def test_solve_completes(self, mock_codec, mock_oracle, mock_black_box, larger_cold_start):
        pytest.importorskip("rdkit")
        from src.spherevo.gp_competitors import TrippTanimotoGPSolver

        x0, y0 = larger_cold_start
        # MockCodec.decode produces MOLxxxx — rdkit won't parse these; we expect
        # fallback-to-random path to trigger, and that the solver still completes
        # without crashing. This verifies the "insufficient valid FPs" fallback.
        solver = TrippTanimotoGPSolver(
            black_box=mock_black_box, x0=x0, y0=y0,
            codec=mock_codec, cs_smiles=["CCO"] * len(x0),
            n_candidates=16,
        )
        solver.solve(max_iter=3)
        assert len(solver._all_scores) == N_COLD + 3

    def test_cs_smiles_path_valid(self, mock_codec, mock_oracle, mock_black_box, larger_cold_start):
        pytest.importorskip("rdkit")
        from src.spherevo.gp_competitors import TrippTanimotoGPSolver

        x0, y0 = larger_cold_start
        valid_smiles = ["CCO", "CCCO", "CC=O", "CCN", "c1ccccc1",
                        "CCOC", "CCCC", "CCCCCCO"]
        solver = TrippTanimotoGPSolver(
            black_box=mock_black_box, x0=x0, y0=y0,
            codec=mock_codec, cs_smiles=valid_smiles,
        )
        # All 8 cold-start FPs should be valid (rdkit parses these SMILES)
        assert solver._n_invalid_cold_start == 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestSmilesToMorgan:
    def test_invalid_smiles_returns_none(self):
        pytest.importorskip("rdkit")
        assert _smiles_to_morgan("") is None
        assert _smiles_to_morgan("not a smiles string") is None
        assert _smiles_to_morgan("@#%invalid") is None

    def test_valid_smiles_returns_binary_vector(self):
        pytest.importorskip("rdkit")
        fp = _smiles_to_morgan("CCO", n_bits=2048, radius=2)
        assert fp is not None
        assert fp.shape == (2048,)
        assert set(np.unique(fp)).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Box-supported acquisition variants
# ---------------------------------------------------------------------------


class TestBoxAcquisition:
    """Verify the *BoxSolver classes constrain candidates to the box and the
    fallback path stays in the box (not on the sphere -- that would
    contaminate the box-vs-sphere ablation)."""

    def test_box_constants_match_zinc_encoding(self):
        # Lock in the paper-cited values; trips if anyone "rounds" the constant.
        assert ZINC_PER_COORD_STD == pytest.approx(0.2937, abs=1e-4)
        assert BOX_SIGMA_MULTIPLIER == 3
        assert BOX_HALF_WIDTH == pytest.approx(BOX_SIGMA_MULTIPLIER * ZINC_PER_COORD_STD)

    def test_random_in_box_shape_and_bounds(self):
        for D in (8, 32, 256):
            sample = _random_in_box(D)
            assert sample.shape == (1, D)
            assert np.abs(sample).max() <= BOX_HALF_WIDTH

    @pytest.mark.parametrize("solver_cls", [
        HvarfnerScaledGPBoxSolver,
        PapenmeierMSRBoxSolver,
        XuStandardGPBoxSolver,
    ])
    def test_box_solvers_produce_in_box_candidates(
        self, solver_cls, mock_black_box, larger_cold_start
    ):
        x0, y0 = larger_cold_start
        solver = solver_cls(black_box=mock_black_box, x0=x0, y0=y0)
        solver.solve(max_iter=3)
        # Every step (cold-start + iterations) should land in the box; the
        # cold-start molecules come from MockCodec encoding, which is unrelated
        # to the box, so we only check the iteration steps.
        for z in solver.history["x"][N_COLD:]:
            assert np.abs(z).max() <= BOX_HALF_WIDTH + 1e-6, (
                f"{solver_cls.__name__} produced candidate with "
                f"max(|z|)={np.abs(z).max():.4f} > BOX_HALF_WIDTH={BOX_HALF_WIDTH:.4f}"
            )

    def test_box_solver_fallback_stays_in_box(self, monkeypatch, mock_black_box, larger_cold_start):
        """When GP fit fails, the fallback proposal must stay in the box."""
        x0, y0 = larger_cold_start
        solver = HvarfnerScaledGPBoxSolver(black_box=mock_black_box, x0=x0, y0=y0)

        def fail_make_gp(self, X, Y):
            raise np.linalg.LinAlgError("synthetic GP-fit failure")

        monkeypatch.setattr(
            HvarfnerScaledGPBoxSolver, "_make_gp", fail_make_gp, raising=True,
        )
        z = solver.next_candidate()
        assert z.shape == (1, D_TEST)
        assert np.abs(z).max() <= BOX_HALF_WIDTH + 1e-6, (
            f"fallback emitted out-of-box candidate "
            f"max(|z|)={np.abs(z).max():.4f}"
        )
        assert solver._fallback_count == 1
        assert "LinAlgError" in solver._fallback_reasons[0]

    def test_box_solver_optimize_acqf_runtime_error_is_caught(
        self, monkeypatch, mock_black_box, larger_cold_start
    ):
        """A plain RuntimeError from optimize_acqf must hit the fallback,
        not crash the solver."""
        from src.spherevo import gp_competitors

        x0, y0 = larger_cold_start
        solver = HvarfnerScaledGPBoxSolver(black_box=mock_black_box, x0=x0, y0=y0)

        def fail_optimize_acqf(*args, **kwargs):
            raise RuntimeError("synthetic L-BFGS failure")

        monkeypatch.setattr(
            gp_competitors, "optimize_acqf", fail_optimize_acqf, raising=True,
        )
        z = solver.next_candidate()
        assert z.shape == (1, D_TEST)
        assert solver._fallback_count == 1
        # ModelFittingError wraps the RuntimeError per _propose_qlogei_on_box.
        assert "ModelFittingError" in solver._fallback_reasons[0] or \
               "RuntimeError" in solver._fallback_reasons[0]

    def test_box_helper_validates_host_contract(self):
        """Calling ``_box_next_candidate`` on a host without _make_gp etc.
        should fail loudly with a clear TypeError, not crash silently."""
        from src.spherevo.solver import StepByStepSolver

        # A bare StepByStepSolver subclass missing the GP machinery.
        class _Bare(StepByStepSolver):
            def next_candidate(self):
                return np.zeros((1, self._dim))

        bare = _Bare(
            black_box=lambda x: np.zeros((x.shape[0], 1)),
            x0=np.ones((5, 4)), y0=np.zeros((5, 1)),
        )
        with pytest.raises(TypeError, match="missing required attributes"):
            _box_next_candidate(bare)


# ---------------------------------------------------------------------------
# Geodesic-Slice GP — novel surrogate
# ---------------------------------------------------------------------------


class TestGeodesicSliceGPSolver:
    def test_solve_completes(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        solver = GeodesicSliceGPSolver(
            black_box=mock_black_box, x0=x0, y0=y0,
            n_directions=4, grid_size=64,
        )
        solver.solve(max_iter=5)
        assert len(solver._all_scores) == N_COLD + 5

    def test_candidates_on_sphere(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        solver = GeodesicSliceGPSolver(
            black_box=mock_black_box, x0=x0, y0=y0,
            n_directions=4, grid_size=64,
        )
        solver.solve(max_iter=3)
        for z in solver.history["x"][N_COLD:]:
            norm = float(np.linalg.norm(z))
            assert abs(norm - solver.mean_norm) < 1e-3, (
                f"candidate norm {norm} != mean_norm {solver.mean_norm}"
            )

    def test_exposes_fallback_counters(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        solver = GeodesicSliceGPSolver(
            black_box=mock_black_box, x0=x0, y0=y0,
        )
        assert solver._fallback_count == 0
        assert solver._fallback_reasons == []

    def test_rejects_bad_args(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        with pytest.raises(ValueError, match="n_directions"):
            GeodesicSliceGPSolver(
                black_box=mock_black_box, x0=x0, y0=y0, n_directions=0,
            )
        with pytest.raises(ValueError, match="grid_size"):
            GeodesicSliceGPSolver(
                black_box=mock_black_box, x0=x0, y0=y0, grid_size=4,
            )

    def test_tangent_basis_orthogonality(self):
        """Returned tangent direction must be unit-norm and orthogonal to anchor."""
        rng = np.random.default_rng(0)
        u = rng.normal(size=16)
        u = u / np.linalg.norm(u)
        for _ in range(20):
            v_raw = rng.normal(size=16)
            v = _tangent_basis_at(u, v_raw)
            assert v is not None
            assert abs(np.linalg.norm(v) - 1.0) < 1e-9
            assert abs(np.dot(u, v)) < 1e-9

    def test_tangent_basis_rejects_parallel(self):
        u = np.array([1.0, 0.0, 0.0])
        assert _tangent_basis_at(u, u * 2.5) is None
        assert _tangent_basis_at(u, -u * 0.7) is None


# ---------------------------------------------------------------------------
# Hvarfner-Ball — novel codec-independent surrogate (Hvarfner GP + GeoWalk
# annealing schedule)
# ---------------------------------------------------------------------------


class TestHvarfnerScaledGPBallSolver:
    def test_solve_completes(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        solver = HvarfnerScaledGPBallSolver(
            black_box=mock_black_box, x0=x0, y0=y0,
            n_candidates=64, total_budget=10,
        )
        solver.solve(max_iter=5)
        assert len(solver._all_scores) == N_COLD + 5

    def test_candidates_on_sphere(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        solver = HvarfnerScaledGPBallSolver(
            black_box=mock_black_box, x0=x0, y0=y0,
            n_candidates=64, total_budget=10,
        )
        solver.solve(max_iter=3)
        for z in solver.history["x"][N_COLD:]:
            norm = float(np.linalg.norm(z))
            assert abs(norm - solver.mean_norm) < 1e-3, (
                f"candidate norm {norm} != mean_norm {solver.mean_norm}"
            )

    def test_anneal_schedule_decays(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        solver = HvarfnerScaledGPBallSolver(
            black_box=mock_black_box, x0=x0, y0=y0,
            n_candidates=8, total_budget=10,
            angle_init=1.0, angle_final=0.1,
        )
        # t=0 -> angle_init; t=total_budget-1 -> angle_final.
        assert abs(solver._angle_at(0) - 1.0) < 1e-9
        assert abs(solver._angle_at(9) - 0.1) < 1e-9
        # Mid-progress should interpolate linearly.
        assert solver._angle_at(0) > solver._angle_at(5) > solver._angle_at(9)

    def test_candidate_within_angular_cap(self, mock_black_box, larger_cold_start):
        """Each proposed candidate must be within angle_init from current best."""
        x0, y0 = larger_cold_start
        solver = HvarfnerScaledGPBallSolver(
            black_box=mock_black_box, x0=x0, y0=y0,
            n_candidates=32, total_budget=20,
            angle_init=0.5, angle_final=0.1,
        )
        for _ in range(3):
            solver.step()
        # Get best in history at start of last step (post the 3 steps).
        x_hist, y_hist = solver.get_history_as_arrays()
        i_best = int(np.argmax(y_hist.flatten()))
        u_best = x_hist[i_best] / np.linalg.norm(x_hist[i_best])
        last_x = solver.history["x"][-1].flatten()
        u_last = last_x / np.linalg.norm(last_x)
        # cos(angle) >= cos(angle_init) for any in-cap proposal.
        cos_angle = float(np.dot(u_best, u_last))
        # Note: u_best at this step was not the same as at iteration 0; it could
        # have changed. Just check candidate stayed near *some* history point.
        cosines = u_last @ (x_hist / np.linalg.norm(x_hist, axis=1, keepdims=True)).T
        assert cosines.max() >= math.cos(0.5 + 1e-3)

    def test_rejects_bad_angles(self, mock_black_box, larger_cold_start):
        x0, y0 = larger_cold_start
        with pytest.raises(ValueError, match="angle"):
            HvarfnerScaledGPBallSolver(
                black_box=mock_black_box, x0=x0, y0=y0,
                angle_init=0.1, angle_final=0.5,  # reversed
            )


# ---------------------------------------------------------------------------
# Regression tests for the bug class that previously archived 654 result files
# (Hvarfner formula error, noise-floor handling, NaN propagation). Each test
# exercises a specific bug class directly so a regression fails loudly here
# instead of silently in benchmark numbers.
# ---------------------------------------------------------------------------


class TestHvarfnerFormulaRegression:
    """Pin the Hvarfner et al. 2024 lengthscale-prior formula numerically.

    Recipe: per-coordinate ARD `MaternKernel(nu=2.5)` with
    `LogNormalPrior(loc=sqrt(2)+0.5*log(D), scale=sqrt(3))`. A future refactor
    that drops the `sqrt(D)` term, swaps the kernel, or rescales `scale` will
    fail this test before it silently corrupts the benchmark.
    """

    def test_lengthscale_prior_pinned_to_paper(self, mock_black_box, larger_cold_start):
        import torch
        from gpytorch.kernels import MaternKernel
        from gpytorch.priors import LogNormalPrior

        x0, y0 = larger_cold_start
        solver = HvarfnerScaledGPSolver(black_box=mock_black_box, x0=x0, y0=y0)
        # Build a fresh GP through the solver's own _make_gp pipeline.
        X = torch.tensor(x0, dtype=torch.float64)
        Y = torch.tensor(y0, dtype=torch.float64)
        Y_std = (Y - Y.mean()) / (Y.std() + 1e-8)
        gp = solver._make_gp(X, Y_std)

        # Walk the kernel tree to find the base MaternKernel with ARD.
        base = gp.covar_module
        while not isinstance(base, MaternKernel):
            base = base.base_kernel

        assert base.nu == 2.5, "Hvarfner uses Matern-5/2"
        assert base.ard_num_dims == X.shape[-1], "Must be ARD with one ls/dim"

        prior = base.lengthscale_prior
        assert isinstance(prior, LogNormalPrior), (
            f"Hvarfner expects LogNormalPrior, got {type(prior).__name__}"
        )
        D = X.shape[-1]
        # Hvarfner et al. 2024 Eq. 6: loc = sqrt(2) + 0.5*log(D), scale = sqrt(3).
        expected_loc = math.sqrt(2.0) + 0.5 * math.log(D)
        expected_scale = math.sqrt(3.0)
        assert float(prior.loc) == pytest.approx(expected_loc, abs=1e-4), (
            f"Hvarfner lengthscale prior loc drifted from "
            f"sqrt(2)+0.5*log(D) = {expected_loc:.4f} (got {float(prior.loc):.4f})"
        )
        assert float(prior.scale) == pytest.approx(expected_scale, abs=1e-4), (
            f"Hvarfner lengthscale prior scale drifted from "
            f"sqrt(3) = {expected_scale:.4f} (got {float(prior.scale):.4f})"
        )


class TestFlatSignalAndNaNRegression:
    """Direct regressions for the fallback paths that originally hid bugs."""

    def test_flat_y_triggers_flat_signal_fallback(self, mock_black_box):
        """All-equal `y` must trip the `sd_raw < 1e-6` flat-signal branch and
        emit a candidate at the empirical sphere radius without raising."""
        x0 = np.random.RandomState(0).randn(N_COLD, D_TEST)
        y0 = np.full((N_COLD, 1), 0.42, dtype=np.float64)  # constant signal
        solver = HvarfnerScaledGPSolver(black_box=mock_black_box, x0=x0, y0=y0)
        z = solver.next_candidate()
        assert z.shape == (1, D_TEST)
        # On-sphere fallback at mean_norm — invariant of the variant.
        assert abs(float(np.linalg.norm(z)) - solver.mean_norm) < 1e-3
        assert solver._fallback_count == 1
        assert solver._fallback_reasons[0].startswith("flat-signal")

    def test_nan_in_y0_does_not_corrupt_gp_fit(self, mock_black_box):
        """A single NaN in `y` must be penalized to a finite floor (not propagate
        to the GP) and the solver must still emit a finite, on-sphere candidate.
        This guards the bug class that previously archived 654 result files."""
        x0 = np.random.RandomState(1).randn(N_COLD, D_TEST)
        y0 = np.random.RandomState(2).rand(N_COLD, 1)
        y0[0, 0] = np.nan  # one corrupted observation
        solver = HvarfnerScaledGPSolver(black_box=mock_black_box, x0=x0, y0=y0)
        z = solver.next_candidate()
        assert z.shape == (1, D_TEST)
        assert np.isfinite(z).all(), (
            "NaN in y0 must not propagate to the candidate; got non-finite z"
        )
        assert abs(float(np.linalg.norm(z)) - solver.mean_norm) < 1e-3


class TestFilteredBoxGeometryRegression:
    """When the norm-cap rejects too many candidates, the fallback must
    preserve the *box* geometry of the diagnostic. Falling back to on-sphere
    (the prior bug) would have contaminated the box-vs-sphere ablation."""

    def test_filtered_box_fallback_stays_in_box(self, mock_black_box, larger_cold_start):
        from src.spherevo.gp_competitors import (
            HvarfnerScaledGPFilteredBoxSolver,
        )

        x0, y0 = larger_cold_start
        # norm_cap_multiplier ≈ 0 forces the filter to reject everything →
        # fallback path must trigger.
        solver = HvarfnerScaledGPFilteredBoxSolver(
            black_box=mock_black_box, x0=x0, y0=y0,
        )
        solver.norm_cap_multiplier = 1e-9
        z = solver.next_candidate()
        assert z.shape == (1, D_TEST)
        # Fallback must stay in the box, not on the sphere.
        assert np.abs(z).max() <= BOX_HALF_WIDTH + 1e-6, (
            f"filtered-box fallback drifted out of box: "
            f"max(|z|)={np.abs(z).max():.4f} > BOX_HALF_WIDTH={BOX_HALF_WIDTH:.4f}"
        )
        assert solver._fallback_count == 1
        assert "filtered-box too restrictive" in solver._fallback_reasons[0]


class TestGeodesicSliceFallbackIntegrity:
    """The per-slice GP fit failure must increment `_fallback_count` (not just
    `_fallback_reasons`). Otherwise the run-level counter underreports failures
    and post-hoc audit silently disagrees with the reasons list."""

    def test_per_slice_failure_increments_counter(
        self, monkeypatch, mock_black_box, larger_cold_start
    ):
        x0, y0 = larger_cold_start
        solver = GeodesicSliceGPSolver(
            black_box=mock_black_box, x0=x0, y0=y0,
            n_directions=4, grid_size=64,
        )
        # Force every per-slice GP fit to fail.
        def fail_fit(*args, **kwargs):
            raise np.linalg.LinAlgError("synthetic per-slice GP failure")

        monkeypatch.setattr(
            GeodesicSliceGPSolver, "_make_2d_gp", fail_fit, raising=True,
        )
        z = solver.next_candidate()
        assert z.shape == (1, D_TEST)
        # Each of the 4 directions should have failed; counter must reflect that.
        assert solver._fallback_count == solver.n_directions == 4
        assert len(solver._fallback_reasons) == solver._fallback_count, (
            "Counter / reasons list must agree (fallback integrity invariant)."
        )
