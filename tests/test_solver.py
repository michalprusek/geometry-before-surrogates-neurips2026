"""Smoke tests for solver classes.

Tests the solver interface, helper functions, and task-name mappings
without requiring GPU, VAE weights, or external packages.
All solvers use the unified poli interface: (black_box, x0, y0).
"""

import numpy as np
import pytest
import torch

from src.analysis.config import GUACAMOL_TO_POLI, POLI_TO_GUACAMOL, TASK_DISPLAY_NAMES
from src.spherevo.solver import (
    CMAESSubspaceSolver,
    GPFreeSphericalSolver,
    GeoWalkNoAnchorSolver,
    GeoWalkNoTangentSolver,
    GeoWalkSolver,
    HeSBOSolver,
    LQCMAESSubspaceSolver,
    NoiseGPSphericalSolver,
    PACESubspaceSolver,
    REMBOSolver,
    ROSESubspaceSolver,
    RandomSearchSolver,
    StepByStepSolver,
    _lift_to_ambient,
    _normalize,
)


# ---------------------------------------------------------------------------
# _normalize helper
# ---------------------------------------------------------------------------


class TestNormalize:
    """Tests for the _normalize() helper function."""

    def test_produces_unit_vectors(self):
        x = np.random.randn(10, 32)
        normed = _normalize(x)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_single_row(self):
        x = np.array([[3.0, 4.0]])
        normed = _normalize(x)
        expected = np.array([[0.6, 0.8]])
        np.testing.assert_allclose(normed, expected, atol=1e-7)

    def test_zero_vector_does_not_crash(self):
        """Near-zero vectors should not produce NaN (guarded by max(norm, 1e-8))."""
        x = np.array([[0.0, 0.0, 0.0]])
        normed = _normalize(x)
        assert not np.any(np.isnan(normed))

    def test_preserves_direction(self):
        x = np.array([[1.0, 2.0, 3.0]])
        normed = _normalize(x)
        # Direction should be preserved (cosine similarity = 1)
        cos_sim = (x @ normed.T) / (np.linalg.norm(x) * np.linalg.norm(normed))
        np.testing.assert_allclose(cos_sim, 1.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Task-name mapping
# ---------------------------------------------------------------------------


class TestTaskNameMapping:
    """Tests for GUACAMOL_TO_POLI and POLI_TO_GUACAMOL."""

    def test_mappings_are_inverses(self):
        """Round-tripping through both dicts should return the original key."""
        for short_code, poli_name in GUACAMOL_TO_POLI.items():
            assert POLI_TO_GUACAMOL[poli_name] == short_code

    def test_reverse_mapping_completeness(self):
        """Every value in GUACAMOL_TO_POLI is a key in POLI_TO_GUACAMOL."""
        for poli_name in GUACAMOL_TO_POLI.values():
            assert poli_name in POLI_TO_GUACAMOL

    def test_forward_mapping_completeness(self):
        """Every value in POLI_TO_GUACAMOL is a key in GUACAMOL_TO_POLI."""
        for short_code in POLI_TO_GUACAMOL.values():
            assert short_code in GUACAMOL_TO_POLI

    def test_all_12_tasks_present(self):
        expected = {"adip", "dhop", "fexo", "med1", "med2", "osmb",
                    "pdop", "rano", "shop", "siga", "valt", "zale"}
        assert set(GUACAMOL_TO_POLI.keys()) == expected

    def test_same_length(self):
        assert len(GUACAMOL_TO_POLI) == len(POLI_TO_GUACAMOL)

    def test_osmb_typo(self):
        """Poli spells osimertinib as 'osimetrinib' (known typo). Mapping must match."""
        assert GUACAMOL_TO_POLI["osmb"] == "osimetrinib_mpo"


# ---------------------------------------------------------------------------
# StepByStepSolver ABC
# ---------------------------------------------------------------------------


class TestStepByStepSolverABC:
    """Test that the ABC cannot be directly instantiated."""

    def test_cannot_instantiate(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        with pytest.raises(TypeError, match="abstract"):
            StepByStepSolver(mock_black_box, x0, y0)


# ---------------------------------------------------------------------------
# CMAESSubspaceSolver smoke test
# ---------------------------------------------------------------------------


class TestCMAESSubspaceSolver:
    """Smoke test: CMAESSubspaceSolver with mock black box."""

    def test_runs_one_step(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = CMAESSubspaceSolver(
            mock_black_box, x0, y0,
            subspace_dim=8, population_size=5,
        )

        score = solver.step()
        assert isinstance(score, float)
        assert solver.iteration == 1

    def test_history_grows(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = CMAESSubspaceSolver(
            mock_black_box, x0, y0,
            subspace_dim=8, population_size=5,
        )

        n_cold = len(x0)
        n_steps = 3
        for _ in range(n_steps):
            solver.step()

        assert len(solver.all_scores) == n_cold + n_steps

    def test_best_score_is_max(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = CMAESSubspaceSolver(
            mock_black_box, x0, y0,
            subspace_dim=8, population_size=5,
        )

        for _ in range(5):
            solver.step()

        assert solver.best_score == max(solver.all_scores)

    def test_candidate_shape(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = CMAESSubspaceSolver(
            mock_black_box, x0, y0,
            subspace_dim=8, population_size=5,
        )

        candidate = solver.next_candidate()
        assert candidate.shape == (1, 32)

    def test_subspace_projection_dimensions(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = CMAESSubspaceSolver(
            mock_black_box, x0, y0,
            subspace_dim=8, population_size=5,
        )

        # A should be [D, d] = [32, 8]
        assert solver._A.shape == (32, 8)
        # A should be orthonormal columns
        AtA = solver._A.T @ solver._A
        np.testing.assert_allclose(AtA, np.eye(8), atol=1e-10)

    def test_solve_method(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = CMAESSubspaceSolver(
            mock_black_box, x0, y0,
            subspace_dim=8, population_size=5,
        )

        solver.solve(max_iter=5)
        assert solver.iteration == 5
        assert len(solver.all_scores) == len(x0) + 5

    def test_get_best_performance(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = CMAESSubspaceSolver(
            mock_black_box, x0, y0,
            subspace_dim=8, population_size=5,
        )

        solver.solve(max_iter=3)
        perf = solver.get_best_performance()
        assert perf.shape == (1, 1)
        assert float(perf.flatten()[0]) == solver.best_score


# ---------------------------------------------------------------------------
# CMAESSubspaceSolver adaptive norm mode
# ---------------------------------------------------------------------------


class TestCMAESSubspaceAdaptiveNorm:
    """Tests for CMAESSubspaceSolver with norm_mode='adaptive'."""

    def test_adaptive_norm_cma_dim(self, mock_black_box, cold_start_data):
        """CMA-ES dimension should be d+1 in adaptive mode."""
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = CMAESSubspaceSolver(
            mock_black_box, x0, y0,
            subspace_dim=8, population_size=5,
            norm_mode="adaptive",
        )

        # CMA-ES internal dimension is d+1 = 9
        assert solver._cma.N == 9

    def test_adaptive_norm_candidate_shape(self, mock_black_box, cold_start_data):
        """Candidate shape should still be [1, D] in ambient space."""
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = CMAESSubspaceSolver(
            mock_black_box, x0, y0,
            subspace_dim=8, population_size=5,
            norm_mode="adaptive",
        )

        candidate = solver.next_candidate()
        assert candidate.shape == (1, 32)

    def test_adaptive_norm_varies(self, mock_black_box, cold_start_data):
        """Different candidates can have different norms (not locked to mean_norm)."""
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = CMAESSubspaceSolver(
            mock_black_box, x0, y0,
            subspace_dim=8, population_size=5,
            norm_mode="adaptive",
        )

        # Use step() to properly feed back scores (deferred-tell protocol)
        for _ in range(10):
            solver.step()

        # Collect norms from norm_trace (logged during _cma_to_ambient)
        norms = solver._norm_trace
        assert len(norms) >= 10
        # Not all norms should be exactly equal (CMA-ES explores)
        assert len(set(round(n, 6) for n in norms)) > 1

    def test_fixed_norm_backward_compat(self, mock_black_box, cold_start_data):
        """norm_mode='fixed' produces candidates with norm == mean_norm."""
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = CMAESSubspaceSolver(
            mock_black_box, x0, y0,
            subspace_dim=8, population_size=5,
            norm_mode="fixed",
        )

        candidate = solver.next_candidate()
        np.testing.assert_allclose(
            np.linalg.norm(candidate), solver.mean_norm, atol=1e-6
        )

    def test_adaptive_project_shape(self, mock_black_box, cold_start_data):
        """_project_to_cma_space should return [N, d+1] in adaptive mode."""
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = CMAESSubspaceSolver(
            mock_black_box, x0, y0,
            subspace_dim=8, population_size=5,
            norm_mode="adaptive",
        )

        proj = solver._project_to_cma_space(x0)
        assert proj.shape == (len(x0), 9)  # d + 1 = 8 + 1

    def test_adaptive_five_steps_smoke(self, mock_black_box, cold_start_data):
        """Integration: 5 CMA-ES steps complete without error."""
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = CMAESSubspaceSolver(
            mock_black_box, x0, y0,
            subspace_dim=8, population_size=5,
            norm_mode="adaptive",
        )

        solver.solve(max_iter=5)
        assert solver.iteration == 5
        assert len(solver.all_scores) == len(x0) + 5
        # norm_trace should have entries from candidate generation
        assert len(solver._norm_trace) > 0


# ---------------------------------------------------------------------------
# LQCMAESSubspaceSolver smoke tests
# ---------------------------------------------------------------------------


class TestLQCMAESSubspaceSolver:
    """Smoke tests for lq-CMA-ES subspace solver."""

    def test_runs_five_steps(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = LQCMAESSubspaceSolver(
            mock_black_box, x0, y0,
            subspace_dim=8, population_size=5,
        )

        for _ in range(5):
            score = solver.step()
            assert isinstance(score, float)
        assert solver.iteration == 5
        assert len(solver.all_scores) == len(x0) + 5

    def test_model_data_grows(self, mock_black_box, cold_start_data):
        """Model training data should accumulate after each generation."""
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = LQCMAESSubspaceSolver(
            mock_black_box, x0, y0,
            subspace_dim=8, population_size=5,
        )

        n_cold = len(x0)
        assert len(solver._model_X) == n_cold

        # 6 steps: 5 complete gen1, 6th triggers generation boundary
        for _ in range(6):
            solver.step()

        assert len(solver._model_X) == n_cold + 5

    def test_gating_starts_disabled(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = LQCMAESSubspaceSolver(
            mock_black_box, x0, y0,
            subspace_dim=8, population_size=5,
        )

        assert solver._model_active is False

    def test_candidate_shape(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = LQCMAESSubspaceSolver(
            mock_black_box, x0, y0,
            subspace_dim=8, population_size=5,
        )

        candidate = solver.next_candidate()
        assert candidate.shape == (1, 32)


# ---------------------------------------------------------------------------
# RandomSearchSolver smoke test
# ---------------------------------------------------------------------------


class TestRandomSearchSolver:
    """Smoke test: RandomSearchSolver with mock black box."""

    def test_runs_one_step(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = RandomSearchSolver(mock_black_box, x0, y0)

        score = solver.step()
        assert isinstance(score, float)
        assert solver.iteration == 1

    def test_candidate_has_correct_norm(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = RandomSearchSolver(mock_black_box, x0, y0)

        candidate = solver.next_candidate()
        norm = np.linalg.norm(candidate)
        # Should be close to mean_norm of cold-start vectors
        np.testing.assert_allclose(norm, solver.mean_norm, atol=1e-6)

    def test_candidate_shape(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = RandomSearchSolver(mock_black_box, x0, y0)

        candidate = solver.next_candidate()
        assert candidate.shape == (1, 32)


# ---------------------------------------------------------------------------
# GPFreeSphericalSolver smoke test
# ---------------------------------------------------------------------------


class TestGPFreeSphericalSolver:
    """Smoke test: GPFreeSphericalSolver with mock black box."""

    def test_runs_one_step(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        solver = GPFreeSphericalSolver(
            mock_black_box, x0, y0,
            n_subspaces=4, subspace_dim=8,
        )

        score = solver.step()
        assert isinstance(score, float)
        assert solver.iteration == 1

    def test_rotates_subspaces(self, mock_black_box, cold_start_data):
        """Active subspace index should cycle through K subspaces."""
        x0, y0 = cold_start_data
        np.random.seed(42)
        torch.manual_seed(42)

        K = 4
        solver = GPFreeSphericalSolver(
            mock_black_box, x0, y0,
            n_subspaces=K, subspace_dim=8,
        )

        # After K steps, should cycle back to subspace 0
        for _ in range(K):
            solver.step()

        assert solver._active_idx == 0


# ---------------------------------------------------------------------------
# _lift_to_ambient helper
# ---------------------------------------------------------------------------


class TestLiftToAmbient:
    """Tests for the _lift_to_ambient() module-level helper."""

    def test_output_norm_equals_mean_norm(self):
        """Output vector should have norm equal to mean_norm."""
        D, d = 32, 8
        A, _ = np.linalg.qr(np.random.randn(D, d))
        v = np.random.randn(d)
        mean_norm = 5.0
        result = _lift_to_ambient(v, A, mean_norm)
        np.testing.assert_allclose(np.linalg.norm(result), mean_norm, atol=1e-6)

    def test_near_zero_input_safe(self):
        """Near-zero input should not produce NaN."""
        D, d = 32, 8
        A, _ = np.linalg.qr(np.random.randn(D, d))
        v = np.zeros(d)
        result = _lift_to_ambient(v, A, 1.0)
        assert not np.any(np.isnan(result))

    def test_output_shape(self):
        D, d = 32, 8
        A, _ = np.linalg.qr(np.random.randn(D, d))
        v = np.random.randn(d)
        result = _lift_to_ambient(v, A, 1.0)
        assert result.shape == (D,)


# ---------------------------------------------------------------------------
# mean_norm base class computation
# ---------------------------------------------------------------------------


class TestMeanNorm:
    """Tests for mean_norm computed in StepByStepSolver base class."""

    def test_mean_norm_equals_expected(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = RandomSearchSolver(mock_black_box, x0, y0)
        expected = float(np.mean(np.linalg.norm(x0, axis=1)))
        np.testing.assert_allclose(solver.mean_norm, expected, atol=1e-6)

    def test_mean_norm_fallback_for_zero_vectors(self, mock_black_box):
        """If all cold-start vectors are near-zero, mean_norm should be 1.0."""
        x0 = np.zeros((3, 16))
        y0 = np.array([[0.1], [0.2], [0.3]])
        solver = RandomSearchSolver(mock_black_box, x0, y0)
        assert solver.mean_norm == 1.0


# ---------------------------------------------------------------------------
# ROSE, PACE, NoiseGP smoke tests
# ---------------------------------------------------------------------------


class TestROSESubspaceSolver:
    """Smoke test: ROSESubspaceSolver with mock black box."""

    def test_runs_one_step(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = ROSESubspaceSolver(
            mock_black_box, x0, y0, subspace_dim=8,
        )
        score = solver.step()
        assert isinstance(score, float)
        assert solver.iteration == 1

    def test_candidate_shape(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = ROSESubspaceSolver(
            mock_black_box, x0, y0, subspace_dim=8,
        )
        candidate = solver.next_candidate()
        assert candidate.shape == (1, 32)


class TestPACESubspaceSolver:
    """Smoke test: PACESubspaceSolver with mock black box."""

    def test_runs_one_step(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = PACESubspaceSolver(
            mock_black_box, x0, y0, subspace_dim=8,
        )
        score = solver.step()
        assert isinstance(score, float)
        assert solver.iteration == 1

    def test_candidate_shape(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = PACESubspaceSolver(
            mock_black_box, x0, y0, subspace_dim=8,
        )
        candidate = solver.next_candidate()
        assert candidate.shape == (1, 32)


class TestNoiseGPSphericalSolver:
    """Smoke test: NoiseGPSphericalSolver with mock black box."""

    def test_runs_one_step(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = NoiseGPSphericalSolver(
            mock_black_box, x0, y0,
            n_subspaces=4, subspace_dim=8,
        )
        score = solver.step()
        assert isinstance(score, float)
        assert solver.iteration == 1

    def test_inherits_gpfree(self):
        """NoiseGPSphericalSolver should inherit from GPFreeSphericalSolver."""
        assert issubclass(NoiseGPSphericalSolver, GPFreeSphericalSolver)


# ---------------------------------------------------------------------------
# REMBO solver
# ---------------------------------------------------------------------------


try:
    import botorch as _botorch  # noqa: F401
    _has_botorch = True
except ImportError:
    _has_botorch = False


@pytest.mark.skipif(not _has_botorch, reason="botorch not installed")
class TestREMBOSolver:
    """Smoke test: REMBOSolver with mock black box."""

    def test_runs_one_step(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = REMBOSolver(
            mock_black_box, x0, y0, subspace_dim=8,
        )
        score = solver.step()
        assert isinstance(score, float)
        assert solver.iteration == 1

    def test_candidate_shape(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = REMBOSolver(
            mock_black_box, x0, y0, subspace_dim=8,
        )
        candidate = solver.next_candidate()
        assert candidate.shape == (1, 32)

    def test_inherits_stepbystep(self):
        assert issubclass(REMBOSolver, StepByStepSolver)


# ---------------------------------------------------------------------------
# HeSBO solver
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_botorch, reason="botorch not installed")
class TestHeSBOSolver:
    """Smoke test: HeSBOSolver with mock black box."""

    def test_runs_one_step(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = HeSBOSolver(
            mock_black_box, x0, y0, subspace_dim=8,
        )
        score = solver.step()
        assert isinstance(score, float)
        assert solver.iteration == 1

    def test_candidate_shape(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = HeSBOSolver(
            mock_black_box, x0, y0, subspace_dim=8,
        )
        candidate = solver.next_candidate()
        assert candidate.shape == (1, 32)

    def test_count_sketch_sparsity(self, mock_black_box, cold_start_data):
        """Count-sketch matrix should have exactly one nonzero per column."""
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = HeSBOSolver(mock_black_box, x0, y0, subspace_dim=8)
        # Each column of S should have exactly 1 nonzero
        nnz_per_col = np.count_nonzero(solver._S, axis=0)
        np.testing.assert_array_equal(nnz_per_col, 1)

    def test_inherits_stepbystep(self):
        assert issubclass(HeSBOSolver, StepByStepSolver)


# ---------------------------------------------------------------------------
# GeoWalkSolver smoke tests
# ---------------------------------------------------------------------------


class TestGeoWalkSolver:
    """Smoke tests for GeoWalkSolver — geodesic random walk on the sphere."""

    def test_runs_one_step(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = GeoWalkSolver(mock_black_box, x0, y0)
        score = solver.step()
        assert isinstance(score, float)
        assert solver.iteration == 1

    def test_candidate_shape(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = GeoWalkSolver(mock_black_box, x0, y0)
        candidate = solver.next_candidate()
        assert candidate.shape == (1, 32)

    def test_history_grows(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = GeoWalkSolver(mock_black_box, x0, y0)
        n_cold = len(x0)
        n_steps = 4
        for _ in range(n_steps):
            solver.step()
        assert len(solver.all_scores) == n_cold + n_steps

    def test_best_score_is_max(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = GeoWalkSolver(mock_black_box, x0, y0)
        for _ in range(5):
            solver.step()
        assert solver.best_score == max(solver.all_scores)

    def test_solve_method(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = GeoWalkSolver(mock_black_box, x0, y0)
        solver.solve(max_iter=10)
        assert solver.iteration == 10
        assert len(solver.all_scores) == len(x0) + 10

    def test_get_best_performance(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = GeoWalkSolver(mock_black_box, x0, y0)
        solver.solve(max_iter=3)
        best = solver.get_best_performance()
        assert best.shape == (1, 1)
        assert best[0, 0] == solver.best_score

    def test_solve_full_25_iterations(self, mock_black_box, cold_start_data):
        """Run the full 25-iter budget — covers both phases (16 random + 9 exploit)."""
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = GeoWalkSolver(mock_black_box, x0, y0)
        solver.solve(max_iter=25)
        assert solver.iteration == 25
        assert len(solver.all_scores) == len(x0) + 25

    def test_exploit_phase_uses_best_anchor(self, mock_black_box, cold_start_data):
        """After 16 random steps, exploit candidates should be on the unit sphere
        scaled by mean_norm — verify shape and norm."""
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = GeoWalkSolver(mock_black_box, x0, y0)
        solver.solve(max_iter=20)  # 16 random + 4 exploit
        candidate = solver.next_candidate()  # would be exploit step #5
        assert candidate.shape == (1, 32)
        norm = float(np.linalg.norm(candidate))
        np.testing.assert_allclose(norm, solver.mean_norm, rtol=1e-6)

    def test_angle_schedule_boundaries(self, mock_black_box, cold_start_data):
        """Angular schedule: 0.60 at start, 0.15 at end, clamped beyond."""
        x0, y0 = cold_start_data
        solver = GeoWalkSolver(mock_black_box, x0, y0)
        assert solver._angle(0) == pytest.approx(0.60)
        assert solver._angle(8) == pytest.approx(0.15, abs=1e-6)
        assert solver._angle(99) == pytest.approx(0.15, abs=1e-6)  # clamped

    def test_angle_schedule_with_total_budget(self, mock_black_box, cold_start_data):
        """When total_budget is set, angle schedule spans the full refinement phase."""
        x0, y0 = cold_start_data
        solver = GeoWalkSolver(mock_black_box, x0, y0, total_budget=100)
        # denom = 100 - 16 - 1 = 83
        assert solver._angle(0) == pytest.approx(0.60)
        assert solver._angle(83) == pytest.approx(0.15, abs=1e-6)

    def test_direction_accumulation_runs(self, mock_black_box, cold_start_data):
        """SRDA (direction accumulation) variant completes without error."""
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = GeoWalkSolver(mock_black_box, x0, y0, direction_accumulation=True)
        solver.solve(max_iter=25)
        assert solver.iteration == 25
        assert len(solver.all_scores) == len(x0) + 25
        assert solver._srda_path.shape == (32,)

    def test_inherits_stepbystep(self):
        assert issubclass(GeoWalkSolver, StepByStepSolver)


class TestGeoWalkNoAnchorSolver:
    """Ablation: tangent walk from fresh random sphere point (no argmax memory)."""

    def test_solve_completes(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = GeoWalkNoAnchorSolver(mock_black_box, x0=x0, y0=y0)
        solver.solve(max_iter=25)
        assert solver.iteration == 25
        assert len(solver.all_scores) == len(x0) + 25

    def test_candidates_on_sphere(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(0)
        solver = GeoWalkNoAnchorSolver(mock_black_box, x0=x0, y0=y0)
        solver.solve(max_iter=20)
        # Phase 2 candidate should be on sphere at mean_norm
        cand = solver.next_candidate()
        assert cand.shape == (1, 32)
        np.testing.assert_allclose(
            float(np.linalg.norm(cand)), solver.mean_norm, rtol=1e-6,
        )

    def test_anchor_does_not_depend_on_history(self, mock_black_box, cold_start_data):
        """No-anchor variant: identical history but different RNG should give
        different Phase 2 candidates (confirming random anchor, not argmax)."""
        x0, y0 = cold_start_data
        s1 = GeoWalkNoAnchorSolver(mock_black_box, x0=x0, y0=y0)
        s2 = GeoWalkNoAnchorSolver(mock_black_box, x0=x0, y0=y0)
        # Seed Phase 2 step directly
        s1.iteration = s2.iteration = s1.explore_steps
        np.random.seed(1)
        c1 = s1.next_candidate()
        np.random.seed(2)
        c2 = s2.next_candidate()
        assert not np.allclose(c1, c2)


class TestGeoWalkNoTangentSolver:
    """Ablation: uniform sphere sampling at mean_norm for full budget."""

    def test_solve_completes(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(42)
        solver = GeoWalkNoTangentSolver(mock_black_box, x0=x0, y0=y0)
        solver.solve(max_iter=25)
        assert solver.iteration == 25

    def test_all_candidates_on_sphere(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(0)
        solver = GeoWalkNoTangentSolver(mock_black_box, x0=x0, y0=y0)
        solver.solve(max_iter=25)
        for z in solver.history["x"][len(x0):]:
            np.testing.assert_allclose(
                float(np.linalg.norm(z)), solver.mean_norm, rtol=1e-6,
            )


# ---------------------------------------------------------------------------
# TASK_DISPLAY_NAMES consistency
# ---------------------------------------------------------------------------


class TestConfigConsistency:
    """Verify SSOT constants are internally consistent."""

    def test_task_display_names_keys_match_poli_values(self):
        """TASK_DISPLAY_NAMES keys must match GUACAMOL_TO_POLI values."""
        assert set(TASK_DISPLAY_NAMES.keys()) == set(GUACAMOL_TO_POLI.values())

    def test_task_display_names_has_12_entries(self):
        assert len(TASK_DISPLAY_NAMES) == 12
