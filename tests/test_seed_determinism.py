"""Seed determinism tests.

Verifies that solvers produce identical results given the same seed,
and different results given different seeds. Uses mock black box
(no GPU, VAE weights, or external packages needed).
All solvers use the unified poli interface: (black_box, x0, y0).
"""

import numpy as np
import pytest
import torch

from src.spherevo.solver import CMAESSubspaceSolver, RandomSearchSolver

from conftest import MockBlackBox, MockCodec, MockOracle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cold_start():
    """Create cold-start (x0, y0) arrays and black box."""
    codec = MockCodec(dim=32)
    oracle = MockOracle()
    smiles = ["CCO", "CCCO", "CC=O", "C1CCCCC1", "c1ccccc1"]
    x0 = codec.encode(smiles).numpy()
    y0 = np.array([oracle(s) for s in smiles]).reshape(-1, 1)
    bb = MockBlackBox(codec, oracle)
    return bb, x0, y0


def _run_cmaes(seed: int, n_steps: int = 3) -> list[float]:
    """Run CMAESSubspaceSolver with given seed, return scores."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    bb, x0, y0 = _make_cold_start()

    solver = CMAESSubspaceSolver(
        bb, x0, y0,
        subspace_dim=8, population_size=5,
    )

    result_scores = []
    for _ in range(n_steps):
        score = solver.step()
        result_scores.append(score)

    return result_scores


def _run_random(seed: int, n_steps: int = 3) -> list[float]:
    """Run RandomSearchSolver with given seed, return scores."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    bb, x0, y0 = _make_cold_start()

    solver = RandomSearchSolver(bb, x0, y0)

    result_scores = []
    for _ in range(n_steps):
        score = solver.step()
        result_scores.append(score)

    return result_scores


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCMAESSeedDeterminism:
    """CMA-ES solver: same seed -> identical results."""

    def test_same_seed_identical_scores(self):
        scores1 = _run_cmaes(seed=42, n_steps=3)
        scores2 = _run_cmaes(seed=42, n_steps=3)
        np.testing.assert_array_equal(
            scores1, scores2,
            err_msg="CMAESSubspaceSolver produced different scores with same seed"
        )

    def test_different_seed_different_scores(self):
        scores1 = _run_cmaes(seed=42, n_steps=3)
        scores2 = _run_cmaes(seed=43, n_steps=3)
        assert not np.array_equal(scores1, scores2), (
            "CMAESSubspaceSolver produced identical scores with different seeds"
        )


class TestRandomSearchSeedDeterminism:
    """Random search solver: same seed -> identical results."""

    def test_same_seed_identical_scores(self):
        scores1 = _run_random(seed=42, n_steps=3)
        scores2 = _run_random(seed=42, n_steps=3)
        np.testing.assert_array_equal(
            scores1, scores2,
            err_msg="RandomSearchSolver produced different scores with same seed"
        )

    def test_different_seed_different_scores(self):
        scores1 = _run_random(seed=42, n_steps=3)
        scores2 = _run_random(seed=43, n_steps=3)
        assert not np.array_equal(scores1, scores2), (
            "RandomSearchSolver produced identical scores with different seeds"
        )


class TestCMAESMultipleRuns:
    """Verify determinism over longer runs (10 steps)."""

    def test_10_steps_deterministic(self):
        scores1 = _run_cmaes(seed=42, n_steps=10)
        scores2 = _run_cmaes(seed=42, n_steps=10)
        np.testing.assert_array_equal(
            scores1, scores2,
            err_msg="CMAESSubspaceSolver diverged after 10 steps with same seed"
        )

    def test_qr_subspace_deterministic(self):
        """The random QR subspace matrix should be identical given the same seed."""
        np.random.seed(42)
        torch.manual_seed(42)

        bb, x0, y0 = _make_cold_start()
        solver1 = CMAESSubspaceSolver(
            bb, x0, y0, subspace_dim=8, population_size=5,
        )
        A1 = solver1._A.copy()

        np.random.seed(42)
        torch.manual_seed(42)

        bb2, x0_2, y0_2 = _make_cold_start()
        solver2 = CMAESSubspaceSolver(
            bb2, x0_2, y0_2, subspace_dim=8, population_size=5,
        )
        A2 = solver2._A.copy()

        np.testing.assert_array_equal(A1, A2, err_msg="QR subspace differs with same seed")
