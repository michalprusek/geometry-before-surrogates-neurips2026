"""``geowalk_one_fifth``: GeoWalk with success-rule angle adaptation.

Camera-ready amendment arm (docs/camera_ready/RUN_RULES.md, amendment log).
The tests pin

* the rule: first refinement angle 0.30, x1.2 after an improving refinement
  step, x0.85 otherwise, clamp [0.05, 0.50] (choices O1-O3 of the class);
* that only the angle differs from GeoWalk: same exploration length, same RNG
  stream, so the cold start and the exploration phase are bit-identical to
  ``geowalk`` under the same seed (O4, O5);
* registration in the benchmark runner.

The rule is a success-based angle rule, not a 1/5 rule: with x1.2 / x0.85 the
angle is stationary at a success rate of about 0.47 (O6).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.spherevo.solver import GeoWalkOneFifthSolver, GeoWalkSolver

BUDGET = 25         # N = 30 protocol: 5 cold-start points + 25 steps
EXPLORE = 16         # int(0.64 * 25), GeoWalk's exploration length


class SequenceBlackBox:
    """Returns a scripted score sequence, one value per call."""

    def __init__(self, scores):
        self._scores = list(scores)
        self.calls = 0

    def __call__(self, z):
        s = self._scores[self.calls]
        self.calls += 1
        return np.array([[s]], dtype=np.float64)


def _expected_angles(scores_all: list[float], n_cold: int, explore: int,
                     n_steps: int) -> list[float]:
    """Re-derive the angle sequence from the score history (independent code)."""
    angles, a = [], 0.30
    for step in range(explore, n_steps):
        angles.append(a)
        idx = n_cold + step                          # index of this step's score
        before = [s for s in scores_all[:idx] if np.isfinite(s)]
        improved = np.isfinite(scores_all[idx]) and scores_all[idx] > max(before) + 1e-10
        a = min(max(a * (1.2 if improved else 0.85), 0.05), 0.50)
    return angles


def test_first_refinement_angle_is_0p30_and_the_rule_is_applied(
    cold_start_data, mock_black_box,
):
    x0, y0 = cold_start_data
    np.random.seed(3)
    solver = GeoWalkOneFifthSolver(mock_black_box, x0, y0, total_budget=BUDGET)
    solver.solve(max_iter=BUDGET)

    assert solver.explore_steps == EXPLORE
    assert len(solver.all_scores) == len(y0) + BUDGET
    assert len(solver.angle_trace) == BUDGET - EXPLORE
    assert solver.angle_trace[0] == 0.30
    expected = _expected_angles(solver.all_scores, len(y0), EXPLORE, BUDGET)
    assert solver.angle_trace == pytest.approx(expected, rel=0, abs=1e-15)
    assert all(0.05 <= a <= 0.50 for a in solver.angle_trace)


def test_angle_grows_on_improvements_and_is_clamped_at_0p50(cold_start_data):
    x0, y0 = cold_start_data
    explore = 2
    scores = list(np.linspace(1.0, 2.0, 12))       # every step improves
    solver = GeoWalkOneFifthSolver(
        SequenceBlackBox(scores), x0, y0, explore_steps=explore, total_budget=12,
    )
    np.random.seed(0)
    solver.solve(max_iter=12)
    # 0.30 -> 0.36 -> 0.432 -> 0.5184 clamped to 0.50, then stays at the cap
    assert solver.angle_trace[:4] == pytest.approx([0.30, 0.36, 0.432, 0.50])
    assert solver.angle_trace[4:] == pytest.approx([0.50] * (len(solver.angle_trace) - 4))


def test_angle_shrinks_on_failures_and_is_clamped_at_0p05(cold_start_data):
    x0, y0 = cold_start_data
    n_steps = 20
    scores = [-1.0] * n_steps                       # never beats the cold start
    solver = GeoWalkOneFifthSolver(
        SequenceBlackBox(scores), x0, y0, explore_steps=1, total_budget=n_steps,
    )
    np.random.seed(0)
    solver.solve(max_iter=n_steps)
    expected, a = [], 0.30
    for _ in range(n_steps - 1):
        expected.append(a)
        a = max(a * 0.85, 0.05)
    assert solver.angle_trace == pytest.approx(expected)
    assert solver.angle_trace[-1] == 0.05


def test_a_nan_score_is_a_failure_and_a_tie_is_not_an_improvement(cold_start_data):
    x0, y0 = cold_start_data
    best0 = float(np.max(y0))
    # explore 1 step, then: NaN (failure), tie with the incumbent (failure),
    # a real improvement (success).
    scores = [best0 - 1.0, float("nan"), best0, best0 + 0.5, best0 + 0.5]
    solver = GeoWalkOneFifthSolver(
        SequenceBlackBox(scores), x0, y0, explore_steps=1, total_budget=len(scores),
    )
    np.random.seed(0)
    solver.solve(max_iter=len(scores))
    assert solver.angle_trace == pytest.approx([0.30, 0.255, 0.21675, 0.2601])


def test_exploration_and_rng_stream_are_geowalks(cold_start_data, mock_black_box):
    """Same seed: cold start + all 16 exploration points are bit-identical to
    GeoWalk, and the first refinement step uses GeoWalk's random tangent
    direction with the adaptive angle in place of GeoWalk's 0.60."""
    x0, y0 = cold_start_data
    runs = {}
    for cls in (GeoWalkSolver, GeoWalkOneFifthSolver):
        np.random.seed(11)
        s = cls(mock_black_box, x0, y0, total_budget=BUDGET)
        s.solve(max_iter=EXPLORE + 1)
        runs[cls.__name__] = s
    gw, of = runs["GeoWalkSolver"], runs["GeoWalkOneFifthSolver"]
    X_gw = np.concatenate(gw.history["x"], axis=0)
    X_of = np.concatenate(of.history["x"], axis=0)
    n_same = len(y0) + EXPLORE
    assert np.array_equal(X_gw[:n_same], X_of[:n_same])
    assert gw.all_scores[:n_same] == of.all_scores[:n_same]

    # Reconstruct both first refinement proposals from one tangent direction.
    ys = np.asarray(gw.all_scores[:n_same])
    anc = X_gw[int(np.argmax(ys))]
    anc_n = anc / (np.linalg.norm(anc) + 1e-10)

    def tangent_of(x, angle):
        u = x / np.linalg.norm(x)
        return (u - np.cos(angle) * anc_n) / np.sin(angle)

    v_gw = tangent_of(X_gw[n_same], gw._angle(0))
    v_of = tangent_of(X_of[n_same], 0.30)
    assert np.allclose(v_gw, v_of, atol=1e-9)
    assert of.angle_trace == [0.30]


def test_registered_in_the_benchmark_runner(cold_start_data, mock_black_box):
    from scripts.run_benchmark import OUR_SOLVERS, _create_solver

    x0, y0 = cold_start_data
    assert "geowalk_one_fifth" in OUR_SOLVERS
    solver = _create_solver(
        solver_name="geowalk_one_fifth", black_box=mock_black_box, x0=x0, y0=y0,
        n_dimensions=x0.shape[1], subspace_dim=16, n_subspaces=16, seed=42,
        n_initial_points=len(y0), codec=None, norm_mode="fixed", n_iterations=BUDGET,
        cs_smiles=None,
    )
    assert type(solver) is GeoWalkOneFifthSolver
    assert solver.explore_steps == EXPLORE
    assert (solver.angle_start, solver.angle_up, solver.angle_down) == (0.30, 1.2, 0.85)
    assert (solver.angle_min, solver.angle_max) == (0.05, 0.50)
