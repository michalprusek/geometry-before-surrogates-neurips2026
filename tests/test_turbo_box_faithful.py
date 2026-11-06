"""TuRBO-1 on the fixed [p01, p99] box: ``turbo_box_faithful``.

The camera-ready prints a TuRBO row from this arm, so the tests pin

* the harness contract (25 steps = 5 + 25 evaluations, fallback bookkeeping),
* the **fixed** search domain — the defect of the submitted wrapper
  (``src/spherevo/turbo_baseline.py``, ``_normalize_z``) is that its unit cube
  is the running min/max of the observed latents, so it moves after every
  evaluation; here the box comes from the bounds file and never moves,
* every candidate lying inside that box,
* the published trust-region update rules,
* Thompson sampling as the acquisition step, and
* determinism under a fixed seed.

dim=32 ``MockBlackBox``. The 256-D ZINC bounds file is not tracked in git, so
each test writes its own 32-D bounds, deliberately NARROWER than the cold start
(as [p01, p99] is for real ZINC encodings) to exercise the centre projection.
``n_candidates`` is lowered from the published ``min(100*D, 5000)`` wherever a
test actually runs steps; the default is asserted separately.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

import gpytorch  # noqa: E402

from src.spherevo import turbo_box_faithful as tbf
from src.spherevo.gp_competitors import (
    HvarfnerScaledGPBoxFaithfulSolver,
    HvarfnerScaledGPSolver,
)
from src.spherevo.turbo_baseline import TurboState
from src.spherevo.turbo_box_faithful import (
    TURBO_LENGTH_INIT,
    TURBO_LENGTH_MAX,
    TURBO_LENGTH_MIN,
    TURBO_SUCCESS_TOLERANCE,
    TurboBoxFaithfulSolver,
    turbo_failure_tolerance,
    turbo_n_candidates,
)

N_STEPS = 25  # paper protocol: 5 cold-start points + 25 steps = N = 30
N_CAND_TEST = 48  # keep the per-step Cholesky small in the test suite


def _narrow_box(x0: np.ndarray) -> dict:
    """32-D per-coordinate bounds; [p01, p99] excludes part of the cold start."""
    lo, hi = x0.min(axis=0), x0.max(axis=0)
    w = hi - lo
    return {
        "min": (lo - 0.05 * w).tolist(),
        "max": (hi + 0.05 * w).tolist(),
        "p01": (lo + 0.10 * w).tolist(),
        "p99": (hi - 0.10 * w).tolist(),
    }


@pytest.fixture
def bounds_file(tmp_path, cold_start_data):
    x0, _ = cold_start_data
    per_coord = _narrow_box(x0)
    path = tmp_path / "bounds32.json"
    path.write_text(json.dumps({"per_coord": per_coord}))
    return (
        str(path),
        np.asarray(per_coord["p01"], dtype=np.float64),
        np.asarray(per_coord["p99"], dtype=np.float64),
    )


@pytest.fixture
def ts_spy(monkeypatch):
    """Record every ``MaxPosteriorSampling`` construction and call.

    Each record also carries the gpytorch ``max_cholesky_size`` setting active
    while the Thompson sample is drawn, which pins choice C5
    (``max_cholesky_size(inf)``; a mutation to gpytorch's default 800 must fail
    the suite even though the tests use fewer than 800 candidates).
    """
    real = tbf.MaxPosteriorSampling
    calls: list[dict] = []

    class Spy:
        def __init__(self, *, model, replacement):
            self._inner = real(model=model, replacement=replacement)
            self._rec = {"replacement": replacement}
            calls.append(self._rec)

        def __call__(self, X, num_samples):
            self._rec["candidates"] = X.detach().clone()
            self._rec["num_samples"] = num_samples
            self._rec["active_max_cholesky_size"] = gpytorch.settings.max_cholesky_size.value()
            out = self._inner(X, num_samples=num_samples)
            self._rec["selected"] = out.detach().clone()
            return out

    monkeypatch.setattr(tbf, "MaxPosteriorSampling", Spy, raising=True)
    return calls


def _make_solver(bounds_path, x0, y0, black_box, *, seed=7, **kwargs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    kwargs.setdefault("n_candidates", N_CAND_TEST)
    return TurboBoxFaithfulSolver(
        black_box=black_box, x0=x0, y0=y0, bounds_path=bounds_path, **kwargs,
    )


# --- published settings ----------------------------------------------------


def test_published_candidate_set_size_and_failure_tolerance():
    assert turbo_n_candidates(256) == 5000          # min(100*D, 5000)
    assert turbo_n_candidates(32) == 3200
    assert turbo_failure_tolerance(256) == 256      # ceil(max(4/b, D/b)), b = 1
    assert turbo_failure_tolerance(2) == 4
    assert (TURBO_LENGTH_INIT, TURBO_LENGTH_MIN, TURBO_LENGTH_MAX) == (
        0.8, 0.5**7, 1.6,
    )
    assert TURBO_SUCCESS_TOLERANCE == 3


def test_default_state_and_candidate_count_follow_the_paper(
    bounds_file, cold_start_data, mock_black_box,
):
    path, _, _ = bounds_file
    x0, y0 = cold_start_data
    solver = TurboBoxFaithfulSolver(
        black_box=mock_black_box, x0=x0, y0=y0, bounds_path=path,
    )
    D = x0.shape[1]
    assert solver.n_candidates == turbo_n_candidates(D)
    assert solver.turbo_state.length == TURBO_LENGTH_INIT
    assert solver.turbo_state.batch_size == 1
    assert solver.turbo_state.failure_tolerance == turbo_failure_tolerance(D)
    assert solver.turbo_state.success_tolerance == TURBO_SUCCESS_TOLERANCE
    assert solver.turbo_state.best_value == pytest.approx(float(np.max(y0)))


def test_gp_is_the_submitted_wrappers_gp(bounds_file, cold_start_data, mock_black_box):
    """The GP is the submitted TuRBO wrapper's (``turbo_baseline.py:232-240``).

    Lengthscale (0.005, 4.0) and noise (1e-8, 1e-3) are the BoTorch tutorial's,
    outputscale (0.05, 20.0) is Eriksson et al. 2019's; the combination is
    neither the tutorial GP nor the paper GP (paper: lengthscale (0.005, 2.0),
    noise (5e-4, 0.1)). Decision D2 keeps it so the camera-ready TuRBO row
    differs from the submitted one only in the box and duplicate accounting.
    """
    path, _, _ = bounds_file
    x0, y0 = cold_start_data
    solver = _make_solver(path, x0, y0, mock_black_box)
    D = x0.shape[1]
    X = torch.rand(8, D, dtype=torch.float64)
    Y = torch.randn(8, 1, dtype=torch.float64)
    gp = solver._make_gp(X, Y)

    base = gp.covar_module.base_kernel
    assert base.nu == 2.5
    assert base.ard_num_dims == D
    ls_c = base.raw_lengthscale_constraint
    assert float(ls_c.lower_bound) == pytest.approx(0.005)
    assert float(ls_c.upper_bound) == pytest.approx(4.0)
    os_c = gp.covar_module.raw_outputscale_constraint
    assert float(os_c.lower_bound) == pytest.approx(0.05)
    assert float(os_c.upper_bound) == pytest.approx(20.0)
    noise_c = gp.likelihood.noise_covar.raw_noise_constraint
    assert float(noise_c.lower_bound) == pytest.approx(1e-8)
    assert float(noise_c.upper_bound) == pytest.approx(1e-3)
    # The Hvarfner recipe is NOT inherited: constraints only, no priors.
    assert not list(base.named_priors())
    assert not list(gp.likelihood.named_priors())
    assert type(solver)._make_gp is not HvarfnerScaledGPSolver._make_gp

    # ...and it is literally the wrapper's constraint set (turbo_baseline.py).
    import inspect

    from src.spherevo import turbo_baseline

    wrapper_src = inspect.getsource(turbo_baseline.TuRBOBaseline)
    assert "GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))" in wrapper_src
    assert "lengthscale_constraint=Interval(0.005, 4.0)" in wrapper_src
    assert "outputscale_constraint=Interval(0.05, 20.0)" in wrapper_src
    assert tbf.TURBO_NOISE_INTERVAL == (1e-8, 1e-3)
    assert tbf.TURBO_LENGTHSCALE_INTERVAL == (0.005, 4.0)
    assert tbf.TURBO_OUTPUTSCALE_INTERVAL == (0.05, 20.0)


# --- harness contract ------------------------------------------------------


def test_runs_25_steps_with_one_thompson_sample_per_step(
    ts_spy, bounds_file, cold_start_data, mock_black_box,
):
    path, _, _ = bounds_file
    x0, y0 = cold_start_data
    solver = _make_solver(path, x0, y0, mock_black_box)
    solver.solve(max_iter=N_STEPS)

    assert solver.iteration == N_STEPS
    assert len(solver.all_scores) == len(y0) + N_STEPS
    assert solver._fallback_count == 0, solver._fallback_reasons
    assert len(ts_spy) == N_STEPS
    for rec in ts_spy:
        assert rec["replacement"] is False        # TuRBO draws without replacement
        assert rec["num_samples"] == 1            # TuRBO-1, batch size 1
        assert rec["candidates"].shape == (N_CAND_TEST, x0.shape[1])
        # choice C5: the Thompson sample comes from a Cholesky factor, never Lanczos
        assert rec["active_max_cholesky_size"] == float("inf")


def test_selected_point_is_one_of_the_thompson_candidates(
    ts_spy, bounds_file, cold_start_data, mock_black_box,
):
    path, _, _ = bounds_file
    x0, y0 = cold_start_data
    solver = _make_solver(path, x0, y0, mock_black_box)
    solver.solve(max_iter=3)

    for rec in ts_spy:
        cand, sel = rec["candidates"], rec["selected"].reshape(1, -1)
        assert ((cand == sel).all(dim=1)).any(), "selection left the candidate set"


def test_fallback_is_recorded_and_stays_in_the_box(
    monkeypatch, bounds_file, cold_start_data, mock_black_box,
):
    path, p01, p99 = bounds_file
    x0, y0 = cold_start_data
    solver = _make_solver(path, x0, y0, mock_black_box)

    def boom(*_a, **_k):
        raise tbf.ModelFittingError("induced numerical failure")

    monkeypatch.setattr(tbf, "fit_gpytorch_mll", boom, raising=True)
    z = solver.next_candidate()

    assert solver._fallback_count == 1
    assert "ModelFittingError" in solver._fallback_reasons[0]
    assert np.all(z >= p01 - 1e-9) and np.all(z <= p99 + 1e-9)


# --- the fix: a FIXED box ---------------------------------------------------


def test_box_comes_from_the_bounds_file_not_from_the_observations(
    bounds_file, cold_start_data, mock_black_box,
):
    path, p01, p99 = bounds_file
    x0, y0 = cold_start_data
    solver = _make_solver(path, x0, y0, mock_black_box)

    lo = torch.tensor(p01, dtype=torch.float64, device=solver._device)
    hi = torch.tensor(p99, dtype=torch.float64, device=solver._device)
    assert torch.allclose(solver._rescale_to_unit_cube(lo), torch.zeros_like(lo))
    assert torch.allclose(solver._rescale_to_unit_cube(hi), torch.ones_like(hi))
    # The cold start is NOT contained in [p01, p99]: the arm must not derive its
    # frame from the data (that is the submitted wrapper's defect).
    assert (x0.min(axis=0) < p01).any() or (x0.max(axis=0) > p99).any()


def test_unit_cube_frame_does_not_move_as_observations_arrive(
    bounds_file, cold_start_data, mock_black_box,
):
    path, _, _ = bounds_file
    x0, y0 = cold_start_data
    solver = _make_solver(path, x0, y0, mock_black_box)
    probe = torch.tensor(x0, dtype=torch.float64, device=solver._device)

    before = solver._rescale_to_unit_cube(probe).clone()
    solver.solve(max_iter=3)
    after = solver._rescale_to_unit_cube(probe)

    assert torch.equal(before, after)


def test_every_candidate_and_proposal_lies_inside_the_fixed_box(
    ts_spy, bounds_file, cold_start_data, mock_black_box,
):
    path, p01, p99 = bounds_file
    x0, y0 = cold_start_data
    solver = _make_solver(path, x0, y0, mock_black_box)
    solver.solve(max_iter=5)

    for rec in ts_spy:
        cand = rec["candidates"]
        assert float(cand.min()) >= 0.0 and float(cand.max()) <= 1.0

    x_hist, _ = solver.get_history_as_arrays()
    proposed = x_hist[len(y0):]           # cold start may lie outside the box
    assert len(proposed) == 5
    assert np.all(proposed >= p01 - 1e-9) and np.all(proposed <= p99 + 1e-9)


def test_trust_region_is_centred_on_the_clamped_incumbent(
    bounds_file, cold_start_data, mock_black_box,
):
    path, _, _ = bounds_file
    x0, y0 = cold_start_data
    solver = _make_solver(path, x0, y0, mock_black_box)

    X_unit = solver._rescale_to_unit_cube(
        torch.tensor(x0, dtype=torch.float64, device=solver._device)
    )
    Y = torch.tensor(
        y0.flatten(), dtype=torch.float64, device=solver._device,
    ).unsqueeze(-1)
    centre = solver._tr_centre(X_unit, Y)
    best = int(np.argmax(y0.flatten()))

    assert float(centre.min()) >= 0.0 and float(centre.max()) <= 1.0
    assert torch.equal(centre, X_unit[best].clamp(0.0, 1.0))
    # Cold-start encodings really do leave the box, so the clamp matters.
    assert float(X_unit.min()) < 0.0 or float(X_unit.max()) > 1.0


def test_trust_region_width_follows_the_lengthscales_and_the_tr_length(
    bounds_file, cold_start_data, mock_black_box,
):
    path, _, _ = bounds_file
    x0, y0 = cold_start_data
    solver = _make_solver(path, x0, y0, mock_black_box)
    D = x0.shape[1]

    X = torch.rand(8, D, dtype=torch.float64, device=solver._device)
    Y = torch.randn(8, 1, dtype=torch.float64, device=solver._device)
    gp = solver._make_gp(X, Y)
    centre = torch.full((D,), 0.5, dtype=torch.float64, device=solver._device)

    ls = gp.covar_module.base_kernel.lengthscale.detach().reshape(-1)
    w = ls / ls.mean()
    w = w / torch.prod(w.pow(1.0 / D))
    for length in (0.8, 0.4):
        solver.turbo_state.length = length
        lb, ub = solver.trust_region_bounds(gp, centre)
        expected_lb = torch.clamp(centre - w * length / 2.0, 0.0, 1.0)
        expected_ub = torch.clamp(centre + w * length / 2.0, 0.0, 1.0)
        assert torch.allclose(lb, expected_lb)
        assert torch.allclose(ub, expected_ub)
        assert float(lb.min()) >= 0.0 and float(ub.max()) <= 1.0


def test_perturbation_mask_leaves_most_coordinates_at_the_centre(
    bounds_file, cold_start_data, mock_black_box,
):
    """Each coordinate is perturbed with probability min(1, 20/D)."""
    path, _, _ = bounds_file
    x0, y0 = cold_start_data
    solver = _make_solver(path, x0, y0, mock_black_box, n_candidates=512)
    D = x0.shape[1]

    X = torch.rand(8, D, dtype=torch.float64, device=solver._device)
    Y = torch.randn(8, 1, dtype=torch.float64, device=solver._device)
    gp = solver._make_gp(X, Y)
    centre = torch.full((D,), 0.5, dtype=torch.float64, device=solver._device)

    cand = solver.generate_candidates(gp, centre)
    changed = (cand != centre).sum(dim=1)
    assert int(changed.min()) >= 1                       # never an empty mask
    expected = min(20.0 / D, 1.0) * D
    assert abs(float(changed.double().mean()) - expected) < 0.25 * expected


# --- trust-region update rules ---------------------------------------------


def test_turbo_state_doubles_after_three_successes_and_caps_at_length_max():
    state = TurboState(dim=32, batch_size=1, length=0.8)
    state.best_value = 0.0
    for i in range(TURBO_SUCCESS_TOLERANCE):
        state.update(1.0 + i)       # each a strict improvement
    assert state.length == pytest.approx(1.6)
    assert state.success_counter == 0
    state.best_value = 0.0
    for i in range(TURBO_SUCCESS_TOLERANCE):
        state.update(10.0 + i)
    assert state.length == pytest.approx(TURBO_LENGTH_MAX)  # min(2*1.6, 1.6)


def test_turbo_state_halves_after_failure_tolerance_and_restarts_at_the_floor():
    state = TurboState(dim=32, batch_size=1, length=0.8, failure_tolerance=2)
    state.best_value = 1.0
    state.update(0.5)
    assert state.length == pytest.approx(0.8) and state.failure_counter == 1
    state.update(0.5)
    assert state.length == pytest.approx(0.4) and state.failure_counter == 0
    assert not state.restart_triggered
    while not state.restart_triggered:
        state.update(0.5)
    assert state.length < TURBO_LENGTH_MIN


def test_turbo_state_counts_a_tie_as_a_failure():
    """Improvement needs y > best + 1e-3*|best| (Eriksson et al. 2019)."""
    state = TurboState(dim=32, batch_size=1, length=0.8, failure_tolerance=2)
    state.best_value = 1.0
    state.update(1.0)
    assert state.failure_counter == 1 and state.success_counter == 0


def test_solver_shrinks_and_restarts_the_trust_region_through_post_update(
    bounds_file, cold_start_data, mock_black_box,
):
    path, _, _ = bounds_file
    x0, y0 = cold_start_data
    # D = 32 gives failure_tolerance = 32 > 25 steps, so production runs can
    # never shrink (choice C7); lower it to exercise the branch.
    solver = _make_solver(path, x0, y0, mock_black_box, failure_tolerance=2)
    worse = np.array([[float(np.min(y0)) - 1.0]])
    x = np.zeros((1, x0.shape[1]))

    solver.post_update(x, worse)
    assert solver.turbo_state.length == pytest.approx(TURBO_LENGTH_INIT)
    solver.post_update(x, worse)
    assert solver.turbo_state.length == pytest.approx(TURBO_LENGTH_INIT / 2)

    while solver._tr_restarts == 0:
        solver.post_update(x, worse)
    # A restart resets the state and keeps the history (choice C8).
    assert solver.turbo_state.length == pytest.approx(TURBO_LENGTH_INIT)
    assert solver.turbo_state.restart_triggered is False
    assert solver._tr_restarts == 1


def test_nan_evaluation_counts_as_a_failure_and_is_not_the_incumbent(
    bounds_file, cold_start_data, mock_black_box,
):
    path, _, _ = bounds_file
    x0, y0 = cold_start_data
    solver = _make_solver(path, x0, y0, mock_black_box, failure_tolerance=1)
    before = solver.turbo_state.best_value

    solver.post_update(np.zeros((1, x0.shape[1])), np.array([[float("nan")]]))

    assert solver.turbo_state.best_value == pytest.approx(before)
    assert solver.turbo_state.length == pytest.approx(TURBO_LENGTH_INIT / 2)


def test_trust_region_length_is_doubled_after_real_improvements(
    bounds_file, cold_start_data, mock_black_box,
):
    path, _, _ = bounds_file
    x0, y0 = cold_start_data
    solver = _make_solver(path, x0, y0, mock_black_box)
    x = np.zeros((1, x0.shape[1]))
    base = float(np.max(y0))

    for i in range(TURBO_SUCCESS_TOLERANCE):
        solver.post_update(x, np.array([[base + 1.0 + i]]))
    assert solver.turbo_state.length == pytest.approx(2 * TURBO_LENGTH_INIT)


# --- determinism -----------------------------------------------------------


def test_same_seed_gives_identical_runs(bounds_file, cold_start_data, mock_black_box):
    path, _, _ = bounds_file
    x0, y0 = cold_start_data

    def run(seed):
        solver = _make_solver(path, x0, y0, mock_black_box, seed=seed)
        solver.solve(max_iter=4)
        x_hist, _ = solver.get_history_as_arrays()
        return list(solver.all_scores), x_hist.copy()

    scores_a, x_a = run(11)
    scores_b, x_b = run(11)
    scores_c, x_c = run(12)

    assert scores_a == scores_b
    assert np.array_equal(x_a, x_b)
    # A different seed must actually move the trajectory (otherwise the test
    # above would pass even for a solver that ignores its RNG).
    assert not np.array_equal(x_a[len(y0):], x_c[len(y0):])


# --- wiring ----------------------------------------------------------------


def test_reuses_the_box_faithful_domain_machinery():
    for name in ("_rescale_to_unit_cube", "_rescale_from_unit_cube"):
        assert getattr(TurboBoxFaithfulSolver, name) is getattr(
            HvarfnerScaledGPBoxFaithfulSolver, name
        )
    assert (
        TurboBoxFaithfulSolver.next_candidate
        is not HvarfnerScaledGPBoxFaithfulSolver.next_candidate
    )
    assert math.isclose(TURBO_LENGTH_MIN, 0.5**7)


def test_registered_in_the_benchmark_runner(
    monkeypatch, bounds_file, cold_start_data, mock_black_box,
):
    from scripts.run_benchmark import SUPPORTED_SOLVERS, _create_solver

    path, _, _ = bounds_file
    x0, y0 = cold_start_data
    # SELFIES codec path: _create_solver passes bounds_path=None, so the class
    # default is used; point it at the 32-D test bounds.
    monkeypatch.setattr(
        TurboBoxFaithfulSolver, "default_bounds_path", path, raising=True,
    )
    assert "turbo_box_faithful" in SUPPORTED_SOLVERS
    solver = _create_solver(
        solver_name="turbo_box_faithful", black_box=mock_black_box, x0=x0, y0=y0,
        n_dimensions=x0.shape[1], subspace_dim=16, n_subspaces=16, seed=42,
        n_initial_points=len(y0), codec=None, n_iterations=N_STEPS,
    )
    assert type(solver) is TurboBoxFaithfulSolver
    assert solver.n_candidates == turbo_n_candidates(x0.shape[1])
