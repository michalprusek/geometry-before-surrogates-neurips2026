"""Published Papenmeier MSR arms: papenmeier_msr_published[_norm_box].

The camera-ready prints a row labelled "MSR" from these arms, so the tests pin
the acquisition step to the published call (reference implementation
github.com/lpapenme/understanding-hdbo-code, src/understanding_hdbo/bo.py
lines 170-190): analytic LogEI, q=1, 5 restarts, 512 raw samples and options
batch_limit=1, sample_around_best=True, sample_around_best_sigma=1e-3, plus
L-BFGS-B maxiter 2000 (BoTorch's default, passed explicitly).

dim=32 MockBlackBox. The 256-D ZINC bounds files are not tracked in git, so
every test writes or patches in its own 32-D bounds.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

import gpytorch  # noqa: E402

from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.exceptions.errors import BotorchError
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf as real_optimize_acqf
from botorch.optim.initializers import sample_points_around_best
from gpytorch.mlls import ExactMarginalLogLikelihood

from src.spherevo import gp_competitors
from src.spherevo.gp_competitors import (
    MSR_PUBLISHED_MAXITER,
    PapenmeierMSRBoxFaithfulSolver,
    PapenmeierMSRPublishedNormBoxSolver,
    PapenmeierMSRPublishedSolver,
    PapenmeierMSRSolver,
    _BoxFaithfulMixin,
)

N_STEPS = 25  # paper protocol: 5 cold-start points + 25 steps = N = 30
PUBLISHED_OPTIONS = {
    "batch_limit": 1,
    "sample_around_best": True,
    "sample_around_best_sigma": 1e-3,
    "maxiter": 2000,
}


def _box_from(x0: np.ndarray) -> dict:
    """32-D per-coordinate bounds derived from the cold start.

    [min, max] contains every cold-start point (as the ZINC [min, max] box
    contains ZINC encodings). [p01, p99] is NARROWER than the cold start, so
    some cold-start coordinates fall outside it -- the case in which the RAASP
    centres have to be projected onto the box.
    """
    lo, hi = x0.min(axis=0), x0.max(axis=0)
    w = hi - lo
    return {
        "min": (lo - 0.05 * w).tolist(),
        "max": (hi + 0.05 * w).tolist(),
        "p01": (lo + 0.10 * w).tolist(),
        "p99": (hi - 0.10 * w).tolist(),
    }


@pytest.fixture
def acqf_spy(monkeypatch):
    """Record every optimize_acqf call made by gp_competitors, then delegate.

    Each record also carries the gpytorch ``cholesky_max_tries`` setting that is
    active during the call, so the published ``cholesky_max_tries(9)`` context
    is pinned (a mutation 9 -> 3 must fail the suite).
    """
    calls: list[dict] = []

    def spy(*args, **kwargs):
        assert not args, "the published arm must call optimize_acqf by keyword"
        rec = dict(kwargs)
        rec["active_cholesky_max_tries"] = gpytorch.settings.cholesky_max_tries.value()
        calls.append(rec)
        return real_optimize_acqf(**kwargs)

    monkeypatch.setattr(gp_competitors, "optimize_acqf", spy, raising=True)
    return calls


def _make_arm(arm, x0, y0, black_box, tmp_path, monkeypatch):
    """Instantiate one published arm on 32-D bounds; return (solver, low, high)."""
    per_coord = _box_from(x0)
    if arm == "published":
        path = tmp_path / "bounds32.json"
        path.write_text(json.dumps({"per_coord": per_coord}))
        solver = PapenmeierMSRPublishedSolver(
            black_box=black_box, x0=x0, y0=y0, bounds_path=str(path),
        )
        return solver, np.asarray(per_coord["p01"]), np.asarray(per_coord["p99"])
    low, high = np.asarray(per_coord["min"]), np.asarray(per_coord["max"])
    monkeypatch.setattr(
        gp_competitors, "_load_zinc_per_coord_bounds",
        lambda *, mode="minmax": (low, high), raising=True,
    )
    solver = PapenmeierMSRPublishedNormBoxSolver(black_box=black_box, x0=x0, y0=y0)
    return solver, low, high


@pytest.mark.parametrize("arm", ["published", "published_norm_box"])
def test_runs_25_steps_with_the_published_acquisition_call(
    arm, acqf_spy, monkeypatch, tmp_path, mock_black_box, cold_start_data,
):
    x0, y0 = cold_start_data
    D = x0.shape[1]
    torch.manual_seed(0)
    np.random.seed(0)
    solver, low, high = _make_arm(arm, x0, y0, mock_black_box, tmp_path, monkeypatch)
    solver.solve(max_iter=N_STEPS)

    assert solver.iteration == N_STEPS
    assert len(solver.all_scores) == len(y0) + N_STEPS
    assert solver._fallback_count == 0, solver._fallback_reasons
    assert len(acqf_spy) == N_STEPS  # one published optimize_acqf call per step

    unit = torch.stack([
        torch.zeros(D, dtype=torch.float64), torch.ones(D, dtype=torch.float64),
    ])
    for kw in acqf_spy:
        # analytic LogEI (published), not the MC qLogEI of the box_faithful arm
        assert type(kw["acq_function"]) is LogExpectedImprovement
        assert kw["q"] == 1
        assert kw["num_restarts"] == 5
        assert kw["raw_samples"] == 512
        assert kw["options"] == PUBLISHED_OPTIONS
        assert kw["options"]["sample_around_best"] is True
        assert kw["options"]["sample_around_best_sigma"] == 1e-3
        assert MSR_PUBLISHED_MAXITER == 2000
        assert torch.equal(kw["bounds"].cpu(), unit)
        # the reference wraps the acquisition call in cholesky_max_tries(9)
        assert kw["active_cholesky_max_tries"] == 9

    # every proposal is inside the box the arm optimises over
    for z in solver.history["x"][len(y0):]:
        u = (np.asarray(z).ravel() - low) / (high - low)
        assert u.min() >= -1e-9 and u.max() <= 1.0 + 1e-9


def test_raasp_centres_are_the_observations_projected_onto_the_box(
    acqf_spy, monkeypatch, tmp_path, mock_black_box, cold_start_data,
):
    x0, y0 = cold_start_data
    torch.manual_seed(1)
    solver, _, _ = _make_arm("published", x0, y0, mock_black_box, tmp_path, monkeypatch)
    solver.step()

    acqf = acqf_spy[0]["acq_function"]
    X_train = acqf.model.train_inputs[0]
    # the GP itself is fitted on the unprojected data, as in box_faithful ...
    assert ((X_train < 0) | (X_train > 1)).any(), "test needs out-of-box cold start"
    # ... and only the RAASP centres are projected
    assert torch.equal(acqf.X_baseline, X_train.clamp(0.0, 1.0))
    assert solver._fallback_count == 0


def test_without_projection_botorch_breaks_on_out_of_box_centres():
    """Why the projection exists: BoTorch's RAASP assumes centres inside the
    box. With a centre coordinate at 1.3 the truncated-normal perturbations put
    it at 0.0 (opposite face) and the subset perturbations leave it at 1.3,
    which gen_candidates_scipy (botorch 0.17.2) rejects as a starting point.

    Deterministic on any BLAS: the centre is pinned via X_baseline (no
    dependence on the GP's posterior-mean ranking), the face jump is exact
    float64 arithmetic (Phi(-300) == 0, icdf(0) == -inf, clamp -> 0.0), and the
    rejection is the unconditional bounds check on supplied starting points.
    """
    D, N = 32, 10
    g = torch.Generator().manual_seed(0)
    X = torch.rand(N, D, generator=g, dtype=torch.float64)
    X[0, :3] = torch.tensor([1.3, 1.05, 1.6], dtype=torch.float64)
    X[0, 3:5] = torch.tensor([-0.2, -0.5], dtype=torch.float64)
    Y = torch.randn(N, 1, generator=g, dtype=torch.float64)
    Y[0] = Y.max() + 2.0
    Y = (Y - Y.mean()) / Y.std()
    gp = PapenmeierMSRSolver._make_gp(object.__new__(PapenmeierMSRSolver), X, Y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    gp.eval()
    best_f = float(Y.max())
    bounds = torch.stack([
        torch.zeros(D, dtype=torch.float64), torch.ones(D, dtype=torch.float64),
    ])

    # (a) RAASP around the out-of-box observation, as BoTorch does it
    torch.manual_seed(0)
    raw = LogExpectedImprovement(model=gp, best_f=best_f)
    raw.X_baseline = X[:1]
    pts = sample_points_around_best(raw, n_discrete_points=512, sigma=1e-3, bounds=bounds)
    out = ((pts < 0) | (pts > 1)).any(dim=-1)
    assert int(out.sum()) >= 5                     # subset half keeps 1.3, -0.5, ...
    assert int((pts[:, 0] == 0.0).sum()) >= 256    # truncated-normal half: 1.3 -> 0.0

    # (b) such a point as an L-BFGS-B starting point is rejected
    with pytest.raises(BotorchError, match="out of bounds"):
        real_optimize_acqf(
            acq_function=LogExpectedImprovement(model=gp, best_f=best_f),
            bounds=bounds, q=1, num_restarts=5, raw_samples=512,
            options=dict(PUBLISHED_OPTIONS),
            batch_initial_conditions=pts[out][:5].unsqueeze(1),
        )

    # (c) with the centre projected onto the box everything stays inside it
    torch.manual_seed(0)
    projected = LogExpectedImprovement(model=gp, best_f=best_f)
    projected.X_baseline = X[:1].clamp(0.0, 1.0)
    pts = sample_points_around_best(projected, n_discrete_points=512, sigma=1e-3, bounds=bounds)
    assert ((pts >= 0) & (pts <= 1)).all()
    assert int((pts[:, 0] == 0.0).sum()) == 0      # perturbed around 1.0, not 0.0

    # (d) and the full published call with every observation projected (what
    # the solver does) runs: no starting point can leave the box any more
    for seed in range(3):
        torch.manual_seed(seed)
        acqf = LogExpectedImprovement(model=gp, best_f=best_f)
        acqf.X_baseline = X.clamp(0.0, 1.0)
        cand, _ = real_optimize_acqf(
            acq_function=acqf, bounds=bounds, q=1, num_restarts=5,
            raw_samples=512, options=dict(PUBLISHED_OPTIONS),
        )
        assert ((cand >= 0) & (cand <= 1)).all()


def test_flat_signal_falls_back_in_the_box_without_optimising(
    acqf_spy, monkeypatch, tmp_path, mock_black_box, cold_start_data,
):
    x0, _ = cold_start_data
    y_flat = np.full((x0.shape[0], 1), 0.5)
    for arm in ("published", "published_norm_box"):
        solver, low, high = _make_arm(arm, x0, y_flat, mock_black_box, tmp_path, monkeypatch)
        z = np.asarray(solver.next_candidate()).ravel()
        u = (z - low) / (high - low)
        assert u.min() >= 0.0 and u.max() <= 1.0
        assert solver._fallback_count == 1
        assert solver._fallback_reasons[0].startswith("flat-signal")
    assert acqf_spy == []


def test_only_the_acquisition_step_differs_from_the_existing_box_arms():
    # same GP construction and the same box rescaling as box_faithful
    assert issubclass(PapenmeierMSRPublishedSolver, _BoxFaithfulMixin)
    assert issubclass(PapenmeierMSRPublishedSolver, PapenmeierMSRSolver)
    assert PapenmeierMSRPublishedSolver._make_gp is PapenmeierMSRSolver._make_gp
    assert PapenmeierMSRPublishedNormBoxSolver._make_gp is PapenmeierMSRSolver._make_gp
    for name in ("__init__", "_rescale_to_unit_cube", "_rescale_from_unit_cube"):
        assert getattr(PapenmeierMSRPublishedSolver, name) is getattr(
            PapenmeierMSRBoxFaithfulSolver, name
        )
    assert (
        PapenmeierMSRPublishedSolver.next_candidate
        is not PapenmeierMSRBoxFaithfulSolver.next_candidate
    )


def test_registered_in_the_benchmark_runner(monkeypatch, tmp_path, mock_black_box, cold_start_data):
    from scripts.run_benchmark import SUPPORTED_SOLVERS, _create_solver

    x0, y0 = cold_start_data
    path = tmp_path / "bounds32.json"
    path.write_text(json.dumps({"per_coord": _box_from(x0)}))
    # SELFIES codec path: _create_solver passes bounds_path=None, so the class
    # default is used; point it at the 32-D test bounds.
    monkeypatch.setattr(
        PapenmeierMSRPublishedSolver, "default_bounds_path", str(path), raising=True,
    )
    common = dict(
        black_box=mock_black_box, x0=x0, y0=y0, n_dimensions=x0.shape[1],
        subspace_dim=16, n_subspaces=16, seed=42, n_initial_points=len(y0),
        codec=None, n_iterations=N_STEPS,
    )
    for name, cls in (
        ("papenmeier_msr_published", PapenmeierMSRPublishedSolver),
        ("papenmeier_msr_published_norm_box", PapenmeierMSRPublishedNormBoxSolver),
    ):
        assert name in SUPPORTED_SOLVERS
        assert type(_create_solver(solver_name=name, **common)) is cls
