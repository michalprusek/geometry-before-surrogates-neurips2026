"""The GP sphere-radius arms of the 2026-10-12 amendment resolve and propose
on the sphere their name says.

``hvarfner_scaled_gp_sphere05`` / ``_sphere15`` / ``_sphere20`` are
``hvarfner_scaled_gp`` (Hvarfner GP, qLogEI argmax over uniform candidates on a
sphere) with the radius set to 0.5 / 1.5 / 2.0 times the cold-start mean
latent norm. The multiplier is not a field of the result file; it is the
class attribute ``sphere_radius_multiplier`` and is encoded in the solver
name, so these tests pin that mapping.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.spherevo.gp_competitors import (
    HvarfnerScaledGPSphere05Solver,
    HvarfnerScaledGPSphere15Solver,
    HvarfnerScaledGPSphere20Solver,
)

ARMS = {
    "hvarfner_scaled_gp_sphere05": (HvarfnerScaledGPSphere05Solver, 0.5),
    "hvarfner_scaled_gp_sphere15": (HvarfnerScaledGPSphere15Solver, 1.5),
    "hvarfner_scaled_gp_sphere20": (HvarfnerScaledGPSphere20Solver, 2.0),
}


def _create(name, black_box, x0, y0):
    from scripts.run_benchmark import _create_solver

    return _create_solver(
        solver_name=name, black_box=black_box, x0=x0, y0=y0,
        n_dimensions=x0.shape[1], subspace_dim=16, n_subspaces=16, seed=42,
        n_initial_points=len(y0), codec=None, norm_mode="fixed", n_iterations=25,
        cs_smiles=None,
    )


@pytest.mark.parametrize("name", sorted(ARMS))
def test_the_arm_resolves_to_its_class_and_multiplier(name, cold_start_data, mock_black_box):
    from scripts.run_benchmark import OUR_SOLVERS, SUPPORTED_SOLVERS

    cls, multiplier = ARMS[name]
    assert name in OUR_SOLVERS and name in SUPPORTED_SOLVERS
    x0, y0 = cold_start_data
    solver = _create(name, mock_black_box, x0, y0)
    assert type(solver) is cls
    assert solver.sphere_radius_multiplier == multiplier
    assert float(name[-2:]) / 10 == multiplier          # the name encodes it
    assert solver.n_candidates == 2048                   # hvarfner_scaled_gp's candidate set


@pytest.mark.parametrize("name", sorted(ARMS))
def test_gp_proposals_lie_on_the_scaled_sphere(name, cold_start_data, mock_black_box):
    x0, y0 = cold_start_data
    solver = _create(name, mock_black_box, x0, y0)
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    np.random.seed(0)
    solver.solve(max_iter=2)
    radius = ARMS[name][1] * solver.mean_norm
    for z in solver.history["x"][len(x0):]:
        assert np.linalg.norm(z) == pytest.approx(radius, rel=1e-9)
    assert solver._fallback_count == 0


@pytest.mark.parametrize("name", sorted(ARMS))
def test_the_flat_signal_fallback_uses_the_same_radius(name, cold_start_data):
    x0, _ = cold_start_data
    y0 = np.full((len(x0), 1), 0.5)

    def flat(z):
        return np.full((np.atleast_2d(z).shape[0], 1), 0.5)

    solver = _create(name, flat, x0, y0)
    np.random.seed(0)
    solver.solve(max_iter=1)
    assert solver._fallback_count == 1 and solver._fallback_reasons[0].startswith("flat-signal")
    radius = ARMS[name][1] * solver.mean_norm
    assert np.linalg.norm(solver.history["x"][-1]) == pytest.approx(radius, rel=1e-9)
