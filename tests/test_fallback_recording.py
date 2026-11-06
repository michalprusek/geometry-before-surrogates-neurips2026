"""Every solver's GP fallbacks reach the result file (rule R-5).

Before the camera-ready successor commit, ``run_benchmark`` wrote only
``_fallback_count`` / ``_fallback_reasons`` (the GP competitors' counters), so
GaBO, Matern-GaBO, REMBO and HeSBO -- which count in ``gp_fallback_count`` --
always reported ``surrogate_fallback_count = 0``, and GaBO's bare
``except Exception`` turned any error into a silent random step. These tests
force every one of those except branches and check that the count and the
reason reach the solver, the shared helper and the result dict. The branches
themselves are unchanged (same fallback proposal, same RNG draws).
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.run_benchmark import _surrogate_fallbacks
from src.spherevo.riemannian_bo import GaBOSolver, MaternGaBOSolver
from src.spherevo.solver import GPGuidedSphericalSolver, HeSBOSolver, REMBOSolver

D = 32


def _raise(exc):
    def boom(*args, **kwargs):
        raise exc
    return boom


# --- the GaBO family: `except Exception` -> random sphere sample -------------


@pytest.mark.parametrize("cls", [GaBOSolver, MaternGaBOSolver])
@pytest.mark.parametrize("exc", [
    RuntimeError("forced GP failure"),
    ValueError("forced pymanopt failure"),
    np.linalg.LinAlgError("forced linalg failure"),
])
def test_gabo_family_records_every_fallback_and_its_reason(
    cls, exc, monkeypatch, cold_start_data, mock_black_box,
):
    x0, y0 = cold_start_data
    solver = cls(black_box=mock_black_box, x0=x0, y0=y0)
    monkeypatch.setattr(cls, "_build_gp_and_acq", _raise(exc), raising=True)
    np.random.seed(0)
    z = solver.next_candidate()

    assert z.shape == (1, D)
    assert np.linalg.norm(z) == pytest.approx(solver.mean_norm)  # random sphere sample
    reason = f"{type(exc).__name__}: {exc}"
    assert solver.gp_fallback_count == 1
    assert solver.gp_fallback_reasons == [reason]
    assert _surrogate_fallbacks(solver) == (1, [reason])


def test_gabo_family_still_reraises_memory_errors(monkeypatch, cold_start_data, mock_black_box):
    x0, y0 = cold_start_data
    solver = GaBOSolver(black_box=mock_black_box, x0=x0, y0=y0)
    monkeypatch.setattr(GaBOSolver, "_build_gp_and_acq", _raise(MemoryError()), raising=True)
    with pytest.raises(MemoryError):
        solver.next_candidate()
    assert solver.gp_fallback_count == 0 and solver.gp_fallback_reasons == []


def test_gabo_fallback_proposal_is_unchanged(monkeypatch, cold_start_data, mock_black_box):
    """Recording the reason draws no random numbers: same proposal as before."""
    x0, y0 = cold_start_data
    solver = GaBOSolver(black_box=mock_black_box, x0=x0, y0=y0)
    monkeypatch.setattr(GaBOSolver, "_build_gp_and_acq", _raise(RuntimeError("x")), raising=True)
    np.random.seed(9)
    z = solver.next_candidate()
    np.random.seed(9)
    ref = np.random.randn(D)
    ref = (ref / (np.linalg.norm(ref) + 1e-10) * solver.mean_norm).reshape(1, D)
    assert np.array_equal(z, ref)


# --- REMBO, HeSBO, GP-guided: `except (RuntimeError, LinAlgError)` -----------


@pytest.mark.parametrize("cls,kwargs", [
    (REMBOSolver, {"subspace_dim": 8}),
    (HeSBOSolver, {"subspace_dim": 8}),
    (GPGuidedSphericalSolver, {"n_subspaces": 4, "subspace_dim": 8}),
])
@pytest.mark.parametrize("exc", [
    RuntimeError("forced fit failure"),
    np.linalg.LinAlgError("forced linalg failure"),
])
def test_embedding_solvers_record_every_fallback_and_its_reason(
    cls, kwargs, exc, monkeypatch, cold_start_data, mock_black_box,
):
    import botorch.fit

    x0, y0 = cold_start_data
    np.random.seed(0)
    solver = cls(black_box=mock_black_box, x0=x0, y0=y0, **kwargs)
    # next_candidate does `from botorch.fit import fit_gpytorch_mll` at call time
    monkeypatch.setattr(botorch.fit, "fit_gpytorch_mll", _raise(exc), raising=True)
    for _ in range(2):
        z = solver.next_candidate()
        assert z.shape == (1, D) and np.all(np.isfinite(z))

    reason = f"{type(exc).__name__}: {exc}"
    assert solver.gp_fallback_count == 2
    assert solver.gp_fallback_reasons == [reason, reason]
    assert _surrogate_fallbacks(solver) == (2, [reason, reason])


# --- the shared helper --------------------------------------------------------


def test_helper_reads_the_gp_competitor_counters_too(cold_start_data, mock_black_box):
    from src.spherevo.gp_competitors import HvarfnerScaledGPSolver

    x0, y0 = cold_start_data
    flat = np.full_like(y0, 0.5)  # flat signal -> the competitor's own fallback
    solver = HvarfnerScaledGPSolver(black_box=mock_black_box, x0=x0, y0=flat)
    solver.next_candidate()
    count, reasons = _surrogate_fallbacks(solver)
    assert count == solver._fallback_count == 1
    assert reasons == solver._fallback_reasons and reasons[0].startswith("flat-signal")


def test_helper_on_a_solver_without_any_counter(cold_start_data, mock_black_box):
    from src.spherevo.solver import RandomSearchSolver

    x0, y0 = cold_start_data
    assert _surrogate_fallbacks(RandomSearchSolver(mock_black_box, x0, y0)) == (0, [])


# --- end to end: the result dict that run_benchmark returns ---------------------


class _BB:
    """Latent black box stub with the attributes run_benchmark reads."""

    def __init__(self, inner):
        self._inner = inner
        self.n_failures = 0

    def __call__(self, z):
        return self._inner(z)

    def terminate(self):
        pass


@pytest.mark.parametrize("solver_name", ["gabo", "matern_gabo", "rembo", "hesbo"])
def test_run_benchmark_writes_the_fallbacks_of_every_solver(
    solver_name, monkeypatch, cold_start_data, mock_black_box,
):
    import botorch.fit

    import scripts.run_benchmark as rb
    import src.oracle.guacamol as guacamol_mod

    x0, y0 = cold_start_data
    bb = _BB(mock_black_box)

    class StubOracle:
        def __init__(self, task_id):
            self.task_id = task_id

        def score(self, smiles):
            return 0.0

    monkeypatch.setattr(rb, "make_our_latent_bb", lambda *a, **k: (bb, None, D))
    monkeypatch.setattr(guacamol_mod, "GuacaMolOracle", StubOracle)
    monkeypatch.setattr(
        rb, "generate_cold_start_ours",
        lambda *a, **k: (x0.copy(), y0.copy(), ["C"] * len(y0)),
    )
    exc = RuntimeError(f"forced {solver_name} failure")
    if solver_name in ("gabo", "matern_gabo"):
        cls = GaBOSolver if solver_name == "gabo" else MaternGaBOSolver
        monkeypatch.setattr(cls, "_build_gp_and_acq", _raise(exc), raising=True)
    else:
        monkeypatch.setattr(botorch.fit, "fit_gpytorch_mll", _raise(exc), raising=True)

    res = rb.run_benchmark(
        mode="ours", solver_name=solver_name, task_name="ranolazine_mpo", seed=42,
        max_iter=3, n_initial_points=10, n_cold_start=len(y0), latent_dim=128,
        subspace_dim=8, n_subspaces=16,
    )
    assert res["status"] == "completed", res["crash_info"]
    assert res["n_evaluated"] == len(y0) + 3
    assert res["surrogate_fallback_count"] == 3
    assert res["surrogate_fallback_reasons"] == [f"RuntimeError: {exc}"] * 3
