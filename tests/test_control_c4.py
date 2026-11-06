"""Control C4 of the camera-ready run rules (amendment of 2026-10-14):
``hvarfner_scaled_gp_sphere_lbfgs``, the shell x continuous cell of the
decomposition with the maximiser of the box arm ``hvarfner_scaled_gp_box_faithful``.

Pinned here: the arm resolves to its class and keeps the Adam arm's
(``hvarfner_scaled_gp_sphere_continuous``) constructor, GP, radius and
fallback point; ``next_candidate`` is the Adam arm's except the caught tuple;
every proposal lies on the shell; the maximiser is the box arm's
``optimize_acqf`` call (same q, restarts and raw samples, no options), with
infinite bounds on v (``gen_candidates_scipy`` gets no bounds, the same batch
of 10 starts and the same options), Gaussian raw samples and the box arm's
start rule (``initialize_q_batch`` with its default eta); the composed
acquisition is scale-invariant in v with a tangential gradient; the result is
at least as good as the best raw sample; errors fall back as in the box arm;
a GP step uses torch's CPU generator only and is reproducible.

The problem is the deterministic one of ``tests/test_control_block.py``
(cold start inside a box, smooth black box, D = 32).
"""
from __future__ import annotations

import difflib
import inspect
import json
import textwrap

import numpy as np
import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement

from src.spherevo import gp_competitors as gc
from src.spherevo.gp_competitors import (
    HvarfnerScaledGPBoxFaithfulSolver,
    HvarfnerScaledGPSolver,
    HvarfnerScaledGPSphereContinuousSolver,
    HvarfnerScaledGPSphereLBFGSSolver,
    _ShellAcquisition,
)

C4 = HvarfnerScaledGPSphereLBFGSSolver
ADAM = HvarfnerScaledGPSphereContinuousSolver
BOX = HvarfnerScaledGPBoxFaithfulSolver
NAME = "hvarfner_scaled_gp_sphere_lbfgs"
D = 32
N0 = 5


def black_box(z: np.ndarray) -> np.ndarray:
    """Deterministic, smooth, not flat."""
    z = np.atleast_2d(z)
    return np.sin(z[:, :4].sum(axis=1)).reshape(-1, 1)


@pytest.fixture
def problem(tmp_path):
    rng = np.random.RandomState(0)
    p01 = rng.uniform(-3.0, -0.2, size=D)
    p99 = p01 + rng.uniform(0.05, 4.0, size=D)
    path = tmp_path / "bounds.json"
    path.write_text(json.dumps({"per_coord": {
        "p01": p01.tolist(), "p99": p99.tolist(),
        "min": (p01 - 1).tolist(), "max": (p99 + 1).tolist(),
    }}))
    x0 = p01 + np.random.RandomState(100).uniform(0.1, 0.9, size=(N0, D)) * (p99 - p01)
    return {"x0": x0, "y0": black_box(x0), "bp": str(path)}


@pytest.fixture
def fit_retries(monkeypatch):
    """Calls of BoTorch's ``sample_all_priors`` (a fit retry), recorded."""
    import botorch.fit

    calls = []
    original = botorch.fit.sample_all_priors

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(botorch.fit, "sample_all_priors", spy)
    return calls


def _c4(problem, *, bb=black_box, y0=None):
    return C4(black_box=bb, x0=problem["x0"], y0=problem["y0"] if y0 is None else y0)


def _spy(monkeypatch, module, name):
    """Replace ``module.name`` by a recorder that calls the original."""
    calls = []
    original = getattr(module, name)

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        calls.append({"args": args, "kwargs": kwargs, "out": out})
        return out

    monkeypatch.setattr(module, name, spy)
    return calls


def _on_shell(z, radius) -> bool:
    return np.linalg.norm(z) == pytest.approx(radius, rel=1e-12)


# =========================================================================
# registration and construction
# =========================================================================


def test_the_arm_resolves_to_its_class_and_proposes_on_the_shell(problem):
    from scripts.run_benchmark import (
        OUR_SOLVERS, SUPPORTED_SOLVERS, _create_solver, _surrogate_fallbacks,
    )

    assert NAME in OUR_SOLVERS and NAME in SUPPORTED_SOLVERS
    s = _create_solver(
        solver_name=NAME, black_box=black_box, x0=problem["x0"], y0=problem["y0"],
        n_dimensions=D, subspace_dim=16, n_subspaces=16, seed=42,
        n_initial_points=N0, codec=None, norm_mode="fixed", n_iterations=25,
        cs_smiles=None,
    )
    assert type(s) is C4
    torch.manual_seed(0)
    np.random.seed(0)
    s.solve(max_iter=3)
    assert len(s.all_scores) == N0 + 3
    assert _surrogate_fallbacks(s) == (0, [])
    for z in s.history["x"][N0:]:
        assert _on_shell(z, s.mean_norm)


def test_c4_keeps_the_adam_arms_constructor_gp_radius_and_counts(problem):
    assert issubclass(C4, HvarfnerScaledGPSolver) and not issubclass(C4, ADAM)
    assert C4.__init__ is ADAM.__init__
    assert C4._make_gp is ADAM._make_gp is HvarfnerScaledGPSolver._make_gp
    assert (C4.n_raw_samples, C4.n_restarts) == (ADAM.n_raw_samples, ADAM.n_restarts) == (512, 10)

    def methods(cls):
        return {k for k, v in vars(cls).items() if callable(v) and not k.startswith("__")}

    assert methods(C4) == {"_maximize_on_sphere", "next_candidate"}
    a = ADAM(black_box=black_box, x0=problem["x0"], y0=problem["y0"])
    c = _c4(problem)
    assert a.mean_norm == c.mean_norm and a._device == c._device


def test_next_candidate_is_the_adam_arms_except_the_caught_tuple():
    def body(cls):
        return textwrap.dedent(inspect.getsource(cls.next_candidate)).splitlines()

    diff = [line for line in difflib.unified_diff(body(ADAM), body(C4), lineterm="", n=0)
            if line[:1] in "+-" and not line.startswith(("+++", "---"))]
    assert diff == ["-    except GP_FIT_ERRORS as e:",
                    "+    except _BOX_FALLBACK_ERRORS as e:"]


# =========================================================================
# the maximiser: the box arm's optimize_acqf call
# =========================================================================


def test_the_maximiser_is_the_box_arms_optimize_acqf_call(problem, monkeypatch, fit_retries):
    import botorch.optim.initializers as bo_init
    import botorch.optim.optimize as bo_opt

    acqf_calls = _spy(monkeypatch, gc, "optimize_acqf")
    gen_calls = _spy(monkeypatch, bo_opt, "gen_candidates_scipy")
    init_calls = _spy(monkeypatch, bo_init, "initialize_q_batch")
    other_init = [_spy(monkeypatch, bo_init, n)
                  for n in ("initialize_q_batch_nonneg", "initialize_q_batch_topn")]

    box = BOX(black_box=black_box, x0=problem["x0"], y0=problem["y0"], bounds_path=problem["bp"])
    torch.manual_seed(0)
    box.next_candidate()
    n_box_gen, n_box_init = len(gen_calls), len(init_calls)
    c4 = _c4(problem)
    torch.manual_seed(0)
    z = c4.next_candidate()
    assert not fit_retries
    assert box._fallback_count == c4._fallback_count == 0
    assert _on_shell(z, c4.mean_norm)

    # optimize_acqf: the same q, restarts and raw samples, no options
    (kb, kc) = (acqf_calls[0]["kwargs"], acqf_calls[1]["kwargs"])
    assert len(acqf_calls) == 2
    shared = {"q": 1, "num_restarts": 10, "raw_samples": 512}
    assert {k: kb[k] for k in shared} == {k: kc[k] for k in shared} == shared
    assert set(kb) == {"acq_function", "bounds", *shared}
    assert set(kc) == {"acq_function", "bounds", "ic_generator", *shared}
    acqf = kc["acq_function"]
    assert isinstance(acqf, _ShellAcquisition) and acqf.radius == c4.mean_norm
    assert type(acqf.inner) is type(kb["acq_function"]) is qLogExpectedImprovement
    assert kc["bounds"].shape == (2, D) and torch.isinf(kc["bounds"]).all()
    assert (kc["bounds"][0] < 0).all() and (kc["bounds"][1] > 0).all()

    # gen_candidates_scipy: one batch of the 10 starts, the same (default)
    # options; the box arm's [0, 1] bounds, none for v
    for calls in (gen_calls[:n_box_gen], gen_calls[n_box_gen:]):
        assert calls
        for call in calls:
            assert call["args"][0].shape == (10, 1, D)
            assert call["kwargs"]["options"] == {}
            assert call["kwargs"]["timeout_sec"] is None
    for call in gen_calls[:n_box_gen]:
        assert torch.equal(call["kwargs"]["lower_bounds"].cpu(), torch.zeros(D, dtype=torch.float64))
        assert torch.equal(call["kwargs"]["upper_bounds"].cpu(), torch.ones(D, dtype=torch.float64))
    for call in gen_calls[n_box_gen:]:
        assert call["kwargs"]["lower_bounds"] is None and call["kwargs"]["upper_bounds"] is None

    # the start rule: initialize_q_batch with its default eta in both arms
    assert all(not calls for calls in other_init)
    for calls in (init_calls[:n_box_init], init_calls[n_box_init:]):
        assert calls
        for call in calls:
            assert set(call["kwargs"]) == {"X", "acq_vals", "n"} and call["kwargs"]["n"] == 10
            assert call["kwargs"]["X"].shape == (512, 1, D)
    X_box = init_calls[0]["kwargs"]["X"]
    assert (X_box >= 0).all() and (X_box <= 1).all()
    # C4's raw samples are i.i.d. standard Gaussian v (uniform directions), not shell points
    V = init_calls[-1]["kwargs"]["X"].reshape(512, D)
    assert V.dtype == torch.float64 and V.device.type == "cpu"
    assert abs(float(V.mean())) < 0.05
    assert abs(float(V.pow(2).sum(-1).mean()) - D) < 0.1 * D
    assert float(V.norm(dim=-1).std()) > 0.1


def test_the_result_is_at_least_as_good_as_the_best_raw_sample(problem, monkeypatch):
    import botorch.optim.initializers as bo_init

    acqf_calls = _spy(monkeypatch, gc, "optimize_acqf")
    init_calls = _spy(monkeypatch, bo_init, "initialize_q_batch")
    c4 = _c4(problem)
    torch.manual_seed(1)
    c4.next_candidate()
    _, value = acqf_calls[-1]["out"]
    best_raw = float(init_calls[-1]["kwargs"]["acq_vals"].max())
    # the starts include the best raw sample and L-BFGS-B does not go downhill
    assert float(value) >= best_raw - 1e-9
    v_starts = init_calls[-1]["out"][0].reshape(10, D)
    assert any(torch.equal(v, init_calls[-1]["kwargs"]["X"].reshape(512, D)[
        int(init_calls[-1]["kwargs"]["acq_vals"].argmax())]) for v in v_starts)


def test_the_shell_acquisition_is_scale_invariant_with_a_tangential_gradient(problem):
    s = _c4(problem)
    X = torch.tensor(problem["x0"], dtype=torch.float64, device=s._device)
    Y = torch.tensor(problem["y0"].flatten(), dtype=torch.float64, device=s._device).unsqueeze(-1)
    Y_std, *_ = gc._standardize(Y)
    gp = s._make_gp(X, Y_std)
    gp.eval()
    acqf = _ShellAcquisition(qLogExpectedImprovement(model=gp, best_f=float(Y_std.max())),
                             radius=s.mean_norm)
    torch.manual_seed(3)
    v = torch.randn(6, 1, D, dtype=torch.float64, device=s._device, requires_grad=True)
    val = acqf(v)
    val.sum().backward()
    g = v.grad
    cos = ((g * v).sum(-1) / (g.norm(dim=-1) * v.norm(dim=-1))).detach()
    assert float(cos.abs().max()) < 1e-10
    with torch.no_grad():
        x = acqf.to_shell(v)
        assert torch.allclose(x.norm(dim=-1), torch.full_like(x.norm(dim=-1), s.mean_norm),
                              rtol=1e-12, atol=0)
        assert torch.equal(acqf(v), acqf.inner(x))
        assert torch.allclose(acqf(3.7 * v), val, rtol=0, atol=1e-10)
        # the same operation order as the Adam arm's to_sphere
        assert torch.equal(x, v / v.norm(dim=-1, keepdim=True) * s.mean_norm)


# =========================================================================
# fallbacks and random numbers
# =========================================================================


def test_the_flat_signal_fallback_is_one_point_on_the_shell(problem):
    y0 = np.full((N0, 1), 0.5)

    def flat(z):
        return np.full((np.atleast_2d(z).shape[0], 1), 0.5)

    s = _c4(problem, bb=flat, y0=y0)
    np.random.seed(2)
    s.solve(max_iter=1)
    assert s._fallback_count == 1 and s._fallback_reasons[0].startswith("flat-signal")
    assert _on_shell(s.history["x"][-1], s.mean_norm)


@pytest.mark.parametrize("failure,reason", [
    ("runtime", "ModelFittingError: optimize_acqf failed: forced"),
    ("nonfinite", "ModelFittingError: optimize_acqf returned non-finite candidate"),
    ("torch_linalg", "LinAlgError: forced"),
])
def test_a_numerical_failure_is_a_recorded_fallback_on_the_shell(problem, monkeypatch,
                                                                failure, reason):
    from scripts.run_benchmark import _surrogate_fallbacks

    if failure == "runtime":
        def broken(**kwargs):
            raise RuntimeError("forced")
        monkeypatch.setattr(gc, "optimize_acqf", broken)
    elif failure == "nonfinite":
        def broken(**kwargs):
            return torch.full((1, D), float("nan"), dtype=torch.float64), torch.tensor(0.0)
        monkeypatch.setattr(gc, "optimize_acqf", broken)
    else:
        def broken(self, X, Y):
            raise torch.linalg.LinAlgError("forced")
        monkeypatch.setattr(HvarfnerScaledGPSolver, "_make_gp", broken)
    s = _c4(problem)
    np.random.seed(0)
    torch.manual_seed(0)
    s.solve(max_iter=2)
    count, reasons = _surrogate_fallbacks(s)
    assert count == 2 and all(r.endswith(reason) for r in reasons), reasons
    for z in s.history["x"][N0:]:
        assert _on_shell(z, s.mean_norm)


def test_torchs_linalg_error_is_caught_as_in_the_box_arm_not_the_adam_arm(problem, monkeypatch):
    def broken(self, X, Y):
        raise torch.linalg.LinAlgError("forced")

    monkeypatch.setattr(HvarfnerScaledGPSolver, "_make_gp", broken)
    with pytest.raises(torch.linalg.LinAlgError):
        ADAM(black_box=black_box, x0=problem["x0"], y0=problem["y0"]).next_candidate()
    assert torch.linalg.LinAlgError in gc._BOX_FALLBACK_ERRORS
    assert torch.linalg.LinAlgError not in gc.GP_FIT_ERRORS


def test_a_programming_error_propagates(problem, monkeypatch):
    def bug(self, X, Y):
        raise TypeError("a programming error")

    monkeypatch.setattr(HvarfnerScaledGPSolver, "_make_gp", bug)
    with pytest.raises(TypeError, match="programming error"):
        _c4(problem).next_candidate()


def test_a_gp_step_uses_torchs_cpu_generator_only_and_is_reproducible(problem, fit_retries):
    zs = []
    for _ in range(2):
        s = _c4(problem)
        torch.manual_seed(7)
        np.random.seed(7)
        np_state = np.random.get_state()[1].copy()
        cpu_state = torch.get_rng_state()
        dev_state = (torch.cuda.get_rng_state(s._device)
                     if s._device.type == "cuda" else None)
        zs.append(s.next_candidate())
        assert s._fallback_count == 0 and not fit_retries
        assert np.array_equal(np.random.get_state()[1], np_state)
        assert not torch.equal(torch.get_rng_state(), cpu_state)
        if dev_state is not None:
            assert torch.equal(torch.cuda.get_rng_state(s._device), dev_state)
    assert np.array_equal(zs[0], zs[1])
