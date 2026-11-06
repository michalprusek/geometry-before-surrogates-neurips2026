"""Device-safe GP priors on CUDA (camera-ready commit, decisions D1 and D5).

gpytorch's ``LogNormalPrior`` / ``HalfNormalPrior`` keep their parameters in
``base_dist``; ``Module.to(device)`` moves only the registered ``_buffered_*``
copies, so after ``gp.to(cuda)`` a plain prior still samples on the CPU.
BoTorch samples priors only in ``sample_all_priors``, i.e. when a
``fit_gpytorch_mll`` attempt fails and the fit is retried; assigning the CPU
sample to a CUDA hyperparameter raised "Expected all tensors to be on the same
device" (21 crashed ``hvarfner_scaled_gp_box_faithful`` cells in E-1). The fix
(``_DeviceSafeLogNormalPrior`` / ``_DeviceSafeHalfNormalPrior``) moves only the
sample. These tests need a GPU (on CPU ``.to(cpu)`` is a no-op and the bug is
invisible) and are skipped without one:

* every device-safe prior of the Hvarfner GP, the Linear BO GP (D1) and the
  Geodesic-Slice GP (D1) samples on the GP's device, and ``sample_all_priors``
  runs;
* a fit whose first attempt raises ``NotPSDError`` is retried and succeeds;
* at solver level the retry yields a candidate, not a fallback and not a crash;
* a first-attempt fit is bitwise identical with the plain and the device-safe
  prior (the fix changes nothing for a cell that never retries).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

from botorch.acquisition.analytic import LogExpectedImprovement  # noqa: E402
from botorch.fit import fit_gpytorch_mll  # noqa: E402
from botorch.models import SingleTaskGP  # noqa: E402
from botorch.optim.fit import fit_gpytorch_mll_scipy  # noqa: E402
from botorch.optim.utils import sample_all_priors  # noqa: E402
from gpytorch.constraints import GreaterThan, Interval  # noqa: E402
from gpytorch.kernels import MaternKernel, ScaleKernel  # noqa: E402
from gpytorch.likelihoods import GaussianLikelihood  # noqa: E402
from gpytorch.means import ConstantMean  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: E402
from gpytorch.priors import GammaPrior, LogNormalPrior  # noqa: E402
from linear_operator.utils.errors import NotPSDError  # noqa: E402

from src.spherevo import gp_competitors as gc  # noqa: E402

DEV = torch.device("cuda")
D = 256


def _data(n: int = 12, d: int = D, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, d, generator=g, dtype=torch.float64).to(DEV)
    Y = torch.randn(n, 1, generator=g, dtype=torch.float64).to(DEV)
    return X, (Y - Y.mean()) / Y.std()


def _hvarfner_gp(X, Y):
    host = object.__new__(gc.HvarfnerScaledGPSolver)
    return gc.HvarfnerScaledGPSolver._make_gp(host, X, Y)


def _linear_bo_gp(X, Y):
    return gc._make_linear_bo_gp(X.clamp(0.0, 1.0), Y, bounds=(0.0, 1.0), device=DEV)


def _geodesic_slice_gp(X, Y):
    host = object.__new__(gc.GeodesicSliceGPSolver)
    return gc.GeodesicSliceGPSolver._make_2d_gp(host, X[:, :2] * math.pi, Y)


BUILDERS = {"hvarfner": _hvarfner_gp, "linear_bo": _linear_bo_gp,
            "geodesic_slice": _geodesic_slice_gp}


def _flaky_optimizer(calls: dict):
    def flaky(mll, closure=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise NotPSDError("induced at attempt 1")
        return fit_gpytorch_mll_scipy(mll, closure=closure, **kw)
    return flaky


@pytest.mark.parametrize("name", list(BUILDERS))
def test_device_safe_priors_sample_on_the_gp_device(name):
    X, Y = _data()
    gp = BUILDERS[name](X, Y)
    safe = [(n, p) for n, _m, p, _c, _s in gp.named_priors()
            if isinstance(p, gc._SampleOnBufferDeviceMixin)]
    assert safe, f"{name}: no device-safe prior"
    for prior_name, prior in safe:
        assert prior.sample().device.type == "cuda", prior_name
        assert prior.rsample().device.type == "cuda", prior_name
    sample_all_priors(gp)  # what every fit retry calls; raised the E-1 error before


def test_linear_bo_noise_prior_is_device_safe():
    """D1: the noise prior was a plain LogNormalPrior at 68292ee."""
    X, Y = _data()
    gp = _linear_bo_gp(X, Y)
    noise_prior = gp.likelihood.noise_covar.noise_prior
    assert isinstance(noise_prior, gc._DeviceSafeLogNormalPrior)
    assert noise_prior.sample().device.type == "cuda"


@pytest.mark.parametrize("name", list(BUILDERS))
def test_a_failed_first_fit_is_retried_on_cuda(name):
    X, Y = _data()
    gp = BUILDERS[name](X, Y)
    calls = {"n": 0}
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp),
                     optimizer=_flaky_optimizer(calls))
    assert calls["n"] == 2  # attempt 1 raised, attempt 2 (after sample_all_priors) fitted
    assert all(p.device.type == "cuda" for p in gp.parameters())


def _bounds_file(tmp_path, x0: np.ndarray) -> str:
    lo, hi = x0.min(axis=0), x0.max(axis=0)
    w = hi - lo
    path = tmp_path / "bounds256.json"
    path.write_text(json.dumps({"per_coord": {
        "p01": (lo + 0.1 * w).tolist(), "p99": (hi - 0.1 * w).tolist(),
        "min": (lo - 0.05 * w).tolist(), "max": (hi + 0.05 * w).tolist(),
    }}))
    return str(path)


@pytest.mark.parametrize("cls_name", ["LinearBOBoxSolver", "HvarfnerScaledGPBoxFaithfulSolver"])
def test_solver_level_retry_returns_a_candidate_not_a_fallback(cls_name, tmp_path, monkeypatch):
    rng = np.random.RandomState(0)
    x0 = rng.randn(12, D)
    y0 = rng.rand(12, 1)
    cls = getattr(gc, cls_name)
    solver = cls(black_box=lambda z: np.zeros((len(z), 1)), x0=x0, y0=y0,
                 bounds_path=_bounds_file(tmp_path, x0))
    assert solver._device.type == "cuda"
    real_fit = gc.fit_gpytorch_mll
    calls = {"n": 0}

    def fit_with_one_failure(mll, **kw):
        return real_fit(mll, optimizer=_flaky_optimizer(calls))

    monkeypatch.setattr(gc, "fit_gpytorch_mll", fit_with_one_failure)
    torch.manual_seed(0)
    np.random.seed(0)
    z = solver.next_candidate()
    assert calls["n"] >= 2
    assert z.shape == (1, D) and np.all(np.isfinite(z))
    assert solver._fallback_count == 0, solver._fallback_reasons


def _plain_linear_bo_gp(X, Y):
    """``_make_linear_bo_gp`` exactly as at 68292ee (plain noise prior)."""
    from src.spherevo.linear_bo_kernel import SphericalLinearKernel

    kernel = SphericalLinearKernel(ard_num_dims=X.shape[-1], bounds=(0.0, 1.0))
    noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(1e-4, initial_value=noise_prior.mode),
    )
    gp = SingleTaskGP(train_X=X, train_Y=Y, mean_module=ConstantMean(),
                      covar_module=kernel, likelihood=likelihood)
    gp.to(X)
    return gp


def _plain_geodesic_slice_gp(X, Y):
    """``GeodesicSliceGPSolver._make_2d_gp`` exactly as at 68292ee."""
    base_kernel = MaternKernel(nu=2.5, ard_num_dims=2,
                               lengthscale_prior=LogNormalPrior(loc=0.0, scale=1.0),
                               lengthscale_constraint=Interval(1e-2, 10.0))
    likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.1, 0.05),
                                    noise_constraint=GreaterThan(1e-5))
    gp = SingleTaskGP(train_X=X, train_Y=Y, likelihood=likelihood,
                      covar_module=ScaleKernel(base_kernel))
    gp.to(X)
    return gp


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n", [6, 17, 29])
@pytest.mark.parametrize("pair", ["linear_bo", "geodesic_slice"])
def test_first_attempt_fits_are_bitwise_identical(pair, n, seed):
    X, Y = _data(n=n, seed=100 * seed + n)
    if pair == "linear_bo":
        builders = (lambda: _plain_linear_bo_gp(X.clamp(0.0, 1.0), Y),
                    lambda: _linear_bo_gp(X, Y))
        d_query = D
    else:
        builders = (lambda: _plain_geodesic_slice_gp(X[:, :2] * math.pi, Y),
                    lambda: _geodesic_slice_gp(X, Y))
        d_query = 2
    fitted = []
    for build in builders:
        torch.manual_seed(seed)
        gp = build()
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        gp.eval()
        g = torch.Generator().manual_seed(seed)
        Xq = torch.rand(64, 1, d_query, generator=g, dtype=torch.float64).to(DEV)
        with torch.no_grad():
            acq = LogExpectedImprovement(gp, best_f=float(Y.max()))(Xq)
        fitted.append((gp.state_dict(), acq))
    (sa, aa), (sb, ab) = fitted
    assert sa.keys() == sb.keys()
    for key in sa:
        assert torch.equal(sa[key], sb[key]), key
    assert torch.equal(aa, ab)
