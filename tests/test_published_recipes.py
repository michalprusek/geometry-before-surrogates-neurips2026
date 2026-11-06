"""Published GP-BO recipes (src/spherevo/published_recipes.py): builders pinned to the official code.

Every expected value below is a LITERAL copied from the official code, with file:line, so that a
mutation of a module constant cannot pass by changing the expectation with it. Official code:

* Hvarfner  github.com/hvarfner/vanilla_bo_in_highdim (+ Ax 0.3.7, which its main.py runs through)
* MSR       github.com/lpapenme/understanding-hdbo-code @292e7251f1, src/understanding_hdbo/
* Xu        github.com/XZT008/Standard-GP-is-all-you-need-for-HDBO
* Doumont   github.com/colmont/linear-bo, src/optimize.py

The builders are small enough to test at D = 32 on the CPU; the 25-step D = 256 runs of the
GuacaMol arms are in tests/test_published_block.py.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

import gpytorch  # noqa: E402
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound  # noqa: E402
from botorch.acquisition.logei import qLogNoisyExpectedImprovement  # noqa: E402
from botorch.exceptions.errors import BotorchError, ModelFittingError  # noqa: E402
from botorch.generation import gen_candidates_scipy  # noqa: E402
from botorch.optim import optimize_acqf as real_optimize_acqf  # noqa: E402
from botorch.optim.fit import fit_gpytorch_mll_scipy  # noqa: E402
from botorch.optim.utils import sample_all_priors  # noqa: E402
from gpytorch.constraints import GreaterThan, Interval, Positive  # noqa: E402
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel  # noqa: E402
from gpytorch.priors import GammaPrior, LogNormalPrior, UniformPrior  # noqa: E402

from src.spherevo import gp_competitors  # noqa: E402
from src.spherevo import published_recipes as pr  # noqa: E402
from src.spherevo.linear_bo_kernel import SphericalLinearKernel  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
D = 32
CUDA = torch.cuda.is_available()
#: the global default once BoTorch is imported (BoTorch raises gpytorch's 3 to 6); the Ax and Xu
#: references fit under this default, MSR under 9 and Doumont under 10
DEFAULT_CHOLESKY_TRIES = gpytorch.settings.cholesky_max_tries.value()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


def _data(D=D, n=8, seed=0, device="cpu", outside=False):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, D, generator=g, dtype=torch.float64)
    if outside:  # GuacaMol adaptation: a cold-start coordinate outside the unit cube
        X[0, :3] = torch.tensor([1.3, -0.2, 1.05], dtype=torch.float64)
    target = torch.rand(D, generator=g, dtype=torch.float64)
    Y = -((X - target) ** 2).sum(-1, keepdim=True)
    return X.to(device), Y.to(device)


@pytest.fixture
def spies(monkeypatch):
    """Record every optimize_acqf / fit_gpytorch_mll call (with the active gpytorch settings) made
    by published_recipes and by the reused MSR port in gp_competitors, then delegate."""
    calls = {"acq": [], "fit": []}

    def acq_spy(*args, **kwargs):
        acqf = kwargs.get("acq_function", args[0] if args else None)
        calls["acq"].append({"acqf": acqf, "args": args, "kwargs": dict(kwargs),
                             "cholesky_max_tries": gpytorch.settings.cholesky_max_tries.value()})
        return real_optimize_acqf(*args, **kwargs)

    real_fit = pr.fit_gpytorch_mll

    def fit_spy(mll, **kwargs):
        calls["fit"].append({
            "kwargs": dict(kwargs),
            "cholesky_max_tries": gpytorch.settings.cholesky_max_tries.value(),
            "max_cholesky_size": gpytorch.settings.max_cholesky_size.value(),
            "fast_log_prob": gpytorch.settings.fast_computations.log_prob.on(),
            "fast_covar_root": gpytorch.settings.fast_computations.covar_root_decomposition.on(),
            "fast_solves": gpytorch.settings.fast_computations.solves.on(),
        })
        return real_fit(mll, **kwargs)

    monkeypatch.setattr(pr, "optimize_acqf", acq_spy)
    monkeypatch.setattr(gp_competitors, "optimize_acqf", acq_spy)
    monkeypatch.setattr(pr, "fit_gpytorch_mll", fit_spy)
    return calls


def _bounds_ok(b, D=D):
    return torch.equal(b.cpu(), torch.stack([torch.zeros(D, dtype=torch.float64),
                                             torch.ones(D, dtype=torch.float64)]))


# ---------------------------------------------------------------------------
# 1) Hvarfner et al. 2024
# ---------------------------------------------------------------------------


def test_hvarfner_model_is_the_default_model_of_the_official_code():
    X, Y = _data()
    m = pr.build_hvarfner_model(X, pr.hvarfner_standardize(Y))
    k = m.covar_module
    # gp_priors.py:91-104: a bare RBFKernel, no ScaleKernel ("outputscale ... is simply 1",
    # default.yaml:4)
    assert type(k) is RBFKernel
    assert k.ard_num_dims == D
    # default.yaml:9-10 loc 1.4, scale 1.73205; gp_priors.py:46-48 loc += 0.5 * log(D)
    assert isinstance(k.lengthscale_prior, LogNormalPrior)
    assert float(k.lengthscale_prior.loc) == pytest.approx(1.4 + 0.5 * math.log(D), abs=1e-6)
    assert float(k.lengthscale_prior.scale) == pytest.approx(1.73205, abs=1e-6)
    # gp_priors.py:55,97: GreaterThan(1e-4) (softplus); no initial value -> softplus(0) + 1e-4
    assert type(k.raw_lengthscale_constraint) is GreaterThan
    assert float(k.raw_lengthscale_constraint.lower_bound) == pytest.approx(1e-4)
    assert torch.allclose(k.lengthscale, torch.full_like(k.lengthscale, math.log(2) + 1e-4))
    # default.yaml:11-13 + gp_priors.py:100-103: noise LogNormal(-4, 1), GreaterThan(1e-4)
    nc = m.likelihood.noise_covar
    assert isinstance(nc.noise_prior, LogNormalPrior)
    assert float(nc.noise_prior.loc) == -4.0 and float(nc.noise_prior.scale) == 1.0
    assert type(nc.raw_noise_constraint) is GreaterThan
    assert float(nc.raw_noise_constraint.lower_bound) == pytest.approx(1e-4)
    # Ax passes no outcome transform (surrogate.py:340-397); the y standardisation is Ax's
    assert getattr(m, "outcome_transform", None) is None


def test_hvarfner_y_standardisation_is_ax_standardize_y():
    # standardize_y.py:152-160: np.std(ddof=1); std < 1e-8 -> 1.0
    Y = torch.tensor([[1.0], [2.0], [4.0]], dtype=torch.float64)
    ys = pr.hvarfner_standardize(Y)
    expected = (Y.numpy() - Y.numpy().mean()) / np.std(Y.numpy(), ddof=1)
    assert np.allclose(ys.numpy(), expected)
    flat = torch.full((4, 1), 3.0, dtype=torch.float64)
    assert torch.equal(pr.hvarfner_standardize(flat), torch.zeros(4, 1, dtype=torch.float64))


def test_hvarfner_acquisition_call_is_the_ax_call(spies, monkeypatch):
    built = []
    real_cls = pr.qLogNoisyExpectedImprovement

    def acq_cls_spy(**kwargs):
        built.append(kwargs)
        return real_cls(**kwargs)

    monkeypatch.setattr(pr, "qLogNoisyExpectedImprovement", acq_cls_spy)
    X, Y = _data()
    x, rec = pr.propose("hvarfner", X, Y)
    assert rec.fallback is None and x.shape == (1, D) and bool(((x >= 0) & (x <= 1)).all())
    # qlognei.yaml:2-5: qLogNEI, prune_baseline True; Ax: X_baseline = unique observed X,
    # ScalarizedPosteriorTransform(weights=[1.]) (acquisition.py:281-317, torch/utils.py:494-496)
    (kw,) = built
    assert kw["prune_baseline"] is True
    assert torch.equal(kw["X_baseline"], torch.unique(X, dim=0))
    assert torch.equal(kw["posterior_transform"].weights.cpu(), torch.ones(1, dtype=torch.float64))
    (call,) = spies["acq"]
    assert type(call["acqf"]) is qLogNoisyExpectedImprovement
    kwargs = call["kwargs"]
    assert kwargs["q"] == 1
    assert kwargs["num_restarts"] == 4                       # highdim.yaml:2
    assert kwargs["raw_samples"] == 512                      # highdim.yaml:1
    assert kwargs["retry_on_optimization_warning"] is False  # highdim.yaml:3
    assert kwargs["sequential"] is True                      # Ax _argparse_base
    assert kwargs["options"] == {
        "init_batch_limit": 32,           # Ax _argparse_base default
        "batch_limit": 64,                # highdim.yaml:9
        "nonnegative": False,             # highdim.yaml:5
        "sample_around_best": True,       # highdim.yaml:6
        "sample_around_best_sigma": 0.1,  # highdim.yaml:7
        "maxiter": 300,                   # highdim.yaml:8
    }
    assert _bounds_ok(kwargs["bounds"])
    # C1: the authors' BoTorch optimises the 4 restarts of the batch_limit-64 chunk as ONE
    # L-BFGS-B problem (botorch 0.10 gen.py:209,230); 0.17.2 does so only with
    # use_parallel_mode=False (gen.py:247-258)
    gc = kwargs["gen_candidates"]
    assert gc.func is gen_candidates_scipy and gc.keywords == {"use_parallel_mode": False}
    # C3: the default initializer, output clamped onto the box (identity here: all in the box)
    assert "ic_generator" in kwargs and not any("clamped" in e for e in rec.events)
    # Ax fits with fit_gpytorch_mll defaults (botorch_modular/utils.py:296-297)
    (fit,) = spies["fit"]
    assert fit["kwargs"] == {} and fit["cholesky_max_tries"] == DEFAULT_CHOLESKY_TRIES != 9


def test_hvarfner_baseline_is_the_in_box_observations_as_ax(spies, monkeypatch):
    """C2: Ax's qLogNEI baseline = torch.unique of the observations inside the bounds (Ax 0.3.7
    acquisition.py:121-163,281-317, torch/utils.py:93-136 _filter_X_observed); the out-of-box
    incumbent stays in the GP data but not in the baseline."""
    built = []
    real_cls = pr.qLogNoisyExpectedImprovement
    monkeypatch.setattr(pr, "qLogNoisyExpectedImprovement",
                        lambda **kw: built.append(kw) or real_cls(**kw))
    X, Y = _data(outside=True)               # row 0 has 3 coordinates outside [0, 1]
    Y[0] = Y.max() + 1.0                     # the out-of-box point is the incumbent
    x, rec = pr.propose("hvarfner", X, Y)
    assert rec.fallback is None, rec.fallback
    (kw,) = built
    assert torch.equal(kw["X_baseline"], torch.unique(X[1:], dim=0))
    assert not any("no observation inside the box" in e for e in rec.events)
    assert bool(((x >= 0) & (x <= 1)).all())


def test_hvarfner_all_out_of_box_falls_back_to_all_observations_recorded(spies):
    """Ax has no path here (acquisition.py:318 not_none raises); ours uses every observation as the
    baseline (recorded as our addition) and clamps the starting points onto the box (C3)."""
    X, Y = _data(outside=True)
    X[:, 0] = 1.2                            # every observation outside the box
    x, rec = pr.propose("hvarfner", X, Y)
    assert rec.fallback is None, rec.fallback
    assert any("no observation inside the box -> all observations" in e for e in rec.events)
    assert rec.info["n_baseline"] >= 1 and bool(((x >= 0) & (x <= 1)).all())


def test_out_of_box_seeding_is_refused_by_botorch_0172_and_clamped_by_the_ic_generator():
    """Why C3 is needed: BoTorch 0.17.2 refuses out-of-box starting points (gen.py:186), the
    authors' BoTorch clamped them silently (botorch 0.10 gen.py:176). The clamped default
    initializer reproduces the latter; in the box it changes nothing."""
    X, Y = _data(outside=True)
    Y[0] = Y.max() + 1.0
    m = pr.build_hvarfner_model(X, pr.hvarfner_standardize(Y))
    pr.fit_hvarfner(m)
    acqf = pr.qLogNoisyExpectedImprovement(       # baseline WITH the out-of-box incumbent
        model=m, X_baseline=torch.unique(X, dim=0), prune_baseline=True,
        posterior_transform=pr.ScalarizedPosteriorTransform(weights=torch.ones(1, dtype=X.dtype)))
    kw = dict(acq_function=acqf, bounds=pr.unit_bounds(D, X.device), q=1, num_restarts=4,
              raw_samples=512, options=pr.hvarfner_options(), retry_on_optimization_warning=False)
    torch.manual_seed(0)
    with pytest.raises(BotorchError, match="out of bounds"):
        real_optimize_acqf(**kw)
    log = []
    torch.manual_seed(0)
    cand, _ = real_optimize_acqf(**kw, ic_generator=pr.clamped_default_ic_generator(log))
    assert sum(log) > 0 and bool(((cand >= 0) & (cand <= 1)).all())


@pytest.mark.parametrize("recipe", ["hvarfner", "msr"])
def test_clamped_initializer_is_the_identity_in_the_authors_setting(recipe, monkeypatch):
    """In the box the clamp changes nothing: the same seed gives the bit-identical candidate with
    and without the wrapper (the authors' setting is unaffected by C3)."""
    X, Y = _data()
    torch.manual_seed(11)
    x_with, rec = pr.propose(recipe, X, Y)
    assert rec.fallback is None and not any("clamped" in e for e in rec.events)
    monkeypatch.setattr(pr, "clamped_default_ic_generator",
                        lambda log=None: pr.gen_batch_initial_conditions)
    torch.manual_seed(11)
    x_without, _ = pr.propose(recipe, X, Y)
    assert torch.equal(x_with, x_without)


def test_msr_reference_call_equals_the_existing_port_in_the_box():
    """C3 for MSR: the new call (RAASP centres at the observed points, clamped starting points)
    gives the bit-identical candidate of gp_competitors._propose_msr_published_on_unit_box (left
    unchanged for the running arms) whenever every observation is in the box."""
    X, Y = _data()
    y_norm = pr.msr_standardize(Y)
    m = pr.build_msr_reference_model(X, y_norm)
    pr.fit_msr_reference(m)
    best = y_norm.max().item()
    torch.manual_seed(4)
    new = pr._propose_msr_reference_on_unit_box(m, best, X)
    torch.manual_seed(4)
    old = gp_competitors._propose_msr_published_on_unit_box(m, best, X)
    assert torch.equal(new, old)


# ---------------------------------------------------------------------------
# 2) Papenmeier et al. 2025: MSR and MLE (scaled)
# ---------------------------------------------------------------------------


def test_msr_reference_model_is_the_reference_gp():
    X, Y = _data()
    m = pr.build_msr_reference_model(X, pr.msr_standardize(Y))
    k = m.covar_module
    # model.py:75-83: ScaleKernel(MaternKernel(nu=2.5, ard, lengthscale_prior=None),
    # outputscale_prior=GammaPrior(2.0, 0.15))
    assert type(k) is ScaleKernel and type(k.base_kernel) is MaternKernel
    assert k.base_kernel.nu == 2.5 and k.base_kernel.ard_num_dims == D
    assert getattr(k.base_kernel, "lengthscale_prior", None) is None
    assert isinstance(k.outputscale_prior, GammaPrior)
    assert float(k.outputscale_prior.concentration) == 2.0
    assert float(k.outputscale_prior.rate) == pytest.approx(0.15)
    # model.py:86-89 + methods.py:43-45: every lengthscale = sqrt(d)/10, default Positive constraint
    assert type(k.base_kernel.raw_lengthscale_constraint) is Positive
    assert torch.allclose(k.base_kernel.lengthscale,
                          torch.full((1, D), math.sqrt(D) / 10.0, dtype=torch.float64))
    # model.py:91-95 with run.py:56-59 default --noiseless (reproduce_fig8.sh never disables it):
    # BoTorch's Gamma(1.1, 0.05) likelihood, noise frozen at 1e-4 (float32 1e-4)
    nc = m.likelihood.noise_covar
    assert isinstance(nc.noise_prior, GammaPrior)
    assert float(nc.noise_prior.concentration) == pytest.approx(1.1)
    assert float(nc.noise_prior.rate) == pytest.approx(0.05)
    assert float(m.likelihood.noise) == pytest.approx(1e-4, rel=1e-6)
    assert nc.raw_noise.requires_grad is False
    # botorch 0.9.5 (the reference pin): no default outcome transform
    assert getattr(m, "outcome_transform", None) is None
    # the frozen noise survives a fit
    pr.fit_msr_reference(m)
    assert float(m.likelihood.noise) == pytest.approx(1e-4, rel=1e-6)


@pytest.mark.parametrize("recipe,options", [
    # bo.py:174-182 (RAASP, sigma 0.001) + maxiter 2000 (the MSR port passes BoTorch's default)
    ("msr", {"batch_limit": 1, "sample_around_best": True, "sample_around_best_sigma": 0.001,
             "maxiter": 2000}),
    # bo.py:174 without RAASP (methods.py:72-78, raasp=False)
    ("mle_scaled", {"batch_limit": 1, "maxiter": 2000}),
])
def test_papenmeier_acquisition_call_is_the_reference_call(spies, recipe, options):
    X, Y = _data()
    x, rec = pr.propose(recipe, X, Y)
    assert rec.fallback is None and x.shape == (1, D)
    (call,) = spies["acq"]
    assert type(call["acqf"]) is LogExpectedImprovement                 # bo.py:173
    assert call["acqf"].best_f.item() == pytest.approx(pr.msr_standardize(Y).max().item())
    assert call["kwargs"]["q"] == 1
    assert call["kwargs"]["num_restarts"] == 5                           # bo.py:76,187
    assert call["kwargs"]["raw_samples"] == 512                          # bo.py:75,188
    assert call["kwargs"]["options"] == options
    assert _bounds_ok(call["kwargs"]["bounds"])
    assert call["cholesky_max_tries"] == 9                               # bo.py:171
    # RAASP centres: BoTorch's get_X_baseline falls back to the GP train inputs (no X_baseline set)
    assert not hasattr(call["acqf"], "X_baseline")
    assert "gen_candidates" not in call["kwargs"]          # batch_limit 1: one restart per problem
    (fit,) = spies["fit"]
    assert fit["cholesky_max_tries"] == 9 and fit["kwargs"] == {}        # fit.py:32-34


def test_msr_fit_failure_takes_the_adam_fallback_not_a_random_point(monkeypatch):
    """fit.py:35-50: ModelFittingError -> Adam(lr 0.01) for 500 steps; the step continues."""
    def failing_fit(mll, **kw):
        raise ModelFittingError("All attempts to fit the model have failed.")

    adam_lrs = []
    real_adam = torch.optim.Adam

    def adam_spy(params, lr):
        adam_lrs.append(lr)
        return real_adam(params, lr=lr)

    monkeypatch.setattr(pr, "fit_gpytorch_mll", failing_fit)
    monkeypatch.setattr(torch.optim, "Adam", adam_spy)
    X, Y = _data()
    x, rec = pr.propose("msr", X, Y)
    assert rec.fallback is None
    assert adam_lrs == [0.01]
    assert any("Adam fallback (500 steps, lr 0.01)" in e for e in rec.events)


def test_msr_acquisition_exception_gives_a_recorded_uniform_point(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("synthetic L-BFGS failure")

    monkeypatch.setattr(pr, "optimize_acqf", broken)
    X, Y = _data()
    torch.manual_seed(5)
    x, rec = pr.propose("msr", X, Y)
    assert rec.fallback is not None and "bo.py:192-194" in rec.fallback
    torch.manual_seed(5)
    assert torch.equal(x, torch.rand(1, D, dtype=torch.float64))   # bo.py:194 torch.rand(1, dim)


# ---------------------------------------------------------------------------
# 3) Xu et al. 2025: SBO-Matern (RI) = GP_ARD_setls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim,box", [(32, (1e-10, 10.0)), (128, (1e-10, 30.0))])
def test_xu_model_is_gp_ard_setls(dim, box):
    X, Y = _data(D=dim)
    m = pr.build_xu_model(X, pr.xu_standardize(Y))
    k = m.covar_module
    # GP.py:206-211: ScaleKernel(MaternKernel(ard, prior, constraint), GammaPrior(2.0, 0.15))
    assert type(k) is ScaleKernel and type(k.base_kernel) is MaternKernel
    assert k.base_kernel.nu == 2.5 and k.base_kernel.ard_num_dims == dim
    assert float(k.outputscale_prior.concentration) == 2.0
    assert float(k.outputscale_prior.rate) == pytest.approx(0.15)
    # GP.py:196-202: Uniform(1e-10, 30) + Interval(1e-10, 30) for D >= 100, else 10.0
    prior, con = k.base_kernel.lengthscale_prior, k.base_kernel.raw_lengthscale_constraint
    assert isinstance(prior, UniformPrior)
    assert float(prior.low) == pytest.approx(box[0]) and float(prior.high) == pytest.approx(box[1])
    assert type(con) is Interval
    assert float(con.lower_bound) == pytest.approx(box[0])
    assert float(con.upper_bound) == pytest.approx(box[1])
    # GP.py:212-214: every lengthscale = sqrt(D)
    assert torch.allclose(k.base_kernel.lengthscale,
                          torch.full((1, dim), math.sqrt(dim), dtype=torch.float64))
    # GP.py:235 SingleTaskGP default likelihood of that botorch: Gamma(1.1, 0.05), GreaterThan(1e-4)
    nc = m.likelihood.noise_covar
    assert isinstance(nc.noise_prior, GammaPrior)
    assert float(nc.noise_prior.concentration) == pytest.approx(1.1)
    assert float(nc.noise_prior.rate) == pytest.approx(0.05)
    assert float(nc.raw_noise_constraint.lower_bound) == pytest.approx(1e-4)
    assert getattr(m, "outcome_transform", None) is None


def test_xu_is_undefined_above_d_900():
    # GP.py:197-199,212-214: sqrt(1000) = 31.6 > 30 -> gpytorch refuses the official init
    assert pr.xu_recipe_defined(900) and not pr.xu_recipe_defined(901)
    X, Y = _data(D=1000, n=3)
    with pytest.raises(pr.PublishedRecipeUndefined):
        pr.build_xu_model(X, pr.xu_standardize(Y))


def test_xu_y_standardisation_is_sklearn_standard_scaler():
    Y = torch.tensor([[1.0], [2.0], [4.0]], dtype=torch.float64)
    expected = (Y.numpy() - Y.numpy().mean()) / np.std(Y.numpy(), ddof=0)   # dataset.py:39-49
    assert np.allclose(pr.xu_standardize(Y).numpy(), expected)


def test_xu_acquisition_call_is_bo_loop_gp_map(spies):
    X, Y = _data()
    x, rec = pr.propose("xu", X, Y)
    assert rec.fallback is None
    (call,) = spies["acq"]
    assert type(call["acqf"]) is UpperConfidenceBound           # BO_loop.py:126-127
    assert float(call["acqf"].beta) == 1.5                       # run_script.py:31
    assert call["acqf"].maximize is True
    assert call["kwargs"]["q"] == 1
    assert call["kwargs"]["num_restarts"] == 10                  # BO_loop.py:137
    assert call["kwargs"]["raw_samples"] == 1000                 # BO_loop.py:138
    assert call["kwargs"]["options"] == {}                       # BO_loop.py:139
    assert _bounds_ok(call["kwargs"]["bounds"])
    # C1: that BoTorch (<= 0.8.x) optimises the 10 restarts jointly (batch_limit = num_restarts)
    gc = call["kwargs"]["gen_candidates"]
    assert gc.func is gen_candidates_scipy and gc.keywords == {"use_parallel_mode": False}
    assert "retry_on_optimization_warning" not in call["kwargs"]   # BoTorch default True
    (fit,) = spies["fit"]
    assert fit["kwargs"] == {}                                   # GP.py:240-241 (LBFGS)
    assert fit["cholesky_max_tries"] == DEFAULT_CHOLESKY_TRIES


# ---------------------------------------------------------------------------
# 4) Doumont et al.: spherical linear kernel
# ---------------------------------------------------------------------------


def test_doumont_model_fit_and_acquisition_are_optimize_py(spies):
    X, Y = _data()
    x, rec = pr.propose("doumont", X, Y)
    assert rec.fallback is None and rec.info["n_inputs_clamped"] == 0
    m = pr.build_doumont_model(X, pr.doumont_standardize(Y))
    # optimize.py:40-53: ConstantMean, SphericalLinearKernel(ard_num_dims=d) on (0, 1) bounds,
    # LogNormal(-4, 1) noise with GreaterThan(1e-4, initial_value=mode); botorch >= 0.13 default
    # outcome transform Standardize
    assert type(m.covar_module) is SphericalLinearKernel
    assert torch.equal(m.covar_module._mins.cpu(), torch.zeros(D, dtype=torch.float64))
    assert torch.equal(m.covar_module._maxs.cpu(), torch.ones(D, dtype=torch.float64))
    nc = m.likelihood.noise_covar
    assert float(nc.noise_prior.loc) == -4.0 and float(nc.noise_prior.scale) == 1.0
    assert float(nc.raw_noise_constraint.lower_bound) == pytest.approx(1e-4)
    assert type(m.outcome_transform).__name__ == "Standardize"
    # optimize.py:78-91
    (fit,) = spies["fit"]
    assert fit["kwargs"] == {"optimizer": fit_gpytorch_mll_scipy}
    assert fit["cholesky_max_tries"] == 10
    assert fit["max_cholesky_size"] == float("inf")
    assert (fit["fast_log_prob"], fit["fast_covar_root"], fit["fast_solves"]) == (True, False, False)
    # optimize.py:164-183
    (call,) = spies["acq"]
    assert type(call["acqf"]) is LogExpectedImprovement
    # C1 decision (text R2-7): the authors' BoTorch is unknown (botorch>=0.13, no lock file; 0.13/0.14
    # joint, >= 0.15 batched); the fresh-install default is kept (the plain function, not a partial)
    assert call["kwargs"]["gen_candidates"] is gen_candidates_scipy
    upm = pr.recipe_settings("doumont")["optimize_acqf"]["use_parallel_mode"]
    assert "unknown" in upm and "postdates" not in upm
    assert call["kwargs"]["raw_samples"] == 512
    assert call["kwargs"]["num_restarts"] == 4
    assert call["kwargs"]["retry_on_optimization_warning"] is False
    assert call["kwargs"]["options"] == {"nonnegative": False, "sample_around_best": True,
                                         "sample_around_best_sigma": 0.1, "maxiter": 300,
                                         "batch_limit": 64}
    assert call["kwargs"]["q"] == 1


def test_linear_bo_kernel_is_invariant_to_rescaling_inputs_and_bounds_together():
    """`run_linear_bo` (LassoBench) feeds [-1,1] inputs with kernel bounds (-1, 1); Doumont feeds
    [0,1] inputs with bounds (0, 1). The global lengthscale is normalised by the box width
    (linear_bo_kernel.py: max_sq_norm), so the two kernels coincide at equal hyperparameters: the
    domain is NOT a difference between `linear_bo` and the builder. What differs is the fit context
    (cholesky_max_tries(10), max_cholesky_size(inf), fast_computations) and
    retry_on_optimization_warning=False (optimize.py:78-91, 168), hence no alias."""
    g = torch.Generator().manual_seed(3)
    U = torch.rand(6, 12, generator=g, dtype=torch.float64)
    k01 = SphericalLinearKernel(ard_num_dims=12, bounds=(0.0, 1.0)).double()
    k11 = SphericalLinearKernel(ard_num_dims=12, bounds=(-1.0, 1.0)).double()
    with torch.no_grad():
        for k in (k01, k11):
            k.lengthscale = torch.linspace(0.05, 3.0, 12, dtype=torch.float64)
            k.raw_glob_ls.fill_(0.7)
            k.raw_coeffs.copy_(torch.tensor([0.3, -0.2], dtype=torch.float64))
    assert torch.allclose(k01(U).to_dense(), k11(2 * U - 1).to_dense(), atol=1e-12)
    src = (REPO / "scripts" / "run_lassobench.py").read_text()
    body = src[src.index("def run_linear_bo("):src.index("def run_gp_bo(")]
    assert "retry_on_optimization_warning" not in body and "cholesky_max_tries" not in body


# ---------------------------------------------------------------------------
# Failure handling: a failed GP fit mid-run is caught per step and recorded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recipe", ["hvarfner", "xu", "doumont"])
def test_forced_fit_failure_mid_run_is_caught_per_step_and_recorded(monkeypatch, recipe):
    real_fit = pr.fit_gpytorch_mll
    n = {"calls": 0}

    def fit_failing_on_third(mll, **kw):
        n["calls"] += 1
        if n["calls"] == 3:
            raise ModelFittingError("All attempts to fit the model have failed.")
        return real_fit(mll, **kw)

    monkeypatch.setattr(pr, "fit_gpytorch_mll", fit_failing_on_third)
    X0, Y0 = _data(n=5)
    target = torch.rand(D, dtype=torch.float64)
    out = pr.run_loop(recipe, lambda x: float(-((x.reshape(-1) - target) ** 2).sum()), X0, Y0, 5)
    assert out["X"].shape == (10, D)
    assert out["fallback_count"] == 1
    assert [s["step"] for s in out["steps"] if s["fallback"]] == [2]
    assert "ModelFittingError" in out["fallback_reasons"][0]


def test_flat_signal_msr_is_a_recorded_uniform_point():
    X, _ = _data()
    x, rec = pr.propose("msr", X, torch.zeros(X.shape[0], 1, dtype=torch.float64))
    assert rec.fallback is not None and rec.fallback.startswith("flat-signal")
    # C8: the reference ends the run (bo.py:150-158 builds the model outside the try; SingleTaskGP
    # raises InputDataError on NaN targets); ours is more lenient and says so
    assert "InputDataError" in rec.fallback and "more lenient" in rec.fallback
    assert x.shape == (1, D)


# ---------------------------------------------------------------------------
# Devices: BoTorch's fit retry (sample_all_priors) must not crash on CUDA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("builder", ["hvarfner", "msr", "xu", "doumont"])
def test_fit_retry_resampling_keeps_every_prior_on_the_model_device(device, builder):
    X, Y = _data(D=128, device=device)
    m = {
        "hvarfner": lambda: pr.build_hvarfner_model(X, pr.hvarfner_standardize(Y)),
        "msr": lambda: pr.build_msr_reference_model(X, pr.msr_standardize(Y)),
        "xu": lambda: pr.build_xu_model(X, pr.xu_standardize(Y)),
        "doumont": lambda: pr.build_doumont_model(X, pr.doumont_standardize(Y)),
    }[builder]()
    sample_all_priors(m)          # what botorch.fit does before a retry
    for name, p in m.named_parameters():
        assert p.device.type == torch.device(device).type, name


def test_device_safe_uniform_prior_draws_like_the_plain_prior():
    a, b = pr._DeviceSafeUniformPrior(1e-10, 30.0), UniformPrior(1e-10, 30.0)
    torch.manual_seed(0)
    sa = a.sample(torch.Size([5]))
    torch.manual_seed(0)
    sb = b.sample(torch.Size([5]))
    assert torch.equal(sa, sb)
    x = torch.tensor([0.5, 29.0, 1e-3])
    assert torch.equal(a.log_prob(x), b.log_prob(x))


# ---------------------------------------------------------------------------
# LassoBench methods (scripts/run_lassobench.py, additive)
# ---------------------------------------------------------------------------


class _FakeLasso:
    """bench.evaluate(x) for x in [-1,1]^D, minimised (LassoBench convention)."""

    def __init__(self, D, fail=False):
        self.t = np.linspace(-0.5, 0.5, D)
        self.n = 0

    def evaluate(self, x):
        self.n += 1
        return float(((np.asarray(x) - self.t) ** 2).sum())


def _lasso_module():
    sys.path.insert(0, str(REPO))
    import scripts.run_lassobench as rl
    return rl


def test_lassobench_methods_are_registered_with_their_recipes():
    rl = _lasso_module()
    assert rl.PUBLISHED_LASSO_METHODS == {
        "hvarfner_published": "hvarfner",
        "papenmeier_published": "msr",
        "papenmeier_mle_scaled_published": "mle_scaled",
        "xu_published": "xu",
        "linear_bo_published": "doumont",
        "turbo_published": "turbo1",
        "baxus_official": "baxus",
        "cmaes_full_published": "cmaes",
        "papenmeier_published_learned": "msr_learned",                     # round 2, R2-6
        "papenmeier_mle_scaled_published_learned": "mle_scaled_learned",
    }


@pytest.mark.parametrize("method", ["hvarfner_published", "papenmeier_published",
                                    "papenmeier_mle_scaled_published", "xu_published",
                                    "linear_bo_published", "turbo_published"])
def test_lassobench_method_runs_and_records(method):
    # cmaes_full_published and baxus_official do not use the shared cold start (round-1 C6;
    # test_round1_lassobench_cmaes_spends_the_whole_budget_from_x0, test_lassobench_baxus_*)
    rl = _lasso_module()
    bench = _FakeLasso(12)
    res = rl.run_published_lasso(method, bench, 12, n_cold=5, n_iter=3, seed=42)
    assert res["status"] == "completed"
    assert len(res["all_scores"]) == 8 == bench.n
    assert len(res["best_trace"]) == 4
    assert res["best_score"] == min(res["all_scores"])
    assert res["surrogate_fallback_count"] == 0
    assert len(res["published_recipe"]["steps"]) == 3
    # cold start: 5 points U[-1,1]^D from RandomState(seed), as every LassoBench arm
    cs = np.random.RandomState(42).uniform(-1, 1, size=(5, 12))
    assert res["all_scores"][:5] == [_FakeLasso(12).evaluate(x) for x in cs]


def test_lassobench_fit_failure_no_longer_loses_the_run(monkeypatch):
    """ANA_runs.md 3.6: one ModelFittingError used to turn the whole Xu run into inf."""
    rl = _lasso_module()

    def always_fail(mll, **kw):
        raise ModelFittingError("All attempts to fit the model have failed.")

    monkeypatch.setattr(pr, "fit_gpytorch_mll", always_fail)
    res = rl.run_published_lasso("xu_published", _FakeLasso(12), 12, n_cold=5, n_iter=4, seed=1)
    assert res["status"] == "completed" and math.isfinite(res["best_score"])
    assert res["surrogate_fallback_count"] == 4
    assert all("ModelFittingError" in r for r in res["surrogate_fallback_reasons"])


def test_lassobench_xu_at_d1000_is_reported_not_applicable():
    rl = _lasso_module()
    bench = _FakeLasso(1000)
    res = rl.run_published_lasso("xu_published", bench, 1000, n_cold=5, n_iter=25, seed=42)
    assert res["status"] == "not_applicable" and bench.n == 5
    assert "Interval(1e-10, 30)" in res["error"]


# ---------------------------------------------------------------------------
# Verification harness (scripts/cr/verify_baselines.py)
# ---------------------------------------------------------------------------


def _vb():
    sys.path.insert(0, str(REPO))
    from scripts.cr import verify_baselines as vb
    return vb


def test_harness_default_initial_designs_are_the_authors():
    vb = _vb()
    assert vb.default_init("hvarfner", 180) == (30, "sobol")
    assert vb.default_init("doumont", 256) == (30, "sobol")
    assert vb.default_init("msr", 180) == (40, "sobol")        # floor(3 * sqrt(180)) = 40
    assert vb.default_init("mle_scaled", 180) == (40, "sobol")
    assert vb.default_init("xu", 180) == (20, "lhs")
    assert vb.default_init("turbo", 20) == (40, "sobol")
    # LassoBench protocol for TuRBO: 0.1 d LHS (synthetic), 100 (real); BAxUS Fig. 3: 10
    assert vb.default_init("turbo_published", 60, "lasso_synt_simple") == (6, "turbo_lhs")
    assert vb.default_init("turbo_published", 180, "lasso_dna") == (100, "turbo_lhs")
    assert vb.default_init("baxus", 1000, "lasso_synt_hard") == (10, "baxus_sobol")
    assert vb.default_init("cmaes", 60) == (0, "none")


def test_harness_maps_every_recipe_to_a_target_of_the_shipped_file():
    vb = _vb()
    targets = vb.load_targets(vb.TARGETS_FILE)
    for recipe, bench in [("hvarfner", "lasso_dna"), ("msr", "lasso_dna"),
                          ("mle_scaled", "lasso_dna"), ("xu", "lasso_dna"),
                          ("doumont", "lasso_dna"), ("doumont", "guacamol:adip"),
                          ("turbo_published", "lasso_synt_simple"),
                          ("cmaes", "lasso_synt_simple"), ("baxus", "lasso_synt_hard"),
                          ("random", "lasso_dna")]:
        assert vb.default_target_id(recipe, bench) in targets, (recipe, bench)
    # R2-10: no mapping to a target that the shipped file does not contain
    assert vb.default_target_id("turbo", "ackley20") is None


def test_shipped_targets_file_is_the_amended_audit_file():
    vb = _vb()
    assert vb.sha256_file(vb.TARGETS_FILE) == vb.TARGETS_SHA256 == (
        "8b0507645cf5141613190ec502a2d4e7a3f4587ef39c6782e44e8b48da47bfc1")
    raw = json.loads(vb.TARGETS_FILE.read_text())
    history = ("a7d960294ad785d935221a4df04456ffc96abaa3c38deb19cd017780ebe8aadc",
               "c1c9031ab86418df1ac8419e9f57c0c8444d2a9cb3fee43272e0e57d66124601",
               "272fc8d1ecf6059a4661475cfeb84b3e0b96f09eb005d74126800602b14c2baf",
               "57083bf841ec8267291a19870ce82c79d1f6d51ab637a2780d9d0b42ef74a395",
               "e56e2c5d8ffa0a8a7965fb42fdb66ba1178517287ad7e9d1c5ee7b9f3eb122a9",
               "b190322f138584c76fd45b4a352cb0ec625c088ad74a5ada3ae04b75736f63b0",
               "b06f72305610ae3e335d6eecf41c94edf5ed9d175d9c169080e007d0aa2c3473",
               "5d72465bed09b703db9756aef7fb6cde97597a5c0f7b511e227ce51088f479a6")
    assert vb.TARGETS_SHA256_HISTORY == history
    # every amendment names the file it was applied to: the history, in order
    assert tuple(a["previous_sha256"] for a in raw["amendments"]) == history
    assert vb.TARGETS_SHA256_AUDIT_ORIGINAL == history[0]
    assert vb.TARGETS_SHA256_PREVIOUS == history[-1]
    assert all(a["before_any_result"] is True for a in raw["amendments"])
    assert [a.get("round") for a in raw["amendments"]] == [None, 1, 2, 2, 2, 2, 2, 2]
    assert len(raw["targets"]) == 68


def test_harness_runs_resumes_and_summarises(tmp_path):
    vb = _vb()
    out = tmp_path / "vb"
    common = ["--benchmark", "ackley20", "--seed", "0", "--out-dir", str(out), "--device", "cpu"]
    assert vb.main(["run", "--recipe", "hvarfner", "--budget", "7", "--n-init", "5", *common]) == 0
    rec = json.loads((out / "hvarfner__ackley20__n5__s0.json").read_text())
    assert rec["status"] == "completed" and len(rec["values"]) == 7 and rec["resumed_at"] == []
    assert rec["best_so_far"] == list(np.maximum.accumulate(rec["values"]))
    assert len(rec["steps"]) == 2 and rec["fallback_count"] == 0
    assert rec["provenance"]["git_commit_full"] and "botorch" in rec["provenance"]["package_versions"]
    assert rec["provenance"]["targets_sha256"] == vb.TARGETS_SHA256
    # a longer budget resumes from the checkpoint instead of starting over
    rec["status"] = "running"
    (out / "hvarfner__ackley20__n5__s0.json").write_text(json.dumps(rec))
    assert vb.main(["run", "--recipe", "hvarfner", "--budget", "9", "--n-init", "5", *common]) == 0
    rec2 = json.loads((out / "hvarfner__ackley20__n5__s0.json").read_text())
    assert rec2["resumed_at"] == [7] and len(rec2["values"]) == 9
    assert rec2["values"][:7] == rec["values"]
    (hist,) = rec2["provenance_history"]                     # B4: the earlier process is kept
    assert hist["provenance"] == rec["provenance"] and hist["evaluations_before_resume"] == 7
    # summary against a figure-read target: one seed of one required -> verdict computed
    tfile = tmp_path / "t.json"
    tfile.write_text(json.dumps({"targets": [{"id": "x", "n_reps": 1, "value_kind": "figure_read",
                                              "statistic": "mean over repetitions of best-so-far",
                                              "values": [{"n": 7, "value": rec["best_so_far"][6],
                                                          "read_unc": 0.1}]}]}))
    assert vb.main(["summarize", "--out-dir", str(out), "--targets", str(tfile),
                    "--target-id", "x", "--write", str(tmp_path / "s.json")]) == 0
    summ = json.loads((tmp_path / "s.json").read_text())
    (v,) = summ["verdicts"]
    assert v["seeds_used"] == [0] and v["rows"][0]["ours"] == pytest.approx(rec["best_so_far"][6])
    assert summ["targets_sha256"] == vb.sha256_file(tfile)
    # B3: a foreign targets file and a target without a protocol never give a verdict
    assert v["pass"] is None and v["n_init"] == 5
    assert any("targets file sha256" in r for r in v["refusals"])
    assert any("no numeric n_init" in r for r in v["refusals"])
    assert summ["summarizer_provenance"]["git_commit_full"] == rec["provenance"]["git_commit_full"]


def test_pass_rule_is_the_run_rules_rule():
    """RUN_RULES 'Published-recipe block' (2): exact -> 2 combined SE; figure -> read_unc + 2 combined
    SE with SE_pub := SE_ours when the band is unstated (round 2, R2-3; the old read_unc + 2 SE_ours is
    pass_strict); LassoBench files -> running minimum of the across-repetition mean."""
    vb = _vb()
    runs = {s: {"recipe": "turbo_published", "sense": "min", "values": vals,
                "best_so_far": list(np.minimum.accumulate(vals))}
            for s, vals in enumerate([[5.0, 3.0, 4.0], [7.0, 5.0, 2.0]])}
    # mean curve 6, 4, 3 -> running min 3 at n=3 (SE of column 3 = sd([4, 2]) / sqrt(2) = 1)
    lb = {"id": "lb", "n_reps": 2, "value_kind": "released_file_exact",
          "statistic": "running minimum over evaluation index of the across-repetition MEAN",
          "values": [{"n": 3, "value": 4.9, "sd_at_x": 0.0}]}
    lb["_norm"] = vb.normalise_values(lb["values"])
    v = vb.verdict_for(lb, None, runs)
    assert v["rows"][0]["ours"] == pytest.approx(3.0) and v["rows"][0]["ours_se"] == pytest.approx(1.0)
    assert v["pass"] is True                       # |3 - 4.9| = 1.9 <= 2 * 1
    lb["values"][0]["value"] = 5.1
    lb["_norm"] = vb.normalise_values(lb["values"])
    assert vb.verdict_for(lb, None, runs)["pass"] is False
    # mean of best-so-far, figure read: best-so-far at n=3 = 3 and 2 -> mean 2.5, SE 0.5
    fig = {"id": "f", "n_reps": 2, "value_kind": "figure_read",
           "statistic": "mean over repetitions of best-so-far",
           "values": [{"n": 3, "value": 3.65, "read_unc": 0.1}]}
    fig["_norm"] = vb.normalise_values(fig["values"])
    v = vb.verdict_for(fig, None, runs)                          # |2.5 - 3.65| = 1.15
    assert v["pass"] is True                                      # <= 0.1 + 2 * sqrt(0.5^2 + 0.5^2)
    assert v["pass_strict"] is False                              # >  0.1 + 2 * 0.5
    fig["values"][0]["value"] = 4.1                               # 1.6 > 1.5142
    fig["_norm"] = vb.normalise_values(fig["values"])
    assert vb.verdict_for(fig, None, runs)["pass"] is False
    fig["values"][0]["value"] = 3.5
    fig["_norm"] = vb.normalise_values(fig["values"])
    assert vb.verdict_for(fig, None, runs)["pass"] is True
    # too few seeds is never a pass: not verified (None), not a fail (round 2, R2-10)
    fig["n_reps"] = 3
    v = vb.verdict_for(fig, None, runs)
    assert v["pass"] is None and v["incomplete"] == ["incomplete: seeds missing [2]"]


def test_harness_runs_the_external_methods(tmp_path):
    vb = _vb()
    out = tmp_path / "vb"
    for recipe, budget in (("turbo_published", 44), ("baxus", 13), ("cmaes", 25)):
        assert vb.main(["run", "--recipe", recipe, "--benchmark", "ackley20", "--seed", "0",
                        "--budget", str(budget), "--out-dir", str(out), "--device", "cpu"]) == 0
        n_init = {"turbo_published": 2, "baxus": 10, "cmaes": 0}[recipe]
        rec = json.loads((out / f"{recipe}__ackley20__n{n_init}__s0.json").read_text())
        assert rec["status"] == "completed" and len(rec["values"]) == budget
        assert rec["fallback_count"] == 0


def test_harness_turbo_runs_the_repository_turbo(tmp_path):
    vb = _vb()
    out = tmp_path / "vb"
    assert vb.main(["run", "--recipe", "turbo", "--benchmark", "ackley20", "--seed", "0",
                    "--budget", "43", "--out-dir", str(out), "--device", "cpu"]) == 0
    rec = json.loads((out / "turbo__ackley20__n40__s0.json").read_text())
    assert rec["n_init"] == 40 and len(rec["values"]) == 43 and rec["fallback_count"] == 0
    assert rec["settings"]["class"] == "TurboBoxFaithfulSolver"


def test_targets_reader_accepts_the_audit_and_extract_layouts(tmp_path):
    vb = _vb()
    ex = vb.load_targets(REPO / "scripts" / "cr" / "baseline_targets.example.json")
    assert ex["turbo1__ackley20__botorch_tutorial"]["_norm"][None][0]["value"] == -8.30
    audit = vb.load_targets(vb.TARGETS_FILE)
    cma = audit["lassobench_pkl__cma-es__synt_simple__noiseless"]["_norm"][None]
    assert {r["n"]: r["x_logged"] for r in cma}[30] == 20.0      # compare at x_logged
    g = audit["doumont_linear__guacamol_aabo__fig13"]["_norm"]
    assert "adip" in g and g["adip"][0]["n"] == 100
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"k": {"values": {"30": {"mean": 0.4, "se": 0.01}}}}))
    (r,) = vb.load_targets(p)["k"]["_norm"][None]
    assert (r["n"], r["value"], r["se"]) == (30.0, 0.4, 0.01)


# ---------------------------------------------------------------------------
# External baselines as released (src/spherevo/published_external.py)
# ---------------------------------------------------------------------------

from src.spherevo import published_external as pe  # noqa: E402


@pytest.mark.parametrize("dim,n_cand,failtol", [(2, 200, 4), (20, 2000, 20), (60, 5000, 60),
                                                (256, 5000, 256)])
def test_turbo_trust_region_constants_are_turbo_1_py(dim, n_cand, failtol):
    t = pe.Turbo1Published(dim=dim, n_init=3)
    assert t.n_cand == n_cand                          # turbo_1.py:105 min(100 d, 5000)
    assert t.failtol == failtol                        # turbo_1.py:106 ceil(max(4/q, d/q))
    assert t.succtol == 3                              # turbo_1.py:107
    assert (t.length_init, t.length_min, t.length_max) == (0.8, 0.5 ** 7, 1.6)   # :111-113


def test_turbo_gp_is_gp_py():
    g = torch.Generator().manual_seed(0)
    X = torch.rand(8, 5, generator=g, dtype=torch.float64)
    y = torch.randn(8, generator=g, dtype=torch.float64)
    adam_lrs = []
    real_adam = torch.optim.Adam

    def adam_spy(params, lr):
        adam_lrs.append(lr)
        return real_adam(params, lr=lr)

    import unittest.mock as um
    with um.patch.object(torch.optim, "Adam", adam_spy):
        gp = pe.turbo_train_gp(X, y)
    assert adam_lrs == [0.1]                                             # gp.py:85
    k = gp.covar_module
    assert type(k) is ScaleKernel and type(k.base_kernel) is MaternKernel and k.base_kernel.nu == 2.5
    for con, lo, hi in ((k.base_kernel.raw_lengthscale_constraint, 0.005, 2.0),   # gp.py:50
                        (k.raw_outputscale_constraint, 0.05, 20.0),               # gp.py:53
                        (gp.likelihood.noise_covar.raw_noise_constraint, 5e-4, 0.2)):  # gp.py:48
        assert float(con.lower_bound) == pytest.approx(lo)
        assert float(con.upper_bound) == pytest.approx(hi)
    assert pe.TURBO_INIT_HYPERS == {"covar_module.outputscale": 1.0,     # gp.py:79-81
                                    "covar_module.base_kernel.lengthscale": 0.5,
                                    "likelihood.noise": 0.005}
    assert pe.TURBO_N_TRAINING_STEPS == 50 and pe.TURBO_MAX_CHOLESKY_SIZE == 2000


def test_turbo_trust_region_rules_and_restart_follow_turbo_1_py():
    t = pe.Turbo1Published(dim=4, n_init=3)
    t.seed_data(np.full((2, 4), 0.5), np.array([1.0, 2.0]))
    t._last_kind = "bo"
    for f in (0.9, 0.8, 0.7):                          # 3 successes (relative 1e-3) -> double
        t.tell(np.full(4, 0.5), f)
    assert t.length == 1.6 and t.succcount == 0
    for _ in range(4):                                 # failtol = 4 failures -> halve
        t.tell(np.full(4, 0.5), 5.0)
    assert t.length == 0.8
    t.length = 0.5 ** 8                                # collapsed -> restart with an LHS design
    np.random.seed(0)
    u, info = t.ask()
    assert info.kind == "init" and t.n_restarts == 1 and len(t._pending_init) == 2
    assert t._X.shape == (0, 4) and t.length == 0.8


def test_turbo_counts_every_proposal_and_reseeds_sobol_from_numpy(monkeypatch):
    seeds = []
    real = pe.SobolEngine

    def sobol_spy(dim, scramble, seed):
        seeds.append(seed)
        return real(dim, scramble=scramble, seed=seed)

    monkeypatch.setattr(pe, "SobolEngine", sobol_spy)
    np.random.seed(1)
    torch.manual_seed(1)
    t = pe.Turbo1Published(dim=6, n_init=4)
    f = lambda u: float(((u - 0.3) ** 2).sum())
    n = 0
    for _ in range(10):
        u, info = t.ask()
        t.tell(u, f(u))
        n += 1
    assert t._X.shape == (n, 6)                        # init 4 + 6 BO steps, all counted
    assert len(seeds) == 6 and len(set(seeds)) == 6    # a fresh numpy-drawn seed per step


def test_turbo_thompson_sample_is_drawn_from_the_posterior_predictive(monkeypatch):
    """turbo_1.py:214-216: y_cand = gp.likelihood(gp(X_cand)).sample(), i.e. noise included."""
    seen = []
    real = pe.GaussianLikelihood

    class LikSpy(real):
        def __call__(self, inp, *a, **k):
            if hasattr(inp, "loc"):
                seen.append(int(inp.loc.shape[-1]))
            return super().__call__(inp, *a, **k)

    monkeypatch.setattr(pe, "GaussianLikelihood", LikSpy)
    np.random.seed(0)
    t = pe.Turbo1Published(dim=5, n_init=4)
    for _ in range(5):
        u, info = t.ask()
        t.tell(u, float((u ** 2).sum()))
    assert seen.count(t.n_cand) == 1                   # one predictive draw over the 500 candidates


def test_baxus_state_and_embedding_are_the_tutorials():
    st = pe.BaxusState(dim=1000, eval_budget=990)
    assert (st.d_init, st.n_splits, st.new_bins_on_split) == (1, 5, 3)
    k = math.floor(math.log(st.length_min / st.length_init, 0.5))
    assert st.failure_tolerance == min(st.target_dim, max(1, math.floor(st.split_budget / k)))
    torch.manual_seed(0)
    S = pe.baxus_embedding_matrix(50, 4, "cpu", torch.double)
    assert S.shape == (4, 50) and torch.equal((S != 0).sum(0), torch.ones(50, dtype=torch.long))
    assert set(S[S != 0].tolist()) <= {-1.0, 1.0}
    X = 2 * torch.rand(7, 4, dtype=torch.double) - 1
    S2, X2 = pe.baxus_increase_embedding_and_observations(S, X, 3)
    assert S2.shape[0] > 4 and torch.allclose(X2 @ S2, X @ S)     # observations are kept exactly


def test_baxus_runs_its_own_design_and_splits_with_a_small_budget():
    torch.manual_seed(0)
    bx = pe.BaxusPublished(dim=40, n_init=5, eval_budget=20, seed=0)
    f = lambda x: -float(((np.asarray(x) - 0.2) ** 2).sum())
    kinds = []
    for _ in range(25):
        x, info = bx.ask()
        assert np.all(np.abs(x) <= 1.0)
        bx.tell(f(x))
        kinds.append(info["kind"])
        assert info["fallback"] is None
    assert kinds[:5] == ["init"] * 5 and set(kinds[5:]) == {"bo"}
    assert bx.n_splits_done >= 1 and bx.S.shape[0] > bx.state.d_init
    assert pe.BAXUS_NOISE_INTERVAL == (1e-8, 1e-3) and pe.BAXUS_ADAM_STEPS == 100


def test_cmaes_uses_the_lassobench_settings(monkeypatch):
    import cma

    made = []
    real = cma.CMAEvolutionStrategy

    def es_spy(x0, sigma0, opts):
        made.append((np.asarray(x0).copy(), sigma0, dict(opts)))
        return real(x0, sigma0, opts)

    monkeypatch.setattr(cma, "CMAEvolutionStrategy", es_spy)
    xs = []
    info = pe.run_cmaes_published(lambda x: float((x ** 2).sum()), 12, 25, seed=4,
                                  on_eval=lambda x, v, i: xs.append(x))
    (x0, sigma0, opts), = made
    # LassoBench example/cma_example.py:45-55: popsize 20, seed, bounds [-1, 1], x0 0, sigma 0.1
    assert np.array_equal(x0, np.zeros(12)) and sigma0 == 0.1
    assert opts["popsize"] == 20 and opts["bounds"] == [-1.0, 1.0] and opts["seed"] == 5
    assert len(xs) == 25 and info["generations_told"] == 1
    assert all(np.all(np.abs(x) <= 1.0) for x in xs)


def test_lassobench_baxus_runs_its_own_design_on_the_box():
    rl = _lasso_module()
    bench = _FakeLasso(40)
    res = rl.run_published_lasso("baxus_official", bench, 40, n_cold=5, n_iter=10, seed=42)
    assert res["status"] == "completed" and bench.n == 15 == len(res["all_scores"])
    cs = np.random.RandomState(42).uniform(-1, 1, size=(5, 40))
    assert res["all_scores"][:5] != [_FakeLasso(40).evaluate(x) for x in cs]
    assert res["published_recipe"]["cold_start"].startswith("own Sobol design")


def test_free_text_repetition_counts_fail_closed_unless_given():
    """The audit's MSR entry says n_reps 'not stated in the Fig. 8 caption (10 in Table 3; ...)':
    no number may be guessed from free text (RUN_RULES (2) needs the target's own count)."""
    vb = _vb()
    targets = vb.load_targets(vb.TARGETS_FILE)
    msr = dict(targets["papenmeier_msr__lasso_dna__fig8"],
               n_reps="not stated in the Fig. 8 caption (10 in Table 3; reproduce_fig8.sh takes REP)")
    assert vb._num(msr["n_reps"]) is None
    assert vb._num(">=10") == 10 and vb._num("20") == 20 and vb._num("final (10 + up to 300)") is None
    runs = {s: {"recipe": "msr", "sense": "min", "values": [0.4] * 1000,
                "best_so_far": [0.4] * 1000} for s in range(10)}
    v = vb.verdict_for(msr, None, runs)
    assert v["pass"] is None and "not a number" in v["note"]
    v = vb.verdict_for(msr, None, runs, n_reps_override=10)
    assert v["seeds_used"] == list(range(10)) and v["rows"]
    assert vb.verdict_for(msr, None, {0: runs[0]}, n_reps_override=10)["pass"] is None
    # the learned-noise variants map to the same targets and design
    assert vb.default_target_id("msr_noise_learned", "lasso_dna") == msr["id"]
    assert vb.default_init("mle_scaled_noise_learned", 180) == (40, "sobol")


def test_msr_learned_noise_variant_trains_the_gamma_noise():
    """model.py:91-95 with --no-noiseless: BoTorch's Gamma(1.1, 0.05) likelihood, noise learned from
    its initial value (the prior mode 2.0); used only by the harness key msr_noise_learned."""
    X, Y = _data()
    m = pr.build_msr_reference_model(X, pr.msr_standardize(Y), noiseless=False)
    assert m.likelihood.noise_covar.raw_noise.requires_grad is True
    assert float(m.likelihood.noise) == pytest.approx(2.0)



def test_pre_run_target_corrections_are_in_the_shipped_file():
    vb = _vb()
    T = vb.load_targets(vb.TARGETS_FILE)
    xu = T["xu_sbo_matern_ri__lasso_dna__fig6"]
    # round 1 (A1): run_script.py:148-152 pickles all X and Y, so x = k is the best after k + 1
    # evaluations; the earlier 20 (BO_loop.py's best_y) is kept as x_eval_offset_previous
    assert xu["x_eval_offset"] == 1 and xu["x_eval_offset_previous"] == 20 and xu["n_init"] == 20
    assert "gate" not in xu and xu["values_curve"].startswith("values = dashed blue")
    assert [r["n"] for r in xu["values"]] == [50, 100, 150, 200, 250, 300]
    assert {k: xu["values"][1][k] for k in ("n", "value", "read_unc")} == {     # dashed (RI)
        "n": 100, "value": 0.3264, "read_unc": 0.0005}
    assert xu["values_solid_sbo_matern"][1]["value"] == 0.3218
    for tid in ("papenmeier_msr__lasso_dna__fig8", "papenmeier_mle_scaled__lasso_dna__fig8"):
        assert T[tid]["n_reps"] == 10 and "Table 3" in T[tid]["n_reps_note"]
        assert "bo.py:210" in T[tid]["x_convention"]
    turbo = [e for k, e in T.items() if k.startswith("lassobench_pkl__turbo__")]
    assert turbo and all(e["batch_size"] == 10 for e in turbo)       # turbo_example.py:78
    cma = [e for k, e in T.items() if k.startswith("lassobench_pkl__cma-es__")]
    assert cma and all(e["cma_logger_column"] == "best" for e in cma)


def test_xu_target_is_compared_n_plus_1_evaluations_in():
    """A1: Fig. 6 plots the running best of ALL evaluations against their 0-based index
    (run_script.py:148-152 pickles X and Y; best_y is discarded): x = 50 is the best after 51."""
    vb = _vb()
    xu = vb.load_targets(vb.TARGETS_FILE)["xu_sbo_matern_ri__lasso_dna__fig6"]
    curve = list(np.linspace(0.40, 0.29, 400))            # best-so-far, strictly decreasing
    runs = {s: {"recipe": "xu", "sense": "min", "values": curve, "best_so_far": curve}
            for s in range(10)}
    v = vb.verdict_for(xu, None, runs)
    row = v["rows"][0]
    assert row["n"] == 50 and row["n_evaluations_compared"] == 51
    assert row["ours"] == pytest.approx(curve[50])        # best after 51 evaluations


def test_turbo_batch_size_must_match_the_target():
    vb = _vb()
    t = vb.load_targets(vb.TARGETS_FILE)["lassobench_pkl__turbo__synt_simple__noiseless"]
    runs = {s: {"recipe": "turbo_published", "sense": "min", "batch_size": 1,
                "values": [1.0] * 600, "best_so_far": [1.0] * 600} for s in range(30)}
    v = vb.verdict_for(t, None, runs)
    assert v["pass"] is None and "--batch-size 10" in v["note"]
    assert any("--batch-size 10" in r for r in v["refusals"])
    for r in runs.values():
        r["batch_size"] = 10
    v = vb.verdict_for(t, None, runs)
    assert v["rows"] and v["refusals"] == []


def test_turbo_batch_q_follows_turbo_1_py():
    t = pe.Turbo1Published(dim=30, n_init=4, batch_size=10)
    assert t.failtol == 3 and t.n_cand == 3000              # ceil(max(4/10, 30/10)); min(100 d, 5000)
    np.random.seed(0)
    torch.manual_seed(0)
    f = lambda U: [float(((u - 0.3) ** 2).sum()) for u in np.asarray(U).reshape(-1, 30)]
    for _ in range(4):                                     # the LHS design, one point per ask
        u, info = t.ask()
        assert info.kind == "init" and u.shape == (1, 30)
        t.tell(u, f(u))
    U, info = t.ask()
    assert info.kind == "bo" and U.shape == (10, 30)
    assert len({tuple(r) for r in U}) == 10                # argmin with removal: 10 distinct rows
    fails = t.failcount
    t.tell(U, [10.0] * 10)                                 # one failed batch = one failure count
    assert t.failcount == fails + 1 and t._X.shape == (14, 30)


def test_production_turbo_paths_run_q_1(monkeypatch, tmp_path):
    seen = []
    real = pe.Turbo1Published.__init__

    def spy(self, *a, **k):
        seen.append(k.get("batch_size"))
        real(self, *a, **k)

    monkeypatch.setattr(pe.Turbo1Published, "__init__", spy)
    rl = _lasso_module()
    rl.run_published_lasso("turbo_published", _FakeLasso(12), 12, n_cold=5, n_iter=2, seed=1)
    from src.spherevo.published_arms import TurboPublishedSolver
    rng = np.random.RandomState(0)
    x0 = rng.randn(5, 12)
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"per_coord": {"p01": (-np.ones(12)).tolist(),
                                           "p99": np.ones(12).tolist()}}))
    TurboPublishedSolver(black_box=lambda z: -np.sum(np.asarray(z) ** 2, 1, keepdims=True),
                         x0=x0, y0=-np.sum(x0 ** 2, 1, keepdims=True), bounds_path=str(p))
    assert seen == [1, 1]                                  # literal q = 1 in both production paths


def test_harness_runs_turbo_with_batch_10(tmp_path):
    vb = _vb()
    out = tmp_path / "vb"
    assert vb.main(["run", "--recipe", "turbo_published", "--benchmark", "ackley20", "--seed", "0",
                    "--budget", "25", "--batch-size", "10", "--out-dir", str(out),
                    "--device", "cpu"]) == 0
    rec = json.loads((out / "turbo_published__ackley20__n2__s0.json").read_text())
    assert rec["batch_size"] == 10 and rec["n_init"] == 2 and len(rec["values"]) == 25
    # B6: the settings record follows q (turbo_1.py:106 failtol = ceil(max(4/q, d/q)) = 2)
    assert rec["settings"]["batch_size"] == 10 and rec["settings"]["failure_tolerance_value"] == 2
    assert [s.get("batch_rows") for s in rec["steps"] if "batch_rows" in s][:3] == [1, 1, 10]


def test_turbo_batch_draws_q_samples_and_removes_each_pick(monkeypatch):
    """turbo_1.py:214-216,226-234: q posterior-predictive draws, then argmin per draw with the picked
    row set to inf. With all q draws forced equal, only the removal yields q distinct rows."""
    shapes = []
    real = pe.GaussianLikelihood

    class _Pred:
        def __init__(self, dist):
            self._d = dist

        def sample(self, shape):
            shapes.append(tuple(shape))
            one = self._d.sample(torch.Size([1]))
            return one.repeat(int(shape[0]), 1)            # q identical draws

    class LikSpy(real):
        def __call__(self, inp, *a, **k):
            out = super().__call__(inp, *a, **k)
            is_cand = hasattr(inp, "loc") and int(inp.loc.shape[-1]) == 800   # min(100 d, 5000)
            return _Pred(out) if is_cand else out

    monkeypatch.setattr(pe, "GaussianLikelihood", LikSpy)
    np.random.seed(3)
    torch.manual_seed(3)
    t = pe.Turbo1Published(dim=8, n_init=4, batch_size=5)
    for _ in range(4):
        u, _ = t.ask()
        t.tell(u, float((u ** 2).sum()))
    U, info = t.ask()
    assert info.kind == "bo" and shapes == [(5,)]           # q draws requested
    assert len({tuple(r) for r in U}) == 5                  # argmin with removal


# ---------------------------------------------------------------------------
# Round 1 (FIXLIST_round1): targets A1-A6, harness B1-B7, external C5-C7, LassoBench C6/C9
# ---------------------------------------------------------------------------


def test_round1_vector_targets_are_in_the_shipped_file():
    """A2-A4: means (and SE where the band is a stated SE) from the vector PDFs, plot-x offsets."""
    vb = _vb()
    T = vb.load_targets(vb.TARGETS_FILE)
    msr, mle = T["papenmeier_msr__lasso_dna__fig8"], T["papenmeier_mle_scaled__lasso_dna__fig8"]
    for e in (msr, mle):
        # arXiv 2502.09198v2 Fig. 8 caption: "confidence bands indicate the standard error of the mean"
        assert e["value_kind"] == "vector_figure_exact" and e["x_eval_offset"] == 1
        assert [r["n"] for r in e["values"]] == [100, 200, 300, 500, 999]
    assert [(r["value"], r["se"]) for r in mle["values"]] == [
        (0.32215, 0.00172), (0.30886, 0.0013), (0.29951, 0.0008), (0.29204, 0.00065),
        (0.28714, 0.00046)]
    # FIXLIST A2 quoted the means at x = n - 1; kept for the record
    assert [r["value"] for r in mle["values_at_x_minus_1"]] == [0.32342, 0.30886, 0.29951, 0.29204,
                                                                0.28714]
    assert [(r["value"], r["se"]) for r in msr["values"]] == [
        (0.32476, 0.00156), (0.30801, 0.00118), (0.29811, 0.00085), (0.29144, 0.00056),
        (0.28778, 0.00072)]
    # Hvarfner Fig. 7: band meaning not stated -> figure_read, SE_pub 0 (no "se" key)
    hv = T["hvarfner_dsp__lasso_dna__fig7"]
    assert hv["value_kind"] == "figure_read" and hv["x_eval_offset"] == 0
    assert [(r["n"], r["value"], r["read_unc"]) for r in hv["values"]] == [
        (50, 0.3236, 0.0001), (100, 0.31115, 0.0001), (200, 0.30084, 0.0001),
        (500, 0.29131, 0.0001), (1000, 0.28574, 0.0001)]
    assert all("se" not in r and r["band_halfwidth"] > 0 for r in hv["values"])
    assert all(r["se"] is None for r in hv["_norm"][None])
    # Doumont Fig. 2 (caption: mean and standard error of the mean): exact with se, 1-based x
    dm = T["doumont_linear__lasso_dna__fig2"]
    assert dm["value_kind"] == "vector_figure_exact" and dm["x_eval_offset"] == 0
    assert [(r["n"], r["value"], r["se"]) for r in dm["values"]][0] == (50, 0.32044, 0.00193)
    # BAxUS Fig. 3 ("mean performance with one standard error"), 0-based x
    for tid, first, last in (("baxus__lasso_hard__fig3", (30, 7.8089, 0.0092), (1000, 1.5722, 0.0634)),
                             ("baxus__lasso_high__fig3", (30, 5.0178, 0.0965), (1000, 1.1153, 0.0105))):
        e = T[tid]
        assert e["value_kind"] == "vector_figure_exact" and e["x_eval_offset"] == 1
        rows = [(r["n"], r["value"], r["se"]) for r in e["values"]]
        assert rows[0] == first and rows[-1] == last and len(rows) == 6
    assert vb.EXACT_KINDS[-1] == "vector_figure_exact"


def test_round1_lassobench_targets_use_the_sd_at_the_argmin_column():
    """A6: the released value is min_{j <= x} mean_j; its SE comes from the SD at j*, not at x
    (literal values from the released res_simple.pkl / res_dna.pkl)."""
    vb = _vb()
    T = vb.load_targets(vb.TARGETS_FILE)
    tu = T["lassobench_pkl__turbo__synt_simple__noiseless"]
    assert [(r["n"], r["argmin_x"], r["sd_at_argmin"]) for r in tu["values"]] == [
        (30, 29.0, 19.92289), (50, 47.0, 13.12712), (100, 93.0, 2.94759), (200, 199.0, 1.67731),
        (500, 497.0, 0.9921)]
    assert tu["values"][0]["sd_at_x"] == 32.96372                    # kept for the record
    (r30,) = [r for r in tu["_norm"][None] if r["n"] == 30]
    assert r30["sd"] == 19.92289 and r30["sd_key"] == "sd_at_argmin"
    runs = {s: {"recipe": "turbo_published", "sense": "min", "batch_size": 10,
                "values": [9.77751] * 600, "best_so_far": [9.77751] * 600} for s in range(30)}
    row = vb.verdict_for(tu, None, runs)["rows"][0]
    assert row["target_se"] == pytest.approx(19.92289 / math.sqrt(30))
    assert row["target_se_from"] == "sd_at_argmin"


def test_round1_cma_dna_and_x1_rows_are_informative():
    """A5: DNA CMA-ES is informative (released x = 1 is not f(0) there); the x = 1 row of every
    CMA-ES entry is an objective-version check with zero tolerance, never gating."""
    vb = _vb()
    T = vb.load_targets(vb.TARGETS_FILE)
    assert T["lassobench_pkl__cma-es__dna__real"]["gate"] is False
    for tid in ("lassobench_pkl__cma-es__synt_simple__noiseless",
                "lassobench_pkl__cma-es__synt_hard__noiseless"):
        assert T[tid].get("gate", True) is not False
        (x1,) = [r for r in T[tid]["values"] if r["n"] == 1]
        assert x1["gate"] is False and x1["sd_at_argmin"] == 0.0
    # verdict: an informative target never passes or fails
    dna = T["lassobench_pkl__cma-es__dna__real"]
    runs = {s: {"recipe": "cmaes", "sense": "min", "cma_popsize": 20, "f_x0": 0.70583,
                "values": [0.3] * 600, "best_so_far": [0.3] * 600} for s in range(30)}
    v = vb.verdict_for(dna, None, runs)
    assert v["pass"] is None and v["gate"] is False and "informative" in v["note"]
    # a gating synthetic target ignores its x = 1 row
    syn = T["lassobench_pkl__cma-es__synt_simple__noiseless"]
    v = vb.verdict_for(syn, None, runs)
    (row1,) = [r for r in v["rows"] if r["n"] == 1]
    assert row1["gate"] is False and row1["tolerance_strict"] == pytest.approx(0.0, abs=1e-12)
    # round 2 (R2-4): the protocol tolerance carries the objective-version allowance delta * |value|
    assert row1["tolerance"] == pytest.approx(0.0032894230634687867 * abs(row1["target"]), rel=1e-9)
    assert v["pass"] is (all(r["pass"] for r in v["rows"] if r["gate"]) and v["refusals"] == [])
    assert v["pass_strict"] is (all(r["pass_strict"] for r in v["rows"] if r["gate"]))


def test_round1_cma_curve_puts_f_x0_at_x1_and_refuses_short_runs():
    """B1: f(x0) (evaluated once, outside the budget) at x = 1, then the best of each generation at
    g * 20; B2: a run shorter than x_logged is never compared."""
    vb = _vb()
    run = {"recipe": "cmaes", "sense": "min", "cma_popsize": 20, "f_x0": 9.0,
           "values": [5.0] * 19 + [4.0] + [3.0] * 20}
    assert vb._cma_logged_curve(run) == {1: 9.0, 20: 4.0, 40: 3.0}
    assert vb._cma_logged_curve({**run, "f_x0": None}) is None
    mean, se = vb.our_statistic([run, run], "lassobench_running_min_of_mean", 40, 40)
    assert mean == 3.0 and se == 0.0
    assert vb.our_statistic([run, run], "lassobench_running_min_of_mean", 1, 1) == (9.0, 0.0)
    assert vb.our_statistic([run, run], "lassobench_running_min_of_mean", 60, 60) == (None, None)
    no_x0 = {k: v for k, v in run.items() if k != "f_x0"}
    assert vb.our_statistic([no_x0, no_x0], "lassobench_running_min_of_mean", 40, 40) == (None, None)


def test_round1_harness_cmaes_records_f_x0_outside_the_budget(tmp_path):
    vb = _vb()
    out = tmp_path / "vb"
    assert vb.main(["run", "--recipe", "cmaes", "--benchmark", "ackley20", "--seed", "0",
                    "--budget", "21", "--out-dir", str(out), "--device", "cpu"]) == 0
    rec = json.loads((out / "cmaes__ackley20__n0__s0.json").read_text())
    bench = vb.Ackley20()
    assert rec["f_x0"] == pytest.approx(bench.value(np.zeros(20) + 2.5))   # x0 = 0 in [-1,1] -> 2.5
    assert len(rec["values"]) == 21                     # f(x0) is not one of the 21
    assert "stop_conditions" in rec["cma"]


def _fake_run(seed, commit="a" * 40, dirty=False, n_init=30, budget=1000, **kw):
    return {"recipe": "hvarfner", "benchmark": "lasso_dna", "seed": seed, "sense": "min",
            "n_init": n_init, "init_type": "sobol", "budget": budget, "batch_size": 1,
            "provenance": {"git_commit_full": commit, "git_dirty": dirty},
            "values": [0.3] * budget, "best_so_far": [0.3] * budget, **kw}


def test_round1_summarize_refusals_are_the_fixlist_b3_list():
    vb = _vb()
    T = vb.load_targets(vb.TARGETS_FILE)
    hv = T["hvarfner_dsp__lasso_dna__fig7"]
    good = [_fake_run(s) for s in range(20)]
    assert vb.provenance_refusals(good) == [] and vb.protocol_refusals(hv, "hvarfner", good) == []
    assert vb.protocol_refusals(hv, "hvarfner", [_fake_run(s, n_init=41) for s in range(20)]) == []
    mixed = good[:10] + [_fake_run(s, commit="b" * 40) for s in range(10, 20)]
    assert any("2 commits" in r for r in vb.provenance_refusals(mixed))
    assert any("dirty" in r for r in vb.provenance_refusals(good[:19] + [_fake_run(19, dirty=True)]))
    assert any("dirty" in r for r in vb.provenance_refusals(good[:19] + [_fake_run(19, dirty=None)]))
    assert any("without a recorded commit" in r
               for r in vb.provenance_refusals(good[:19] + [_fake_run(19, commit="")]))
    resumed = good[:19] + [_fake_run(19, provenance_history=[{"provenance": {
        "git_commit_full": "c" * 40}}])]
    assert any("resumed across commits" in r for r in vb.provenance_refusals(resumed))
    assert any("n_init" in r for r in vb.protocol_refusals(hv, "hvarfner",
                                                           [_fake_run(s, n_init=20) for s in range(20)]))
    assert any("different n_init" in r for r in vb.protocol_refusals(
        hv, "hvarfner", good[:10] + [_fake_run(s, n_init=41) for s in range(10, 20)]))
    assert any("budget" in r for r in vb.protocol_refusals(hv, "hvarfner",
                                                           [_fake_run(s, budget=500) for s in range(20)]))
    assert any("initial design" in r for r in vb.protocol_refusals(
        hv, "hvarfner", [{**_fake_run(s), "init_type": "lhs"} for s in range(20)]))
    bx = T["baxus__lasso_hard__fig3"]
    bruns = [{**_fake_run(s, n_init=10, budget=2000), "recipe": "baxus", "init_type": "baxus_sobol",
              "m_d": 2000 - 10} for s in range(20)]
    assert any("m_D" in r for r in vb.protocol_refusals(bx, "baxus", bruns))
    assert vb.protocol_refusals(bx, "baxus", [{**r, "m_d": 1000} for r in bruns]) == []


def test_round1_git_failure_raises_instead_of_recording_an_empty_commit(monkeypatch):
    vb = _vb()
    import subprocess as sp

    class R:
        returncode, stdout, stderr = 128, "", "fatal: not a git repository"

    monkeypatch.setattr(vb.subprocess, "run", lambda *a, **k: R())
    with pytest.raises(RuntimeError, match="git rev-parse HEAD failed"):
        vb.git_commit_full()
    with pytest.raises(RuntimeError, match="git rev-parse HEAD failed"):
        vb.provenance("cpu")                 # a run never starts with an empty commit
    assert sp.run is vb.subprocess.run


def test_round1_summarize_groups_by_n_init_and_filters(tmp_path):
    vb = _vb()
    out = tmp_path / "vb"
    out.mkdir()
    for n_init in (30, 41):
        for s in range(2):
            r = {**_fake_run(s, n_init=n_init), "status": "completed"}
            (out / f"hvarfner__lasso_dna__n{n_init}__s{s}.json").write_text(json.dumps(r))
    assert vb.main(["summarize", "--out-dir", str(out), "--write", str(tmp_path / "a.json")]) == 0
    v = json.loads((tmp_path / "a.json").read_text())["verdicts"]
    # two verdicts, one per design; only 2 of 20 seeds -> not verified (None), nothing to refuse
    assert sorted(x["n_init"] for x in v) == [30, 41]
    assert all(x["pass"] is None and x["refusals"] == [] for x in v)
    assert all(x["incomplete"] == [f"incomplete: seeds missing {list(range(2, 20))}"] for x in v)
    assert vb.main(["summarize", "--out-dir", str(out), "--n-init-filter", "41",
                    "--write", str(tmp_path / "b.json")]) == 0
    (v41,) = json.loads((tmp_path / "b.json").read_text())["verdicts"]
    assert v41["n_init"] == 41 and v41["seeds_used"] == [0, 1]


def test_round1_resume_refuses_another_commit_or_settings(tmp_path):
    vb = _vb()
    out = tmp_path / "vb"
    common = ["--benchmark", "ackley20", "--seed", "0", "--out-dir", str(out), "--device", "cpu"]
    assert vb.main(["run", "--recipe", "random", "--budget", "6", *common]) == 0
    p = out / "random__ackley20__n0__s0.json"
    rec = json.loads(p.read_text())
    rec["status"] = "running"
    rec["provenance"]["git_commit_full"] = "0" * 40
    p.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="refuse resume.*commit"):
        vb.main(["run", "--recipe", "random", "--budget", "8", *common])
    assert vb.resume_mismatches({**rec, "provenance": {"git_commit_full": "x"}, "n_init": 7},
                                {**rec, "provenance": {"git_commit_full": "x"}}) == ["n_init 7 != 0"]


def test_round1_baxus_eval_budget_is_m_d_minus_n_init(tmp_path, monkeypatch):
    """B5: BAxUS Fig. 3 uses m_D = 1000 with a budget of 2000 (paper Sec. 4); eval_budget = m_D -
    n_init, independent of --budget."""
    vb = _vb()
    seen = []
    real = pe.BaxusPublished.__init__

    def spy(self, *a, **k):
        seen.append(k["eval_budget"])
        real(self, *a, **k)

    monkeypatch.setattr(pe.BaxusPublished, "__init__", spy)
    out = tmp_path / "vb"
    for extra in ([], ["--m-d", "300"]):
        assert vb.main(["run", "--recipe", "baxus", "--benchmark", "ackley20", "--seed", "0",
                        "--budget", "12", "--out-dir", str(out), "--device", "cpu", "--force",
                        *extra]) == 0
    assert seen == [990, 290]
    rec = json.loads((out / "baxus__ackley20__n10__s0.json").read_text())
    assert rec["m_d"] == 300 and rec["settings"]["eval_budget"] == 290
    with pytest.raises(SystemExit, match="baxus only"):
        vb.main(["run", "--recipe", "cmaes", "--benchmark", "ackley20", "--seed", "0",
                 "--budget", "3", "--out-dir", str(out), "--m-d", "100"])


def test_round1_baxus_thompson_candidates_come_from_the_input_dimension(monkeypatch):
    """C5: tutorial cell 21, N_CANDIDATES = min(5000, max(2000, 200 * dim)) with dim the INPUT
    dimension (5000 at D = 40 while the target dimension is 1, not 2000)."""
    sizes = []
    real = pe.SobolEngine

    class SobolSpy(real):
        def draw(self, n, *a, **k):
            sizes.append(int(n))
            return super().draw(n, *a, **k)

    monkeypatch.setattr(pe, "SobolEngine", SobolSpy)
    torch.manual_seed(0)
    bx = pe.BaxusPublished(dim=40, n_init=3, eval_budget=20, seed=0)
    for _ in range(4):
        x, info = bx.ask()
        bx.tell(-float((np.asarray(x) ** 2).sum()))
    assert bx.state.target_dim < 40 and sizes == [3, 5000]       # the design, then one BO step


def test_round1_baxus_adam_fallback_is_recorded(monkeypatch):
    """C7 (E-m4): the tutorial's Adam fallback after ModelFittingError is an event, not a fallback."""
    import botorch.fit

    def failing(mll, **kw):
        raise ModelFittingError("All attempts to fit the model have failed.")

    monkeypatch.setattr(botorch.fit, "fit_gpytorch_mll", failing)
    torch.manual_seed(0)
    bx = pe.BaxusPublished(dim=12, n_init=3, eval_budget=10, seed=0)
    infos = []
    for _ in range(4):
        x, info = bx.ask()
        bx.tell(-float((np.asarray(x) ** 2).sum()))
        infos.append(info)
    assert infos[-1]["fallback"] is None
    assert any("Adam fallback (100 steps, lr 0.1)" in e for e in infos[-1]["events"])


def test_round1_cmaes_records_es_stop_without_stopping_or_changing_the_run(monkeypatch):
    """C7 (E-m5): es.stop() after every told generation is recorded; the run continues to the
    budget, and the evaluated points are those of a run that never asks es.stop()."""
    import cma

    def run(patch_stop):
        if patch_stop:
            monkeypatch.setattr(cma.CMAEvolutionStrategy, "stop", lambda self, *a, **k: {})
        xs = []
        info = pe.run_cmaes_published(lambda x: 1.0, 6, 100, seed=2,     # flat: tolflatfitness
                                      on_eval=lambda x, v, i: xs.append((x, i)))
        return xs, info

    xs, info = run(False)
    assert len(xs) == 100 and info["generations_told"] == 5
    assert any("tolflatfitness" in c["conditions"] for c in info["stop_conditions"])
    assert any(e.startswith("es.stop() after generation") for _, i in xs for e in i["events"])
    xs_plain, info_plain = run(True)
    assert info_plain["stop_conditions"] == []
    assert all(np.array_equal(a, b) for (a, _), (b, _) in zip(xs, xs_plain))


def test_round1_lassobench_cmaes_spends_the_whole_budget_from_x0(monkeypatch):
    """C6: the published LassoBench protocol has no initial design; cmaes_full_published spends the
    n_cold + n_iter evaluations from x0 = 0 and does not evaluate the cold start; round 2 (R2-12):
    evaluation 1 is f(x0), then n_cold + n_iter - 1 pycma samples."""
    rl = _lasso_module()
    calls = []
    real = pe.run_cmaes_published

    def spy(f, dim, n_evals, seed, **kw):
        calls.append((dim, n_evals, seed, kw.get("x0")))
        return real(f, dim, n_evals, seed, **kw)

    monkeypatch.setattr(pe, "run_cmaes_published", spy)
    bench = _FakeLasso(12)
    res = rl.run_published_lasso("cmaes_full_published", bench, 12, n_cold=5, n_iter=25, seed=42)
    assert calls == [(12, 29, 42, None)] and bench.n == 30 == len(res["all_scores"])
    assert len(res["best_trace"]) == 26 and res["published_recipe"]["cold_start"].startswith("none")
    cs = np.random.RandomState(42).uniform(-1, 1, size=(5, 12))
    assert res["all_scores"][:5] != [_FakeLasso(12).evaluate(x) for x in cs]


@pytest.mark.parametrize("method", ["hvarfner_published", "turbo_published", "baxus_official",
                                    "cmaes_full_published"])
def test_round1_lassobench_published_results_carry_provenance(method):
    """C9: commit, dirty flag and versions in every published-method result."""
    rl = _lasso_module()
    res = rl.run_published_lasso(method, _FakeLasso(12), 12, n_cold=5, n_iter=2, seed=1)
    prov = res["provenance"]
    assert prov["git_commit"] and prov["git_dirty"] in (True, False)
    assert {"botorch", "gpytorch", "torch"} <= set(prov["package_versions"])


def test_round1_lassobench_crash_record_survives_a_provenance_failure(monkeypatch):
    """C9 follow-up: the crash handler of run_lassobench.main records the crash even when provenance
    itself fails (e.g. ``src`` not importable because PYTHONPATH is missing, as the usage line of
    run_lassobench.py requires it); the failure is written into the record instead."""
    rl = _lasso_module()

    def broken():
        raise ModuleNotFoundError("No module named 'src'")

    monkeypatch.setattr(rl, "_published_lasso_provenance", broken)
    prov = rl._published_lasso_provenance_or_error()
    assert prov == {"error": "provenance unavailable: ModuleNotFoundError: No module named 'src'"}
    src = (REPO / "scripts" / "run_lassobench.py").read_text()
    handler = src[src.index("elif method in PUBLISHED_LASSO_METHODS:"):]
    handler = handler[:handler.index("else:")]
    assert '"provenance": _published_lasso_provenance_or_error()}' in handler



# ---------------------------------------------------------------------------
# Round 2 (orchestrator decisions, part 1)
# ---------------------------------------------------------------------------


def test_round2_noisy_cma_entries_are_informative():
    """R2-2: under the protocol (E) the CMA-ES gate is the four noiseless synthetic targets; the four
    noisy synthetic entries are gate: false with that reason, the noiseless ones stay gates."""
    vb = _vb()
    T = vb.load_targets(vb.TARGETS_FILE)
    for b in ("synt_simple", "synt_medium", "synt_high", "synt_hard"):
        noisy = T[f"lassobench_pkl__cma-es__{b}__noise"]
        assert noisy["gate"] is False
        assert noisy["gate_reason"] == ("not a gate under the protocol (E): the CMA-ES gate is the "
                                        "four noiseless synthetic targets")
        assert T[f"lassobench_pkl__cma-es__{b}__noiseless"].get("gate", True) is not False
    raw = json.loads(vb.TARGETS_FILE.read_text())
    r22 = [a for a in raw["amendments"] if a.get("item") == "R2-2"]
    assert len(r22) == 1 and r22[0]["round"] == 2
    gates = sorted(k for k, e in T.items() if k.startswith("lassobench_pkl__cma-es__")
                   and e.get("gate", True) is not False)
    assert gates == [f"lassobench_pkl__cma-es__{b}__noiseless"
                     for b in ("synt_hard", "synt_high", "synt_medium", "synt_simple")]


def _two_seed_runs(recipe="hvarfner"):
    # best-so-far at n = 3: 3.0 and 2.0 -> mean 2.5, SE = sd([3, 2]) / sqrt(2) = 0.5
    return {s: {"recipe": recipe, "sense": "min", "values": vals,
                "best_so_far": list(np.minimum.accumulate(vals))}
            for s, vals in enumerate([[5.0, 3.0, 4.0], [7.0, 5.0, 2.0]])}


def test_round2_figure_reading_takes_se_pub_equal_to_ours_when_unstated():
    """R2-3: tol = read_unc + 2 sqrt(SE_ours^2 + SE_pub^2), SE_pub := SE_ours when the source states no
    SE (recorded as assumed_equal_to_ours); pass_strict keeps the old rule (SE_pub = 0). The row at
    value 3.65 passes ONLY with SE_pub = SE_ours."""
    vb = _vb()
    t = {"id": "f", "n_reps": 2, "value_kind": "figure_read",
         "statistic": "mean over repetitions of best-so-far",
         "values": [{"n": 3, "value": 3.65, "read_unc": 0.1}]}
    t["_norm"] = vb.normalise_values(t["values"])
    v = vb.verdict_for(t, None, _two_seed_runs())
    (row,) = v["rows"]
    assert row["ours"] == pytest.approx(2.5) and row["ours_se"] == pytest.approx(0.5)
    assert row["target_se_from"] == "assumed_equal_to_ours" and row["target_se"] == pytest.approx(0.5)
    assert row["tolerance"] == pytest.approx(0.1 + 2 * math.sqrt(0.5))          # 1.5142
    assert row["tolerance_strict"] == pytest.approx(1.1) and row["target_se_strict"] == 0.0
    assert row["pass"] is True and row["pass_strict"] is False
    assert v["pass"] is True and v["pass_strict"] is False
    # a stated SE is used as is, in both readings
    t["values"][0]["se"] = 0.3
    t["_norm"] = vb.normalise_values(t["values"])
    (row,) = vb.verdict_for(t, None, _two_seed_runs())["rows"]
    assert row["target_se_from"] == "se" and row["target_se"] == 0.3
    assert row["tolerance"] == row["tolerance_strict"] == pytest.approx(0.1 + 2 * math.sqrt(0.25 + 0.09))


def test_round2_exact_kinds_keep_their_own_se_pub():
    vb = _vb()
    t = {"id": "e", "n_reps": 2, "value_kind": "table_exact",
         "statistic": "mean over repetitions of best-so-far", "values": [{"n": 3, "value": 3.65}]}
    t["_norm"] = vb.normalise_values(t["values"])
    (row,) = vb.verdict_for(t, None, _two_seed_runs())["rows"]
    assert row["target_se_from"] == "none (0)" and row["target_se"] == 0.0
    assert row["tolerance"] == row["tolerance_strict"] == pytest.approx(1.0)       # 2 * 0.5
    assert row["pass"] is False and row["pass_strict"] is False
    vec = {"id": "v", "n_reps": 2, "value_kind": "vector_figure_exact",
           "statistic": "mean over repetitions of best-so-far",
           "values": [{"n": 3, "value": 3.6, "se": 0.25}]}
    vec["_norm"] = vb.normalise_values(vec["values"])
    (row,) = vb.verdict_for(vec, None, _two_seed_runs())["rows"]
    assert row["tolerance"] == pytest.approx(2 * math.sqrt(0.25 + 0.0625))          # 1.1180
    assert row["pass"] is True and row["pass_strict"] is True


def test_round2_shipped_figure_targets_assume_se_pub_equal_to_ours():
    """Every figure-reading target of the shipped file (Hvarfner Fig. 7, Xu Fig. 6, ...) has no stated
    SE, so the rule assumes SE_pub = SE_ours; the convention text and the amendment say so."""
    vb = _vb()
    T = vb.load_targets(vb.TARGETS_FILE)
    figs = sorted(k for k, e in T.items() if not str(e.get("value_kind", "")).startswith(vb.EXACT_KINDS))
    # after R2-9 the two Doumont Fig. 13 targets are vector_figure_exact
    assert figs == ["cmaes__lasso_dna__hvarfner_fig7", "gabo__ackley_sphere__fig3",
                    "hesbo_ei__branin_D100__fig1", "hvarfner_dsp__lasso_dna__fig7",
                    "saasbo__hartmann6_D100__fig3", "saasbo__lasso_dna__hvarfner_fig7",
                    "turbo1__lasso_dna__hvarfner_fig7", "xu_sbo_matern_ri__lasso_dna__fig6"]
    curve = list(np.linspace(0.40, 0.28, 1100))
    runs = {s: {"recipe": "x", "sense": "min", "values": curve,
                "best_so_far": [c + 1e-4 * s for c in curve]} for s in range(20)}
    rows = vb.verdict_for(T["hvarfner_dsp__lasso_dna__fig7"], None, runs)["rows"]
    assert rows and all(r["target_se_from"] == "assumed_equal_to_ours" for r in rows)
    assert all(r["target_se"] == r["ours_se"] for r in rows)
    raw = json.loads(vb.TARGETS_FILE.read_text())
    assert "SE_pub := SE_repro" in raw["conventions"]["acceptance"]
    (r23,) = [a for a in raw["amendments"] if a.get("item") == "R2-3"]
    assert r23["round"] == 2 and all(f in r23["changes"][1] for f in figs)
    assert "doumont_linear__guacamol_aabo__fig13" in r23["changes"][1]   # it was a figure read then


#: delta_b measured on kajman (/data2/prusek/cr_verify/delta/delta_x0.py, sha256 aaaea09f...,
#: /data2/prusek/NeurIPS/.venv312: LassoBench git cb9702e8, scikit-learn 1.8.0, numpy 2.4.4, celer 0.7.4)
DELTA = {"synt_simple": 0.0032894230634687867, "synt_medium": 0.0004503141645827083,
         "synt_high": 0.00012309060631257167, "synt_hard": 0.00020740230852339518,
         "dna": 0.0032894230634687867}


def test_round2_objective_version_allowance_is_in_the_shipped_file():
    """R2-4: delta_b = |f_ours(0) - released CMA-ES x = 1| / |released| per noiseless synthetic benchmark,
    delta_dna = max_b (assumption: DNA's released x = 1 is not f(0)); provenance of the measurement."""
    vb = _vb()
    raw = json.loads(vb.TARGETS_FILE.read_text())
    a = raw["objective_version_allowance"]
    assert a["delta"] == DELTA
    meas = a["measurement"]
    assert (meas["synt_simple"]["released_cma_x1_mean"], meas["synt_simple"]["f_ours_x0_zero"]) == (
        4.6922853242769795, 4.676850412710927)
    assert (meas["synt_medium"]["released_cma_x1_mean"], meas["synt_medium"]["f_ours_x0_zero"]) == (
        4.617671223573554, 4.619750626332915)
    assert (meas["synt_high"]["released_cma_x1_mean"], meas["synt_high"]["f_ours_x0_zero"]) == (
        5.3837408761571925, 5.383078188228517)
    assert (meas["synt_hard"]["released_cma_x1_mean"], meas["synt_hard"]["f_ours_x0_zero"]) == (
        7.876416298523859, 7.878049885447064)
    for b, m in meas.items():
        assert m["delta"] == pytest.approx(abs(m["f_ours_x0_zero"] - m["released_cma_x1_mean"])
                                           / abs(m["released_cma_x1_mean"]), rel=1e-12)
        assert m["f_ours_deterministic"] is True
    prov = a["provenance"]
    assert prov["script_sha256"] == "aaaea09fdb10f715c1e3261ae77c1a0cd7bb243deb2509d04d2c7c2d37821c07"
    assert prov["host"] == "kajman.utia.cas.cz"
    assert prov["packages"]["LassoBench_direct_url"]["vcs_info"]["commit_id"] == (
        "cb9702e87c07b83e4c175ab88155ad8272edea0c")
    assert (prov["packages"]["scikit-learn"], prov["packages"]["numpy"], prov["packages"]["celer"]) == (
        "1.8.0", "2.4.4", "0.7.4")
    assert "assumption" in a["dna_assumption"]
    (r24,) = [x for x in raw["amendments"] if x.get("item") == "R2-4"]
    assert r24["round"] == 2 and r24["before_any_result"] is True


def test_round2_allowance_covers_every_lassobench_objective_target():
    """Every target on a LassoBench objective is either mapped to a delta or listed with a reason;
    GuacaMol and other targets get none."""
    vb = _vb()
    raw = json.loads(vb.TARGETS_FILE.read_text())
    a = raw["objective_version_allowance"]
    tb, na = a["target_benchmark"], a["not_applied"]
    assert not set(tb) & set(na)
    for e in raw["targets"]:
        on_lasso = "lasso" in e["id"].lower() or "LassoBench" in str(e.get("benchmark"))
        assert on_lasso == (e["id"] in tb or e["id"] in na), e["id"]
    assert tb["hvarfner_dsp__lasso_dna__fig7"] == "dna"
    assert tb["xu_sbo_matern_ri__lasso_dna__fig6"] == "dna"
    assert tb["lassobench_pkl__cma-es__dna__real"] == "dna"
    assert tb["baxus__lasso_hard__fig3"] == "synt_hard" and tb["baxus__lasso_high__fig3"] == "synt_high"
    assert tb["lassobench_pkl__turbo__synt_medium__noiseless"] == "synt_medium"
    assert sorted(na) == ["lassobench__lassocv_reference", "lassobench__random_uniform_mean",
                          "lassobench_pkl__hesbo__leukemia__real",
                          "lassobench_pkl__random_search__leukemia__real",
                          "lassobench_pkl__turbo__leukemia__real"]
    T = vb.load_targets(vb.TARGETS_FILE)
    assert T["baxus__lasso_high__fig3"]["_objective_delta"] == DELTA["synt_high"]
    assert T["doumont_linear__guacamol_aabo__fig13"]["_objective_delta"] == 0.0


def test_round2_a_row_passes_only_with_the_objective_allowance(tmp_path):
    """R2-4: tol += delta * |value|. Ours 2.5 +- 0.5 against an exact 3.55 with SE_pub 0: the plain
    tolerance 1.0 fails (diff 1.05), the allowance 0.02 * 3.55 = 0.071 makes it pass; pass_strict keeps
    delta = 0. The delta reaches the verdict through the targets file (load_targets)."""
    vb = _vb()
    f = tmp_path / "t.json"
    f.write_text(json.dumps({
        "targets": [{"id": "lasso_x", "n_reps": 2, "value_kind": "released_file_exact",
                     "statistic": "mean over repetitions of best-so-far",
                     "values": [{"n": 3, "value": 3.55}]}],
        "objective_version_allowance": {"delta": {"synt_simple": 0.02},
                                        "target_benchmark": {"lasso_x": "synt_simple"}}}))
    t = vb.load_targets(f)["lasso_x"]
    assert t["_objective_delta"] == 0.02 and t["_objective_benchmark"] == "synt_simple"
    v = vb.verdict_for(t, None, _two_seed_runs())
    (row,) = v["rows"]
    assert row["objective_allowance"] == pytest.approx(0.071)
    assert row["tolerance"] == pytest.approx(1.071) and row["tolerance_strict"] == pytest.approx(1.0)
    assert row["pass"] is True and row["pass_strict"] is False
    assert v["pass"] is True and v["pass_strict"] is False and v["objective_delta"] == 0.02
    # the same target without the allowance fails
    t0 = dict(t, _objective_delta=0.0)
    assert vb.verdict_for(t0, None, _two_seed_runs())["pass"] is False



# ---------------------------------------------------------------------------
# Round 2, part 2a (GP reviewer): R2-5 hvarfner_allobs
# ---------------------------------------------------------------------------


def test_round2_hvarfner_allobs_equals_hvarfner_in_the_box(spies):
    """R2-5: with every observation in the box the variant's proposal is bit-identical to hvarfner."""
    X, Y = _data()
    torch.manual_seed(21)
    x_ref, rec_ref = pr.propose("hvarfner", X, Y)
    torch.manual_seed(21)
    x_all, rec_all = pr.propose("hvarfner_allobs", X, Y)
    assert torch.equal(x_ref, x_all) and rec_ref.fallback is None and rec_all.fallback is None
    a, b = spies["acq"]
    assert torch.equal(a["acqf"].X_baseline, b["acqf"].X_baseline)
    assert a["kwargs"]["options"] == b["kwargs"]["options"]
    assert a["kwargs"]["gen_candidates"].keywords == b["kwargs"]["gen_candidates"].keywords


def test_round2_hvarfner_allobs_baseline_holds_every_observation(monkeypatch):
    """R2-5: with out-of-box observations the variant's qLogNEI baseline is torch.unique of ALL rows
    (pre-C2), hvarfner's only the in-box ones; no fallback event for the variant."""
    built = []
    real_cls = pr.qLogNoisyExpectedImprovement
    monkeypatch.setattr(pr, "qLogNoisyExpectedImprovement",
                        lambda **kw: built.append(kw) or real_cls(**kw))
    X, Y = _data(outside=True)               # row 0 out of the box
    Y[0] = Y.max() + 1.0
    _, rec_all = pr.propose("hvarfner_allobs", X, Y)
    _, rec_ref = pr.propose("hvarfner", X, Y)
    assert torch.equal(built[0]["X_baseline"], torch.unique(X, dim=0))      # all 8 rows
    assert built[0]["X_baseline"].shape[0] == 8
    assert torch.equal(built[1]["X_baseline"], torch.unique(X[1:], dim=0))  # in-box rows only
    X2 = X.clone()
    X2[:, 0] = 1.2                                                          # every row out of the box
    _, rec_out = pr.propose("hvarfner_allobs", X2, Y)
    assert not any("no observation inside the box" in e for e in rec_out.events)
    assert rec_all.recipe == "hvarfner_allobs" and rec_out.fallback is None
    s = pr.recipe_settings("hvarfner_allobs", D=D)
    assert s["recipe"] == "hvarfner_allobs" and s["x_baseline"].startswith("torch.unique of ALL")
    assert {k: v for k, v in s.items() if k not in ("recipe", "variant", "x_baseline")} == {
        k: v for k, v in pr.recipe_settings("hvarfner", D=D).items()
        if k not in ("recipe", "x_baseline")}



# ---------------------------------------------------------------------------
# Round 2, part 2a: R2-6 learned-noise Papenmeier variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frozen,learned", [("msr", "msr_learned"), ("mle_scaled", "mle_scaled_learned")])
def test_round2_learned_variant_learns_the_noise_and_the_frozen_one_does_not(monkeypatch, frozen,
                                                                             learned):
    """R2-6: model.py:91-95 noiseless -> noise 1e-4 frozen; --no-noiseless -> Gamma(1.1, 0.05) likelihood,
    noise trainable (initial value 2.0 = the prior mode)."""
    models = {}
    real = pr.build_msr_reference_model

    def spy(X, Y, *, noiseless=True):
        m = real(X, Y, noiseless=noiseless)
        models[noiseless] = m
        return m

    monkeypatch.setattr(pr, "build_msr_reference_model", spy)
    X, Y = _data()
    pr.propose(frozen, X, Y)
    _, rec = pr.propose(learned, X, Y)
    assert rec.recipe == learned and rec.fallback is None
    fz, lr = models[True], models[False]
    assert fz.likelihood.noise_covar.raw_noise.requires_grad is False
    assert float(fz.likelihood.noise) == pytest.approx(1e-4, rel=1e-6)
    assert lr.likelihood.noise_covar.raw_noise.requires_grad is True
    s = pr.recipe_settings(learned, D=D)
    assert "noise LEARNED" in s["likelihood"] and s["recipe"] == learned
    assert {k: v for k, v in s.items() if k not in ("recipe", "variant", "likelihood")} == {
        k: v for k, v in pr.recipe_settings(frozen, D=D).items() if k not in ("recipe", "likelihood")}


@pytest.mark.parametrize("frozen,learned", [("msr", "msr_learned"), ("mle_scaled", "mle_scaled_learned")])
def test_round2_learned_and_frozen_variants_propose_identically_given_the_same_model(monkeypatch,
                                                                                    frozen, learned):
    """R2-6: apart from the likelihood the two variants are one recipe: with the same fitted model
    (build and fit replaced by a fixed, fitted GP) the proposals are bit-identical."""
    X, Y = _data()
    fixed = pr.build_msr_reference_model(X, pr.msr_standardize(Y))
    pr.fit_msr_reference(fixed)
    calls = []
    monkeypatch.setattr(pr, "build_msr_reference_model",
                        lambda X, Y, *, noiseless=True: calls.append(noiseless) or fixed)
    monkeypatch.setattr(pr, "fit_msr_reference", lambda model, rec=None: None)
    torch.manual_seed(13)
    x_f, _ = pr.propose(frozen, X, Y)
    torch.manual_seed(13)
    x_l, _ = pr.propose(learned, X, Y)
    assert calls == [True, False] and torch.equal(x_f, x_l)


def test_round2_learned_variant_is_the_harness_noise_learned_call(monkeypatch, tmp_path):
    """R2-6: production calls _propose_papenmeier exactly as the harness keys msr_noise_learned /
    mle_scaled_noise_learned do (raasp, noiseless=False)."""
    seen = []
    real = pr._propose_papenmeier

    def spy(X, Y, *, raasp, noiseless=True):
        seen.append((raasp, noiseless))
        return real(X, Y, raasp=raasp, noiseless=noiseless)

    monkeypatch.setattr(pr, "_propose_papenmeier", spy)
    X, Y = _data()
    pr.propose("msr_learned", X, Y)
    pr.propose("mle_scaled_learned", X, Y)
    vb = _vb()
    for key in ("msr_noise_learned", "mle_scaled_noise_learned"):
        assert vb.main(["run", "--recipe", key, "--benchmark", "ackley20", "--seed", "0",
                        "--budget", "6", "--n-init", "5", "--out-dir", str(tmp_path / "vb"),
                        "--device", "cpu"]) == 0
    assert seen == [(True, False), (False, False), (True, False), (False, False)]


def test_round2_lassobench_learned_methods_run():
    rl = _lasso_module()
    for method in ("papenmeier_published_learned", "papenmeier_mle_scaled_published_learned"):
        bench = _FakeLasso(12)
        res = rl.run_published_lasso(method, bench, 12, n_cold=5, n_iter=2, seed=3)
        assert res["status"] == "completed" and bench.n == 7
        assert "noise LEARNED" in res["published_recipe"]["settings"]["likelihood"]



# ---------------------------------------------------------------------------
# Round 2, part 2b: R2-8 Xu Fig. 6 band = stated SD
# ---------------------------------------------------------------------------


def test_round2_xu_rows_use_the_stated_sd_band_where_it_can_be_read():
    """R2-8: Xu Sec. 5 states the band is the SD across the 10 runs, so SE_pub = half-width / sqrt(10)
    where the dashed (RI) band can be separated (n = 100, 200); elsewhere the R2-3 assumption."""
    vb = _vb()
    xu = vb.load_targets(vb.TARGETS_FILE)["xu_sbo_matern_ri__lasso_dna__fig6"]
    by_n = {r["n"]: r for r in xu["values"]}
    assert (by_n[100]["band_halfwidth"], by_n[100]["band_halfwidth_read_unc"], by_n[100]["se"]) == (
        0.002, 0.00074, 0.00063)
    assert (by_n[200]["band_halfwidth"], by_n[200]["band_halfwidth_read_unc"], by_n[200]["se"]) == (
        0.0021, 0.00047, 0.00066)
    assert by_n[100]["se"] == round(0.002 / math.sqrt(10), 5)
    assert all(by_n[n]["se_from_rule"] == "assumed_equal_to_ours" and "se" not in by_n[n]
               for n in (50, 150, 250, 300))
    assert "standard deviation across all runs" in xu["band_meaning"]
    curve = list(np.linspace(0.40, 0.29, 400))
    runs = {s: {"recipe": "xu", "sense": "min", "values": curve,
                "best_so_far": [c + 2e-4 * s for c in curve]} for s in range(10)}
    rows = {r["n"]: r for r in vb.verdict_for(xu, None, runs)["rows"]}
    r100 = rows[100]
    assert r100["target_se_from"] == "band_sd_over_sqrt_n" and r100["target_se"] == 0.00063
    se = r100["ours_se"]
    assert r100["tolerance"] == pytest.approx(
        0.0005 + 2 * math.sqrt(se ** 2 + 0.00063 ** 2) + DELTA["dna"] * 0.3264)
    assert r100["tolerance_strict"] == pytest.approx(0.0005 + 2 * math.sqrt(se ** 2 + 0.00063 ** 2))
    assert rows[50]["target_se_from"] == "assumed_equal_to_ours"
    assert rows[50]["target_se"] == rows[50]["ours_se"]



def test_round2_doumont_fig13_targets_are_vector_exact():
    """R2-9: Doumont Sec. 4 / App. C state SE for every plot and page 20 is vector: both Fig. 13 targets are
    vector_figure_exact with the band centre and half-width (literal values from page 20)."""
    vb = _vb()
    T = vb.load_targets(vb.TARGETS_FILE)
    lin, rbf = T["doumont_linear__guacamol_aabo__fig13"], T["hvarfner_dsp__guacamol_aabo__doumont_fig13"]
    for e in (lin, rbf):
        assert e["value_kind"] == "vector_figure_exact" and e["x_eval_offset"] == 0
        assert sorted(e["values"]) == ["adip", "fexo", "med1", "med2", "osmb", "pdop", "rano", "shop", "zale"]
        assert "standard error of the mean" in e["source"] and "page 20" in e["source"]
        assert e["values_previous_pixel_read"]["adip"][0]["value"] in (0.4727, 0.2662)
    assert [(r["n"], r["value"], r["se"]) for r in lin["values"]["adip"]] == [
        (100, 0.47263, 0.00994), (200, 0.50289, 0.01138), (500, 0.51645, 0.00817), (999, 0.53545, 0.00535)]
    assert [(r["n"], r["value"], r["se"]) for r in rbf["values"]["fexo"]] == [
        (100, 0.52918, 0.02146), (200, 0.55434, 0.02438), (500, 0.63277, 0.01872), (1000, 0.64857, 0.01736)]
    # the harness reads the per-task records with their se
    (r100,) = [r for r in lin["_norm"]["adip"] if r["n"] == 100]
    assert r100["se"] == 0.00994 and r100["value"] == 0.47263



# ---------------------------------------------------------------------------
# Round 2, part 2b: R2-10 harness minors
# ---------------------------------------------------------------------------


def test_round2_incomplete_verdicts_are_none_not_fail():
    """R2-10 (m4): missing seeds, or a gating row without a statistic, give pass None with the reason."""
    vb = _vb()
    t = {"id": "f", "n_reps": 2, "value_kind": "figure_read",
         "statistic": "mean over repetitions of best-so-far",
         "values": [{"n": 3, "value": 9.0, "read_unc": 0.1}, {"n": 4, "value": 9.0, "read_unc": 0.1}]}
    t["_norm"] = vb.normalise_values(t["values"])
    runs = _two_seed_runs()                       # 3 evaluations each: n = 4 has no statistic
    v = vb.verdict_for(t, None, runs)
    assert v["pass"] is None and v["pass_strict"] is None
    assert v["incomplete"] == ["no statistic at n = 4"] and v["note"].startswith("not verified")
    assert v["rows"][0]["pass"] is False          # the row itself is computed and fails
    v = vb.verdict_for(dict(t, n_reps=3), None, runs)
    assert v["incomplete"][0] == "incomplete: seeds missing [2]" and v["pass"] is None
    t2 = dict(t, values=t["values"][:1])
    t2["_norm"] = vb.normalise_values(t2["values"])
    assert vb.verdict_for(t2, None, runs)["pass"] is False    # complete and wrong: a fail


def test_round2_fallbacks_are_noted_not_refused():
    """R2-10 (m7): runs with fallbacks give a note on the verdict, not a refusal."""
    vb = _vb()
    t = {"id": "f", "n_reps": 2, "value_kind": "figure_read",
         "statistic": "mean over repetitions of best-so-far",
         "values": [{"n": 3, "value": 2.5, "read_unc": 0.1}]}
    t["_norm"] = vb.normalise_values(t["values"])
    runs = _two_seed_runs()
    runs[1]["fallback_count"] = 2
    v = vb.verdict_for(t, None, runs)
    assert v["pass"] is True and v["refusals"] == [] and v["fallback_note"] == "fallbacks in 1 runs"
    runs[1]["fallback_count"] = 0
    assert "fallback_note" not in vb.verdict_for(t, None, runs)



# ---------------------------------------------------------------------------
# Round 2, part 2c: R2-11 start-point rows
# ---------------------------------------------------------------------------


def test_round2_synt_hard_start_point_rows_are_informative():
    """R2-11: on synt_hard the released CMA-ES running minimum at n = 30 and 50 is the x = 1 value
    f(x0) = 7.87642 (argmin_x 1, sd 0): f(x0) against f(x0), informative; n >= 100 still gate."""
    vb = _vb()
    T = vb.load_targets(vb.TARGETS_FILE)
    for v in ("noiseless", "noise"):
        e = T[f"lassobench_pkl__cma-es__synt_hard__{v}"]
        by_n = {r["n"]: r for r in e["values"]}
        for n in (30, 50):
            assert by_n[n]["argmin_x"] == 1.0 and by_n[n]["gate"] is False
            assert by_n[n]["gate_reason"].startswith("start point")
        assert all(by_n[n].get("gate", True) is not False for n in (100, 200, 500))
    assert T["lassobench_pkl__cma-es__synt_hard__noiseless"]["values"][0]["value"] == 7.87642
    # every other CMA-ES record with argmin_x 1 is an n = 1 row (A5)
    for k, e in T.items():
        if k.startswith("lassobench_pkl__cma-es__") and "synt_hard" not in k:
            assert [r["n"] for r in e["values"] if r.get("argmin_x") == 1.0] == [1]
    # through the verdict: 30 CMA-ES runs, our minimum after x >= 100 well below f(x0)
    e = T["lassobench_pkl__cma-es__synt_hard__noiseless"]
    runs = {s: {"recipe": "cmaes", "sense": "min", "cma_popsize": 20, "f_x0": 7.878049885447064,
                "values": [9.0] * 60 + [3.0] * 540, "best_so_far": [3.0] * 600} for s in range(30)}
    rows = {r["n"]: r for r in vb.verdict_for(e, None, runs)["rows"]}
    assert [n for n, r in rows.items() if r["gate"]] == [100, 200, 500]
    # logged x = 1, 20, 40, ...: generations 1-3 at 9.0, from generation 4 (x = 80) at 3.0
    assert rows[30]["ours_argmin_x"] == 1.0 and rows[100]["ours_argmin_x"] == 80.0


def test_round2_harness_makes_any_both_at_x1_row_non_gating():
    """R2-11, durable: a row whose published argmin is x = 1 and whose own running-minimum argmin is
    x = 1 does not gate, even if the targets file does not mark it."""
    vb = _vb()
    t = {"id": "c", "n_reps": 2, "value_kind": "released_file_exact",
         "statistic": "running minimum over evaluation index of the across-repetition MEAN",
         "values": [{"n": 40, "x_logged": 40, "value": 5.0, "argmin_x": 1.0, "sd_at_argmin": 0.0},
                    {"n": 60, "x_logged": 60, "value": 2.0, "argmin_x": 60.0, "sd_at_argmin": 0.5}]}
    t["_norm"] = vb.normalise_values(t["values"])
    runs = {s: {"recipe": "cmaes", "sense": "min", "cma_popsize": 20, "f_x0": 5.0 + 1e-9,
                "values": [6.0] * 40 + [2.0 + 0.1 * s] * 20, "best_so_far": [0.0] * 60} for s in range(2)}
    v = vb.verdict_for(t, None, runs)
    r40, r60 = v["rows"]
    assert r40["start_point"] is True and r40["gate"] is False and r40["ours_argmin_x"] == 1.0
    assert r60["gate"] is True and "start_point" not in r60
    assert v["pass"] is True              # decided by the n = 60 row alone



def test_round2_lassobench_cmaes_evaluates_x0_first(monkeypatch):
    """R2-12: the published CMA-ES curve starts with f(x0); the production arm evaluates x0 = 0 as
    evaluation 1 of 30, then 29 pycma samples (popsize 20, sigma0 0.1, x0 = 0)."""
    rl = _lasso_module()
    made = []
    import cma
    real_es = cma.CMAEvolutionStrategy

    def es_spy(x0, sigma0, opts):
        made.append((np.asarray(x0).copy(), sigma0, opts["popsize"]))
        return real_es(x0, sigma0, opts)

    monkeypatch.setattr(cma, "CMAEvolutionStrategy", es_spy)
    bench = _FakeLasso(12)
    res = rl.run_published_lasso("cmaes_full_published", bench, 12, n_cold=5, n_iter=25, seed=42)
    f0 = _FakeLasso(12).evaluate(np.zeros(12))
    assert res["all_scores"][0] == f0 and len(res["all_scores"]) == 30 == bench.n
    (x0, sigma0, pop), = made
    assert np.array_equal(x0, np.zeros(12)) and sigma0 == 0.1 and pop == 20
    rec = res["published_recipe"]
    assert rec["full_best_trace"][0] == f0 and len(rec["full_best_trace"]) == 30
    assert rec["full_best_trace"] == list(np.minimum.accumulate(res["all_scores"]))
    assert res["best_trace"][0] == min(res["all_scores"][:5])       # shared 26-entry format
    assert rec["cma"]["generations_told"] == 1 and len(rec["steps"]) == 30
    assert "evaluation 1 is f(x0 = 0)" in rec["cold_start"]



def test_round2_cmaes_published_result_reports_the_arm_popsize():
    """R2-14: the top-level pop_size of a cmaes_full_published result is the arm's 20; the CLI value
    (default 5, used by the submission-era cmaes_full) is kept under a separate, labelled key."""
    src = (REPO / "scripts" / "run_lassobench.py").read_text()
    block = src[src.index('if method == "cmaes_full_published":'):]
    block = block[:block.index("with open(result_file")]
    assert 'result["pop_size_cli_argument_unused"] = args.pop_size' in block
    assert 'result["pop_size"] = ' in block and '"popsize")' in block
    rl = _lasso_module()
    res = rl.run_published_lasso("cmaes_full_published", _FakeLasso(6), 6, n_cold=5, n_iter=2, seed=1)
    assert ((res.get("published_recipe") or {}).get("cma") or {}).get("popsize") == 20
    # the submission-era arm is untouched: its line still writes the CLI value
    assert '"pop_size": args.pop_size if "cmaes" in method else None,' in src
    pe_src = (REPO / "src" / "spherevo" / "published_external.py").read_text()
    assert "turbo_1.py:284-292" in pe_src and "turbo_1.py:240-246" not in pe_src
