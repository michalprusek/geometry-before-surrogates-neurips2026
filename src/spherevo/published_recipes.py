"""Published GP-BO recipes as pure builder functions (camera-ready, published-recipe block).

One recipe = output standardisation, model construction, fitting with the reference's failure
handling, acquisition function, acquisition optimisation. Every function works on inputs in the
unit cube ``[0, 1]^D`` and on raw objective values ``Y`` (to be MAXIMISED), on any device. The same
functions are used by

* the GuacaMol arms (``src/spherevo/published_arms.py``),
* the LassoBench methods ``*_published`` (``scripts/run_lassobench.py``),
* the verification harness ``scripts/cr/verify_baselines.py`` (the authors' own setting).

Recipes (``RECIPES``) and their official code:

``hvarfner``    Vanilla BO, Hvarfner, Hellsten & Nardi, ICML 2024. github.com/hvarfner/
                vanilla_bo_in_highdim: ``benchmarking/gp_priors.py`` model ``default``,
                ``configs/acq/qlognei.yaml``, ``configs/acq_opt/highdim.yaml``, run through Ax
                (``main.py``). What Ax adds is reproduced here explicitly (``hvarfner_*``).
``msr``         MSR, Papenmeier, Poloczek & Nardi, ICML 2025. github.com/lpapenme/
                understanding-hdbo-code: ``model.py``, ``fit.py``, ``bo.py``, ``methods.py``; the
                acquisition call is ``_propose_msr_reference_on_unit_box`` (below): the published
                call of the existing port ``_propose_msr_published_on_unit_box``
                (``gp_competitors.py``, left unchanged for the arms that use it) with the RAASP
                centres at the observed points and the starting points clamped onto the box, as
                the authors' BoTorch does (round-1 fix C3).
``mle_scaled``  "MLE (scaled)": the same GP and fit, acquisition without RAASP (``methods.py:72-78``).
``xu``          SBO-Matern (RI), Xu, Wang, Phillips & Zhe, ICLR 2025. github.com/XZT008/
                Standard-GP-is-all-you-need-for-HDBO: ``GP_ARD_setls`` = ``BO_loop_GP_MAP`` with
                ``GP_MAP_Wrapper(ls_prior_type="Uniform", set_ls=True, if_matern=True)``.
``doumont``     Spherical linear kernel, Doumont et al. github.com/colmont/linear-bo:
                ``src/optimize.py`` (kernel vendored in ``src/spherevo/linear_bo_kernel.py``).

Failure handling. A numerical failure never ends a run: the step returns a uniform point in the
unit cube and records why (``StepRecord.fallback``). Events that the reference treats as normal
operation (MSR's Adam fit fallback, starting points clamped onto the box, clamped kernel inputs)
are recorded in ``StepRecord.events`` and are NOT fallbacks. Where the reference itself has no per-step handling
(Hvarfner via Ax, Xu's ``BO_loop_GP_MAP``, Doumont) an exception would end the authors' run; we are
more lenient and say so in the record.
"""

from __future__ import annotations

import contextlib
import functools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import gpytorch
import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.fit import fit_gpytorch_mll
from botorch.generation import gen_candidates_scipy
from botorch.models import SingleTaskGP
from botorch.models.utils.gpytorch_modules import get_gaussian_likelihood_with_gamma_prior
from botorch.optim import optimize_acqf
from botorch.optim.fit import fit_gpytorch_mll_scipy
from botorch.optim.initializers import gen_batch_initial_conditions
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior, UniformPrior

from src.spherevo.gp_competitors import (
    _BOX_FALLBACK_ERRORS,
    MSR_PUBLISHED_CHOLESKY_MAX_TRIES,
    MSR_PUBLISHED_MAXITER,
    MSR_PUBLISHED_NUM_RESTARTS,
    MSR_PUBLISHED_RAW_SAMPLES,
    ModelFittingError,
    NotPSDError,
    _DeviceSafeLogNormalPrior,
    _make_linear_bo_gp,
    _msr_published_options,
)

RECIPES = ("hvarfner", "msr", "mle_scaled", "xu", "doumont")
#: production variants (round 2): R2-5 Hvarfner with every observation in the qLogNEI baseline;
#: R2-6 MSR / MLE (scaled) with the Gaussian noise LEARNED (understanding-hdbo --no-noiseless)
RECIPE_VARIANTS = ("hvarfner_allobs", "msr_learned", "mle_scaled_learned")

#: Numerical failures that turn a step into a recorded uniform draw. Same set as the existing box
#: arms (``gp_competitors._BOX_FALLBACK_ERRORS``); a RuntimeError raised by ``optimize_acqf`` is
#: converted to ModelFittingError at its source, any other RuntimeError (device or shape bug)
#: propagates.
STEP_ERRORS: tuple[type[BaseException], ...] = _BOX_FALLBACK_ERRORS

DTYPE = torch.float64


class PublishedRecipeUndefined(ValueError):
    """The published recipe is not defined for this problem (e.g. Xu at D > 900)."""


# ---------------------------------------------------------------------------
# Device-safe priors
# ---------------------------------------------------------------------------


class _DeviceSafeUniformPrior(UniformPrior):
    """UniformPrior whose samples follow ``.to(device)``.

    ``UniformPrior`` does not bufferize ``low``/``high``, so ``.to(cuda)`` leaves them on the CPU and
    ``sample()`` (called only by BoTorch's ``sample_all_priors`` on a fit retry) returns a CPU
    tensor that the CUDA lengthscale constraint rejects (probe, PUB_build.md section 0). The draw is
    unchanged (same CPU generator, same dtype); only its device follows the anchor buffer.
    ``log_prob`` is untouched.
    """

    def __init__(self, a, b, validate_args=None, transform=None):
        super().__init__(a, b, validate_args=validate_args, transform=transform)
        self.register_buffer("_device_anchor", torch.zeros(()))

    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).to(self._device_anchor.device)

    def rsample(self, sample_shape=torch.Size()):
        return super().rsample(sample_shape).to(self._device_anchor.device)


@contextlib.contextmanager
def _torch_default_dtype(dtype: torch.dtype):
    """Temporarily set torch's default dtype (Xu's ``run_script.py:59`` sets DoubleTensor)."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def unit_bounds(D: int, device: torch.device) -> torch.Tensor:
    return torch.stack([
        torch.zeros(D, dtype=DTYPE, device=device),
        torch.ones(D, dtype=DTYPE, device=device),
    ])


def uniform_unit_point(D: int, device: torch.device) -> torch.Tensor:
    """Uniform point in [0,1]^D drawn on the CPU generator, as the references do
    (``bo.py:194`` ``torch.rand(1, dim)``; Xu ``BO_loop.py:56`` ``torch.rand(dim)``)."""
    return torch.rand(1, D, dtype=DTYPE).to(device)


def joint_gen_candidates_scipy() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """``gen_candidates_scipy`` with ``use_parallel_mode=False`` (round-1 fix C1).

    BoTorch >= 0.15.0 optimises the restarts as independent, batched L-BFGS-B problems
    (botorch 0.17.2 ``generation/gen.py:225-246``). The authors' BoTorch (Hvarfner: the version Ax
    0.3.7 pins; Xu: <= 0.8.x) optimises all restarts of a ``batch_limit`` chunk JOINTLY, as one
    L-BFGS-B problem on the summed acquisition with one ``maxiter`` and one stopping test. With
    ``use_parallel_mode=False`` 0.17.2 aggregates the whole chunk into one problem
    (``gen.py:247-258``), which is that joint problem.
    """
    return functools.partial(gen_candidates_scipy, use_parallel_mode=False)


def clamped_default_ic_generator(log: Optional[list[int]] = None) -> Callable[..., torch.Tensor]:
    """BoTorch's default initializer with its output clamped onto ``bounds`` (round-1 fix C3).

    The authors' BoTorch ranks the seeding centres at the OBSERVED points, perturbs around them,
    scores the raw points unprojected and then clamps the chosen starting points silently in
    ``gen_candidates_scipy`` (botorch 0.9.5 / 0.10 ``generation/gen.py:176``, no
    ``raise_on_violation``); 0.17.2 raises instead. Clamping the default initializer's output
    reproduces the pinned behaviour. When every starting point is inside the box (always, in the
    authors' setting) the clamp is the identity. ``log`` receives the number of clamped
    coordinates per call.
    """
    def ic_generator(acq_function, bounds, q, num_restarts, raw_samples, **kwargs):
        ics = gen_batch_initial_conditions(acq_function, bounds, q, num_restarts, raw_samples,
                                           **kwargs)
        out = torch.max(torch.min(ics, bounds[1]), bounds[0])
        if log is not None:
            log.append(int((out != ics).sum()))
        return out
    return ic_generator


# ---------------------------------------------------------------------------
# Step record
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    recipe: str
    fallback: Optional[str] = None
    events: list[str] = field(default_factory=list)
    t_fit_s: float = 0.0
    t_acq_s: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, **extra: Any) -> dict[str, Any]:
        d = {"fallback": self.fallback, "events": list(self.events),
             "t_fit_s": round(self.t_fit_s, 4), "t_acq_s": round(self.t_acq_s, 4)}
        d.update(self.info)
        d.update(extra)
        return d


def _guard_oom(e: BaseException) -> None:
    if isinstance(e, (torch.cuda.OutOfMemoryError, MemoryError)):
        raise e


# ===========================================================================
# 1) Hvarfner et al. 2024, Vanilla BO -- the ``default`` model, through Ax
# ===========================================================================
#
# Official code, file:line (vanilla_bo_in_highdim):
#   configs/model/default.yaml:9-13   ls loc 1.4, scale 1.73205; noise loc -4, scale 1
#   benchmarking/gp_priors.py:34-37   DIM_SCALING["default"] = (0.5, 0)
#   benchmarking/gp_priors.py:46-48   loc += log(D)*0.5; scale = (scale**2 + log(D)*0)**0.5
#   benchmarking/gp_priors.py:55,57   ls / noise constraint default 1e-4
#   benchmarking/gp_priors.py:91-104  RBFKernel(ard, LogNormalPrior, GreaterThan) + GaussianLikelihood
#                                     (LogNormalPrior noise, GreaterThan), no ScaleKernel
#   configs/acq/qlognei.yaml:1-5      qLogNEI, prune_baseline True
#   configs/acq_opt/highdim.yaml:1-9  raw 512, restarts 4, retry_on_optimization_warning False,
#                                     options nonnegative False, sample_around_best True,
#                                     sigma 0.1, maxiter 300, batch_limit 64
#   main.py:94-111                    Ax BOTORCH_MODULAR step with Surrogate(SingleTaskGP, ...)
# What Ax 0.3.7 adds (ext_ax/0.3.7 and the 0.3.7 wheel):
#   registry.py:88-111,170            transforms Cont_X_trans + Y_trans: UnitX (box -> [0,1]^D),
#                                     StandardizeY (np.std ddof=1; std < 1e-8 -> 1.0,
#                                     standardize_y.py:152-160); the others are no-ops here
#   surrogate.py:296-314,340-397      SingleTaskGP(train_X, train_Y, covar_module, likelihood),
#                                     no outcome/input transform (none configured), fresh kernel
#                                     from deep-copied options at every refit
#   botorch_modular/utils.py:296-297  fit_gpytorch_mll(ExactMarginalLogLikelihood) with defaults
#   acquisition.py:121-163,281-317    X_baseline = torch.unique(in-bounds observed X, dim=0)
#                                     (torch/utils.py:93-136 _filter_X_observed, model_utils.py:562)
#   torch/utils.py:494-496            posterior_transform = ScalarizedPosteriorTransform([1.0])
#   optimizer_argparse.py (_argparse_base)  sequential True; options init_batch_limit 32 and
#                                     batch_limit 5 merged UNDER the yaml options (batch_limit 64)

HVARFNER_LS_LOC = 1.4
HVARFNER_LS_SCALE = 1.73205
HVARFNER_LS_DIM_SCALING = (0.5, 0.0)
HVARFNER_LS_CONSTRAINT = 1e-4
HVARFNER_NOISE_LOC = -4.0
HVARFNER_NOISE_SCALE = 1.0
HVARFNER_NOISE_CONSTRAINT = 1e-4
HVARFNER_NUM_RESTARTS = 4
HVARFNER_RAW_SAMPLES = 512
HVARFNER_RETRY_ON_OPTIMIZATION_WARNING = False
HVARFNER_SEQUENTIAL = True


def hvarfner_options() -> dict[str, Any]:
    """``optimize_acqf`` options as Ax 0.3.7 builds them from ``highdim.yaml``."""
    return {
        "init_batch_limit": 32,          # Ax _argparse_base default
        "batch_limit": 64,               # highdim.yaml:9 (overrides Ax's 5)
        "nonnegative": False,            # highdim.yaml:5
        "sample_around_best": True,      # highdim.yaml:6
        "sample_around_best_sigma": 0.1,  # highdim.yaml:7
        "maxiter": 300,                  # highdim.yaml:8
    }


def hvarfner_lengthscale_prior_params(D: int) -> tuple[float, float]:
    loc = HVARFNER_LS_LOC + math.log(D) * HVARFNER_LS_DIM_SCALING[0]
    scale = (HVARFNER_LS_SCALE ** 2 + math.log(D) * HVARFNER_LS_DIM_SCALING[1]) ** 0.5
    return loc, scale


def hvarfner_standardize(Y: torch.Tensor) -> torch.Tensor:
    """Ax ``StandardizeY``: mean and Bessel-corrected std; std < 1e-8 (or n = 1) -> 1.0."""
    mean = Y.mean()
    std = float(Y.std()) if Y.shape[0] > 1 else 0.0
    if not std >= 1e-8:
        std = 1.0
    return (Y - mean) / std


def build_hvarfner_model(X: torch.Tensor, Y_std: torch.Tensor) -> SingleTaskGP:
    D = X.shape[-1]
    loc, scale = hvarfner_lengthscale_prior_params(D)
    covar = RBFKernel(
        ard_num_dims=D,
        lengthscale_prior=_DeviceSafeLogNormalPrior(loc=loc, scale=scale),
        lengthscale_constraint=GreaterThan(HVARFNER_LS_CONSTRAINT),
    )
    likelihood = GaussianLikelihood(
        noise_prior=_DeviceSafeLogNormalPrior(loc=HVARFNER_NOISE_LOC, scale=HVARFNER_NOISE_SCALE),
        noise_constraint=GreaterThan(HVARFNER_NOISE_CONSTRAINT),
    )
    model = SingleTaskGP(
        train_X=X, train_Y=Y_std, covar_module=covar, likelihood=likelihood,
        outcome_transform=None,
    )
    return model.to(X)


def fit_hvarfner(model: SingleTaskGP) -> None:
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))


def hvarfner_x_baseline(X: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Ax's qLogNEI baseline: ``torch.unique`` of the observations INSIDE the box (round-1 fix C2;
    Ax ``_filter_X_observed``, also with ``fit_out_of_design=True``). When no observation is in the
    box Ax raises (``acquisition.py:318``, ``not_none``); we fall back to all observations instead
    and say so (second return value True; our addition, recorded per step)."""
    inside = ((X >= 0.0) & (X <= 1.0)).all(dim=-1)
    if bool(inside.any()):
        return torch.unique(X[inside], dim=0), False
    return torch.unique(X, dim=0), True


def build_hvarfner_acqf(model: SingleTaskGP, X: torch.Tensor, *, all_observations: bool = False,
                        ) -> qLogNoisyExpectedImprovement:
    """qLogNEI as Ax builds it. ``all_observations`` (variant ``hvarfner_allobs``, round 2 R2-5):
    the baseline is ``torch.unique`` of every observation (the pre-C2 behaviour) instead of the
    in-box ones; identical whenever every observation is in the box."""
    weights = torch.ones(1, dtype=X.dtype, device=X.device)
    X_baseline = torch.unique(X, dim=0) if all_observations else hvarfner_x_baseline(X)[0]
    return qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=X_baseline,
        prune_baseline=True,
        posterior_transform=ScalarizedPosteriorTransform(weights=weights),
    )


def optimize_hvarfner_acqf(
    acqf: qLogNoisyExpectedImprovement, D: int, device: torch.device,
    *, ic_generator: Optional[Callable[..., torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs: dict[str, Any] = {}
    if ic_generator is not None:
        kwargs["ic_generator"] = ic_generator
    try:
        return optimize_acqf(
            acq_function=acqf,
            bounds=unit_bounds(D, device),
            q=1,
            num_restarts=HVARFNER_NUM_RESTARTS,
            raw_samples=HVARFNER_RAW_SAMPLES,
            options=hvarfner_options(),
            sequential=HVARFNER_SEQUENTIAL,
            retry_on_optimization_warning=HVARFNER_RETRY_ON_OPTIMIZATION_WARNING,
            gen_candidates=joint_gen_candidates_scipy(),
            **kwargs,
        )
    except RuntimeError as e:
        _guard_oom(e)
        raise ModelFittingError(f"optimize_acqf failed: {e}") from e


def propose_hvarfner(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, StepRecord]:
    return _propose_hvarfner(X, Y, all_observations=False)


def propose_hvarfner_allobs(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, StepRecord]:
    """Variant ``hvarfner_allobs`` (GuacaMol arm ``hvarfner_published_allobs``): every observation in
    the qLogNEI baseline; everything else as ``propose_hvarfner``."""
    return _propose_hvarfner(X, Y, all_observations=True)


def _propose_hvarfner(X: torch.Tensor, Y: torch.Tensor, *, all_observations: bool,
                      ) -> tuple[torch.Tensor, StepRecord]:
    rec = StepRecord("hvarfner_allobs" if all_observations else "hvarfner")
    D, device = X.shape[-1], X.device
    try:
        t0 = time.perf_counter()
        model = build_hvarfner_model(X, hvarfner_standardize(Y))
        fit_hvarfner(model)
        rec.t_fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, all_out = hvarfner_x_baseline(X)
        if all_out and not all_observations:
            rec.events.append("qLogNEI baseline: no observation inside the box -> all observations "
                              "(our addition; Ax raises here)")
        acqf = build_hvarfner_acqf(model, X, all_observations=all_observations)
        rec.info["n_baseline"] = int(acqf.X_baseline.shape[0])
        n_clamped: list[int] = []
        cand, _ = optimize_hvarfner_acqf(acqf, D, device,
                                         ic_generator=clamped_default_ic_generator(n_clamped))
        if sum(n_clamped):
            rec.events.append(f"starting points clamped onto the box: {sum(n_clamped)} coordinates "
                              f"(as the authors' BoTorch, gen.py:176)")
        rec.t_acq_s = time.perf_counter() - t0
        if not torch.isfinite(cand).all():
            raise ModelFittingError("optimize_acqf returned a non-finite candidate")
        return cand.detach().reshape(1, D), rec
    except STEP_ERRORS as e:
        rec.fallback = f"{type(e).__name__}: {e} (reference: no per-step handling; Ax ends the run)"
        return uniform_unit_point(D, device), rec


# ===========================================================================
# 2) Papenmeier et al. 2025, MSR and MLE (scaled) -- reference GP, fit, acquisition
# ===========================================================================
#
# Official code, file:line (understanding-hdbo-code @292e7251f1):
#   methods.py:43-45      l_init = sqrt(d) / 10            methods.py:65-78  msr / mle_scaled
#   model.py:75-79        MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_prior=None)
#   model.py:80-83        ScaleKernel(outputscale_prior=GammaPrior(2.0, 0.15))
#   model.py:86-89        lengthscale = torch.full((dim,), init_ls)  (default Positive constraint)
#   model.py:91-95        get_gaussian_likelihood_with_gamma_prior(); noiseless -> noise 1e-4, frozen
#   run.py:56-59          --noiseless default True; scripts/reproduce_fig8.sh:38-41 never passes
#                         --no-noiseless, so the Figure-8 path (incl. lasso-dna) is noiseless
#   model.py:97-102       SingleTaskGP(..., covar_module, likelihood), botorch 0.9.5: no outcome tf
#   bo.py:143             y_norm = (y - mean) / std (torch, Bessel)
#   fit.py:29-53          fit_gpytorch_mll under cholesky_max_tries(9); ModelFittingError ->
#                         Adam(lr 0.01) 500 steps; NotPSDError in Adam -> skip
#   bo.py:171-194         LogEI(best_f=y_norm.max()); optimize_acqf(q=1, num_restarts=5,
#                         raw_samples=512, options={"batch_limit": 1} (+ RAASP)); any exception
#                         -> torch.rand(1, dim)

MSR_LS_INIT_FACTOR = 0.1
MSR_OUTPUTSCALE_PRIOR = (2.0, 0.15)
MSR_NOISELESS_NOISE = 1e-4
MSR_ADAM_STEPS = 500
MSR_ADAM_LR = 0.01


def msr_standardize(Y: torch.Tensor) -> torch.Tensor:
    return (Y - Y.mean()) / Y.std()


def build_msr_reference_model(X: torch.Tensor, Y_norm: torch.Tensor, *, noiseless: bool = True,
                              ) -> SingleTaskGP:
    D = X.shape[-1]
    base_kernel = MaternKernel(nu=2.5, ard_num_dims=D, lengthscale_prior=None)
    covar_module = ScaleKernel(
        base_kernel=base_kernel, outputscale_prior=GammaPrior(*MSR_OUTPUTSCALE_PRIOR),
    )
    covar_module.base_kernel.lengthscale = torch.full(
        (D,), MSR_LS_INIT_FACTOR * math.sqrt(D), dtype=DTYPE, device=X.device,
    )
    likelihood = get_gaussian_likelihood_with_gamma_prior()
    if noiseless:
        likelihood.noise = torch.tensor(MSR_NOISELESS_NOISE)
        likelihood.noise.requires_grad = False
    model = SingleTaskGP(
        X, Y_norm, covar_module=covar_module, likelihood=likelihood, outcome_transform=None,
    )
    return model.to(device=X.device, dtype=DTYPE)


def fit_msr_reference(model: SingleTaskGP, rec: Optional[StepRecord] = None) -> None:
    """``fit.py:29-53``: BoTorch fit under cholesky_max_tries(9) with the Adam fallback."""
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with gpytorch.settings.cholesky_max_tries(MSR_PUBLISHED_CHOLESKY_MAX_TRIES):
        try:
            fit_gpytorch_mll(mll)
        except ModelFittingError as e:
            if rec is not None:
                rec.events.append(f"fit: ModelFittingError -> Adam fallback "
                                  f"({MSR_ADAM_STEPS} steps, lr {MSR_ADAM_LR}): {e}")
            try:
                train_x = model.train_inputs[0]
                train_y = model.train_targets
                model.train()
                optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=MSR_ADAM_LR)
                for _ in range(MSR_ADAM_STEPS):
                    optimizer.zero_grad()
                    output = model(train_x)
                    loss = -mll(output, train_y.flatten())
                    loss.backward()
                    optimizer.step()
                model.eval()
            except NotPSDError as nerr:
                if rec is not None:
                    rec.events.append(f"fit: Adam fallback NotPSDError -> fit skipped: {nerr}")
                model.eval()


def mle_scaled_options() -> dict[str, Any]:
    """``bo.py:174`` without RAASP; maxiter 2000 passed explicitly as in the MSR port."""
    return {"batch_limit": 1, "maxiter": MSR_PUBLISHED_MAXITER}


def _propose_mle_scaled_on_unit_box(gp: SingleTaskGP, best_y: float, D: int) -> torch.Tensor:
    device = gp.train_inputs[0].device
    with gpytorch.settings.cholesky_max_tries(MSR_PUBLISHED_CHOLESKY_MAX_TRIES):
        acqf = LogExpectedImprovement(model=gp, best_f=best_y)
        try:
            candidate, acq_value = optimize_acqf(
                acq_function=acqf,
                bounds=unit_bounds(D, device),
                q=1,
                num_restarts=MSR_PUBLISHED_NUM_RESTARTS,
                raw_samples=MSR_PUBLISHED_RAW_SAMPLES,
                options=mle_scaled_options(),
            )
        except RuntimeError as e:
            _guard_oom(e)
            raise ModelFittingError(f"optimize_acqf failed: {e}") from e
    if not torch.isfinite(candidate).all():
        raise ModelFittingError("optimize_acqf returned a non-finite candidate")
    return candidate.detach().reshape(1, D)


def _propose_msr_reference_on_unit_box(gp: SingleTaskGP, best_y: float, X_unit: torch.Tensor,
                                       rec: Optional[StepRecord] = None) -> torch.Tensor:
    """The published MSR call (``bo.py:171-190``) with the authors' BoTorch seeding semantics.

    Same call as ``gp_competitors._propose_msr_published_on_unit_box`` (LogEI, 5 restarts, 512 raw
    samples, ``_msr_published_options()``: batch_limit 1, RAASP sigma 1e-3, maxiter 2000, under
    cholesky_max_tries(9)), with two differences that matter only for out-of-box observations
    (GuacaMol adaptation, round-1 fix C3): ``acqf.X_baseline`` is NOT set, so BoTorch's
    ``get_X_baseline`` takes the GP training inputs, i.e. the RAASP centres are ranked and
    perturbed at the OBSERVED points (as in the reference); and the default initializer's output is
    clamped onto the box (``clamped_default_ic_generator``). For in-box observations both calls give
    identical candidates (the projection and the clamp are identities).
    """
    D = X_unit.shape[-1]
    n_clamped: list[int] = []
    with gpytorch.settings.cholesky_max_tries(MSR_PUBLISHED_CHOLESKY_MAX_TRIES):
        acqf = LogExpectedImprovement(model=gp, best_f=best_y)
        try:
            candidate, acq_value = optimize_acqf(
                acq_function=acqf,
                bounds=unit_bounds(D, X_unit.device),
                q=1,
                num_restarts=MSR_PUBLISHED_NUM_RESTARTS,
                raw_samples=MSR_PUBLISHED_RAW_SAMPLES,
                options=_msr_published_options(),
                ic_generator=clamped_default_ic_generator(n_clamped),
            )
        except RuntimeError as e:
            _guard_oom(e)
            raise ModelFittingError(f"optimize_acqf failed: {e}") from e
    if rec is not None and sum(n_clamped):
        rec.events.append(f"starting points clamped onto the box: {sum(n_clamped)} coordinates "
                          f"(as the authors' BoTorch, gen.py:176)")
    if not torch.isfinite(candidate).all():
        raise ModelFittingError(
            f"optimize_acqf returned non-finite candidate (acq_value={float(acq_value):.3g})")
    return candidate.detach().reshape(1, D)


def _propose_papenmeier(X: torch.Tensor, Y: torch.Tensor, *, raasp: bool, noiseless: bool = True,
                        ) -> tuple[torch.Tensor, StepRecord]:
    rec = StepRecord("msr" if raasp else "mle_scaled")
    D, device = X.shape[-1], X.device
    y_norm = msr_standardize(Y)
    if not torch.isfinite(y_norm).all():
        rec.fallback = ("flat-signal: y.std() = 0 -> NaN y_norm. The reference ENDS the run here: "
                        "bo.py:150-158 builds the model outside its try block and SingleTaskGP "
                        "raises InputDataError on NaN targets. Ours draws a uniform point instead "
                        "(more lenient, recorded)")
        return uniform_unit_point(D, device), rec
    try:
        t0 = time.perf_counter()
        model = build_msr_reference_model(X, y_norm, noiseless=noiseless)
        fit_msr_reference(model, rec)
        rec.t_fit_s = time.perf_counter() - t0
    except STEP_ERRORS as e:
        rec.fallback = (f"fit raised {type(e).__name__}: {e} (outside the reference's handling, "
                        f"which would end the run)")
        return uniform_unit_point(D, device), rec
    t0 = time.perf_counter()
    try:
        best_y = y_norm.max().item()
        if raasp:
            cand = _propose_msr_reference_on_unit_box(model, best_y, X, rec)
        else:
            cand = _propose_mle_scaled_on_unit_box(model, best_y, D)
        rec.t_acq_s = time.perf_counter() - t0
        return cand, rec
    except Exception as e:  # noqa: BLE001 -- bo.py:192-194 catches every exception
        _guard_oom(e)
        rec.t_acq_s = time.perf_counter() - t0
        rec.fallback = f"acquisition raised {type(e).__name__}: {e} (bo.py:192-194: uniform point)"
        return uniform_unit_point(D, device), rec


def propose_msr(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, StepRecord]:
    return _propose_papenmeier(X, Y, raasp=True)


def propose_mle_scaled(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, StepRecord]:
    return _propose_papenmeier(X, Y, raasp=False)


def propose_msr_learned(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, StepRecord]:
    """MSR with the noise LEARNED (round 2, R2-6): the call of the harness key ``msr_noise_learned``
    (``scripts/cr/verify_baselines.py``), ``_propose_papenmeier(X, Y, raasp=True, noiseless=False)``."""
    x, rec = _propose_papenmeier(X, Y, raasp=True, noiseless=False)
    rec.recipe = "msr_learned"
    return x, rec


def propose_mle_scaled_learned(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, StepRecord]:
    """MLE (scaled) with the noise LEARNED (round 2, R2-6), as the harness key
    ``mle_scaled_noise_learned``."""
    x, rec = _propose_papenmeier(X, Y, raasp=False, noiseless=False)
    rec.recipe = "mle_scaled_learned"
    return x, rec


# ===========================================================================
# 3) Xu et al. 2025, SBO-Matern (RI) = GP_ARD_setls
# ===========================================================================
#
# Official code, file:line (Standard-GP-is-all-you-need-for-HDBO):
#   baselines/run_script.py:59        torch.set_default_tensor_type(torch.DoubleTensor)
#   baselines/run_script.py:134-138   GP_ARD_setls -> BO_loop_GP_MAP(beta, if_ard=True,
#                                     optim_type="LBFGS", acqf_type="UCB", ls_prior_type="Uniform",
#                                     set_ls=True, if_matern=True); beta 1.5 (run_script.py:31)
#   baselines/GP.py:196-202           D >= 100: UniformPrior(1e-10, 30), Interval(1e-10, 30);
#                                     else UniformPrior(1e-10, 10.0), Interval(1e-10, 10.0)
#   baselines/GP.py:206-214           ScaleKernel(MaternKernel(ard, prior, constraint),
#                                     outputscale_prior=GammaPrior(2.0, 0.15)); ls = sqrt(D)
#   baselines/GP.py:235               SingleTaskGP(X, y, covar_module) -> that botorch's default
#                                     likelihood (Gamma(1.1, 0.05), App. B.3) and no outcome tf
#   baselines/GP.py:237-241           fit_gpytorch_mll(ExactMarginalLogLikelihood) (LBFGS)
#   dataset.py:39-55                  y standardised with sklearn StandardScaler (population std)
#   baselines/BO_loop.py:118          X, Y = dataset.get_data(normalize=True)
#   baselines/BO_loop.py:126-127      UpperConfidenceBound(model, beta=beta, maximize=True)
#   baselines/BO_loop.py:133-140      optimize_acqf(bounds [0,1]^d, q=1, num_restarts=10,
#                                     raw_samples=1000, options={}); no try/except in this loop;
#                                     that BoTorch (<= 0.8.x) optimises the 10 restarts jointly
#                                     (batch_limit default = num_restarts) -> C1, and its
#                                     retry_on_optimization_warning (default True) reacts to a
#                                     warning of that joint problem
#   dataset.py:57-73                  the proposal is clipped to [0,1]^d before evaluation

XU_HIGH_D_THRESHOLD = 100
XU_UNIFORM_HIGH_D = (1e-10, 30.0)
XU_UNIFORM_LOW_D = (1e-10, 10.0)
XU_OUTPUTSCALE_PRIOR = (2.0, 0.15)
XU_BETA = 1.5
XU_NUM_RESTARTS = 10
XU_RAW_SAMPLES = 1000


def xu_lengthscale_box(D: int) -> tuple[float, float]:
    return XU_UNIFORM_HIGH_D if D >= XU_HIGH_D_THRESHOLD else XU_UNIFORM_LOW_D


def xu_recipe_defined(D: int) -> bool:
    """The official init sqrt(D) must lie inside the official constraint (GP.py:196-214)."""
    return math.sqrt(D) <= xu_lengthscale_box(D)[1]


def xu_standardize(Y: torch.Tensor) -> torch.Tensor:
    """sklearn ``StandardScaler``: population std (ddof=0); a zero std is replaced by 1."""
    std = float(Y.std(unbiased=False))
    if std == 0.0:
        std = 1.0
    return (Y - Y.mean()) / std


def build_xu_model(X: torch.Tensor, Y_std: torch.Tensor) -> SingleTaskGP:
    D = X.shape[-1]
    if not xu_recipe_defined(D):
        raise PublishedRecipeUndefined(
            f"Xu SBO-Matern (RI) is undefined at D={D}: the official initial lengthscale "
            f"sqrt(D)={math.sqrt(D):.2f} lies outside the official constraint "
            f"Interval{xu_lengthscale_box(D)} (GP.py:196-214); gpytorch refuses to set it.")
    lo, hi = xu_lengthscale_box(D)
    with _torch_default_dtype(DTYPE):
        ls_prior = _DeviceSafeUniformPrior(lo, hi)
        ls_constraint = Interval(lower_bound=lo, upper_bound=hi)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(ard_num_dims=D, lengthscale_prior=ls_prior,
                                     lengthscale_constraint=ls_constraint),
            outputscale_prior=GammaPrior(*XU_OUTPUTSCALE_PRIOR),
        )
        ls = torch.ones_like(covar_module.base_kernel.lengthscale) * math.sqrt(D)
        covar_module.base_kernel._set_lengthscale(ls)
        likelihood = get_gaussian_likelihood_with_gamma_prior()
    model = SingleTaskGP(
        X, Y_std, covar_module=covar_module, likelihood=likelihood, outcome_transform=None,
    )
    return model.to(X)


def fit_xu(model: SingleTaskGP) -> None:
    model.train()
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))


def build_xu_acqf(model: SingleTaskGP) -> UpperConfidenceBound:
    return UpperConfidenceBound(model=model, beta=XU_BETA, maximize=True)


def optimize_xu_acqf(acqf: UpperConfidenceBound, D: int, device: torch.device,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return optimize_acqf(
            acq_function=acqf,
            bounds=unit_bounds(D, device),
            q=1,
            num_restarts=XU_NUM_RESTARTS,
            raw_samples=XU_RAW_SAMPLES,
            options={},
            gen_candidates=joint_gen_candidates_scipy(),
        )
    except RuntimeError as e:
        _guard_oom(e)
        raise ModelFittingError(f"optimize_acqf failed: {e}") from e


def propose_xu(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, StepRecord]:
    rec = StepRecord("xu")
    D, device = X.shape[-1], X.device
    if not xu_recipe_defined(D):
        raise PublishedRecipeUndefined(f"Xu SBO-Matern (RI) is undefined at D={D}")
    try:
        t0 = time.perf_counter()
        model = build_xu_model(X, xu_standardize(Y))
        fit_xu(model)
        rec.t_fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        cand, _ = optimize_xu_acqf(build_xu_acqf(model), D, device)
        rec.t_acq_s = time.perf_counter() - t0
        if not torch.isfinite(cand).all():
            raise ModelFittingError("optimize_acqf returned a non-finite candidate")
        return cand.detach().reshape(1, D).clamp(0.0, 1.0), rec
    except STEP_ERRORS as e:
        rec.fallback = (f"{type(e).__name__}: {e} (reference BO_loop_GP_MAP has no per-step "
                        f"handling; we draw uniformly as its sibling BO_loop_GP does, "
                        f"BO_loop.py:54-56)")
        return uniform_unit_point(D, device), rec


# ===========================================================================
# 4) Doumont et al., spherical linear kernel (linear-bo, src/optimize.py)
# ===========================================================================
#
# Official code, file:line (colmont/linear-bo):
#   src/optimize.py:38-53     ConstantMean, SphericalLinearKernel(ard_num_dims=d) (bounds (0, 1)),
#                             GaussianLikelihood(LogNormalPrior(-4, 1), GreaterThan(1e-4,
#                             initial_value=mode)), SingleTaskGP (botorch >= 0.13: default
#                             Standardize outcome transform)
#   src/optimize.py:71-91     fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_scipy) under
#                             cholesky_max_tries(10), max_cholesky_size(inf),
#                             fast_computations(log_prob=True, covar_root_decomposition=False,
#                             solves=False)
#   src/optimize.py:117-120   inputs normalised to [0,1]^d
#   src/optimize.py:159-183   standardize(Y); LogEI(model, standardize(Y).max()); optimize_acqf(
#                             q=1, gen_candidates=gen_candidates_scipy, raw_samples=512,
#                             num_restarts=4, retry_on_optimization_warning=False, options=
#                             {nonnegative False, sample_around_best True, sigma 0.1,
#                             maxiter 300, batch_limit 64}); no try/except
#   src/benchmarks/guacamol_task.py:611-616  molecular box [(-8, 8)] * 256
# The model builder is the repository's ``_make_linear_bo_gp`` (gp_competitors.py:2894-2923),
# which builds exactly this GP with a device-safe noise prior.
# Restart mode (round-1 fix C1, text round 2 R2-7): the BoTorch version behind the paper's curves is
# unknown. linear-bo is one commit with `botorch>=0.13` (pyproject.toml, requirements.txt) and no lock
# file, environment file or Dockerfile. The floor admits 0.13 / 0.14, which optimise the restarts
# jointly, and >= 0.15, which batches them independently by default (0.15.x / 0.16.x only with scipy
# 1.13-1.16; 0.17.2 with scipy < 1.18). We keep the fresh-install default (batched, no
# use_parallel_mode=False), which optimises every restart at least as well, i.e. is not weaker;
# Hvarfner and Xu, whose pinned BoTorch is known to be joint, get use_parallel_mode=False.

DOUMONT_RAW_SAMPLES = 512
DOUMONT_NUM_RESTARTS = 4
DOUMONT_RETRY_ON_OPTIMIZATION_WARNING = False
DOUMONT_CHOLESKY_MAX_TRIES = 10
DOUMONT_MOLECULAR_BOX = (-8.0, 8.0)


def doumont_options() -> dict[str, Any]:
    return {
        "nonnegative": False,
        "sample_around_best": True,
        "sample_around_best_sigma": 0.1,
        "maxiter": 300,
        "batch_limit": 64,
    }


def doumont_standardize(Y: torch.Tensor) -> torch.Tensor:
    from botorch.utils.transforms import standardize
    return standardize(Y)


def build_doumont_model(X: torch.Tensor, Y_std: torch.Tensor) -> SingleTaskGP:
    return _make_linear_bo_gp(X, Y_std, bounds=(0.0, 1.0), device=X.device)


def fit_doumont(model: SingleTaskGP) -> None:
    model.train()
    with contextlib.ExitStack() as es:
        es.enter_context(gpytorch.settings.cholesky_max_tries(DOUMONT_CHOLESKY_MAX_TRIES))
        es.enter_context(gpytorch.settings.max_cholesky_size(float("inf")))
        es.enter_context(gpytorch.settings.fast_computations(
            log_prob=True, covar_root_decomposition=False, solves=False))
        mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
        fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_scipy)


def build_doumont_acqf(model: SingleTaskGP, Y_std: torch.Tensor) -> LogExpectedImprovement:
    return LogExpectedImprovement(model, Y_std.max())


def optimize_doumont_acqf(acqf: LogExpectedImprovement, D: int, device: torch.device,
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return optimize_acqf(
            acqf,
            bounds=unit_bounds(D, device),
            q=1,
            gen_candidates=gen_candidates_scipy,
            raw_samples=DOUMONT_RAW_SAMPLES,
            num_restarts=DOUMONT_NUM_RESTARTS,
            retry_on_optimization_warning=DOUMONT_RETRY_ON_OPTIMIZATION_WARNING,
            options=doumont_options(),
        )
    except RuntimeError as e:
        _guard_oom(e)
        raise ModelFittingError(f"optimize_acqf failed: {e}") from e


def propose_doumont(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, StepRecord]:
    rec = StepRecord("doumont")
    D, device = X.shape[-1], X.device
    outside = (X < 0.0) | (X > 1.0)
    if bool(outside.any()):
        # The kernel asserts its inputs lie in its bounds; the authors' observations always do.
        rec.events.append(f"kernel inputs clamped onto the box: {int(outside.sum())} coordinates")
        X = X.clamp(0.0, 1.0)
    rec.info["n_inputs_clamped"] = int(outside.sum())
    try:
        t0 = time.perf_counter()
        Y_std = doumont_standardize(Y)
        model = build_doumont_model(X, Y_std)
        fit_doumont(model)
        model.eval()
        rec.t_fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        cand, _ = optimize_doumont_acqf(build_doumont_acqf(model, Y_std), D, device)
        rec.t_acq_s = time.perf_counter() - t0
        if not torch.isfinite(cand).all():
            raise ModelFittingError("optimize_acqf returned a non-finite candidate")
        return cand.detach().reshape(1, D), rec
    except STEP_ERRORS as e:
        rec.fallback = f"{type(e).__name__}: {e} (reference: no per-step handling)"
        return uniform_unit_point(D, device), rec


# ===========================================================================
# Dispatch, settings record, generic loop
# ===========================================================================

PROPOSE: dict[str, Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, StepRecord]]] = {
    "hvarfner": propose_hvarfner,
    "hvarfner_allobs": propose_hvarfner_allobs,
    "msr": propose_msr,
    "mle_scaled": propose_mle_scaled,
    "msr_learned": propose_msr_learned,
    "mle_scaled_learned": propose_mle_scaled_learned,
    "xu": propose_xu,
    "doumont": propose_doumont,
}


def propose(recipe: str, X_unit: torch.Tensor, Y: torch.Tensor,
            ) -> tuple[torch.Tensor, StepRecord]:
    """One BO step of ``recipe``: ``X_unit`` (n x D, unit-cube coordinates, may lie outside the
    cube in the GuacaMol adaptation), ``Y`` (n x 1, maximised). Returns a 1 x D point in [0,1]^D."""
    if recipe not in PROPOSE:
        raise ValueError(f"unknown recipe {recipe!r}; choose from {RECIPES + RECIPE_VARIANTS}")
    X_unit = X_unit.to(dtype=DTYPE)
    Y = Y.to(dtype=DTYPE, device=X_unit.device).reshape(-1, 1)
    return PROPOSE[recipe](X_unit, Y)


def recipe_settings(recipe: str, D: Optional[int] = None) -> dict[str, Any]:
    """The literal settings of a recipe, written once into every result record."""
    if recipe == "hvarfner_allobs":
        s = recipe_settings("hvarfner", D=D)
        s["variant"] = ("hvarfner_allobs (GuacaMol arm hvarfner_published_allobs, round 2 R2-5): as "
                        "hvarfner except the qLogNEI baseline")
        s["x_baseline"] = ("torch.unique of ALL observations, out-of-box ones included (the pre-C2 "
                           "behaviour); Ax 0.3.7 filters the baseline to the box in both "
                           "fit_out_of_design modes")
        s["recipe"] = recipe
        return s
    if recipe in ("msr_learned", "mle_scaled_learned"):
        s = recipe_settings(recipe[:-len("_learned")], D=D)
        s["variant"] = (f"{recipe} (round 2, R2-6): as {recipe[:-len('_learned')]} except the noise, as "
                        f"the harness key {recipe[:-len('_learned')]}_noise_learned")
        s["likelihood"] = ("get_gaussian_likelihood_with_gamma_prior(), noise LEARNED "
                           "(understanding-hdbo --no-noiseless)")
        s["recipe"] = recipe
        return s
    if recipe == "hvarfner":
        s = {"model": "SingleTaskGP, ConstantMean, RBFKernel ARD (no ScaleKernel)",
             "lengthscale_prior": "LogNormal(1.4 + 0.5 ln D, 1.73205)",
             "lengthscale_constraint": "GreaterThan(1e-4)",
             "noise_prior": "LogNormal(-4, 1)", "noise_constraint": "GreaterThan(1e-4)",
             "outcome_transform": None, "y_standardisation": "Ax StandardizeY (ddof 1)",
             "fit": "fit_gpytorch_mll defaults",
             "acquisition": "qLogNoisyExpectedImprovement(prune_baseline=True, "
                            "ScalarizedPosteriorTransform([1]))",
             "optimize_acqf": {"q": 1, "num_restarts": 4, "raw_samples": 512, "sequential": True,
                               "retry_on_optimization_warning": False,
                               "options": hvarfner_options(),
                               "gen_candidates": "gen_candidates_scipy(use_parallel_mode=False): "
                                                 "the restarts of a batch_limit chunk optimised "
                                                 "jointly, as the authors' BoTorch"},
             "x_baseline": "torch.unique of the in-box observations (Ax bounds filter); none in "
                           "the box -> all observations (our addition, recorded)",
             "initial_conditions": "BoTorch default initializer, output clamped onto the box (the "
                                   "authors' BoTorch clamps silently, gen.py:176; identity when "
                                   "every observation is in the box)"}
        if D is not None:
            s["lengthscale_prior_params"] = list(hvarfner_lengthscale_prior_params(D))
    elif recipe in ("msr", "mle_scaled"):
        opts = ({"batch_limit": 1, "sample_around_best": True, "sample_around_best_sigma": 1e-3,
                 "maxiter": MSR_PUBLISHED_MAXITER} if recipe == "msr" else mle_scaled_options())
        s = {"model": "SingleTaskGP, ConstantMean, ScaleKernel(Matern-5/2 ARD)",
             "lengthscale_init": "sqrt(D)/10", "lengthscale_constraint": "Positive (default)",
             "outputscale_prior": "Gamma(2, 0.15)",
             "likelihood": "get_gaussian_likelihood_with_gamma_prior(), noise frozen at 1e-4",
             "outcome_transform": None, "y_standardisation": "(y - mean) / std (Bessel)",
             "fit": "fit_gpytorch_mll under cholesky_max_tries(9); Adam(500, lr 0.01) fallback",
             "acquisition": "LogExpectedImprovement(best_f = max y_norm)",
             "optimize_acqf": {"q": 1, "num_restarts": 5, "raw_samples": 512, "options": opts},
             "acquisition_context": "cholesky_max_tries(9)",
             "on_acquisition_exception": "uniform point",
             "on_flat_y": "uniform point, recorded (the reference ends the run: InputDataError)"}
        if recipe == "msr":
            s["initial_conditions"] = ("RAASP centres at the observed points (GP train inputs); "
                                       "BoTorch default initializer, output clamped onto the box "
                                       "(identity when every observation is in the box)")
    elif recipe == "xu":
        s = {"model": "SingleTaskGP, ConstantMean, ScaleKernel(Matern-5/2 ARD)",
             "outputscale_prior": "Gamma(2, 0.15)", "lengthscale_init": "sqrt(D)",
             "lengthscale_prior": "Uniform(1e-10, 30) if D >= 100 else Uniform(1e-10, 10)",
             "lengthscale_constraint": "Interval(1e-10, 30) if D >= 100 else Interval(1e-10, 10)",
             "likelihood": "Gamma(1.1, 0.05), GreaterThan(1e-4) (botorch < 0.12 default)",
             "outcome_transform": None, "y_standardisation": "StandardScaler (ddof 0)",
             "fit": "fit_gpytorch_mll defaults",
             "acquisition": "UpperConfidenceBound(beta=1.5)",
             "optimize_acqf": {"q": 1, "num_restarts": 10, "raw_samples": 1000, "options": {},
                               "retry_on_optimization_warning": True,
                               "gen_candidates": "gen_candidates_scipy(use_parallel_mode=False): "
                                                 "the 10 restarts optimised jointly, as the "
                                                 "authors' BoTorch"}}
    elif recipe == "doumont":
        s = {"model": "SingleTaskGP, ConstantMean, SphericalLinearKernel(bounds (0, 1))",
             "noise_prior": "LogNormal(-4, 1)", "noise_constraint": "GreaterThan(1e-4, init mode)",
             "outcome_transform": "Standardize (botorch default)",
             "y_standardisation": "botorch.utils.standardize",
             "fit": "fit_gpytorch_mll(optimizer=fit_gpytorch_mll_scipy) under "
                    "cholesky_max_tries(10), max_cholesky_size(inf), fast_computations(log_prob)",
             "acquisition": "LogExpectedImprovement(best_f = max standardised y)",
             "optimize_acqf": {"q": 1, "num_restarts": 4, "raw_samples": 512,
                               "gen_candidates": "gen_candidates_scipy",
                               "retry_on_optimization_warning": False,
                               "options": doumont_options(),
                               "use_parallel_mode": "BoTorch default (batched independent "
                                                    "L-BFGS-B). The authors' version is unknown: "
                                                    "botorch>=0.13, no lock file; 0.13/0.14 joint, "
                                                    ">= 0.15 batched by default. Fresh-install "
                                                    "default kept (not weaker)"}}
    else:
        raise ValueError(f"unknown recipe {recipe!r}")
    s["recipe"] = recipe
    return s


def run_loop(
    recipe: str,
    evaluate: Callable[[torch.Tensor], float],
    X0_unit: torch.Tensor,
    Y0: torch.Tensor,
    n_steps: int,
    *,
    on_step: Optional[Callable[[int, torch.Tensor, float, StepRecord], None]] = None,
) -> dict[str, Any]:
    """Generic BO loop on the unit cube (LassoBench methods, verification harness).

    ``evaluate(x_unit)`` maps a 1 x D unit-cube point to the objective value to MAXIMISE.
    The GP is rebuilt and refitted from scratch at every step, as in every reference.
    """
    X = X0_unit.to(dtype=DTYPE).clone()
    Y = Y0.to(dtype=DTYPE, device=X.device).reshape(-1, 1).clone()
    steps: list[dict[str, Any]] = []
    reasons: list[str] = []
    for i in range(n_steps):
        x_next, rec = propose(recipe, X, Y)
        y_next = float(evaluate(x_next))
        X = torch.cat([X, x_next.to(X)], dim=0)
        Y = torch.cat([Y, torch.tensor([[y_next]], dtype=DTYPE, device=X.device)], dim=0)
        if rec.fallback is not None:
            reasons.append(rec.fallback)
        steps.append(rec.to_dict(step=i))
        if on_step is not None:
            on_step(i, x_next, y_next, rec)
    return {"X": X, "Y": Y, "steps": steps, "fallback_count": len(reasons),
            "fallback_reasons": reasons}
