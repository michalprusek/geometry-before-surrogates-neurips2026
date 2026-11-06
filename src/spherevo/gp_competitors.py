"""Surrogate competitors for HDBO comparison on VAE latent spaces.

Three recent GP recipes (Hvarfner 2024, Papenmeier 2025, Xu 2025) each provided
in two acquisition variants -- *on-sphere* (our adaptation) and *box*
(reproducing the original publications) -- plus Tripp's Tanimoto-GP and a
Random Forest non-Lipschitz ablation. All benchmarked under our 40-seed pipeline.

GP recipes (each in two variants — on-sphere and box):
1. HvarfnerScaledGP (2024) -- Matern-5/2 + LogNormal sqrt(D) prior + HalfNormal
   noise per Eq.3 / Sec.4 of PMLR 235:20793.
2. PapenmeierMSR (2025) -- Matern-5/2 + MLE-initialized lengthscale at 0.1*sqrt(D)
   per arXiv:2502.09198 Sec.4. Base class omits RAASP. PapenmeierMSRRaaspSolver
   is a variant (TuRBO-style trust region + RAASP-style sparse perturbations +
   candidate argmax), NOT the published acquisition maximiser; the published
   MSR call (analytic LogEI, multi-start L-BFGS-B started from Sobol points and
   BoTorch's sample_around_best RAASP points) is PapenmeierMSRPublishedSolver
   and PapenmeierMSRPublishedNormBoxSolver at the end of this module.
3. XuStandardGP (2025) -- Matern-5/2 ARD + c*sqrt(D) init per arXiv:2402.02746.

Acquisition scaffolds:
- *Solver  (on-sphere) -- argmax of qLogEI over uniform candidates on the
  empirical-radius sphere. Used as the primary VAE-latent comparison.
- *BoxSolver -- continuous L-BFGS via BoTorch optimize_acqf over the
  [-3sigma, +3sigma]^D box (sigma = per-coord std from ZINC-250K). Reproduces
  the published acquisition recipe; used to ablate the on-sphere scaffold.

Other surrogates:
4. TrippTanimotoGPSolver (2024) -- Tanimoto kernel on Morgan fingerprints
   (operates in FP space, not latent). Ref: arXiv:2406.07709.
5. RandomForestBOSolver -- sklearn RandomForestRegressor with Thompson
   sampling, partition-based (NOT Lipschitz). Used as a non-Lipschitz
   surrogate ablation.

All solvers expose `_fallback_count` and `_fallback_reasons` so the benchmark
pipeline can persist them to result JSON. Runs with nonzero fallback counts
should be excluded from aggregate statistics by analysis scripts.

All GP solvers are full-D GP-BO (no subspace projection): the competing claim
is that proper priors/kernels rescue GP-BO in high dim, so we test that claim
directly on our D=256 VAE latent space.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Optional, Protocol, runtime_checkable

import gpytorch
import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior, HalfNormalPrior, LogNormalPrior
from sklearn.ensemble import RandomForestRegressor

from src.analysis.config import (
    BOX_HALF_WIDTH,
    BOX_SIGMA_MULTIPLIER,
    ZINC_PER_COORD_BOUNDS_PATH,
    ZINC_PER_COORD_STD,
)
from src.spherevo.solver import StepByStepSolver

logger = logging.getLogger(__name__)


# GP fit exceptions we consider "numerical failure, fall back to random".
# All other exceptions (AttributeError, TypeError, ValueError from programming
# bugs, etc.) propagate — we want loud failures for coding regressions.
try:
    from linear_operator.utils.errors import NanError, NotPSDError
except ImportError:  # older gpytorch/linear_operator
    try:
        from gpytorch.utils.errors import NanError, NotPSDError
    except ImportError:
        NanError = NotPSDError = RuntimeError  # pragma: no cover
try:
    from botorch.exceptions.errors import ModelFittingError
except ImportError:
    ModelFittingError = RuntimeError  # pragma: no cover

GP_FIT_ERRORS = (NotPSDError, NanError, ModelFittingError, np.linalg.LinAlgError)

# Box-acquisition variants additionally catch torch's LinAlgError (a distinct
# class from numpy's). We deliberately do NOT catch bare RuntimeError here:
# every acquisition path that can raise a *numerical* RuntimeError (L-BFGS in
# _propose_qlogei_on_box, optimize_acqf in LinearBOBoxSolver, the acqf eval in
# PapenmeierMSRRaaspSolver) re-raises it as ModelFittingError at its source, so
# genuine numerical failures still fall back, while a real bug (shape/device
# mismatch, CUDA assert) propagates loudly instead of being silently converted
# into a random candidate.
_BOX_FALLBACK_ERRORS: tuple[type[BaseException], ...] = (
    *GP_FIT_ERRORS,
    torch.linalg.LinAlgError,
)


# ---------------------------------------------------------------------------
# Common GP acquisition helper
# ---------------------------------------------------------------------------


def _propose_qlogei_on_box(
    gp,
    best_y: float,
    D: int,
    device: torch.device,
    *,
    half_width: float = BOX_HALF_WIDTH,
    num_restarts: int = 10,
    raw_samples: int = 512,
) -> np.ndarray:
    """Propose next point via qLogExpectedImprovement on an axis-aligned box.

    Reproduces the original Hvarfner / Papenmeier / Xu acquisition recipe:
    Sobol initialization + multi-restart L-BFGS continuous optimization
    via BoTorch's `optimize_acqf`. Hvarfner's published recipe uses the
    unit hypercube [0,1]^D for unit-variance functions; we apply the
    analogous 3-sigma per-coordinate coverage rule for VAE latents whose
    empirical per-coord std is ZINC_PER_COORD_STD ~= 0.294, giving bounds
    [-3*0.294, +3*0.294]^D ~= [-0.881, +0.881]^D.

    Raises ModelFittingError when L-BFGS fails or returns a non-finite
    candidate, so the caller's GP_FIT_ERRORS handler can record a fallback
    instead of crashing the entire optimization run.
    """
    bounds = torch.tensor(
        [[-half_width] * D, [+half_width] * D],
        dtype=torch.float64, device=device,
    )
    acqf = qLogExpectedImprovement(model=gp, best_f=best_y)
    try:
        candidate, acq_value = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    except RuntimeError as e:
        # L-BFGS / scipy numerical failure -- not always a NotPSDError /
        # NanError / ModelFittingError, so without this re-raise the outer
        # GP_FIT_ERRORS handler would miss it and the run would crash.
        raise ModelFittingError(f"optimize_acqf failed: {e}") from e
    cand_np = candidate.detach().cpu().numpy().reshape(1, -1)
    if not np.all(np.isfinite(cand_np)):
        raise ModelFittingError(
            f"optimize_acqf returned non-finite candidate "
            f"(acq_value={float(acq_value):.3g}, "
            f"n_nan={int(np.isnan(cand_np).sum())})"
        )
    return cand_np


# ---------------------------------------------------------------------------
# [0,1]^D normalized-box acquisition (faithful Hvarfner/Papenmeier/Xu protocol)
#
# The original papers run GP on inputs normalized to [0,1]^D using per-coordinate
# data bounds (see Standard-GP-is-all-you-need-for-HDBO/baselines/BO_loop.py:
# `dataset.get_data(normalize=True)` then `bounds=[[0]*dim, [1]*dim]`).
#
# For VAE latents we use ZINC-250K encoded per-coord [p01, p99] as the natural
# data range. The Hvarfner LogNormal(sqrt(2)+log(D)/2, sqrt(3)) prior is
# calibrated for unit-width inputs, so an unnormalized box of width
# 6*sigma_per_coord ~= 1.76 mis-scales the lengthscale prior by the same factor.
# ---------------------------------------------------------------------------

_ZINC_BOUNDS_CACHE: dict[str, np.ndarray] | None = None


_ZINC_DEAD_SPAN_FLOOR = 1e-3  # span floor for dead dims (131/256 collapse to ~0)


def _load_zinc_per_coord_bounds(
    *, mode: str = "minmax",
) -> tuple[np.ndarray, np.ndarray]:
    """Load per-coordinate [low, high] bounds from precomputed JSON.

    mode: "minmax" uses observed [min, max] across ZINC-50K (faithful to
    Hvarfner's `dataset.get_data(normalize=True)` which min-max scales raw
    inputs); "p1p99" uses robust percentiles. The SELFIES VAE has 131/256
    collapsed dims (std<1e-3); for those we floor span at 1e-3 around the
    observed mean so normalization does not divide by zero — those dims
    freeze near zero, which matches the VAE's training distribution.
    Cached at module level.
    """
    global _ZINC_BOUNDS_CACHE
    if _ZINC_BOUNDS_CACHE is None or _ZINC_BOUNDS_CACHE.get("mode") != mode:
        from pathlib import Path
        path = Path(ZINC_PER_COORD_BOUNDS_PATH)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            path = project_root / path
        with open(path) as f:
            data = json.load(f)
        if mode == "minmax":
            low = np.asarray(data["per_coord"]["min"], dtype=np.float64)
            high = np.asarray(data["per_coord"]["max"], dtype=np.float64)
        elif mode == "p1p99":
            low = np.asarray(data["per_coord"]["p01"], dtype=np.float64)
            high = np.asarray(data["per_coord"]["p99"], dtype=np.float64)
        else:
            raise ValueError(f"unknown mode={mode!r}; want 'minmax' or 'p1p99'")
        if low.shape != high.shape:
            raise ValueError(f"low/high shape mismatch: {low.shape} vs {high.shape}")
        # Floor span at _ZINC_DEAD_SPAN_FLOOR around the midpoint to avoid
        # division-by-zero on collapsed dims.
        span = high - low
        bad = span < _ZINC_DEAD_SPAN_FLOOR
        if bad.any():
            mid = 0.5 * (low + high)
            half = 0.5 * _ZINC_DEAD_SPAN_FLOOR
            low = np.where(bad, mid - half, low)
            high = np.where(bad, mid + half, high)
        _ZINC_BOUNDS_CACHE = {"low": low, "high": high, "mode": mode}
    return _ZINC_BOUNDS_CACHE["low"], _ZINC_BOUNDS_CACHE["high"]


def _propose_qlogei_on_normalized_box(
    gp,
    best_y: float,
    D: int,
    device: torch.device,
    *,
    low: np.ndarray,
    high: np.ndarray,
    num_restarts: int = 10,
    raw_samples: int = 512,
) -> np.ndarray:
    """Original-recipe acquisition: GP fitted in [0,1]^D, candidate de-normalized.

    The GP is expected to have been trained on x_normalized = (x - low) / (high - low).
    We optimize qLogEI on bounds=[[0]*D, [1]*D] and de-normalize the result.
    """
    span = high - low
    if (span <= 0).any():
        raise ValueError(f"per-coord span has non-positive entries: min={span.min():.3g}")

    bounds = torch.tensor(
        [[0.0] * D, [1.0] * D], dtype=torch.float64, device=device,
    )
    acqf = qLogExpectedImprovement(model=gp, best_f=best_y)
    try:
        candidate, acq_value = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    except RuntimeError as e:
        raise ModelFittingError(f"optimize_acqf failed: {e}") from e
    cand_norm = candidate.detach().cpu().numpy().reshape(1, -1)
    if not np.all(np.isfinite(cand_norm)):
        raise ModelFittingError(
            f"optimize_acqf returned non-finite candidate "
            f"(acq_value={float(acq_value):.3g})"
        )
    return cand_norm * span + low  # de-normalize to raw VAE latent space


def _random_in_normalized_box(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Single uniform sample in raw latent space matching the [0,1]^D box."""
    u = np.random.uniform(0.0, 1.0, size=low.shape[0])
    return (u * (high - low) + low).reshape(1, -1)


def _random_in_box(D: int, half_width: float = BOX_HALF_WIDTH) -> np.ndarray:
    """Return a single uniform sample from the box [-half_width, +half_width]^D.

    Used by box-supported solvers as the GP-fit fallback so a fallback step
    stays in the same regime as a successful step. Falling back to an on-sphere
    sample (radius mean_norm) would mix scaffolds and contaminate a box-vs-
    sphere ablation.
    """
    return np.random.uniform(-half_width, +half_width, size=(1, D))


def _propose_qlogei_on_sphere(
    gp,
    best_y: float,
    n_candidates: int,
    D: int,
    mean_norm: float,
    device: torch.device,
) -> np.ndarray:
    """Propose next point via qLogExpectedImprovement over ON-SPHERE candidates.

    Samples candidates uniformly on S^{D-1} at `mean_norm` (matching the VAE's
    training-data latent distribution), NOT in a hypercube. Box sampling
    produces z-vectors with norm ~6x mean_norm that decode to out-of-distribution
    garbage SMILES; this fair-comparison fix constrains candidates to the same
    manifold the VAE was trained on.
    """
    u = torch.randn(n_candidates, D, dtype=torch.float64, device=device)
    u = u / u.norm(dim=-1, keepdim=True)
    X_cand = u * mean_norm

    acqf = qLogExpectedImprovement(model=gp, best_f=best_y)
    with torch.no_grad():
        acq_vals = acqf(X_cand.unsqueeze(1))
    best_idx = int(torch.argmax(acq_vals))
    return X_cand[best_idx].detach().cpu().numpy().reshape(1, -1)


def _propose_qlogei_on_prior(
    gp,
    best_y: float,
    n_candidates: int,
    D: int,
    device: torch.device,
) -> np.ndarray:
    """Propose next point via qLogEI over candidates drawn from N(0, I_D).

    Tests "the natural alternative isn't a hand-crafted box, it's the VAE's own
    Gaussian prior." Candidates have norms concentrated near sqrt(D), not at the
    empirical encoded latents' radius mean_norm.
    """
    X_cand = torch.randn(n_candidates, D, dtype=torch.float64, device=device)
    acqf = qLogExpectedImprovement(model=gp, best_f=best_y)
    with torch.no_grad():
        acq_vals = acqf(X_cand.unsqueeze(1))
    best_idx = int(torch.argmax(acq_vals))
    return X_cand[best_idx].detach().cpu().numpy().reshape(1, -1)


def _propose_qlogei_on_angular_box_at_r(
    gp,
    best_y: float,
    n_candidates: int,
    D: int,
    mean_norm: float,
    device: torch.device,
    *,
    half_width: float = BOX_HALF_WIDTH,
) -> np.ndarray:
    """Propose next point via qLogEI over candidates uniform in box, renormalized to r̄.

    Isolates angular-distribution effect: sample u uniformly in [-h,+h]^D, then
    project onto sphere at radius mean_norm. The radial coordinate matches
    on-sphere candidate generation exactly; only the angular distribution
    differs (biased toward box-corner directions, not isotropic).
    """
    u = (torch.rand(n_candidates, D, dtype=torch.float64, device=device) * 2.0 - 1.0) * half_width
    u = u / u.norm(dim=-1, keepdim=True)
    X_cand = u * mean_norm

    acqf = qLogExpectedImprovement(model=gp, best_f=best_y)
    with torch.no_grad():
        acq_vals = acqf(X_cand.unsqueeze(1))
    best_idx = int(torch.argmax(acq_vals))
    return X_cand[best_idx].detach().cpu().numpy().reshape(1, -1)


def _random_from_prior(D: int) -> np.ndarray:
    """Return a single sample from N(0, I_D), shape (1, D)."""
    return np.random.randn(1, D)


def _random_on_angular_box_at_r(
    D: int, mean_norm: float, half_width: float = BOX_HALF_WIDTH
) -> np.ndarray:
    """Return a single uniform-in-box sample renormalized to radius `mean_norm`."""
    u = np.random.uniform(-half_width, +half_width, size=D)
    u = u / np.linalg.norm(u)
    return (u * mean_norm).reshape(1, -1)


def _standardize(y: torch.Tensor) -> tuple[torch.Tensor, float, float, float]:
    """Standardize y to zero-mean unit-variance.

    Returns ``(y_standardized, mu, sd_used, sd_raw)`` where ``sd_used`` is
    floored to 1e-8 to avoid division by zero, and ``sd_raw`` is the actual
    sample std-dev. Callers should check ``sd_raw < 1e-6`` and treat that
    iteration as a fallback (incrementing ``_fallback_count``), since a
    flat signal means the GP would otherwise train on essentially-zero y
    and emit garbage proposals via 1e8-amplified standardization.
    """
    mu = float(y.mean())
    sd_raw = float(y.std())
    sd = max(sd_raw, 1e-8)
    return (y - mu) / sd, mu, sd, sd_raw


def _random_on_sphere(D: int, mean_norm: float) -> np.ndarray:
    """Return a single uniform point on S^{D-1} at radius `mean_norm`."""
    u = np.random.randn(D)
    u = u / np.linalg.norm(u)
    return (u * mean_norm).reshape(1, -1)


# ---------------------------------------------------------------------------
# Device-safe sampling for gpytorch's TransformedDistribution priors
# ---------------------------------------------------------------------------
# LogNormalPrior and HalfNormalPrior keep their parameters in ``base_dist``.
# ``Module.to(device)`` moves only the registered ``_buffered_*`` copies, so after
# ``gp.to(X)`` puts the GP on CUDA, ``sample()`` still draws on the CPU. BoTorch
# calls ``sample()`` only from ``sample_all_priors``, i.e. when a
# ``fit_gpytorch_mll`` attempt fails and the fit is retried; assigning that CPU
# sample to a CUDA hyperparameter raised "Expected all tensors to be on the same
# device" (21 crashed hvarfner_scaled_gp_box_faithful cells in E-1).
# Only the device of the sample changes: same draws from the same generator, and
# ``log_prob`` (the prior's only role inside the MLL) is untouched, so a fit that
# succeeds on its first attempt is computed exactly as before.


class _SampleOnBufferDeviceMixin:
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).to(self._buffered_scale.device)

    def rsample(self, sample_shape=torch.Size()):
        return super().rsample(sample_shape).to(self._buffered_scale.device)


class _DeviceSafeLogNormalPrior(_SampleOnBufferDeviceMixin, LogNormalPrior):
    """LogNormalPrior whose samples follow ``.to(device)``."""


class _DeviceSafeHalfNormalPrior(_SampleOnBufferDeviceMixin, HalfNormalPrior):
    """HalfNormalPrior whose samples follow ``.to(device)``."""


# ---------------------------------------------------------------------------
# 1) Hvarfner 2024 — sqrt(D)-scaled lengthscale prior
# ---------------------------------------------------------------------------


class HvarfnerScaledGPSolver(StepByStepSolver):
    """GP-BO with Hvarfner 2024 LogNormal lengthscale prior (sqrt(D) scaling).

    Matern-5/2 kernel. Lengthscale prior LogNormal(mu=sqrt(2)+0.5*log(D), sigma=sqrt(3))
    per Hvarfner et al. Eq. 3. Noise: half-normal prior per Hvarfner Section 4.
    Acquisition: qLogExpectedImprovement on uniform on-sphere candidates.

    Ref: Hvarfner et al. 2024 "Vanilla BO Performs Great in High Dimensions",
    ICML 2024 / PMLR 235:20793.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        n_candidates: int = 2048,
    ):
        super().__init__(black_box, x0, y0)
        if self._dim < 2:
            raise ValueError(f"Hvarfner GP requires D>=2, got D={self._dim}")
        if self.mean_norm <= 0:
            raise ValueError(f"mean_norm must be positive, got {self.mean_norm}")
        self.n_candidates = n_candidates
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fallback_count = 0
        self._fallback_reasons: list[str] = []

    def _make_gp(self, X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
        D = X.shape[-1]
        # Hvarfner 2024 Eq. 3: LogNormal(mu=sqrt(2)+log(D)/2, sigma=sqrt(3))
        ls_mean = math.sqrt(2.0) + 0.5 * math.log(D)
        ls_prior = _DeviceSafeLogNormalPrior(loc=ls_mean, scale=math.sqrt(3.0))

        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=D,
            lengthscale_prior=ls_prior,
        )
        covar = ScaleKernel(base_kernel)
        # Half-normal noise prior per Hvarfner Section 4 (scale = 1.0 nominal)
        likelihood = GaussianLikelihood(
            noise_prior=_DeviceSafeHalfNormalPrior(scale=1.0),
            noise_constraint=GreaterThan(1e-5),
        )

        gp = SingleTaskGP(
            train_X=X,
            train_Y=Y,
            likelihood=likelihood,
            covar_module=covar,
        )
        gp.to(X)
        return gp

    def next_candidate(self) -> np.ndarray:
        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        D = x_hist.shape[-1]

        X = torch.tensor(x_hist, dtype=torch.float64, device=self._device)
        Y = torch.tensor(y_hist.flatten(), dtype=torch.float64, device=self._device).unsqueeze(-1)
        Y_std, _, _, sd_raw = _standardize(Y)
        if sd_raw < 1e-6:
            self._fallback_count += 1
            reason = f"flat-signal: sd={sd_raw:.2e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"{type(self).__name__} flat-signal at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            return _random_on_sphere(D, self.mean_norm)

        try:
            gp = self._make_gp(X, Y_std)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()
            best_y_std = float(Y_std.max())
            return _propose_qlogei_on_sphere(
                gp, best_y_std, self.n_candidates, D, self.mean_norm, self._device
            )
        except GP_FIT_ERRORS as e:
            self._fallback_count += 1
            reason = f"{type(e).__name__}: {e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"Hvarfner GP fit FAILED at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            return _random_on_sphere(D, self.mean_norm)
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            raise


# ---------------------------------------------------------------------------
# 2) Papenmeier 2025 — MLE-initialized lengthscale at 0.1*sqrt(D)
# ---------------------------------------------------------------------------


class PapenmeierMSRSolver(StepByStepSolver):
    """GP-BO with Papenmeier MLE-Scaled (MSR) lengthscale initialization.

    Matern-5/2 kernel. Lengthscale INITIALIZED at 0.1*sqrt(D) before MLE (per
    Papenmeier Sec. 4: "initial length scale 0.1 scaled with sqrt(d)"), with
    Interval(0.01, 100) constraint during fit (no prior). Noise Gamma(1.1, 0.05).
    Acquisition: qLogEI on on-sphere candidates.

    Note: full Papenmeier-MSR additionally uses RAASP (Random Adaptive
    Axis-Subspace Projection) restarts. RAASP is NOT implemented here — this
    is the MLE-initialization variant only. Results should be interpreted as
    "Papenmeier init recipe without subspace restart".

    Ref: Papenmeier et al. 2025 "Understanding HDBO" arXiv:2502.09198.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        n_candidates: int = 2048,
    ):
        super().__init__(black_box, x0, y0)
        if self._dim < 2:
            raise ValueError(f"Papenmeier MSR requires D>=2, got D={self._dim}")
        if self.mean_norm <= 0:
            raise ValueError(f"mean_norm must be positive, got {self.mean_norm}")
        self.n_candidates = n_candidates
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fallback_count = 0
        self._fallback_reasons: list[str] = []

    def _make_gp(self, X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
        D = X.shape[-1]
        # Papenmeier et al. 2025 (arXiv:2502.09198), Sec. 4: "we choose the
        # initial length scale as 0.1 and scale with sqrt(d)", i.e. ell_0 =
        # 0.1*sqrt(D) (= sqrt(D)/10). Earlier code used sqrt(D) (10x too large),
        # which reviewer 6qvY flagged; corrected here.
        ls_init = 0.1 * math.sqrt(D)

        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=D,
            lengthscale_constraint=Interval(0.01, 100.0),
        )
        base_kernel.lengthscale = torch.full(
            (1, D), ls_init, device=X.device, dtype=X.dtype
        )

        covar = ScaleKernel(base_kernel)
        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(1.1, 0.05),
            noise_constraint=GreaterThan(1e-5),
        )

        gp = SingleTaskGP(
            train_X=X,
            train_Y=Y,
            likelihood=likelihood,
            covar_module=covar,
        )
        gp.to(X)
        return gp

    def next_candidate(self) -> np.ndarray:
        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        D = x_hist.shape[-1]

        X = torch.tensor(x_hist, dtype=torch.float64, device=self._device)
        Y = torch.tensor(y_hist.flatten(), dtype=torch.float64, device=self._device).unsqueeze(-1)
        Y_std, _, _, sd_raw = _standardize(Y)
        if sd_raw < 1e-6:
            self._fallback_count += 1
            reason = f"flat-signal: sd={sd_raw:.2e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"{type(self).__name__} flat-signal at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            return _random_on_sphere(D, self.mean_norm)

        try:
            gp = self._make_gp(X, Y_std)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()
            best_y_std = float(Y_std.max())
            return _propose_qlogei_on_sphere(
                gp, best_y_std, self.n_candidates, D, self.mean_norm, self._device
            )
        except GP_FIT_ERRORS as e:
            self._fallback_count += 1
            reason = f"{type(e).__name__}: {e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"Papenmeier MSR GP fit FAILED at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            return _random_on_sphere(D, self.mean_norm)
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            raise


# ---------------------------------------------------------------------------
# 3) Xu 2025 — Standard GP + c*sqrt(D) lengthscale init
# ---------------------------------------------------------------------------


class XuStandardGPSolver(StepByStepSolver):
    """GP-BO with Xu 2025 standard Matern-5/2 + c*sqrt(D) lengthscale init.

    Per Xu et al. ICLR 2025 "Standard Gaussian Process is All You Need for HDBO"
    (arXiv:2402.02746): default BoTorch SingleTaskGP with Matern-5/2 lengthscale
    initialized at c*sqrt(D), c = 1.0 by default. Lengthscale is softly
    constrained to [0.01, 100] during MLE to prevent drift to numerical extremes.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        n_candidates: int = 2048,
        c: float = 1.0,
    ):
        super().__init__(black_box, x0, y0)
        if self._dim < 2:
            raise ValueError(f"Xu GP requires D>=2, got D={self._dim}")
        if self.mean_norm <= 0:
            raise ValueError(f"mean_norm must be positive, got {self.mean_norm}")
        if c <= 0:
            raise ValueError(f"c must be positive, got {c}")
        self.n_candidates = n_candidates
        self.c = c
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fallback_count = 0
        self._fallback_reasons: list[str] = []

    def _make_gp(self, X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
        D = X.shape[-1]
        ls_init = self.c * math.sqrt(D)

        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=D,
            lengthscale_constraint=Interval(0.01, 100.0),
        )
        base_kernel.lengthscale = torch.full(
            (1, D), ls_init, device=X.device, dtype=X.dtype
        )
        covar = ScaleKernel(base_kernel)
        # Explicit noise config matches Papenmeier's for numerical-stability fairness
        # across GP competitors. Without explicit prior, BoTorch's default effective
        # noise floor is 1e-4 which causes MLE stagnation on plateau data typical of
        # BO exploitation (25s per fit at N=390 vs 11s with this config). The recipe
        # difference between Xu and Papenmeier remains in the lengthscale init
        # (Xu: c*sqrt(D) with c=1; Papenmeier: 0.1*sqrt(D) — Xu is 10x Papenmeier's init).
        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(1.1, 0.05),
            noise_constraint=GreaterThan(1e-5),
        )
        gp = SingleTaskGP(
            train_X=X, train_Y=Y, likelihood=likelihood, covar_module=covar
        )
        gp.to(X)
        return gp

    def next_candidate(self) -> np.ndarray:
        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        D = x_hist.shape[-1]

        X = torch.tensor(x_hist, dtype=torch.float64, device=self._device)
        Y = torch.tensor(y_hist.flatten(), dtype=torch.float64, device=self._device).unsqueeze(-1)
        Y_std, _, _, sd_raw = _standardize(Y)
        if sd_raw < 1e-6:
            self._fallback_count += 1
            reason = f"flat-signal: sd={sd_raw:.2e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"{type(self).__name__} flat-signal at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            return _random_on_sphere(D, self.mean_norm)

        try:
            gp = self._make_gp(X, Y_std)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()
            best_y_std = float(Y_std.max())
            return _propose_qlogei_on_sphere(
                gp, best_y_std, self.n_candidates, D, self.mean_norm, self._device
            )
        except GP_FIT_ERRORS as e:
            self._fallback_count += 1
            reason = f"{type(e).__name__}: {e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"Xu standard GP fit FAILED at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            return _random_on_sphere(D, self.mean_norm)
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            raise


# ---------------------------------------------------------------------------
# 4) Tripp 2024 — Tanimoto GP on Morgan fingerprints
# ---------------------------------------------------------------------------


class TanimotoKernel(gpytorch.kernels.Kernel):
    """Tanimoto (Jaccard) kernel for binary fingerprint vectors.

    k(x, y) = <x, y> / (||x||^2 + ||y||^2 - <x, y>)

    Ref: Ralaivola et al. 2005, "Graph kernels for chemical informatics".
    """

    has_lengthscale = False

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            dot = (x1 * x2).sum(dim=-1)
            sq1 = (x1 * x1).sum(dim=-1)
            sq2 = (x2 * x2).sum(dim=-1)
        else:
            dot = x1 @ x2.transpose(-2, -1)
            sq1 = (x1 * x1).sum(dim=-1, keepdim=True)
            sq2 = (x2 * x2).sum(dim=-1, keepdim=True).transpose(-2, -1)
        denom = sq1 + sq2 - dot
        return dot / denom.clamp_min(1e-6)


def _smiles_to_morgan(
    smiles: str, n_bits: int = 2048, radius: int = 2
) -> Optional[np.ndarray]:
    """Compute Morgan fingerprint of a SMILES string.

    Returns None for empty or unparseable SMILES. Callers must filter these
    rather than feed zero-vectors into the Tanimoto GP (a zero fingerprint
    has similarity 0 to every other point, which pollutes the kernel).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
    except ImportError:
        raise ImportError("rdkit is required for Tanimoto-GP; pip install rdkit")

    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float64)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class TrippTanimotoGPSolver(StepByStepSolver):
    """GP-BO on Morgan fingerprints with Tanimoto kernel (Tripp 2024).

    Operates in fingerprint space: the GP is fit on Morgan(SMILES(z)) vs the
    oracle scores; acquisition is evaluated on candidate fingerprints obtained
    by sampling candidate z uniformly on S^{D-1} at `mean_norm` and decoding.

    Parameters
    ----------
    cs_smiles : list[str] | None
        Cold-start SMILES (from CSV/Zinc). If provided, fingerprints are
        computed directly from these strings (correct). If None, falls back
        to `codec.decode(x0)` which may round-trip incorrectly for non-roundtrip
        VAEs — a warning is emitted in that case.

    Ref: Tripp & Hernandez-Lobato 2024 "Diagnosing and fixing common problems
    in BO for molecule design", arXiv:2406.07709.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        codec: Any = None,
        cs_smiles: list[str] | None = None,
        n_candidates: int = 64,
        n_bits: int = 2048,
        radius: int = 2,
    ):
        super().__init__(black_box, x0, y0)
        if codec is None:
            raise ValueError("TrippTanimotoGPSolver requires codec for SMILES decoding")
        if self.mean_norm <= 0:
            raise ValueError(f"mean_norm must be positive, got {self.mean_norm}")
        self.codec = codec
        self.n_candidates = n_candidates
        self.n_bits = n_bits
        self.radius = radius
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fallback_count = 0
        self._fallback_reasons: list[str] = []
        self._n_invalid_cold_start = 0

        # Cold-start fingerprints: prefer original SMILES (exact labels); only
        # decode x0 if caller forgot to pass them (non-roundtrip VAEs bias FPs).
        if cs_smiles is None:
            logger.warning(
                "TrippTanimotoGPSolver: cs_smiles not provided, falling back to "
                "codec.decode(x0). For non-roundtrip VAEs the FPs will not match "
                "the y0 labels; pass cs_smiles through the factory for correct "
                "Tripp 2024 semantics."
            )
            x0_t = torch.tensor(x0, dtype=torch.float32, device=self._device)
            cs_smiles = codec.decode(x0_t)
        self._history_fps: list[Optional[np.ndarray]] = [
            _smiles_to_morgan(s, n_bits, radius) for s in cs_smiles
        ]
        self._n_invalid_cold_start = sum(
            1 for fp in self._history_fps if fp is None
        )
        if self._n_invalid_cold_start > 0:
            logger.warning(
                f"TrippTanimotoGPSolver: {self._n_invalid_cold_start}/"
                f"{len(cs_smiles)} cold-start SMILES were invalid "
                f"(excluded from GP training)."
            )

        # Cache for candidate selection: (smiles, fp) set by next_candidate,
        # consumed by post_update to avoid a second decode pass.
        self._pending_smiles: Optional[str] = None
        self._pending_fp: Optional[np.ndarray] = None

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        # Use cached SMILES/fingerprint from next_candidate if available,
        # otherwise decode on demand (should only happen on external caller).
        if self._pending_fp is not None:
            self._history_fps.append(self._pending_fp)
            self._pending_smiles = None
            self._pending_fp = None
        else:
            x_t = torch.tensor(x, dtype=torch.float32, device=self._device)
            s = self.codec.decode(x_t)[0]
            self._history_fps.append(_smiles_to_morgan(s, self.n_bits, self.radius))

    def next_candidate(self) -> np.ndarray:
        _, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        D = self._dim
        mean_norm = self.mean_norm

        # Filter out invalid (None) history fingerprints — mis-parsed SMILES
        # would produce zero kernel similarity and bias the GP posterior.
        valid_hist = [(fp, y_hist[i, 0]) for i, fp in enumerate(self._history_fps)
                      if fp is not None]
        if len(valid_hist) < 3:
            self._fallback_count += 1
            self._fallback_reasons.append(
                f"insufficient_valid_fps: {len(valid_hist)}/{len(self._history_fps)}"
            )
            logger.error(
                f"Tripp Tanimoto-GP: only {len(valid_hist)} valid fingerprints "
                f"out of {len(self._history_fps)} history entries; falling back "
                f"to random on-sphere."
            )
            self._pending_smiles = None
            self._pending_fp = None
            return _random_on_sphere(D, mean_norm)

        fps_arr = np.stack([fp for fp, _ in valid_hist])
        y_arr = np.array([y for _, y in valid_hist], dtype=np.float64)

        # Sample uniformly on S^{D-1} at mean_norm (no anchor — pure GP
        # proposer, matching Tripp 2024 fair-comparison protocol).
        u = np.random.randn(self.n_candidates, D)
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        z_cand = u * mean_norm

        z_cand_t = torch.tensor(z_cand, dtype=torch.float32, device=self._device)
        smiles_cand = self.codec.decode(z_cand_t)
        fps_cand_raw = [_smiles_to_morgan(s, self.n_bits, self.radius) for s in smiles_cand]

        # Only consider candidates with valid fingerprints
        valid_cand_idx = [i for i, fp in enumerate(fps_cand_raw) if fp is not None]
        if not valid_cand_idx:
            # All candidates decoded to invalid SMILES; fall back to random
            self._fallback_count += 1
            self._fallback_reasons.append(
                f"all_candidates_invalid_smiles (n={self.n_candidates})"
            )
            logger.error(
                f"Tripp Tanimoto-GP: all {self.n_candidates} candidates decoded "
                f"to invalid SMILES at iter {self.iteration}; falling back to random."
            )
            self._pending_smiles = None
            self._pending_fp = None
            return _random_on_sphere(D, mean_norm)

        valid_fps_cand = np.stack([fps_cand_raw[i] for i in valid_cand_idx])

        X = torch.tensor(fps_arr, dtype=torch.float64, device=self._device)
        Y = torch.tensor(y_arr, dtype=torch.float64, device=self._device).unsqueeze(-1)
        Y_std, _, _, sd_raw = _standardize(Y)
        if sd_raw < 1e-6:
            self._fallback_count += 1
            reason = f"flat-signal: sd={sd_raw:.2e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"Tripp Tanimoto-GP flat-signal at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            first_valid = valid_cand_idx[0]
            self._pending_smiles = smiles_cand[first_valid]
            self._pending_fp = fps_cand_raw[first_valid]
            return z_cand[first_valid : first_valid + 1]

        try:
            kernel = ScaleKernel(TanimotoKernel())
            gp = SingleTaskGP(train_X=X, train_Y=Y_std, covar_module=kernel)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()

            X_cand = torch.tensor(valid_fps_cand, dtype=torch.float64, device=self._device)
            best_y_std = float(Y_std.max())
            acqf = qLogExpectedImprovement(model=gp, best_f=best_y_std)
            with torch.no_grad():
                acq_vals = acqf(X_cand.unsqueeze(1))
            best_valid_i = int(torch.argmax(acq_vals))
            best_orig_idx = valid_cand_idx[best_valid_i]

            # Cache the SMILES/FP for post_update
            self._pending_smiles = smiles_cand[best_orig_idx]
            self._pending_fp = fps_cand_raw[best_orig_idx]

            return z_cand[best_orig_idx : best_orig_idx + 1]
        except GP_FIT_ERRORS as e:
            self._fallback_count += 1
            reason = f"{type(e).__name__}: {e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"Tripp Tanimoto GP fit FAILED at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            # Fallback: return first valid candidate (keeps the candidate SMILES
            # on-distribution; random would be even worse)
            first_valid = valid_cand_idx[0]
            self._pending_smiles = smiles_cand[first_valid]
            self._pending_fp = fps_cand_raw[first_valid]
            return z_cand[first_valid : first_valid + 1]
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            raise


# ---------------------------------------------------------------------------
# 5) Non-Lipschitz surrogate ablation — Random Forest BO
# ---------------------------------------------------------------------------


class RandomForestBOSolver(StepByStepSolver):
    """Random-Forest BO: partition-based surrogate, NOT Lipschitz in feature map.

    Tests Proposition 3 (Surrogate Futility) scope empirically. RF does not
    satisfy the Lipschitz assumption of Proposition 3, yet we expect BBP-driven
    collapse to occur anyway. Acquisition is Thompson sampling via bootstrap
    of tree predictions (pick one tree uniformly, argmax its predictions over
    on-sphere candidates).

    Ref: Hutter et al. 2011 "SMAC" (RF for sequential model-based optimization);
    non-Lipschitz analog to GP-TS.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        n_candidates: int = 2048,
        n_estimators: int = 100,
    ):
        super().__init__(black_box, x0, y0)
        if self._dim < 2:
            raise ValueError(f"RF-BO requires D>=2, got D={self._dim}")
        if self.mean_norm <= 0:
            raise ValueError(f"mean_norm must be positive, got {self.mean_norm}")
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >=1, got {n_estimators}")
        self.n_candidates = n_candidates
        self.n_estimators = n_estimators
        self._fallback_count = 0
        self._fallback_reasons: list[str] = []

    def next_candidate(self) -> np.ndarray:
        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        D = x_hist.shape[-1]
        y_flat = y_hist.flatten()

        u = np.random.randn(self.n_candidates, D)
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        z_cand = u * self.mean_norm

        try:
            # Draw RF seed from the (per-seed-seeded) global numpy RNG so
            # runs with different outer seeds get independent RF bootstraps,
            # not seed-independent iteration-indexed ones.
            rf_seed = int(np.random.randint(0, 2**31 - 1))
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_features="sqrt",
                n_jobs=-1,
                random_state=rf_seed,
            )
            rf.fit(x_hist, y_flat)

            # Thompson sampling over trees
            tree_idx = int(np.random.randint(self.n_estimators))
            tree_preds = rf.estimators_[tree_idx].predict(z_cand)
            best_idx = int(np.argmax(tree_preds))
            return z_cand[best_idx : best_idx + 1]
        except ValueError as e:
            # sklearn raises ValueError for several distinct conditions; we
            # treat empty-history / shape mismatches as legitimate fallbacks
            # but re-raise NaN/inf in the input as programming bugs (history
            # corruption upstream). Otherwise a corrupted history silently
            # turns into a degraded random-search run.
            msg = str(e).lower()
            if "nan" in msg or "infinity" in msg or "infinite" in msg:
                raise
            self._fallback_count += 1
            reason = f"{type(e).__name__}: {e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"RF-BO fit FAILED at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            return _random_on_sphere(D, self.mean_norm)
        except (MemoryError, KeyboardInterrupt):
            raise


# ---------------------------------------------------------------------------
# Box-supported variants — reproduce the original papers' acquisition recipe
# (Sobol + L-BFGS over a hyper-rectangle, no on-sphere candidate sampling).
# Used to ablate the on-sphere scaffold against the box scaffold the original
# papers were published with. GP/prior/kernel components are inherited from
# the on-sphere variants and remain unchanged.
# ---------------------------------------------------------------------------


# Required attributes/methods on the host class for the box-acquisition free
# helper to function. Verified at first call to ``_box_next_candidate``.
_BOX_HOST_REQUIREMENTS = (
    "_make_gp", "_device", "_fallback_count", "_fallback_reasons",
    "mean_norm", "iteration", "get_history_as_arrays",
)


def _box_next_candidate(host: StepByStepSolver) -> np.ndarray:
    """Box-supported acquisition step, shared by all *BoxSolver classes.

    Replaces argmax over uniform on-sphere candidates with continuous L-BFGS
    via ``optimize_acqf`` over a hyper-rectangle of half-width
    ``host.box_half_width`` (default ``BOX_HALF_WIDTH`` = 3 sigma per coord).
    Reproduces the original Hvarfner / Papenmeier / Xu acquisition recipe so
    we can isolate the on-sphere scaffold's contribution from the GP recipe's
    contribution.

    The ``host.box_half_width`` attribute lets a subclass run the same recipe
    with a different box width for the box-radius sweep (k = 1, 2, 4, 5).
    """
    missing = [name for name in _BOX_HOST_REQUIREMENTS if not hasattr(host, name)]
    if missing:
        raise TypeError(
            f"{type(host).__name__} cannot use _box_next_candidate: "
            f"host class is missing required attributes: {missing}. "
            f"Box solvers must subclass a GP solver (HvarfnerScaledGPSolver / "
            f"PapenmeierMSRSolver / XuStandardGPSolver) that initializes "
            f"_make_gp, _device, _fallback_count, _fallback_reasons in __init__."
        )

    half_width = float(getattr(host, "box_half_width", BOX_HALF_WIDTH))

    x_hist, y_hist = host.get_history_as_arrays(penalize_nans_with=-1.0)
    D = x_hist.shape[-1]

    X = torch.tensor(x_hist, dtype=torch.float64, device=host._device)
    Y = torch.tensor(
        y_hist.flatten(), dtype=torch.float64, device=host._device
    ).unsqueeze(-1)
    Y_std, _, _, sd_raw = _standardize(Y)
    if sd_raw < 1e-6:
        host._fallback_count += 1
        reason = f"flat-signal: sd={sd_raw:.2e}"
        host._fallback_reasons.append(reason)
        logger.error(
            f"{type(host).__name__} flat-signal at iter {host.iteration} "
            f"(fallback #{host._fallback_count}): {reason}"
        )
        return _random_in_box(D, half_width=half_width)

    try:
        gp = host._make_gp(X, Y_std)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()
        best_y_std = float(Y_std.max())
        return _propose_qlogei_on_box(
            gp, best_y_std, D, host._device, half_width=half_width,
        )
    except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
        raise
    except _BOX_FALLBACK_ERRORS as e:
        host._fallback_count += 1
        reason = f"{type(e).__name__}: {e}"
        host._fallback_reasons.append(reason)
        logger.error(
            f"{type(host).__name__} GP fit FAILED at iter {host.iteration} "
            f"(fallback #{host._fallback_count}): {reason}"
        )
        return _random_in_box(D, half_width=half_width)


class HvarfnerScaledGPBoxSolver(HvarfnerScaledGPSolver):
    """Hvarfner 2024 with original box-supported acquisition (Sobol + L-BFGS).

    GP recipe identical to ``HvarfnerScaledGPSolver``: same Mat\u00e9rn-5/2 ARD
    kernel, same LogNormal sqrt(D) lengthscale prior, same HalfNormal noise
    prior. Only the acquisition step differs -- ``optimize_acqf`` over
    ``[-3*sigma, +3*sigma]^D`` instead of argmax over uniform on-sphere
    candidates. Reproduces the published acquisition setup so we can isolate
    the on-sphere scaffold's contribution from the GP recipe's contribution.

    Subclasses can override ``box_half_width`` (set in ``__init__``) for the
    box-radius sweep that tests whether the box-vs-sphere effect is driven
    by box bounds being mis-calibrated to the empirical sphere radius.
    """

    box_half_width: float = BOX_HALF_WIDTH  # k = 3 (default = 3*sigma_per_coord)

    def next_candidate(self) -> np.ndarray:
        return _box_next_candidate(self)


# Box-radius sweep variants: same Hvarfner GP recipe, only the box width
# differs. k = sigma multiplier; default Hvarfner-on-box is k = 3 (see
# BOX_SIGMA_MULTIPLIER in src/analysis/config.py). Expected uniform-sample
# norm scales as k * sigma * sqrt(D / 3); for SELFIES (sigma = 0.294, D = 256)
# the empirical sphere is at r = 7.98 and k = 3 already matches at norm ~8.15.
# Sweep tests k in {1, 2, 4, 5} -> norms ~ {2.7, 5.4, 10.9, 13.6} to dis-
# entangle "right radius" from "sphere-vs-box shape".


class HvarfnerScaledGPBox1Solver(HvarfnerScaledGPBoxSolver):
    """Hvarfner-on-box at k = 1 (box [-1*sigma, +1*sigma]^D, expected norm ~2.7)."""
    box_half_width: float = 1.0 * ZINC_PER_COORD_STD


class HvarfnerScaledGPBox2Solver(HvarfnerScaledGPBoxSolver):
    """Hvarfner-on-box at k = 2 (box [-2*sigma, +2*sigma]^D, expected norm ~5.4)."""
    box_half_width: float = 2.0 * ZINC_PER_COORD_STD


class HvarfnerScaledGPBox4Solver(HvarfnerScaledGPBoxSolver):
    """Hvarfner-on-box at k = 4 (box [-4*sigma, +4*sigma]^D, expected norm ~10.9)."""
    box_half_width: float = 4.0 * ZINC_PER_COORD_STD


class HvarfnerScaledGPBox5Solver(HvarfnerScaledGPBoxSolver):
    """Hvarfner-on-box at k = 5 (box [-5*sigma, +5*sigma]^D, expected norm ~13.6)."""
    box_half_width: float = 5.0 * ZINC_PER_COORD_STD


class _BoxFaithfulMixin:
    """Faithful published-domain box acquisition on a per-coord-rescaled [0,1]^D.

    Shared by every GP recipe whose paper calibrates its lengthscale prior for the
    unit hypercube (Hvarfner, Papenmeier MSR, Xu). The recipe-specific part is
    ``_make_gp``, inherited from whichever GP solver the concrete class also
    subclasses; everything here -- bounds loading, rescaling, and the acquisition
    step -- is recipe-independent.

    Mix in BEFORE the GP solver so this ``__init__``/``next_candidate`` win:
    ``class XBoxFaithfulSolver(_BoxFaithfulMixin, XSolver)``.
    """

    #: Overridden per codec by run_benchmark.py via the ``bounds_path`` kwarg.
    default_bounds_path: Optional[str] = None

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        n_candidates: int = 2048,
        bounds_path: Optional[str] = None,
    ):
        super().__init__(
            black_box=black_box, x0=x0, y0=y0, n_candidates=n_candidates,
        )
        if bounds_path is None:
            bounds_path = self.default_bounds_path
        if bounds_path is None:
            from src.analysis.config import ZINC_PER_COORD_BOUNDS_PATH
            bounds_path = ZINC_PER_COORD_BOUNDS_PATH
        import json as _json
        from pathlib import Path
        path = Path(bounds_path)
        if not path.is_absolute():
            for parent in (Path(__file__).resolve().parent.parent.parent,):
                cand = parent / bounds_path
                if cand.exists():
                    path = cand
                    break
        if not path.exists():
            raise FileNotFoundError(
                f"Per-coord bounds file not found at {bounds_path}; "
                f"run scripts/analysis/zinc_per_coord_minmax.py to compute it."
            )
        meta = _json.loads(path.read_text())
        per_coord = meta["per_coord"]
        p01 = np.array(per_coord["p01"], dtype=np.float64)
        p99 = np.array(per_coord["p99"], dtype=np.float64)
        if p01.shape[0] != self._dim or p99.shape[0] != self._dim:
            raise ValueError(
                f"Bounds file dim {p01.shape[0]} != solver dim {self._dim}"
            )
        width = p99 - p01
        width = np.where(width > 1e-6, width, 1e-6)
        self._lo = torch.tensor(p01, dtype=torch.float64, device=self._device)
        self._hi = torch.tensor(p99, dtype=torch.float64, device=self._device)
        self._width = torch.tensor(width, dtype=torch.float64, device=self._device)

    def _rescale_to_unit_cube(self, x: torch.Tensor) -> torch.Tensor:
        """Map x in raw latent space to [0,1]^D coords."""
        return (x - self._lo) / self._width

    def _rescale_from_unit_cube(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Inverse: [0,1]^D coords back to raw latent space."""
        return self._lo + x_norm * self._width

    def next_candidate(self) -> np.ndarray:
        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        D = x_hist.shape[-1]

        X_raw = torch.tensor(x_hist, dtype=torch.float64, device=self._device)
        X_unit = self._rescale_to_unit_cube(X_raw)
        Y = torch.tensor(
            y_hist.flatten(), dtype=torch.float64, device=self._device,
        ).unsqueeze(-1)
        Y_std, _, _, sd_raw = _standardize(Y)
        if sd_raw < 1e-6:
            self._fallback_count += 1
            reason = f"flat-signal: sd={sd_raw:.2e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"{type(self).__name__} flat-signal at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            u_unit = torch.rand(1, D, dtype=torch.float64, device=self._device)
            return self._rescale_from_unit_cube(u_unit).cpu().numpy()

        try:
            gp = self._make_gp(X_unit, Y_std)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()
            best_y_std = float(Y_std.max())
            unit_bounds = torch.tensor(
                [[0.0] * D, [1.0] * D],
                dtype=torch.float64, device=self._device,
            )
            acqf = qLogExpectedImprovement(model=gp, best_f=best_y_std)
            try:
                cand_unit, _ = optimize_acqf(
                    acq_function=acqf, bounds=unit_bounds, q=1,
                    num_restarts=10, raw_samples=512,
                )
            except RuntimeError as e:
                raise ModelFittingError(f"optimize_acqf failed: {e}") from e
            if not torch.isfinite(cand_unit).all():
                raise ModelFittingError("optimize_acqf returned non-finite candidate")
            cand_raw = self._rescale_from_unit_cube(cand_unit)
            return cand_raw.detach().cpu().numpy().reshape(1, -1)
        except _BOX_FALLBACK_ERRORS as e:
            self._fallback_count += 1
            reason = f"{type(e).__name__}: {e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"{type(self).__name__} GP fit FAILED at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            u_unit = torch.rand(1, D, dtype=torch.float64, device=self._device)
            return self._rescale_from_unit_cube(u_unit).cpu().numpy()
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            raise


class _BoxFaithfulArgmaxMixin(_BoxFaithfulMixin):
    """Faithful [0,1]^D box, but the acquisition is maximised by ARGMAX over a
    finite uniform candidate set instead of continuous multi-start L-BFGS.

    This is the missing cell of the support x acquisition-optimizer factorial.
    The submitted box-vs-sphere ablation changes both factors at once: the
    on-sphere arm draws a candidate set and takes the argmax, while the box arm
    runs ``optimize_acqf``. Everything here is identical to
    ``_BoxFaithfulMixin`` -- same GP, same unit-cube rescaling, same qLogEI --
    except the acquisition maximiser, so the difference between this solver and
    its parent isolates the optimizer, and the difference between this solver
    and the on-sphere arm isolates the support.
    """

    def next_candidate(self) -> np.ndarray:
        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        D = x_hist.shape[-1]
        X_unit = self._rescale_to_unit_cube(
            torch.tensor(x_hist, dtype=torch.float64, device=self._device)
        )
        Y = torch.tensor(
            y_hist.flatten(), dtype=torch.float64, device=self._device,
        ).unsqueeze(-1)
        Y_std, _, _, sd_raw = _standardize(Y)

        def _fallback() -> np.ndarray:
            u = torch.rand(1, D, dtype=torch.float64, device=self._device)
            return self._rescale_from_unit_cube(u).cpu().numpy()

        if sd_raw < 1e-6:
            self._fallback_count += 1
            self._fallback_reasons.append(f"flat-signal: sd={sd_raw:.2e}")
            return _fallback()
        try:
            gp = self._make_gp(X_unit, Y_std)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            gp.eval()
            cand = torch.rand(
                self.n_candidates, D, dtype=torch.float64, device=self._device,
            )
            acqf = qLogExpectedImprovement(model=gp, best_f=float(Y_std.max()))
            with torch.no_grad():
                acq_vals = acqf(cand.unsqueeze(1))
            best = cand[int(torch.argmax(acq_vals))].unsqueeze(0)
            return self._rescale_from_unit_cube(best).cpu().numpy().reshape(1, -1)
        except _BOX_FALLBACK_ERRORS as e:
            self._fallback_count += 1
            self._fallback_reasons.append(f"{type(e).__name__}: {e}")
            return _fallback()
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            raise


class HvarfnerScaledGPBoxArgmaxSolver(_BoxFaithfulArgmaxMixin, HvarfnerScaledGPSolver):
    """Hvarfner GP on the faithful [0,1]^D box, acquisition argmax over samples."""


# ---------------------------------------------------------------------------
# Control block of the camera-ready run rules (docs/camera_ready/RUN_RULES.md,
# amendment of 2026-10-12): C1 (input-transform control) and C2a (anisotropy
# control, GP arm). C2b, the surrogate-free twin of C2a, is
# ``src.spherevo.solver.RandomBoxDirSolver``.
# ---------------------------------------------------------------------------


class HvarfnerScaledGPBoxArgmaxRawSolver(HvarfnerScaledGPBoxArgmaxSolver):
    """C1 ``hvarfner_scaled_gp_box_argmax_raw``: the box x candidate-set cell
    with the GP fitted on RAW latents (input-transform control).

    ``hvarfner_scaled_gp_box_argmax`` fits its GP on per-coordinate rescaled
    inputs ``(x - p01) / width``, while the shell arm ``hvarfner_scaled_gp``
    fits the same GP constructor on raw latents, so their difference mixes the
    candidate support with the GP's input transform. This arm is
    ``hvarfner_scaled_gp_box_argmax`` with the GP fitted on raw latents, exactly
    as the shell arm fits it. Unchanged from ``hvarfner_scaled_gp_box_argmax``:
    ``__init__`` (``_BoxFaithfulMixin``: the [p01, p99] bounds of
    ``results/precomputed/zinc_per_coord_bounds.json``, width floored at 1e-6),
    the GP constructor (``HvarfnerScaledGPSolver._make_gp``), ``n_candidates``
    = 2048, the candidate draw, the qLogEI arg-max, the flat-signal rule, the
    caught exceptions, the fallback point, the fallback counters and the use of
    the random-number generators.

    Lines that differ from ``_BoxFaithfulArgmaxMixin.next_candidate`` (the
    ``hvarfner_scaled_gp_box_argmax`` step); every other line is the same:

    1. GP inputs: ``X = torch.tensor(x_hist, ...)`` instead of
       ``X_unit = self._rescale_to_unit_cube(torch.tensor(x_hist, ...))``, and
       ``self._make_gp(X, Y_std)`` instead of ``self._make_gp(X_unit, Y_std)``.
    2. Candidates: the draw ``u = torch.rand(self.n_candidates, D)`` (the
       parent's ``cand``) is mapped to raw latent coordinates,
       ``cand = self._candidates_from_unit(u)`` = ``p01 + u * width``, before the
       acquisition function sees it; the parent evaluates qLogEI on the
       unit-cube draw itself, which is its GP's input space. Both arms
       therefore score the same points of the latent space: uniform in the
       [p01, p99] box.
    3. Return value: the arg-max row of ``cand``, already raw, instead of
       ``self._rescale_from_unit_cube(<arg-max row of the draw>)``; both are
       ``p01 + u_i * width`` computed by the same element-wise operations.
    4. Fallback point: ``self._candidates_from_unit(torch.rand(1, D))``, which
       for this class is the parent's ``self._rescale_from_unit_cube(torch.rand(1, D))``.

    With bounds p01 = 0 and p99 = 1 on every coordinate both rescalings are
    the identity and the two arms compute bit-identical trajectories
    (``tests/test_control_block.py``).

    Random numbers per step (the same calls in the same order as
    ``hvarfner_scaled_gp_box_argmax``): torch's default generator of
    ``self._device`` (the CUDA generator in a GPU run) gives one
    ``torch.rand(self.n_candidates, D)`` per GP step (2048 x D float64
    uniforms) and one ``torch.rand(1, D)`` per fallback step (flat signal, or a
    caught GP error; an error raised while scoring the candidates comes after
    the candidate draw, so such a step consumes both); torch's CPU generator
    gives one ``torch.randint`` per GP step, which BoTorch draws lazily as the
    seed of qLogEI's SobolQMCNormalSampler at the first acquisition call; a fit
    whose first attempt fails is retried by BoTorch after ``sample_all_priors``
    (torch generators), as in every Hvarfner arm. NumPy's global generator is
    not used.
    """

    def _candidates_from_unit(self, u: torch.Tensor) -> torch.Tensor:
        """Uniform draws in [0,1)^D -> candidates in raw latent coordinates."""
        return self._rescale_from_unit_cube(u)

    def next_candidate(self) -> np.ndarray:
        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        D = x_hist.shape[-1]
        X = torch.tensor(x_hist, dtype=torch.float64, device=self._device)
        Y = torch.tensor(
            y_hist.flatten(), dtype=torch.float64, device=self._device,
        ).unsqueeze(-1)
        Y_std, _, _, sd_raw = _standardize(Y)

        def _fallback() -> np.ndarray:
            u = torch.rand(1, D, dtype=torch.float64, device=self._device)
            return self._candidates_from_unit(u).cpu().numpy()

        if sd_raw < 1e-6:
            self._fallback_count += 1
            self._fallback_reasons.append(f"flat-signal: sd={sd_raw:.2e}")
            return _fallback()
        try:
            gp = self._make_gp(X, Y_std)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            gp.eval()
            u = torch.rand(
                self.n_candidates, D, dtype=torch.float64, device=self._device,
            )
            cand = self._candidates_from_unit(u)
            acqf = qLogExpectedImprovement(model=gp, best_f=float(Y_std.max()))
            with torch.no_grad():
                acq_vals = acqf(cand.unsqueeze(1))
            best = cand[int(torch.argmax(acq_vals))].unsqueeze(0)
            return best.cpu().numpy().reshape(1, -1)
        except _BOX_FALLBACK_ERRORS as e:
            self._fallback_count += 1
            self._fallback_reasons.append(f"{type(e).__name__}: {e}")
            return _fallback()
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            raise


class HvarfnerScaledGPBoxDirArgmaxSolver(HvarfnerScaledGPBoxArgmaxRawSolver):
    """C2a ``hvarfner_scaled_gp_boxdir_argmax``: [p01, p99]-box directions at the
    run's shell radius, GP on raw latents, qLogEI arg-max (anisotropy control).

    The candidates are the uniform [p01, p99]-box draws of
    ``hvarfner_scaled_gp_box_argmax`` (the same ``torch.rand(2048, D)`` mapped
    to ``p01 + u * width``), each renormalised to ``self.mean_norm``, the mean
    norm of the run's cold-start latents, which is the radius of the shell arm
    ``hvarfner_scaled_gp``. A candidate thus has the shell arm's norm and a box
    draw's direction: about half the coordinates of the [p01, p99] box are
    nearly collapsed and the others run uniformly to their p01 / p99 tails, so
    these directions are anisotropic, unlike the shell arm's uniform
    directions and unlike ``hvarfner_scaled_gp_angular_box`` (an isotropic cube
    ``[-0.881, 0.881]^D`` renormalised). The GP is fitted on raw latents, as in
    C1 and in the shell arm. The fallback point (flat signal or a caught GP
    error) is one such renormalised box draw, so a fallback step stays on this
    arm's support, as in every other arm.

    Everything else, including the use of the random-number generators, is
    C1 (``HvarfnerScaledGPBoxArgmaxRawSolver``); the only overridden method is
    ``_candidates_from_unit``. The renormalisation has the operation order of
    ``_propose_qlogei_on_angular_box_at_r`` (divide by the row norm, then
    multiply by the radius). A zero-norm draw has probability zero (the box has
    positive volume) and is not special-cased, as in that function.
    """

    def _candidates_from_unit(self, u: torch.Tensor) -> torch.Tensor:
        """Uniform draws in [0,1)^D -> box draws renormalised to ``mean_norm``."""
        x = self._rescale_from_unit_cube(u)
        return x / x.norm(dim=-1, keepdim=True) * self.mean_norm


class PapenmeierMSRBoxFaithfulSolver(_BoxFaithfulMixin, PapenmeierMSRSolver):
    """Papenmeier MSR on the faithful per-coord-rescaled [0,1]^D domain.

    Table-2 companion to the on-sphere ``PapenmeierMSRSolver``. Uses the fixed
    l0 = 0.1*sqrt(D) initialization (arXiv:2502.09198 Sec. 4). Added 2026-07-25:
    the previous Papenmeier box row came from the -3*sigma ``_box`` path, which
    (a) is not the domain Section 4.4 describes and (b) predated the 0.1*sqrt(D)
    fix, making it bit-identical to the Xu box row.
    """


class XuStandardGPBoxFaithfulSolver(_BoxFaithfulMixin, XuStandardGPSolver):
    """Xu standard GP on the faithful per-coord-rescaled [0,1]^D domain.

    Table-2 companion to the on-sphere ``XuStandardGPSolver`` (l0 = c*sqrt(D),
    c = 1, arXiv:2402.02746). See ``PapenmeierMSRBoxFaithfulSolver`` for why the
    old ``_box`` rows had to be replaced.
    """


class HvarfnerScaledGPBoxFaithfulSolver(_BoxFaithfulMixin, HvarfnerScaledGPSolver):
    """Hvarfner-on-box with faithful per-coord min-max rescaling to [0,1]^D.

    Hvarfner et al. 2024 calibrate the LogNormal lengthscale prior
    ``mu = sqrt(2) + log(D)/2`` for the **unit hypercube [0,1]^D**. Our
    default ``HvarfnerScaledGPBoxSolver`` runs L-BFGS on bounds
    ``[-3*sigma, +3*sigma]^D`` (width ~1.76) without rescaling, which puts
    the prior on a different scale than Hvarfner intended. This class fixes
    that "scale fairness" issue by:

    1. Loading per-coordinate ``[p01, p99]`` bounds from
       ``results/precomputed/zinc_per_coord_bounds.json`` (precomputed from
       50K ZINC encodings).
    2. Rescaling history points to [0,1]^D before fitting the GP.
    3. Optimising acquisition on the unit hypercube.
    4. Inverse-rescaling the proposed candidate to raw latent space before
       returning to the decoder.

    This is the *faithful* Hvarfner recipe applied to the VAE latent. Compare
    to ``HvarfnerScaledGPBoxSolver`` to isolate the contribution of correct
    domain rescaling from the contribution of the box-vs-sphere scaffold.
    """

    # __init__, rescaling and next_candidate all come from _BoxFaithfulMixin.

# Sphere-radius sweep variants: same Hvarfner GP recipe, only the radius at
# which uniform on-sphere candidates are sampled differs. Default uses
# ``self.mean_norm`` (cold-start mean encoded latent norm); subclasses
# override ``sphere_radius_multiplier`` to scale that.


class HvarfnerScaledGPBoxFaithfulProjectedSolver(HvarfnerScaledGPBoxFaithfulSolver):
    """[p01,p99] L-BFGS proposal, then PROJECTED onto the empirical shell.

    The shell-support / continuous-optimizer cell of the support x
    acquisition-optimizer factorial. Its proposal lands on exactly the support the
    on-sphere arm uses, so the only thing differing from that arm is how the
    acquisition was maximised.

    Supersedes ``HvarfnerScaledGPBoxProjectedSolver``, which optimises over a
    [-3 sigma] box: paired against a [p01,p99] box arm, that variant varies the
    L-BFGS domain as well as the support, which is the very confound the factorial
    exists to remove.
    """

    def next_candidate(self) -> np.ndarray:
        x = np.asarray(super().next_candidate(), dtype=np.float64).reshape(-1)
        n = float(np.linalg.norm(x))
        if not np.isfinite(n) or n < 1e-12:
            return _random_on_sphere(self._dim, self.mean_norm)
        return (x / n * self.mean_norm).reshape(1, -1)


class HvarfnerScaledGPSphereContinuousSolver(HvarfnerScaledGPSolver):
    """Shell support, acquisition maximised CONTINUOUSLY on the shell.

    The genuinely missing cell of the support x acquisition-optimizer factorial.
    Its two existing neighbours each fail to isolate the optimizer:

      * the on-sphere arm maximises by argmax over a finite candidate set, so it
        differs from the box arm in optimizer *class*, not just in support;
      * ``HvarfnerScaledGPBoxFaithfulProjectedSolver`` maximises on the box and
        then projects, so it never maximises the acquisition *on* the shell at
        all -- it lands on the shell carrying a direction chosen elsewhere.

    Here the constraint is absorbed into the parameterisation, x = r * v/||v||,
    which makes the sphere the whole domain rather than a post-hoc projection.
    The gradient w.r.t. v is automatically tangential (the Jacobian of the
    normalisation annihilates the radial component of grad_x), so plain Adam on v
    is Riemannian ascent on S^{D-1} without an explicit retraction step.

    Multi-start settings mirror the box arm's ``optimize_acqf`` call
    (raw_samples=512 draws, the best num_restarts=10 kept as starts) so that the
    two continuous arms differ in support and nothing else.
    """

    #: Mirrors optimize_acqf(num_restarts=10, raw_samples=512) in _BoxFaithfulMixin.
    n_raw_samples: int = 512
    n_restarts: int = 10
    n_ascent_steps: int = 200
    ascent_lr: float = 0.05

    def _maximize_on_sphere(self, gp, best_y: float, D: int) -> np.ndarray:
        acqf = qLogExpectedImprovement(model=gp, best_f=best_y)

        def to_sphere(v: torch.Tensor) -> torch.Tensor:
            return v / v.norm(dim=-1, keepdim=True) * self.mean_norm

        raw = torch.randn(
            self.n_raw_samples, D, dtype=torch.float64, device=self._device,
        )
        with torch.no_grad():
            raw_vals = acqf(to_sphere(raw).unsqueeze(1))
        top = torch.topk(raw_vals, min(self.n_restarts, self.n_raw_samples)).indices
        v = raw[top].clone().requires_grad_(True)

        opt = torch.optim.Adam([v], lr=self.ascent_lr)
        for _ in range(self.n_ascent_steps):
            opt.zero_grad(set_to_none=True)
            loss = -acqf(to_sphere(v).unsqueeze(1)).sum()
            if not torch.isfinite(loss):
                break
            loss.backward()
            opt.step()

        with torch.no_grad():
            X = to_sphere(v)
            vals = acqf(X.unsqueeze(1))
            # Ascent can diverge on a badly conditioned GP; never return a point
            # worse than the best start, so this arm is a strict improvement over
            # the raw-sample argmax it was seeded from.
            if not torch.isfinite(vals).all():
                finite = torch.isfinite(vals)
                if not finite.any():
                    return to_sphere(raw[top[:1]]).cpu().numpy().reshape(1, -1)
                vals = torch.where(finite, vals, torch.full_like(vals, -float("inf")))
            best = X[int(torch.argmax(vals))]
        return best.detach().cpu().numpy().reshape(1, -1)

    def next_candidate(self) -> np.ndarray:
        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        D = x_hist.shape[-1]
        X = torch.tensor(x_hist, dtype=torch.float64, device=self._device)
        Y = torch.tensor(
            y_hist.flatten(), dtype=torch.float64, device=self._device,
        ).unsqueeze(-1)
        Y_std, _, _, sd_raw = _standardize(Y)
        if sd_raw < 1e-6:
            self._fallback_count += 1
            self._fallback_reasons.append(f"flat-signal: sd={sd_raw:.2e}")
            return _random_on_sphere(D, self.mean_norm)
        try:
            gp = self._make_gp(X, Y_std)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            gp.eval()
            return self._maximize_on_sphere(gp, float(Y_std.max()), D)
        except GP_FIT_ERRORS as e:
            self._fallback_count += 1
            self._fallback_reasons.append(f"{type(e).__name__}: {e}")
            logger.error(
                f"{type(self).__name__} GP fit FAILED at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {e}"
            )
            return _random_on_sphere(D, self.mean_norm)
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            raise


# ---------------------------------------------------------------------------
# Control C4 of the camera-ready run rules (docs/camera_ready/RUN_RULES.md,
# amendment of 2026-10-14): the shell x continuous cell with the box arm's
# maximiser.
# ---------------------------------------------------------------------------


class _ShellAcquisition(AcquisitionFunction):
    """``inner`` composed with v -> radius * v / ||v||, over an unconstrained v.

    Its value at v is ``inner`` at the shell point in v's direction, so it is
    scale-invariant in v and its gradient w.r.t. v is tangential (the Jacobian
    of the normalisation annihilates the radial component). The operation
    order of ``to_shell`` is that of ``to_sphere`` in
    ``HvarfnerScaledGPSphereContinuousSolver`` (divide by the row norm, then
    multiply by the radius).
    """

    def __init__(self, inner: AcquisitionFunction, radius: float):
        super().__init__(model=inner.model)
        self.inner = inner
        self.radius = radius

    def to_shell(self, v: torch.Tensor) -> torch.Tensor:
        return v / v.norm(dim=-1, keepdim=True) * self.radius

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.inner(self.to_shell(X))


class HvarfnerScaledGPSphereLBFGSSolver(HvarfnerScaledGPSolver):
    """C4 ``hvarfner_scaled_gp_sphere_lbfgs``: shell support, acquisition
    maximised on the shell by the box arm's maximiser.

    The decomposition's two continuous cells use different optimisers:
    ``hvarfner_scaled_gp_sphere_continuous`` (shell) runs projected Adam from
    the top 10 of 512 Gaussian starts (lr 0.05, 200 steps), while
    ``hvarfner_scaled_gp_box_faithful`` (box) calls ``optimize_acqf``, which
    selects 10 starts from 512 raw samples by Boltzmann sampling and runs
    BoTorch's batched L-BFGS-B. This arm is the Adam arm with the box arm's
    call, ``optimize_acqf(q=1, num_restarts=10, raw_samples=512)`` and nothing
    else, applied to qLogEI composed with x = r * v / ||v|| (``_ShellAcquisition``),
    v in R^D:

    * the bounds on v are -inf and +inf, so ``gen_candidates_scipy`` runs
      L-BFGS-B without bounds; the composed function is scale-invariant in v,
      so a bound on v would restrict nothing;
    * the raw samples are i.i.d. standard Gaussian v (float64, torch's CPU
      generator), i.e. uniform directions on the shell, drawn through the
      ``generator`` hook of BoTorch's own ``gen_batch_initial_conditions``,
      which then scores them and selects the starts with
      ``initialize_q_batch`` (Boltzmann, eta = 1), the rule it applies to
      qLogEI in the box arm (``is_nonnegative`` is False for both). Sobol
      points in a box of v would give non-uniform directions, the anisotropy
      that control C2 is about;
    * everything after the starts is ``optimize_acqf``'s: one batch of 10,
      L-BFGS-B with maxiter 2000, a retry with new starts after an
      optimisation warning, and the best of the 10 by acquisition value.

    Unchanged from the Adam arm: the constructor, the GP
    (``HvarfnerScaledGPSolver._make_gp`` on raw latents), qLogEI with best_f =
    the maximum of the standardised observations, the radius
    ``self.mean_norm``, the flat-signal rule and the fallback point (a uniform
    point on the shell, NumPy's global generator). Error handling is the box
    arm's (``_BoxFaithfulMixin.next_candidate``): a RuntimeError of
    ``optimize_acqf`` is re-raised as ModelFittingError, ``_BOX_FALLBACK_ERRORS``
    are caught, and a non-finite result is a ModelFittingError too; each
    fallback is counted. ``next_candidate`` is the Adam arm's, line for line,
    except the caught tuple (``_BOX_FALLBACK_ERRORS`` instead of
    ``GP_FIT_ERRORS``); its ``self._maximize_on_sphere`` is this class's.

    Random numbers per GP step: the GP fit as in every Hvarfner arm (a failed
    first attempt is retried after ``sample_all_priors``); torch's CPU
    generator gives one ``torch.randn(512, 1, D)`` (the raw samples), one
    ``torch.randint`` (the seed of qLogEI's SobolQMCNormalSampler, drawn
    lazily at the first acquisition call) and the ``torch.multinomial`` of
    ``initialize_q_batch``; a retry after an optimisation warning draws the
    raw samples and the starts again. NumPy's global generator is used only
    by a fallback step. The device's generator is not used.
    """

    #: The box arm's optimize_acqf(num_restarts=10, raw_samples=512).
    n_raw_samples: int = 512
    n_restarts: int = 10

    def _maximize_on_sphere(self, gp, best_y: float, D: int) -> np.ndarray:
        from botorch.optim.initializers import gen_batch_initial_conditions

        acqf = _ShellAcquisition(
            qLogExpectedImprovement(model=gp, best_f=best_y), radius=self.mean_norm,
        )
        bounds_v = torch.full(
            (2, D), float("inf"), dtype=torch.float64, device=self._device,
        )
        bounds_v[0] = -float("inf")

        def gaussian_v(n: int, q: int, seed: Optional[int]) -> torch.Tensor:
            # ``seed`` is optimize_acqf's options["seed"], None here as in the box arm.
            return torch.randn(n, q, D, dtype=torch.float64)

        def ic_generator(acq_function, bounds, q, num_restarts, raw_samples, **kwargs):
            # gen_batch_initial_conditions refuses infinite bounds; with a
            # generator it uses them only for their shape and device.
            finite = torch.zeros_like(bounds)
            finite[1] = 1.0
            return gen_batch_initial_conditions(
                acq_function=acq_function, bounds=finite, q=q,
                num_restarts=num_restarts, raw_samples=raw_samples,
                generator=gaussian_v, **kwargs,
            )

        try:
            v, _ = optimize_acqf(
                acq_function=acqf, bounds=bounds_v, q=1,
                num_restarts=self.n_restarts, raw_samples=self.n_raw_samples,
                ic_generator=ic_generator,
            )
        except RuntimeError as e:
            raise ModelFittingError(f"optimize_acqf failed: {e}") from e
        x = acqf.to_shell(v.detach()).reshape(1, -1)
        if not torch.isfinite(x).all():
            raise ModelFittingError("optimize_acqf returned non-finite candidate")
        return x.cpu().numpy()

    def next_candidate(self) -> np.ndarray:
        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        D = x_hist.shape[-1]
        X = torch.tensor(x_hist, dtype=torch.float64, device=self._device)
        Y = torch.tensor(
            y_hist.flatten(), dtype=torch.float64, device=self._device,
        ).unsqueeze(-1)
        Y_std, _, _, sd_raw = _standardize(Y)
        if sd_raw < 1e-6:
            self._fallback_count += 1
            self._fallback_reasons.append(f"flat-signal: sd={sd_raw:.2e}")
            return _random_on_sphere(D, self.mean_norm)
        try:
            gp = self._make_gp(X, Y_std)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            gp.eval()
            return self._maximize_on_sphere(gp, float(Y_std.max()), D)
        except _BOX_FALLBACK_ERRORS as e:
            self._fallback_count += 1
            self._fallback_reasons.append(f"{type(e).__name__}: {e}")
            logger.error(
                f"{type(self).__name__} GP fit FAILED at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {e}"
            )
            return _random_on_sphere(D, self.mean_norm)
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            raise


class _HvarfnerScaledGPSphereScaledSolver(HvarfnerScaledGPSolver):
    """Hvarfner-on-sphere at r = sphere_radius_multiplier * empirical mean_norm."""

    sphere_radius_multiplier: float = 1.0  # subclass overrides

    def next_candidate(self) -> np.ndarray:
        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        D = x_hist.shape[-1]
        radius = self.sphere_radius_multiplier * self.mean_norm

        X = torch.tensor(x_hist, dtype=torch.float64, device=self._device)
        Y = torch.tensor(
            y_hist.flatten(), dtype=torch.float64, device=self._device
        ).unsqueeze(-1)
        Y_std, _, _, sd_raw = _standardize(Y)
        if sd_raw < 1e-6:
            self._fallback_count += 1
            reason = f"flat-signal: sd={sd_raw:.2e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"{type(self).__name__} flat-signal at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            return _random_on_sphere(D, radius)

        try:
            gp = self._make_gp(X, Y_std)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()
            best_y_std = float(Y_std.max())
            return _propose_qlogei_on_sphere(
                gp, best_y_std, self.n_candidates, D, radius, self._device,
            )
        except GP_FIT_ERRORS as e:
            self._fallback_count += 1
            reason = f"{type(e).__name__}: {e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"Hvarfner GP fit FAILED at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            return _random_on_sphere(D, radius)
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            raise


class HvarfnerScaledGPBoxProjectedSolver(HvarfnerScaledGPBoxSolver):
    """Mechanism diagnostic: Hvarfner GP + box-supported L-BFGS acquisition,
    then post-hoc project the candidate to the empirical sphere at ``mean_norm``.

    Tests the off-manifold-drift hypothesis: if L-BFGS over a box finds good
    directions but drifts off-manifold (norm > mean_norm), then renormalising
    to the sphere should recover most of the Hvarfner-on-sphere advantage.
    Compare against:
      - HvarfnerScaledGPBoxSolver (no projection, baseline d = -0.25 vs sphere)
      - HvarfnerScaledGPSolver    (on-sphere candidates from the start)
    If projected ≈ on-sphere, the off-manifold drift is the mechanism.
    """

    def next_candidate(self) -> np.ndarray:
        z = _box_next_candidate(self)  # shape [1, D]
        norm = float(np.linalg.norm(z))
        if norm > 1e-9:
            z = z / norm * self.mean_norm
        return z


class HvarfnerScaledGPSphere05Solver(_HvarfnerScaledGPSphereScaledSolver):
    """Hvarfner-on-sphere at r = 0.5 * mean_norm (~3.99 for SELFIES)."""
    sphere_radius_multiplier: float = 0.5


class HvarfnerScaledGPSphere15Solver(_HvarfnerScaledGPSphereScaledSolver):
    """Hvarfner-on-sphere at r = 1.5 * mean_norm (~11.97 for SELFIES)."""
    sphere_radius_multiplier: float = 1.5


class HvarfnerScaledGPSphere20Solver(_HvarfnerScaledGPSphereScaledSolver):
    """Hvarfner-on-sphere at r = 2.0 * mean_norm (~15.96 for SELFIES)."""
    sphere_radius_multiplier: float = 2.0


class PapenmeierMSRBoxSolver(PapenmeierMSRSolver):
    """Papenmeier MSR with original box-supported acquisition (Sobol + L-BFGS).

    GP recipe identical to ``PapenmeierMSRSolver``: same Mat\u00e9rn-5/2 ARD
    kernel, same MLE-initialized lengthscale at 0.1*sqrt(D) per Sec.\u202f4 of
    arXiv:2502.09198 (RAASP restart still NOT implemented). Acquisition
    swapped to box-supported continuous L-BFGS as in the published recipe.
    """

    def next_candidate(self) -> np.ndarray:
        return _box_next_candidate(self)


class XuStandardGPBoxSolver(XuStandardGPSolver):
    """Xu standard GP with original box-supported acquisition (Sobol + L-BFGS).

    GP recipe identical to ``XuStandardGPSolver``: same Mat\u00e9rn-5/2 ARD
    kernel, same c\u00b7sqrt(D) lengthscale init per arXiv:2402.02746.
    Acquisition swapped to box-supported continuous L-BFGS as in the
    published recipe.
    """

    def next_candidate(self) -> np.ndarray:
        return _box_next_candidate(self)


# ---------------------------------------------------------------------------
# Faithful [0,1]^D normalized-box variants — original Hvarfner / Papenmeier /
# Xu protocol with input_normalize=True and bounds=[[0]*D, [1]*D]. Differs
# from *BoxSolver above (which uses an unnormalized [-3sigma, +3sigma]^D box)
# by per-coord min-max normalization of GP inputs to [0,1]^D before fitting.
# Reference: Standard-GP-is-all-you-need-for-HDBO/baselines/BO_loop.py:47.
# ---------------------------------------------------------------------------


def _norm_box_next_candidate(host: StepByStepSolver) -> np.ndarray:
    """[0,1]^D normalized acquisition step shared by all *NormBoxSolver classes.

    Steps:
    1. Load per-coord ZINC [low, high] bounds (cached).
    2. Normalize history X to [0,1]^D via (X - low) / (high - low).
    3. Fit GP on normalized inputs (lengthscale prior calibrated for unit-width).
    4. optimize_acqf with bounds=[[0]*D, [1]*D].
    5. De-normalize candidate back to raw VAE latent space.
    """
    missing = [name for name in _BOX_HOST_REQUIREMENTS if not hasattr(host, name)]
    if missing:
        raise TypeError(
            f"{type(host).__name__} cannot use _norm_box_next_candidate: "
            f"host class is missing required attributes: {missing}."
        )

    low, high = _load_zinc_per_coord_bounds(mode="minmax")

    x_hist, y_hist = host.get_history_as_arrays(penalize_nans_with=-1.0)
    D = x_hist.shape[-1]
    if low.shape[0] != D:
        raise ValueError(
            f"ZINC bounds have D={low.shape[0]} but history has D={D} — "
            f"recompute zinc_per_coord_bounds.json for this codec."
        )
    span = high - low

    x_norm = (x_hist - low) / span  # [N, D] in ~[0, 1]
    X = torch.tensor(x_norm, dtype=torch.float64, device=host._device)
    Y = torch.tensor(
        y_hist.flatten(), dtype=torch.float64, device=host._device
    ).unsqueeze(-1)
    Y_std, _, _, sd_raw = _standardize(Y)
    if sd_raw < 1e-6:
        host._fallback_count += 1
        reason = f"flat-signal: sd={sd_raw:.2e}"
        host._fallback_reasons.append(reason)
        logger.error(
            f"{type(host).__name__} flat-signal at iter {host.iteration} "
            f"(fallback #{host._fallback_count}): {reason}"
        )
        return _random_in_normalized_box(low, high)

    try:
        gp = host._make_gp(X, Y_std)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()
        best_y_std = float(Y_std.max())
        return _propose_qlogei_on_normalized_box(
            gp, best_y_std, D, host._device, low=low, high=high,
        )
    except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
        raise
    except _BOX_FALLBACK_ERRORS as e:
        host._fallback_count += 1
        reason = f"{type(e).__name__}: {e}"
        host._fallback_reasons.append(reason)
        logger.error(
            f"{type(host).__name__} GP fit FAILED at iter {host.iteration} "
            f"(fallback #{host._fallback_count}): {reason}"
        )
        return _random_in_normalized_box(low, high)


def _norm_box_argmax_next_candidate(host: StepByStepSolver) -> np.ndarray:
    """Min/max-normalized [0,1]^D box, acquisition maximised by ARGMAX over samples.

    Companion to ``_norm_box_next_candidate``: identical GP, identical rescaling,
    identical qLogEI, but the acquisition maximiser is an argmax over a uniform
    candidate set instead of continuous multi-start L-BFGS. Together the two
    isolate the acquisition-optimizer factor on the domain Section 4.4 of the
    paper actually specifies (per-coordinate [min, max]).
    """
    low, high = _load_zinc_per_coord_bounds(mode="minmax")
    x_hist, y_hist = host.get_history_as_arrays(penalize_nans_with=-1.0)
    D = x_hist.shape[-1]
    span = high - low
    X = torch.tensor((x_hist - low) / span, dtype=torch.float64, device=host._device)
    Y = torch.tensor(
        y_hist.flatten(), dtype=torch.float64, device=host._device
    ).unsqueeze(-1)
    Y_std, _, _, sd_raw = _standardize(Y)

    def _fallback() -> np.ndarray:
        return _random_in_normalized_box(low, high)

    if sd_raw < 1e-6:
        host._fallback_count += 1
        host._fallback_reasons.append(f"flat-signal: sd={sd_raw:.2e}")
        return _fallback()
    try:
        gp = host._make_gp(X, Y_std)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        gp.eval()
        cand = torch.rand(
            host.n_candidates, D, dtype=torch.float64, device=host._device,
        )
        acqf = qLogExpectedImprovement(model=gp, best_f=float(Y_std.max()))
        with torch.no_grad():
            acq_vals = acqf(cand.unsqueeze(1))
        best = cand[int(torch.argmax(acq_vals))].cpu().numpy()
        return (best * span + low).reshape(1, -1)
    except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
        raise
    except _BOX_FALLBACK_ERRORS as e:
        host._fallback_count += 1
        host._fallback_reasons.append(f"{type(e).__name__}: {e}")
        return _fallback()


class HvarfnerScaledGPNormBoxArgmaxSolver(HvarfnerScaledGPSolver):
    """Min/max [0,1]^D box support, acquisition argmax over a sampled candidate set."""

    def next_candidate(self) -> np.ndarray:
        return _norm_box_argmax_next_candidate(self)


class HvarfnerScaledGPNormBoxProjectedSolver(HvarfnerScaledGPSolver):
    """Min/max [0,1]^D L-BFGS proposal, then PROJECTED onto the empirical shell.

    The shell-support / continuous-optimizer cell of the factorial. Its proposal
    ends up on exactly the support the on-sphere arm uses, so the only thing that
    differs from that arm is how the acquisition was maximised. If it still loses,
    the acquisition optimizer is a separate effect from the support.

    Replaces the earlier projected variant, which optimised over a [-3 sigma] box
    and therefore did not share an L-BFGS domain with the box arm it was compared
    against.
    """

    def next_candidate(self) -> np.ndarray:
        cand = _norm_box_next_candidate(self)
        x = np.asarray(cand, dtype=np.float64).reshape(-1)
        n = float(np.linalg.norm(x))
        if not np.isfinite(n) or n < 1e-12:
            return _random_on_sphere(self._dim, self.mean_norm)
        return (x / n * self.mean_norm).reshape(1, -1)


class HvarfnerScaledGPNormBoxSolver(HvarfnerScaledGPSolver):
    """Hvarfner GP with faithful [0,1]^D normalized-box acquisition.

    Identical GP recipe (Matern-5/2 + LogNormal sqrt(D) lengthscale prior +
    HalfNormal noise) but X is normalized to [0,1]^D via per-coord ZINC
    [min, max] before GP fitting, and acquisition optimizes over [0,1]^D
    bounds. Reproduces the original Hvarfner protocol used in
    Standard-GP-is-all-you-need-for-HDBO/baselines/BO_loop.py.
    """

    def next_candidate(self) -> np.ndarray:
        return _norm_box_next_candidate(self)


class PapenmeierMSRNormBoxSolver(PapenmeierMSRSolver):
    """Papenmeier MSR with faithful [0,1]^D normalized-box acquisition.

    Identical GP recipe (Matern-5/2 + MLE-init at 0.1*sqrt(D)) but X is normalized
    to [0,1]^D via per-coord ZINC [min, max] before GP fitting. Reproduces the
    standard BoTorch normalize=True protocol from arXiv:2502.09198.
    """

    def next_candidate(self) -> np.ndarray:
        return _norm_box_next_candidate(self)


class XuStandardGPNormBoxSolver(XuStandardGPSolver):
    """Xu standard GP with faithful [0,1]^D normalized-box acquisition.

    Identical GP recipe (Matern-5/2 ARD + c*sqrt(D) lengthscale init) but X is
    normalized to [0,1]^D via per-coord ZINC [min, max] before GP fitting.
    Reproduces the protocol from arXiv:2402.02746 (Standard-GP-is-all-you-need).
    """

    def next_candidate(self) -> np.ndarray:
        return _norm_box_next_candidate(self)


# ---------------------------------------------------------------------------
# Reviewer-response variants (Section 4.3 extension): same Hvarfner GP recipe,
# acquisition over candidates drawn from (a) the VAE's own N(0, I) prior, or
# (b) an angular-uniform-in-box distribution renormalized to the empirical
# sphere radius. Together with on-sphere and on-box, these isolate radial vs.
# angular contributions of the on-sphere scaffold.
# ---------------------------------------------------------------------------


def _prior_next_candidate(host: StepByStepSolver) -> np.ndarray:
    """Reviewer-response acquisition step: candidates ~ N(0, I_D), no normalization.

    Tests "the natural alternative isn't a hand-crafted box, it's the VAE's
    own Gaussian prior." Shares the host class's ``_make_gp`` (so the GP
    recipe is identical to its on-sphere parent). Only the candidate
    generator differs.
    """
    missing = [name for name in _BOX_HOST_REQUIREMENTS if not hasattr(host, name)]
    if missing:
        raise TypeError(
            f"{type(host).__name__} cannot use _prior_next_candidate: "
            f"missing required attributes: {missing}."
        )
    x_hist, y_hist = host.get_history_as_arrays(penalize_nans_with=-1.0)
    D = x_hist.shape[-1]

    X = torch.tensor(x_hist, dtype=torch.float64, device=host._device)
    Y = torch.tensor(
        y_hist.flatten(), dtype=torch.float64, device=host._device
    ).unsqueeze(-1)
    Y_std, _, _, sd_raw = _standardize(Y)
    if sd_raw < 1e-6:
        host._fallback_count += 1
        reason = f"flat-signal: sd={sd_raw:.2e}"
        host._fallback_reasons.append(reason)
        logger.error(
            f"{type(host).__name__} flat-signal at iter {host.iteration} "
            f"(fallback #{host._fallback_count}): {reason}"
        )
        return _random_from_prior(D)

    try:
        gp = host._make_gp(X, Y_std)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()
        best_y_std = float(Y_std.max())
        return _propose_qlogei_on_prior(
            gp, best_y_std, host.n_candidates, D, host._device,
        )
    except GP_FIT_ERRORS as e:
        host._fallback_count += 1
        reason = f"{type(e).__name__}: {e}"
        host._fallback_reasons.append(reason)
        logger.error(
            f"{type(host).__name__} GP fit FAILED at iter {host.iteration} "
            f"(fallback #{host._fallback_count}): {reason}"
        )
        return _random_from_prior(D)
    except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
        raise


def _angular_box_next_candidate(host: StepByStepSolver) -> np.ndarray:
    """Reviewer-response acquisition step: candidates uniform-in-box renormalized to r̄.

    Same radial distribution as on-sphere (radius ``host.mean_norm``); only
    the angular distribution differs (biased toward box-corner directions).
    Isolates angular contribution of the on-sphere scaffold.
    """
    missing = [name for name in _BOX_HOST_REQUIREMENTS if not hasattr(host, name)]
    if missing:
        raise TypeError(
            f"{type(host).__name__} cannot use _angular_box_next_candidate: "
            f"missing required attributes: {missing}."
        )
    x_hist, y_hist = host.get_history_as_arrays(penalize_nans_with=-1.0)
    D = x_hist.shape[-1]

    X = torch.tensor(x_hist, dtype=torch.float64, device=host._device)
    Y = torch.tensor(
        y_hist.flatten(), dtype=torch.float64, device=host._device
    ).unsqueeze(-1)
    Y_std, _, _, sd_raw = _standardize(Y)
    if sd_raw < 1e-6:
        host._fallback_count += 1
        reason = f"flat-signal: sd={sd_raw:.2e}"
        host._fallback_reasons.append(reason)
        logger.error(
            f"{type(host).__name__} flat-signal at iter {host.iteration} "
            f"(fallback #{host._fallback_count}): {reason}"
        )
        return _random_on_angular_box_at_r(D, host.mean_norm)

    try:
        gp = host._make_gp(X, Y_std)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()
        best_y_std = float(Y_std.max())
        return _propose_qlogei_on_angular_box_at_r(
            gp, best_y_std, host.n_candidates, D, host.mean_norm, host._device,
        )
    except GP_FIT_ERRORS as e:
        host._fallback_count += 1
        reason = f"{type(e).__name__}: {e}"
        host._fallback_reasons.append(reason)
        logger.error(
            f"{type(host).__name__} GP fit FAILED at iter {host.iteration} "
            f"(fallback #{host._fallback_count}): {reason}"
        )
        return _random_on_angular_box_at_r(D, host.mean_norm)
    except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
        raise


class HvarfnerScaledGPPriorSolver(HvarfnerScaledGPSolver):
    """Hvarfner GP, candidates ~ N(0, I_D) (VAE prior, no normalization)."""

    def next_candidate(self) -> np.ndarray:
        return _prior_next_candidate(self)


class HvarfnerScaledGPAngularBoxSolver(HvarfnerScaledGPSolver):
    """Hvarfner GP, candidates uniform-in-box renormalized to mean_norm."""

    def next_candidate(self) -> np.ndarray:
        return _angular_box_next_candidate(self)


class PapenmeierMSRPriorSolver(PapenmeierMSRSolver):
    """Papenmeier MSR GP, candidates ~ N(0, I_D)."""

    def next_candidate(self) -> np.ndarray:
        return _prior_next_candidate(self)


class PapenmeierMSRAngularBoxSolver(PapenmeierMSRSolver):
    """Papenmeier MSR GP, candidates uniform-in-box renormalized to mean_norm."""

    def next_candidate(self) -> np.ndarray:
        return _angular_box_next_candidate(self)


class XuStandardGPPriorSolver(XuStandardGPSolver):
    """Xu Standard GP, candidates ~ N(0, I_D)."""

    def next_candidate(self) -> np.ndarray:
        return _prior_next_candidate(self)


class XuStandardGPAngularBoxSolver(XuStandardGPSolver):
    """Xu Standard GP, candidates uniform-in-box renormalized to mean_norm."""

    def next_candidate(self) -> np.ndarray:
        return _angular_box_next_candidate(self)


# ---------------------------------------------------------------------------
# Mechanism diagnostic + universal recipe
# ---------------------------------------------------------------------------


def _filtered_box_next_candidate(
    host: StepByStepSolver, norm_cap_multiplier: float = 1.2,
) -> np.ndarray:
    """Sample uniform in box, then REJECT candidates with ||x|| > k*mean_norm.

    Removes the OOD-corner contamination from Hvarfner's box-supported recipe
    while preserving its uniform-in-box angular distribution. Tests the
    norm-variance hypothesis: if filtered-box ≈ on-sphere, the load-bearing
    feature is the radial cap, not the angular shape.

    Default k=1.2 keeps ~95% of box volume but removes box-corner directions
    where ||x|| can reach h*sqrt(D) ~= 1.77 * mean_norm.
    """
    missing = [name for name in _BOX_HOST_REQUIREMENTS if not hasattr(host, name)]
    if missing:
        raise TypeError(
            f"{type(host).__name__} cannot use _filtered_box_next_candidate: "
            f"missing required attributes: {missing}."
        )
    x_hist, y_hist = host.get_history_as_arrays(penalize_nans_with=-1.0)
    D = x_hist.shape[-1]
    half_width = float(getattr(host, "box_half_width", BOX_HALF_WIDTH))
    norm_cap = norm_cap_multiplier * host.mean_norm

    X = torch.tensor(x_hist, dtype=torch.float64, device=host._device)
    Y = torch.tensor(
        y_hist.flatten(), dtype=torch.float64, device=host._device
    ).unsqueeze(-1)
    Y_std, _, _, sd_raw = _standardize(Y)
    if sd_raw < 1e-6:
        host._fallback_count += 1
        host._fallback_reasons.append(f"flat-signal: sd={sd_raw:.2e}")
        return _random_in_box(D, half_width=half_width)

    try:
        gp = host._make_gp(X, Y_std)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()
        best_y_std = float(Y_std.max())

        # Sample a generous pool of candidates from the box, then filter by norm.
        # Pool size 4x normal because filter rejects ~30-50% on average for
        # half_width = 0.881 (default Hvarfner-on-box).
        pool = host.n_candidates * 4
        u = (torch.rand(pool, D, dtype=torch.float64, device=host._device)
             * 2.0 - 1.0) * half_width
        norms = u.norm(dim=-1)
        mask = norms <= norm_cap
        kept = u[mask]
        if kept.shape[0] < 16:
            # Filter rejects too much — fall back to uniform-in-box (NOT
            # on-sphere) so the variant retains its box geometry; an on-sphere
            # fallback here would contaminate the mechanism diagnostic.
            host._fallback_count += 1
            host._fallback_reasons.append(
                f"filtered-box too restrictive: {kept.shape[0]}/{pool} kept"
            )
            return _random_in_box(D, half_width=half_width)

        # Cap to host.n_candidates to keep acquisition cost stable.
        if kept.shape[0] > host.n_candidates:
            idx = torch.randperm(kept.shape[0], device=host._device)[: host.n_candidates]
            kept = kept[idx]

        acqf = qLogExpectedImprovement(model=gp, best_f=best_y_std)
        with torch.no_grad():
            acq_vals = acqf(kept.unsqueeze(1))
        best_idx = int(torch.argmax(acq_vals))
        return kept[best_idx].detach().cpu().numpy().reshape(1, -1)
    except GP_FIT_ERRORS as e:
        host._fallback_count += 1
        host._fallback_reasons.append(f"{type(e).__name__}: {e}")
        return _random_on_sphere(D, host.mean_norm)
    except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
        raise


def _normalized_input_next_candidate(host: StepByStepSolver) -> np.ndarray:
    """Universal recipe: project all training X AND candidates to ||x||=mean_norm.

    Makes the GP intrinsically norm-invariant. Candidates are sampled from
    N(0, I_D) (worst-case raw input distribution); both they and the history
    are renormalized to lie on the empirical sphere before any GP operation.
    The returned candidate is also at ||x||=mean_norm so the actual decoder
    sees an on-shell point regardless of where sampling happened.
    """
    missing = [name for name in _BOX_HOST_REQUIREMENTS if not hasattr(host, name)]
    if missing:
        raise TypeError(
            f"{type(host).__name__} cannot use _normalized_input_next_candidate: "
            f"missing required attributes: {missing}."
        )
    x_hist, y_hist = host.get_history_as_arrays(penalize_nans_with=-1.0)
    D = x_hist.shape[-1]
    r = host.mean_norm

    # Project history rows to ||x||=r (no-op for on-sphere history, otherwise
    # collapses radial information so the GP only learns angular structure).
    x_norms = np.linalg.norm(x_hist, axis=-1, keepdims=True)
    safe = np.where(x_norms > 1e-9, x_norms, 1.0)
    x_hist_proj = x_hist / safe * r

    X = torch.tensor(x_hist_proj, dtype=torch.float64, device=host._device)
    Y = torch.tensor(
        y_hist.flatten(), dtype=torch.float64, device=host._device
    ).unsqueeze(-1)
    Y_std, _, _, sd_raw = _standardize(Y)
    if sd_raw < 1e-6:
        host._fallback_count += 1
        host._fallback_reasons.append(f"flat-signal: sd={sd_raw:.2e}")
        return _random_on_sphere(D, r)

    try:
        gp = host._make_gp(X, Y_std)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()
        best_y_std = float(Y_std.max())

        # Candidates from prior N(0, I_D), then projected to the empirical sphere.
        u = torch.randn(host.n_candidates, D, dtype=torch.float64, device=host._device)
        u = u / u.norm(dim=-1, keepdim=True)
        X_cand = u * r

        acqf = qLogExpectedImprovement(model=gp, best_f=best_y_std)
        with torch.no_grad():
            acq_vals = acqf(X_cand.unsqueeze(1))
        best_idx = int(torch.argmax(acq_vals))
        return X_cand[best_idx].detach().cpu().numpy().reshape(1, -1)
    except GP_FIT_ERRORS as e:
        host._fallback_count += 1
        host._fallback_reasons.append(f"{type(e).__name__}: {e}")
        return _random_on_sphere(D, r)
    except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
        raise


class HvarfnerScaledGPFilteredBoxSolver(HvarfnerScaledGPSolver):
    """Hvarfner GP, sample uniform-in-box but REJECT candidates with ||x|| > 1.2*r̄.

    Mechanism diagnostic: if this matches on-sphere performance, the box-vs-sphere
    loss is driven by **radial-norm variance** (corners with ||x|| ~14 vs r̄=8),
    not by the angular distribution within the cap region.
    """

    box_half_width: float = BOX_HALF_WIDTH
    norm_cap_multiplier: float = 1.2

    def next_candidate(self) -> np.ndarray:
        return _filtered_box_next_candidate(self, self.norm_cap_multiplier)


class HvarfnerScaledGPNormalizedSolver(HvarfnerScaledGPSolver):
    """Universal Hvarfner: sample from N(0, I_D), normalize all inputs to ||x||=r̄.

    Codec-agnostic recipe: the GP is shown only ||x||=mean_norm points, so its
    learned structure depends only on angular relationships. Tests whether
    input-norm projection makes Hvarfner robust to whatever raw input
    distribution the user (or codec) provides.
    """

    def next_candidate(self) -> np.ndarray:
        return _normalized_input_next_candidate(self)


# ---------------------------------------------------------------------------
# SpherePFN-Solver — codec-independent in-context surrogate (offline-pretrained
# transformer). Replaces per-step GP fit with a single forward pass.
# ---------------------------------------------------------------------------


class SpherePFNSolver(StepByStepSolver):
    """In-context PFN surrogate; codec-independent (operates on unit vectors).

    Loads a SpherePFN checkpoint trained offline on synthetic Matern-on-sphere
    tasks. At each iteration:
    1. Normalise history to unit vectors on S^{D-1}.
    2. Sample ``n_candidates`` candidate unit vectors uniformly on S^{D-1}.
    3. For each candidate, build the (N+2)-token context [history, candidate],
       compute the pairwise similarity matrix, run the PFN once per candidate
       (or batched), get (μ, σ²).
    4. Score each candidate by Log-Expected-Improvement under N(μ, σ²); pick
       argmax; emit at radius ``mean_norm``.

    Codec-independent. Works for any spherical scaffold.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        ckpt_path: str,
        n_candidates: int = 256,
        device: Optional[str] = None,
    ):
        super().__init__(black_box, x0, y0)
        if self._dim < 2:
            raise ValueError(f"SpherePFN requires D>=2, got D={self._dim}")
        if self.mean_norm <= 0:
            raise ValueError(f"mean_norm must be positive, got {self.mean_norm}")
        if n_candidates < 1:
            raise ValueError(f"n_candidates must be >=1, got {n_candidates}")

        self.n_candidates = n_candidates
        self._fallback_count = 0
        self._fallback_reasons: list[str] = []
        self._device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Lazy import so the wider codebase stays importable without the PFN deps.
        from src.spherevo.pfn.model import SpherePFN

        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)
        cargs = ckpt.get("args", {})
        self._model = SpherePFN(
            hidden_dim=cargs.get("hidden_dim", 192),
            n_layers=cargs.get("n_layers", 6),
            n_heads=cargs.get("n_heads", 6),
        ).to(self._device)
        self._model.load_state_dict(ckpt["model"])
        self._model.eval()
        self._ckpt_path = ckpt_path

    @torch.no_grad()
    def _predict(
        self, U_hist: torch.Tensor, y_hist: torch.Tensor,
        U_cand: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorised PFN forward over a batch of candidates.

        Each candidate becomes the (last) query token in its own context;
        we stack ``n_candidates`` such contexts into a single batch.
        """
        n = U_hist.shape[0]
        m = U_cand.shape[0]
        # Build [m, n+1, D] by repeating history and tacking each candidate.
        U_hist_rep = U_hist.unsqueeze(0).expand(m, -1, -1)             # [m,n,D]
        U_query = U_cand.unsqueeze(1)                                  # [m,1,D]
        U_full = torch.cat([U_hist_rep, U_query], dim=1)              # [m,n+1,D]
        sim = torch.matmul(U_full, U_full.transpose(-1, -2)).clamp(-1.0, 1.0)
        y_full = torch.zeros(m, n + 1, dtype=U_full.dtype, device=U_full.device)
        y_full[:, :n] = y_hist
        mu, log_var = self._model(sim, y_full, query_idx=n)
        return mu, log_var

    def next_candidate(self) -> np.ndarray:
        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        D = x_hist.shape[-1]

        norms = np.linalg.norm(x_hist, axis=1, keepdims=True)
        safe = np.where(norms > 1e-9, norms, 1.0)
        u_hist = x_hist / safe

        y_arr = y_hist.flatten().astype(np.float64)
        mu_y = float(np.mean(y_arr))
        sd_y = float(np.std(y_arr))
        if sd_y < 1e-6:
            self._fallback_count += 1
            self._fallback_reasons.append(f"flat-signal: sd={sd_y:.2e}")
            return _random_on_sphere(D, self.mean_norm)
        y_std = (y_arr - mu_y) / max(sd_y, 1e-8)

        # Candidate set: uniform on the sphere (no anchor — we want PFN-driven
        # ranking to do the work; anchor-based caps already failed in the
        # geodesic-slice / ball ablations).
        u_cand_np = np.random.randn(self.n_candidates, D)
        u_cand_np /= np.linalg.norm(u_cand_np, axis=1, keepdims=True).clip(1e-9)

        U_hist_t = torch.tensor(u_hist, dtype=torch.float32, device=self._device)
        y_hist_t = torch.tensor(y_std, dtype=torch.float32, device=self._device)
        U_cand_t = torch.tensor(u_cand_np, dtype=torch.float32, device=self._device)

        try:
            mu_pred, log_var_pred = self._predict(U_hist_t, y_hist_t, U_cand_t)
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            raise
        except RuntimeError as e:
            self._fallback_count += 1
            self._fallback_reasons.append(f"PFN forward: {type(e).__name__}: {e}")
            logger.error(
                f"SpherePFN forward FAILED at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {e}"
            )
            return _random_on_sphere(D, self.mean_norm)

        # qLogEI under Gaussian posterior.
        sigma_pred = torch.exp(0.5 * log_var_pred).clamp_min(1e-6)
        best_y_std = float(np.max(y_std))
        diff = mu_pred - best_y_std
        z = diff / sigma_pred
        # Standard EI = (mu-best)*Phi(z) + sigma*phi(z); use logEI for stability.
        # The (mu-best) factor must gate per-candidate: candidates with mu<=best
        # contribute -inf to log_term1, never log(1e-30) ≈ -69 which would
        # spuriously bias argmax toward losing candidates with high log_Phi.
        normal = torch.distributions.Normal(0.0, 1.0)
        log_phi = normal.log_prob(z)
        log_Phi = normal.cdf(z).clamp_min(1e-12).log()
        log_term1 = torch.where(
            diff > 0,
            torch.log(diff.clamp_min(1e-30)) + log_Phi,
            torch.full_like(z, float("-inf")),
        )
        log_term2 = torch.log(sigma_pred) + log_phi
        # log(EI) = logsumexp([log_term1, log_term2]).
        log_ei = torch.logaddexp(log_term1, log_term2)
        idx = int(torch.argmax(log_ei).item())

        return (u_cand_np[idx] * self.mean_norm).reshape(1, -1)


# ---------------------------------------------------------------------------
# Hvarfner-Ball GP — novel codec-independent surrogate that combines
# Hvarfner's GP recipe (best surrogate baseline) with GeoWalk's annealing
# angular schedule (best surrogate-free baseline) for candidate proposal.
# ---------------------------------------------------------------------------


def _propose_qlogei_on_sphere_ball(
    gp,
    best_y: float,
    n_candidates: int,
    D: int,
    mean_norm: float,
    u_anchor: np.ndarray,
    angle_max: float,
    device: torch.device,
) -> np.ndarray:
    """Propose candidate via qLogEI over points uniform-on-S^{D-1} INSIDE an
    angular ball of radius ``angle_max`` around ``u_anchor``.

    Each candidate is parametrized as
        x = mean_norm * (cos(α) u_anchor + sin(α) v),
    with ``v`` a uniform unit tangent at u_anchor and ``α ~ Uniform(0, angle_max)``.
    This is the geodesic analogue of a Euclidean trust region: candidates lie
    within angular distance ``angle_max`` of the current best, so the GP sees
    nearby points (concentration of measure breaks down locally).
    """
    if D < 2:
        raise ValueError(f"Ball acquisition requires D>=2, got D={D}")
    u_t = torch.as_tensor(u_anchor, dtype=torch.float64, device=device)
    if abs(float(u_t.norm()) - 1.0) > 1e-6:
        u_t = u_t / u_t.norm()

    # Random tangent directions: project Gaussian onto T_{u} S^{D-1}, normalise.
    v = torch.randn(n_candidates, D, dtype=torch.float64, device=device)
    proj = (v @ u_t)[:, None] * u_t[None, :]
    v = v - proj
    norms = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v = v / norms

    # Random angles in [0, angle_max].
    alpha = torch.rand(n_candidates, dtype=torch.float64, device=device) * angle_max

    cos_a = torch.cos(alpha).unsqueeze(-1)
    sin_a = torch.sin(alpha).unsqueeze(-1)
    unit = cos_a * u_t[None, :] + sin_a * v
    X_cand = unit * mean_norm

    acqf = qLogExpectedImprovement(model=gp, best_f=best_y)
    with torch.no_grad():
        acq_vals = acqf(X_cand.unsqueeze(1))
    best_idx = int(torch.argmax(acq_vals))
    return X_cand[best_idx].detach().cpu().numpy().reshape(1, -1)


class HvarfnerScaledGPBallSolver(HvarfnerScaledGPSolver):
    """Hvarfner GP recipe + GeoWalk-style annealing angular ball.

    Combines two best-of-class baselines that have never been combined in any
    of the 100+ ablation variants run on this benchmark:
    - Hvarfner GP (LogNormal sqrt(D) lengthscale prior, HalfNormal noise) —
      the best surrogate among standard GP recipes.
    - GeoWalk's angular schedule (linear decay 0.6 -> 0.15 rad over the
      exploit phase) — the best surrogate-free spherical baseline.

    Mechanism. At iteration ``t``, anchor on the current best ``u* = x_best``
    on S^{D-1} and sample ``n_candidates`` candidates uniformly inside the
    spherical cap of angular radius ``angle_max(t) = angle_init - (angle_init
    - angle_final) * t / total_budget``. Hvarfner's qLogEI then re-ranks the
    in-cap candidates.

    Codec-independent: depends only on ambient dimension ``D`` and empirical
    radius ``mean_norm``; no decoder, no SMILES, no chemistry features.

    Hypothesis. Hvarfner-on-sphere generates 2048 *uniform* sphere candidates
    per step. In D=256 with N=30, all such candidates have near-identical
    distance to every history point (concentration of measure), so the GP
    cannot rank them and effectively returns one at random. Restricting to a
    cap around ``u*`` gives candidates with *resolvable* relative distances,
    so the GP's posterior gradient first contains usable signal.
    """

    angle_init: float = 0.60
    angle_final: float = 0.15

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        n_candidates: int = 2048,
        total_budget: int = 25,
        angle_init: Optional[float] = None,
        angle_final: Optional[float] = None,
    ):
        super().__init__(
            black_box=black_box, x0=x0, y0=y0, n_candidates=n_candidates,
        )
        if total_budget < 1:
            raise ValueError(f"total_budget must be >=1, got {total_budget}")
        if angle_init is not None:
            self.angle_init = float(angle_init)
        if angle_final is not None:
            self.angle_final = float(angle_final)
        if not (0.0 < self.angle_final < self.angle_init <= math.pi):
            raise ValueError(
                f"need 0 < angle_final < angle_init <= pi, got "
                f"({self.angle_final}, {self.angle_init})"
            )
        self.total_budget = total_budget

    def _angle_at(self, t: int) -> float:
        progress = min(1.0, max(0.0, t / max(1, self.total_budget - 1)))
        return self.angle_init - (self.angle_init - self.angle_final) * progress

    def next_candidate(self) -> np.ndarray:
        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        D = x_hist.shape[-1]

        norms = np.linalg.norm(x_hist, axis=1, keepdims=True)
        safe = np.where(norms > 1e-9, norms, 1.0)
        u_hist = x_hist / safe
        i_best = int(np.argmax(y_hist.flatten()))
        u_anchor = u_hist[i_best]

        X = torch.tensor(x_hist, dtype=torch.float64, device=self._device)
        Y = torch.tensor(
            y_hist.flatten(), dtype=torch.float64, device=self._device,
        ).unsqueeze(-1)
        Y_std, _, _, sd_raw = _standardize(Y)
        if sd_raw < 1e-6:
            self._fallback_count += 1
            reason = f"flat-signal: sd={sd_raw:.2e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"{type(self).__name__} flat-signal at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            return _random_on_sphere(D, self.mean_norm)

        try:
            gp = self._make_gp(X, Y_std)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()
            best_y_std = float(Y_std.max())
            angle_max = self._angle_at(self.iteration)
            return _propose_qlogei_on_sphere_ball(
                gp, best_y_std, self.n_candidates, D,
                self.mean_norm, u_anchor, angle_max, self._device,
            )
        except GP_FIT_ERRORS as e:
            self._fallback_count += 1
            reason = f"{type(e).__name__}: {e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"{type(self).__name__} GP fit FAILED at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            return _random_on_sphere(D, self.mean_norm)
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            raise


# ---------------------------------------------------------------------------
# Geodesic-Slice GP — novel surrogate that bypasses concentration of measure
# in D=256 by collapsing each step to a 2D fit (arc-length, off-circle distance)
# in the tangent frame at the current best, repeated over K random great-circles.
# ---------------------------------------------------------------------------


def _tangent_basis_at(u: np.ndarray, v_raw: np.ndarray) -> np.ndarray:
    """Project ``v_raw`` to the tangent space of S^{D-1} at ``u`` and unit-normalize.

    Both ``u`` and the returned vector are unit-norm. Returns ``None`` if the
    raw direction is parallel to ``u`` (degenerate, caller resamples).
    """
    v = v_raw - np.dot(v_raw, u) * u
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    return v / n


class GeodesicSliceGPSolver(StepByStepSolver):
    """Novel surrogate: 2D Matern GP per great-circle in the tangent frame at best.

    Mechanism
    ---------
    At each iteration, anchor on the current best ``u* = x_best/||x_best||``.
    Sample ``K`` random unit tangent directions ``{v_k}`` at ``u*`` (so each
    ``span(u*, v_k)`` defines a great-circle on S^{D-1}). For each ``v_k``,
    coordinate every history point ``û_i`` in the orthonormal frame
    ``(u*, v_k, w_perp)``::

        a_i = û_i · u*               # cos-component along anchor
        b_i = û_i · v_k              # cos-component along great-circle
        s_i = atan2(b_i, a_i)        # signed arc-length (rad)
        d_i = arcsin(||r_i||)        # off-circle angular distance (rad),
                                     # r_i = û_i − a_i u* − b_i v_k

    Fit a 2D ``SingleTaskGP`` (Matern-5/2 ARD over (s, d)) to ``{(s_i, d_i, y_i)}``
    and maximise qLogEI along the slice ``d = 0`` over a dense 1D grid in
    ``s ∈ (−π, π]``. Pick the great-circle ``k*`` with the highest acquisition
    value; emit ``x = mean_norm * (cos(s*) u* + sin(s*) v_{k*})``.

    Why this beats raw-D=256 GPs at low budget
    ------------------------------------------
    In D=256 with N=30, pairwise distances ‖z_i − z_j‖ concentrate (σ/μ ≈ 6%);
    Matern lengthscales drift to √D and the GP posterior degenerates to a
    constant. By fitting in (s, d) ∈ ℝ² we side-step the concentration: the
    kernel sees full-range angular signal in s, the GP is well-conditioned at
    N=30 in 2D, and lengthscales are interpretable in radians.

    Parameters
    ----------
    n_directions : int, default 8
        Number of random great-circles ``K`` evaluated per step.
    grid_size : int, default 128
        Dense grid size for 1D acquisition along each slice.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        n_directions: int = 8,
        grid_size: int = 128,
    ):
        super().__init__(black_box, x0, y0)
        if self._dim < 2:
            raise ValueError(f"Geodesic-Slice GP requires D>=2, got D={self._dim}")
        if self.mean_norm <= 0:
            raise ValueError(f"mean_norm must be positive, got {self.mean_norm}")
        if n_directions < 1:
            raise ValueError(f"n_directions must be >=1, got {n_directions}")
        if grid_size < 16:
            raise ValueError(f"grid_size must be >=16, got {grid_size}")
        self.n_directions = n_directions
        self.grid_size = grid_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fallback_count = 0
        self._fallback_reasons: list[str] = []

    def _make_2d_gp(self, X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
        """2D Matern-5/2 ARD GP over (arc, off-circle) in radians.

        Lengthscale prior LogNormal(loc=0, scale=1.0): mean ≈ 1 rad (~57°), heavy
        tail to 3+ rad. Both axes share the same prior because both are angles.
        Noise prior matches Papenmeier/Xu for numerical-stability fairness.
        """
        # Device-safe for the same reason as the Hvarfner and Linear BO priors.
        ls_prior = _DeviceSafeLogNormalPrior(loc=0.0, scale=1.0)
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=2,
            lengthscale_prior=ls_prior,
            lengthscale_constraint=Interval(1e-2, 10.0),
        )
        covar = ScaleKernel(base_kernel)
        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(1.1, 0.05),
            noise_constraint=GreaterThan(1e-5),
        )
        gp = SingleTaskGP(
            train_X=X, train_Y=Y, likelihood=likelihood, covar_module=covar,
        )
        gp.to(X)
        return gp

    def _evaluate_slice(
        self,
        u_anchor: np.ndarray,
        v_dir: np.ndarray,
        u_hist: np.ndarray,
        y_std: torch.Tensor,
        best_y_std: float,
    ) -> tuple[float, float] | None:
        """Fit 2D GP and return ``(best_s, best_acq)`` along the d=0 slice.

        Returns ``None`` on numerical failure (caller skips this direction).
        """
        a = u_hist @ u_anchor                       # [N]
        b = u_hist @ v_dir                          # [N]
        residual = u_hist - a[:, None] * u_anchor - b[:, None] * v_dir
        r_norm = np.linalg.norm(residual, axis=1)   # [N]
        # Numerically clip to [0,1] before arcsin to avoid NaN from float drift.
        r_norm = np.clip(r_norm, 0.0, 1.0)
        s = np.arctan2(b, a)                        # [N], rad
        d = np.arcsin(r_norm)                       # [N], rad >= 0

        X_2d = torch.tensor(
            np.stack([s, d], axis=1),
            dtype=torch.float64, device=self._device,
        )
        try:
            gp = self._make_2d_gp(X_2d, y_std)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()
        except GP_FIT_ERRORS as e:
            # Per-slice failure: increment counter so aggregate `_fallback_count`
            # reflects the true number of failed GP fits across slices, not just
            # the all-slices-failed terminal case at the call site below.
            self._fallback_count += 1
            self._fallback_reasons.append(f"slice GP fit: {type(e).__name__}: {e}")
            return None

        # Dense grid on the d=0 slice.
        s_grid = torch.linspace(
            -math.pi, math.pi, self.grid_size, dtype=torch.float64, device=self._device,
        )
        d_grid = torch.zeros_like(s_grid)
        X_test = torch.stack([s_grid, d_grid], dim=1)        # [G, 2]

        acqf = qLogExpectedImprovement(model=gp, best_f=best_y_std)
        with torch.no_grad():
            acq_vals = acqf(X_test.unsqueeze(1))             # [G]
        idx = int(torch.argmax(acq_vals))
        return float(s_grid[idx].item()), float(acq_vals[idx].item())

    def next_candidate(self) -> np.ndarray:
        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        D = x_hist.shape[-1]

        # Anchor: current best on unit sphere (history may include cold-start
        # vectors with slightly different norms; project all to S^{D-1}).
        norms = np.linalg.norm(x_hist, axis=1, keepdims=True)
        safe = np.where(norms > 1e-9, norms, 1.0)
        u_hist = x_hist / safe                                # [N, D]
        i_best = int(np.argmax(y_hist.flatten()))
        u_anchor = u_hist[i_best]                              # [D]

        Y = torch.tensor(
            y_hist.flatten(), dtype=torch.float64, device=self._device,
        ).unsqueeze(-1)
        Y_std, _, _, sd_raw = _standardize(Y)
        if sd_raw < 1e-6:
            self._fallback_count += 1
            reason = f"flat-signal: sd={sd_raw:.2e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"{type(self).__name__} flat-signal at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            return _random_on_sphere(D, self.mean_norm)

        best_y_std = float(Y_std.max())

        # Sample K candidate directions in the tangent space at u_anchor.
        # We oversample raw directions (1.5x) so degenerate parallel ones can
        # be replaced without falling below n_directions.
        raw = np.random.randn(int(self.n_directions * 1.5) + 2, D)
        directions: list[np.ndarray] = []
        for v_raw in raw:
            v = _tangent_basis_at(u_anchor, v_raw)
            if v is not None:
                directions.append(v)
            if len(directions) == self.n_directions:
                break
        if not directions:  # extremely unlikely (would need exactly parallel raw)
            self._fallback_count += 1
            self._fallback_reasons.append("no valid tangent direction")
            return _random_on_sphere(D, self.mean_norm)

        # Fit 2D GP per direction and pick the great-circle with the highest EI.
        best = None  # (acq, s, v)
        try:
            for v in directions:
                result = self._evaluate_slice(
                    u_anchor, v, u_hist, Y_std, best_y_std,
                )
                if result is None:
                    continue
                s_star, acq_star = result
                if best is None or acq_star > best[0]:
                    best = (acq_star, s_star, v)
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            raise

        if best is None:
            # Per-slice failures already incremented `_fallback_count` and
            # appended to `_fallback_reasons` in `_evaluate_slice`; only log
            # the terminal aggregate here.
            logger.error(
                f"{type(self).__name__} all slices failed GP fit at iter "
                f"{self.iteration} (cumulative fallback #{self._fallback_count})"
            )
            return _random_on_sphere(D, self.mean_norm)

        _, s_star, v_star = best
        u_new = math.cos(s_star) * u_anchor + math.sin(s_star) * v_star
        # Numerical drift: re-normalise to exact unit norm before scaling.
        u_new = u_new / max(np.linalg.norm(u_new), 1e-12)
        return (u_new * self.mean_norm).reshape(1, -1)


# ===========================================================================
# 4) Doumont et al. 2025 — "Linear BO" (SphericalLinearKernel GP)
# ===========================================================================
# Reviewer tPDn requested Linear BO of Doumont et al. (arXiv:2512.00170) as a
# baseline. Linear BO is a GP whose kernel maps box-normalized inputs through an
# inverse stereographic projection onto a sphere, then applies a *linear*
# (dot-product) kernel — a geometry-aware, box-supported recipe. We add it as a
# fourth recipe to the controlled box-vs-sphere ablation, using Doumont's
# UNMODIFIED kernel (src/spherevo/linear_bo_kernel.py, vendored verbatim).
#
# Two variants, parallel to the Hvarfner/Papenmeier/Xu box+sphere pair:
#   LinearBOBoxSolver     — faithful [0,1]^D reproduction (their kernel's native
#                           domain), acquisition on the unit cube.
#   LinearBOSphereSolver  — our on-sphere adaptation (kernel unchanged; only the
#                           bounds arg and candidate support change, exactly as
#                           we adapt the other three recipes to the sphere).


def _make_linear_bo_gp(
    X: torch.Tensor, Y: torch.Tensor, bounds, device: torch.device,
) -> SingleTaskGP:
    """Build Doumont's Linear-BO GP: ConstantMean + SphericalLinearKernel +
    LogNormal(-4, 1) noise prior. ``bounds`` is (lo, hi) scalars or a per-dim
    sequence, passed straight to the vendored kernel.
    """
    from gpytorch.means import ConstantMean

    from src.spherevo.linear_bo_kernel import SphericalLinearKernel

    D = X.shape[-1]
    kernel = SphericalLinearKernel(ard_num_dims=D, bounds=bounds)
    # Device-safe: a plain LogNormalPrior samples on the CPU after gp.to(cuda),
    # so a fit retry (sample_all_priors) crashed the cell on a GPU. Same draws,
    # same log_prob; a first-attempt fit is computed exactly as before.
    noise_prior = _DeviceSafeLogNormalPrior(loc=-4.0, scale=1.0)
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(1e-4, initial_value=noise_prior.mode),
    )
    gp = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        mean_module=ConstantMean(),
        covar_module=kernel,
        likelihood=likelihood,
    )
    gp.to(X)
    return gp


class LinearBOBoxSolver(HvarfnerScaledGPBoxFaithfulSolver):
    """Linear BO (Doumont et al. 2025) — faithful [0,1]^D box reproduction.

    Reuses the faithful per-coord [p01, p99] -> [0,1]^D rescaling of the parent,
    swaps in Doumont's SphericalLinearKernel GP, and reproduces Doumont's own
    acquisition loop: analytic ``LogExpectedImprovement`` optimised with
    ``optimize_acqf`` at ``num_restarts=4``, ``raw_samples=512``,
    ``sample_around_best=True`` (``src/optimize.py`` of colmont/linear-bo). This
    is the method's *published* candidate-generation recipe, so it is the fair
    Table-1 row for Linear BO. Inputs are clamped to [0,1]^D before fitting
    because the kernel asserts inputs lie within its bounds (upstream always
    feeds Sobol-in-[0,1]; our rescaled cold-start tails can fall a hair outside
    p01/p99).
    """

    def _make_gp(self, X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
        X_clamped = X.clamp(0.0, 1.0)
        return _make_linear_bo_gp(X_clamped, Y, bounds=(0.0, 1.0), device=self._device)

    def next_candidate(self) -> np.ndarray:
        from botorch.acquisition.analytic import LogExpectedImprovement

        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        D = x_hist.shape[-1]
        X_raw = torch.tensor(x_hist, dtype=torch.float64, device=self._device)
        X_unit = self._rescale_to_unit_cube(X_raw)
        Y = torch.tensor(
            y_hist.flatten(), dtype=torch.float64, device=self._device
        ).unsqueeze(-1)
        Y_std, _, _, sd_raw = _standardize(Y)
        if sd_raw < 1e-6:
            self._fallback_count += 1
            self._fallback_reasons.append(f"flat-signal: sd={sd_raw:.2e}")
            logger.error(
                f"LinearBO box flat-signal at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): sd={sd_raw:.2e}"
            )
            u_unit = torch.rand(1, D, dtype=torch.float64, device=self._device)
            return self._rescale_from_unit_cube(u_unit).cpu().numpy()

        try:
            gp = self._make_gp(X_unit, Y_std)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()
            # Doumont's exact acquisition: analytic LogEI + sample_around_best.
            acqf = LogExpectedImprovement(model=gp, best_f=float(Y_std.max()))
            unit_bounds = torch.tensor(
                [[0.0] * D, [1.0] * D], dtype=torch.float64, device=self._device,
            )
            try:
                cand_unit, _ = optimize_acqf(
                    acq_function=acqf, bounds=unit_bounds, q=1,
                    num_restarts=4, raw_samples=512,
                    options={
                        "sample_around_best": True,
                        "sample_around_best_sigma": 0.1,
                        "maxiter": 300, "batch_limit": 64,
                    },
                )
            except RuntimeError as e:  # L-BFGS/scipy numerical failure -> fall back
                raise ModelFittingError(f"optimize_acqf failed: {e}") from e
            if not torch.isfinite(cand_unit).all():
                raise ModelFittingError("optimize_acqf returned non-finite candidate")
            return self._rescale_from_unit_cube(cand_unit).detach().cpu().numpy().reshape(1, -1)
        except _BOX_FALLBACK_ERRORS as e:
            self._fallback_count += 1
            self._fallback_reasons.append(f"{type(e).__name__}: {e}")
            logger.error(
                f"LinearBO box fit FAILED at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {type(e).__name__}: {e}"
            )
            u_unit = torch.rand(1, D, dtype=torch.float64, device=self._device)
            return self._rescale_from_unit_cube(u_unit).cpu().numpy()
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            raise


class LinearBOSphereSolver(HvarfnerScaledGPSolver):
    """Linear BO (Doumont et al. 2025) — our on-sphere adaptation.

    Doumont's SphericalLinearKernel is kept UNMODIFIED. Only its ``bounds``
    argument (set to the symmetric bounding box of the data/sphere) and the
    candidate-generation support (uniform on the empirical shell) change —
    exactly the adaptation applied to the Hvarfner/Papenmeier/Xu recipes for
    the on-sphere column of the ablation.
    """

    def _make_gp(self, X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
        # Bounds must bracket both history and the on-sphere candidate pool
        # (radius = mean_norm). Use a symmetric box a touch beyond both.
        max_abs = float(X.abs().max().item()) if X.numel() else self.mean_norm
        B = max(max_abs, self.mean_norm) * 1.1 + 1e-6
        return _make_linear_bo_gp(X, Y, bounds=(-B, B), device=self._device)


class LinearBOSphereCalibratedSolver(LinearBOSphereSolver):
    """Linear BO on the shell with ``bounds`` matched to the on-sphere spread.

    ``LinearBOSphereSolver`` passes the sphere's *bounding box* (+-1.1 r_bar).
    That silently mis-scales the kernel, because the lengthscale cancels out of
    its input normalisation and only the box WIDTH survives:

        ||x_scaled|| = 2 ||x - c|| / (W * sqrt(D/2)),    W = max - min

    On the published [0,1]^D box this gives 0.817, the regime the inverse
    stereographic projection is calibrated for. A radius-7.98 sphere in D=256
    has a bounding box about 5.5x wider than the coordinate spread of points
    lying ON it (sigma = r_bar/sqrt(D) = 0.499), so the same expression gives
    0.080: every projected point lands within 0.16 of the south pole and the
    linear kernel is nearly constant. One learnable scalar (``raw_glob_ls``)
    then has to recover a factor of ten from at most 30 observations.

    Passing the coordinate spread instead of the bounding box puts the sphere
    arm in the same regime as the box arm. This is a fairness fix to OUR
    adaptation of Doumont et al.'s kernel, which is itself unmodified.

    Two caveats we state rather than hide. The kernel asserts its inputs lie
    within ``bounds``, so the width must still bracket the candidate pool; and a
    sphere does NOT fill its bounding box, so no admissible width reproduces the
    box arm's 0.817 exactly. Matching the pool's actual coordinate range gives
    about 0.28 -- roughly 3.5x closer to the calibrated regime than 0.080, with
    no clipping of the inputs. The residual gap is intrinsic to applying a
    box-calibrated kernel to a spherical support, and we report it as such
    rather than clamping coordinates to manufacture the target value.
    """

    #: Half-width in units of the on-sphere coordinate sd, r_bar/sqrt(D). Must
    #: bracket the pool: the max of ~524k standard normals (2048 candidates x
    #: 256 coords) sits near 4.6 sd, so 5.0 leaves headroom without clipping.
    spread_sigmas: float = 5.0

    def _make_gp(self, X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
        sigma = self.mean_norm / math.sqrt(self._dim)
        max_abs = float(X.abs().max().item()) if X.numel() else 0.0
        B = max(self.spread_sigmas * sigma, max_abs) * 1.05 + 1e-6
        return _make_linear_bo_gp(X, Y, bounds=(-B, B), device=self._device)


# ===========================================================================
# Papenmeier MSR variant: trust region + RAASP-style candidates + argmax
# (NOT the published MSR acquisition call; see PapenmeierMSRPublishedSolver)
# ===========================================================================
# Reviewer 6qvY: the published MSR uses (i) ell_0 = 0.1*sqrt(D) init (fixed in
# PapenmeierMSRSolver) AND (ii) RAASP. This class keeps (i) and replaces the
# acquisition step by a TuRBO-style trust region on the [0,1]^D min-max domain
# with RAASP-style sparse perturbations of the incumbent, from which qLogEI picks
# the argmax. It is a variant and is printed only under its own name
# (papenmeier_msr_raasp). The published MSR acquisition maximiser -- analytic
# LogEI optimised by multi-start L-BFGS-B whose starting points include
# BoTorch's sample_around_best RAASP points (Papenmeier et al. 2025, App. B.1:
# Gaussian perturbations with sigma 1e-3 around the top 5 % of observations) --
# is PapenmeierMSRPublishedSolver / PapenmeierMSRPublishedNormBoxSolver below.
#
# The candidate rule this class implements is TuRBO's RAASP (Eriksson et al.
# 2019), which Papenmeier et al. 2025 describe in their background section:
# "With a probability of min(1, 20/d), it replaces the value of each dimension
# of the incumbent solution with a value drawn uniformly at random within the
# TR bounds." That sentence describes TuRBO, not MSR's own RAASP.


class PapenmeierMSRRaaspSolver(PapenmeierMSRSolver):
    """Papenmeier MSR variant: MLE Scaled (0.1*sqrt(D)) GP + trust region + RAASP-style argmax.

    NOT the published MSR acquisition call (that is ``PapenmeierMSRPublishedSolver``).
    GP: inherited PapenmeierMSRSolver._make_gp (Matern-5/2, ell_0 = 0.1*sqrt(D),
    MLE, no prior), fitted on [0,1]^D-normalized inputs (per-coord ZINC min-max).
    Candidate generation: TuRBO-style trust region (L0=0.8, x2 after 3
    successes, /2 after max(4, ceil(D)) failures, lengthscale-weighted per-dim
    side lengths) + TuRBO's RAASP sparse axis-aligned perturbation of the
    incumbent. qLogEI selects the argmax among the candidates.
    """

    _SUCC_TOL = 3
    _L0 = 0.8
    _L_MIN = 0.5 ** 7
    _L_MAX = 1.6

    def __init__(self, black_box, x0, y0, *, n_candidates: int = 5000):
        super().__init__(black_box, x0, y0, n_candidates=n_candidates)
        self._L = self._L0
        self._succ = 0
        self._fail = 0
        # TuRBO failure tolerance ceil(dim / batch), batch = 1.
        self._fail_tol = max(4, int(math.ceil(self._dim)))
        self._tr_best = float(np.max(y0))
        self._low, self._high = _load_zinc_per_coord_bounds(mode="minmax")
        if self._low.shape[0] != self._dim:
            raise ValueError(
                f"ZINC per-coord bounds dim {self._low.shape[0]} != solver dim "
                f"{self._dim} — RAASP box bounds are SELFIES-256D only."
            )
        self._span = np.where(self._high - self._low > 1e-12, self._high - self._low, 1e-12)

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        val = float(np.asarray(y).flatten()[0])
        thresh = self._tr_best + 1e-3 * max(abs(self._tr_best), 1e-3)
        if val > thresh:
            self._succ += 1
            self._fail = 0
        else:
            self._fail += 1
            self._succ = 0
        self._tr_best = max(self._tr_best, val)
        if self._succ >= self._SUCC_TOL:
            self._L = min(self._L * 2.0, self._L_MAX)
            self._succ = 0
        if self._fail >= self._fail_tol:
            self._L = max(self._L / 2.0, self._L_MIN)
            self._fail = 0

    def _to_unit(self, x):
        return (x - self._low) / self._span

    def _from_unit(self, xn):
        return self._low + xn * self._span

    def _raasp_candidates(self, x_c: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
        """RAASP: sparse axis-aligned perturbation of the incumbent within the TR."""
        D = x_c.shape[0]
        lb = np.clip(x_c - weights * self._L / 2.0, 0.0, 1.0)
        ub = np.clip(x_c + weights * self._L / 2.0, 0.0, 1.0)
        prob = min(1.0, 20.0 / D)
        cand = np.tile(x_c, (n, 1))
        mask = np.random.rand(n, D) < prob
        empty = ~mask.any(axis=1)
        if empty.any():
            j = np.random.randint(0, D, size=int(empty.sum()))
            mask[np.where(empty)[0], j] = True
        pert = lb + np.random.rand(n, D) * (ub - lb)
        return np.where(mask, pert, cand)

    def next_candidate(self) -> np.ndarray:
        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        D = x_hist.shape[-1]
        x_norm = np.clip(self._to_unit(x_hist), 0.0, 1.0)
        incumbent = x_norm[int(np.argmax(y_hist.flatten()))]
        X = torch.tensor(x_norm, dtype=torch.float64, device=self._device)
        Y = torch.tensor(y_hist.flatten(), dtype=torch.float64, device=self._device).unsqueeze(-1)
        Y_std, _, _, sd_raw = _standardize(Y)
        if sd_raw < 1e-6:
            self._fallback_count += 1
            self._fallback_reasons.append(f"flat-signal: sd={sd_raw:.2e}")
            cand = self._raasp_candidates(incumbent, np.ones(D), 1)[0]
            return self._from_unit(cand).reshape(1, -1)
        try:
            gp = self._make_gp(X, Y_std)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()
            ls = gp.covar_module.base_kernel.lengthscale.detach().view(-1).double().cpu().numpy()
            ls = np.clip(ls, 1e-6, None)
            weights = ls / ls.mean()
            weights = weights / (np.prod(weights) ** (1.0 / D))
            cands = self._raasp_candidates(incumbent, weights, self.n_candidates)
            Xc = torch.tensor(cands, dtype=torch.float64, device=self._device)
            acqf = qLogExpectedImprovement(model=gp, best_f=float(Y_std.max()))
            try:
                with torch.no_grad():
                    av = acqf(Xc.unsqueeze(1))
            except RuntimeError as e:  # numerical acqf failure -> fall back
                raise ModelFittingError(f"acqf eval failed: {e}") from e
            best = cands[int(torch.argmax(av))]
            return self._from_unit(best).reshape(1, -1)
        except _BOX_FALLBACK_ERRORS as e:
            self._fallback_count += 1
            self._fallback_reasons.append(f"{type(e).__name__}: {e}")
            cand = self._raasp_candidates(incumbent, np.ones(D), 1)[0]
            return self._from_unit(cand).reshape(1, -1)
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            raise


# ===========================================================================
# Papenmeier MSR -- the PUBLISHED acquisition maximiser
# ===========================================================================
# MSR ("MLE Scaled with RAASP"; Papenmeier, Poloczek & Nardi, ICML 2025,
# arXiv:2502.09198, Sec. 4 and App. B) maximises analytic LogEI with BoTorch's
# multi-start L-BFGS-B. The reference implementation
# (github.com/lpapenme/understanding-hdbo-code, src/understanding_hdbo/bo.py,
# lines 170-190, pinned to botorch==0.9.5) makes exactly this call:
#
#     with gpytorch.settings.cholesky_max_tries(9):
#         acq = LogExpectedImprovement(model=model, best_f=y_norm.max().item())
#         options = {"batch_limit": 1, "sample_around_best": True,
#                    "sample_around_best_sigma": 0.001}
#         x_next, _ = optimize_acqf(acq_function=acq, bounds=bounds,  # [0,1]^d
#                                   q=1, num_restarts=5, raw_samples=512,
#                                   options=options)
#
# RAASP is therefore neither a finite candidate set nor a trust region. BoTorch
# draws 512 scrambled Sobol points plus 512 perturbations of the top 5 % of the
# training inputs ranked by posterior mean (half truncated-normal with sigma =
# 1e-3 in every dimension, half with each dimension perturbed with probability
# min(1, 20/d)), picks 5 starting points by Boltzmann sampling on LogEI and runs
# L-BFGS-B from each for at most 2000 iterations. In botorch 0.9.5 batch_limit=1
# is what keeps the 5 runs independent (a batch is otherwise one joint problem);
# 0.17.2 optimises batched starts independently anyway, so here it only splits
# the work into single-start calls and single-point raw-sample evaluations.
#
# PapenmeierMSRRaaspSolver (trust region + candidate argmax) and
# PapenmeierMSRBoxFaithfulSolver (no RAASP, qLogEI, 10 restarts) are NOT this
# procedure. The two classes below keep the repo's MLE-scaled GP
# (PapenmeierMSRSolver._make_gp), cold start, budget, flat-signal check and
# random fallback of the two existing box arms and change only the acquisition
# step to the call above:
#
#   papenmeier_msr_published           [p01, p99] box of papenmeier_msr_box_faithful
#   papenmeier_msr_published_norm_box  [min, max] box of papenmeier_msr_norm_box

#: Settings of the published call. The reference code does not pass maxiter;
#: 2000 is the gen_candidates_scipy default in botorch 0.9.5 (their pin) and in
#: 0.17.2 and matches "up to 2000 iterations" (App. B.2). It is passed
#: explicitly so that a future BoTorch default cannot change this arm silently.
MSR_PUBLISHED_NUM_RESTARTS = 5
MSR_PUBLISHED_RAW_SAMPLES = 512
MSR_PUBLISHED_RAASP_SIGMA = 1e-3
MSR_PUBLISHED_MAXITER = 2000
MSR_PUBLISHED_CHOLESKY_MAX_TRIES = 9


def _msr_published_options() -> dict[str, Any]:
    """Fresh copy of the ``optimize_acqf`` options of the published MSR call."""
    return {
        "batch_limit": 1,
        "sample_around_best": True,
        "sample_around_best_sigma": MSR_PUBLISHED_RAASP_SIGMA,
        "maxiter": MSR_PUBLISHED_MAXITER,
    }


def _propose_msr_published_on_unit_box(
    gp: SingleTaskGP, best_y: float, X_unit: torch.Tensor,
) -> torch.Tensor:
    """Maximise analytic LogEI on [0,1]^D with the published MSR call.

    ``gp`` must have been fitted on ``X_unit`` (training inputs in unit-cube
    coordinates). Returns a ``1 x D`` candidate in unit-cube coordinates.

    RAASP centres: BoTorch perturbs the top 5 % of ``get_X_baseline(acqf)``,
    which is ``acqf.X_baseline`` when set and the GP training inputs otherwise.
    In the paper every observation lies inside [0,1]^d. Here cold-start
    encodings can lie outside the box, and BoTorch then sends an out-of-box
    coordinate of a truncated-normal perturbation to the opposite face of the
    box and keeps it unchanged in a subset perturbation; botorch 0.17.2's
    ``gen_candidates_scipy`` rejects such a starting point with BotorchError
    (0.9.5 clamped it silently). We therefore hand BoTorch the observations
    projected onto the box. For observations inside the box, which includes
    every point this arm proposes, the projection is the identity.

    Two consequences of this call that the paper's deviation list must carry:

    * (D5) BoTorch ranks the RAASP centres by the GP posterior mean evaluated
      **at the projected points**, not at the observed ones. When an
      out-of-box observation is the best by posterior mean, its projection may
      rank differently. Measured on real cold starts (seeds 42-51 x 11 tasks,
      repo MSR GP fitted on the 5 cold-start points): the chosen centre changed
      in 3 of 109 step-1 cold starts, all near-ties (largest posterior-mean
      shift 1.1e-4 in standardised units; cr/H_review_solvers.md section 3.5).
      Exact reference semantics would rank at the observed points and project
      only the chosen centre.
    * (D6) "Top 5 %" is ``n_best = max(1, round(0.05 * n))`` observations, which
      is a single point for every acquisition step of this protocol
      (n = 5 ... 29), so RAASP always perturbs one centre.

    Raises ModelFittingError when the optimiser fails numerically or returns a
    non-finite candidate, so the caller falls back exactly like the box arms.
    """
    D = X_unit.shape[-1]
    unit_bounds = torch.stack([
        torch.zeros(D, dtype=torch.float64, device=X_unit.device),
        torch.ones(D, dtype=torch.float64, device=X_unit.device),
    ])
    with gpytorch.settings.cholesky_max_tries(MSR_PUBLISHED_CHOLESKY_MAX_TRIES):
        acqf = LogExpectedImprovement(model=gp, best_f=best_y)
        acqf.X_baseline = X_unit.detach().clamp(0.0, 1.0)
        try:
            candidate, acq_value = optimize_acqf(
                acq_function=acqf,
                bounds=unit_bounds,
                q=1,
                num_restarts=MSR_PUBLISHED_NUM_RESTARTS,
                raw_samples=MSR_PUBLISHED_RAW_SAMPLES,
                options=_msr_published_options(),
            )
        except RuntimeError as e:
            raise ModelFittingError(f"optimize_acqf failed: {e}") from e
    if not torch.isfinite(candidate).all():
        raise ModelFittingError(
            f"optimize_acqf returned non-finite candidate "
            f"(acq_value={float(acq_value):.3g})"
        )
    return candidate.detach().reshape(1, D)


class PapenmeierMSRPublishedSolver(_BoxFaithfulMixin, PapenmeierMSRSolver):
    """Published MSR on the faithful per-coordinate [p01, p99] box.

    Identical to ``PapenmeierMSRBoxFaithfulSolver`` (same box and rescaling,
    same MLE-scaled GP with l0 = 0.1*sqrt(D), same flat-signal check and
    uniform-in-box fallback) except for the acquisition step, which is the
    published MSR call: analytic LogEI, RAASP via ``sample_around_best``
    (sigma 1e-3), 5 restarts, 512 raw samples, L-BFGS-B with batch_limit=1 and
    maxiter=2000 (see ``_propose_msr_published_on_unit_box``).

    Still differs from the paper in: initial design (5 encoded ZINC molecules,
    not floor(3*sqrt(d)) Sobol points) and budget (N = 30); the domain (a
    rescaled [p01, p99] latent box that observations can lie outside of, so
    RAASP centres are projected into it, and BoTorch ranks them by the
    posterior mean at the projected points -- D5, changed the centre in 3/109
    real step-1 cold starts, all near-ties); "top 5 %" being a single centre
    at every step of this protocol (D6, n <= 29); the repo's GP
    (Interval(0.01, 100) lengthscale constraint, no outputscale prior,
    noise > 1e-5, BoTorch's default Standardize outcome transform on already
    standardised y); and the GP fit (no cholesky_max_tries(9), no Adam
    fallback; numerical failures fall back to a uniform point in the box, as
    in every box arm).
    """

    def next_candidate(self) -> np.ndarray:
        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        D = x_hist.shape[-1]

        X_raw = torch.tensor(x_hist, dtype=torch.float64, device=self._device)
        X_unit = self._rescale_to_unit_cube(X_raw)
        Y = torch.tensor(
            y_hist.flatten(), dtype=torch.float64, device=self._device,
        ).unsqueeze(-1)
        Y_std, _, _, sd_raw = _standardize(Y)
        if sd_raw < 1e-6:
            self._fallback_count += 1
            reason = f"flat-signal: sd={sd_raw:.2e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"{type(self).__name__} flat-signal at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            u_unit = torch.rand(1, D, dtype=torch.float64, device=self._device)
            return self._rescale_from_unit_cube(u_unit).cpu().numpy()

        try:
            gp = self._make_gp(X_unit, Y_std)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()
            best_y_std = float(Y_std.max())
            cand_unit = _propose_msr_published_on_unit_box(gp, best_y_std, X_unit)
            cand_raw = self._rescale_from_unit_cube(cand_unit)
            return cand_raw.detach().cpu().numpy().reshape(1, -1)
        except _BOX_FALLBACK_ERRORS as e:
            self._fallback_count += 1
            reason = f"{type(e).__name__}: {e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"{type(self).__name__} GP fit FAILED at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            u_unit = torch.rand(1, D, dtype=torch.float64, device=self._device)
            return self._rescale_from_unit_cube(u_unit).cpu().numpy()
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            raise


def _norm_box_msr_published_next_candidate(host: StepByStepSolver) -> np.ndarray:
    """``_norm_box_next_candidate`` with the published MSR acquisition step.

    Same ZINC [min, max] box (``_load_zinc_per_coord_bounds(mode="minmax")``),
    rescaling, GP, flat-signal check and uniform-in-box fallback; only
    ``_propose_qlogei_on_normalized_box`` is replaced by
    ``_propose_msr_published_on_unit_box``.
    """
    missing = [name for name in _BOX_HOST_REQUIREMENTS if not hasattr(host, name)]
    if missing:
        raise TypeError(
            f"{type(host).__name__} cannot use _norm_box_msr_published_next_candidate: "
            f"host class is missing required attributes: {missing}."
        )

    low, high = _load_zinc_per_coord_bounds(mode="minmax")

    x_hist, y_hist = host.get_history_as_arrays(penalize_nans_with=-1.0)
    D = x_hist.shape[-1]
    if low.shape[0] != D:
        raise ValueError(
            f"ZINC bounds have D={low.shape[0]} but history has D={D} — "
            f"recompute zinc_per_coord_bounds.json for this codec."
        )
    span = high - low
    if (span <= 0).any():
        raise ValueError(f"per-coord span has non-positive entries: min={span.min():.3g}")

    x_norm = (x_hist - low) / span  # [N, D] in ~[0, 1]
    X = torch.tensor(x_norm, dtype=torch.float64, device=host._device)
    Y = torch.tensor(
        y_hist.flatten(), dtype=torch.float64, device=host._device
    ).unsqueeze(-1)
    Y_std, _, _, sd_raw = _standardize(Y)
    if sd_raw < 1e-6:
        host._fallback_count += 1
        reason = f"flat-signal: sd={sd_raw:.2e}"
        host._fallback_reasons.append(reason)
        logger.error(
            f"{type(host).__name__} flat-signal at iter {host.iteration} "
            f"(fallback #{host._fallback_count}): {reason}"
        )
        return _random_in_normalized_box(low, high)

    try:
        gp = host._make_gp(X, Y_std)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()
        best_y_std = float(Y_std.max())
        cand_unit = _propose_msr_published_on_unit_box(gp, best_y_std, X)
        cand_norm = cand_unit.cpu().numpy().reshape(1, -1)
        return cand_norm * span + low  # de-normalize to raw VAE latent space
    except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
        raise
    except _BOX_FALLBACK_ERRORS as e:
        host._fallback_count += 1
        reason = f"{type(e).__name__}: {e}"
        host._fallback_reasons.append(reason)
        logger.error(
            f"{type(host).__name__} GP fit FAILED at iter {host.iteration} "
            f"(fallback #{host._fallback_count}): {reason}"
        )
        return _random_in_normalized_box(low, high)


class PapenmeierMSRPublishedNormBoxSolver(PapenmeierMSRSolver):
    """Published MSR on the ZINC [min, max] box of ``papenmeier_msr_norm_box``.

    Identical to ``PapenmeierMSRNormBoxSolver`` except for the acquisition step
    (``_propose_msr_published_on_unit_box``). Like the existing norm_box arms it
    always uses the SELFIES [min, max] bounds. Remaining differences to the
    paper are those listed for ``PapenmeierMSRPublishedSolver``.
    """

    def next_candidate(self) -> np.ndarray:
        return _norm_box_msr_published_next_candidate(self)
