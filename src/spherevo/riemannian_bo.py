"""Geometry-aware Bayesian Optimization (GaBO / HD-GaBO / MaternGaBO) on S^(D-1).

Wraps the vendored BoManifolds packages
(``src/external/gabotorch_bm``, ``src/external/matern_gabo_bm``) into our
:class:`~src.spherevo.solver.StepByStepSolver` interface. The methods are
manifold-aware GP-BO baselines on the sphere:

- :class:`GaBOSolver` — Sphere Gaussian kernel + manifold acquisition (CoRL19).
- :class:`HDGaBOSolver` — Nested-sphere kernel learning a low-D latent
  sphere (NeurIPS20). Configurable via ``latent_dim`` (manifold dim, e.g.
  ``2`` for S² as in the original paper or ``16`` to match SphereVO-CMA's
  random R^16 subspace).
- :class:`MaternGaBOSolver` — Sphere Riemannian Matern 5/2 kernel (CoRL21).

All three share the spherical scaffold from CLAUDE.md Lesson #4: project
history to S^(D-1) by L2 normalisation, fit a GP with sphere-aware kernel,
optimise EI on the manifold via pymanopt TrustRegions, lift the proposal
back to ``mean_norm`` × unit-sphere. On any GP/optim failure we fall back to
a uniform random sphere sample (same pattern as ``GPGuidedSphericalSolver``).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import torch

from src.spherevo.solver import StepByStepSolver

logger = logging.getLogger(__name__)

# Extrapolated beta_min for SphereGaussianKernel.
# Their tabulation: dim 51 → 0.21, dim 101 → 0.21, dim<=160 → 0.21.
# For our ambient D=256 (S^255) we extrapolate to the same plateau. If the
# kernel matrix is non-PD with this β floor the GP fit raises and we fall
# back to random-sphere sampling for that step.
_BETA_MIN_HIGH_D = 0.21


def _normalize(x: np.ndarray | torch.Tensor, eps: float = 1e-10):
    if isinstance(x, torch.Tensor):
        return x / (torch.linalg.norm(x, dim=-1, keepdim=True) + eps)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


class _RiemannianBOBase(StepByStepSolver):
    """Common scaffold for the 3 Riemannian BO methods.

    Subclasses implement :meth:`_build_gp_and_acq` (returns ``(gp, acq, manifold)``)
    and :meth:`_propose_on_manifold` (returns proposal as a ``[1, D]`` torch tensor
    on the AMBIENT unit sphere).
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        num_restarts: int = 5,
        raw_samples: int = 100,
        max_pymanopt_iter: int = 200,
    ) -> None:
        super().__init__(black_box, x0, y0)
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.max_pymanopt_iter = max_pymanopt_iter
        self.gp_fallback_count = 0
        #: One "Type: message" entry per fallback step; run_benchmark writes
        #: count and reasons to the result file (``_surrogate_fallbacks``).
        self.gp_fallback_reasons: list[str] = []

    # -- subclass hooks ------------------------------------------------------

    def _build_gp_and_acq(self, X_unit: torch.Tensor, Y_std: torch.Tensor):
        raise NotImplementedError

    def _propose_on_manifold(self, acq_fct, manifold) -> torch.Tensor:
        raise NotImplementedError

    # -- helpers -------------------------------------------------------------

    def _random_sphere_sample(self) -> np.ndarray:
        """Fallback proposal: uniform random point on S^(D-1) × mean_norm."""
        x = np.random.randn(self._dim)
        x = x / (np.linalg.norm(x) + 1e-10) * self.mean_norm
        return x.reshape(1, self._dim)

    def next_candidate(self) -> np.ndarray:
        with warnings.catch_warnings():
            # Old pymanopt 0.2.5 emits SyntaxWarnings under py3.12 + numpy 2.x.
            # Narrow the filter to that specific case so genuine GP-numerical /
            # BoTorch warnings still surface — a blanket ignore here would hide
            # NumericalWarning (= GP degenerated to constant predictor).
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            warnings.filterwarnings(
                "ignore", category=DeprecationWarning, module=r"pymanopt(\..*)?"
            )
            try:
                x_hist, y_hist = self.get_history_as_arrays()
                X_unit = _normalize(x_hist)
                X = torch.tensor(X_unit, dtype=torch.float64)
                Y = torch.tensor(y_hist.flatten(), dtype=torch.float64).unsqueeze(-1)
                # Standardise Y for GP numerical stability.
                Y_std = (Y - Y.mean()) / (Y.std() + 1e-8)

                _, acq_fct, manifold = self._build_gp_and_acq(X, Y_std)
                new_x_unit = self._propose_on_manifold(acq_fct, manifold)
                new_x_unit = new_x_unit.detach().cpu().numpy().reshape(1, -1)
                new_x_unit = _normalize(new_x_unit)
                return new_x_unit * self.mean_norm
            except Exception as e:
                if isinstance(e, (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt)):
                    raise
                # This clause turns ANY exception into a random sphere step, so
                # the reason is always recorded (it reaches the result file).
                self.gp_fallback_count += 1
                self.gp_fallback_reasons.append(f"{type(e).__name__}: {e}")
                logger.warning(
                    "%s fallback #%d: %s: %s — random sphere sample.",
                    type(self).__name__, self.gp_fallback_count,
                    type(e).__name__, e,
                )
                return self._random_sphere_sample()


# ---------------------------------------------------------------------------
# GaBO — CoRL19 Jaquier et al.
# ---------------------------------------------------------------------------


class GaBOSolver(_RiemannianBOBase):
    """GaBO: SphereGaussianKernel + manifold-aware EI on S^(D-1)."""

    def _build_gp_and_acq(self, X, Y):
        import gpytorch
        import botorch
        import pymanopt.manifolds as pm
        from src.external.gabotorch_bm.kernel_utils.kernels_sphere import SphereGaussianKernel

        D = X.shape[-1]
        kernel = gpytorch.kernels.ScaleKernel(
            SphereGaussianKernel(beta_min=_BETA_MIN_HIGH_D),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
        )
        noise_prior = gpytorch.priors.GammaPrior(1.1, 0.05)
        lik = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
            initial_value=(noise_prior.concentration - 1) / noise_prior.rate,
        )
        gp = botorch.models.SingleTaskGP(X, Y, covar_module=kernel, likelihood=lik)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
        botorch.fit.fit_gpytorch_mll(mll)
        gp.eval()

        best_y = float(Y.max().item())
        acq = botorch.acquisition.ExpectedImprovement(model=gp, best_f=best_y, maximize=True)
        manifold = pm.Sphere(D)
        return gp, acq, manifold

    def _propose_on_manifold(self, acq_fct, manifold):
        import pymanopt.solvers as ps
        from src.external.gabotorch_bm.manifold_optimization.manifold_optimize import (
            joint_optimize_manifold,
        )

        D = manifold.dim + 1  # pymanopt Sphere(N) has manifold dim N-1, ambient N.
        bounds = torch.stack([
            -torch.ones(D, dtype=torch.float64),
            torch.ones(D, dtype=torch.float64),
        ])
        solver = ps.TrustRegions(maxiter=self.max_pymanopt_iter, logverbosity=0)
        new_x = joint_optimize_manifold(
            acq_fct, manifold, solver, q=1,
            num_restarts=self.num_restarts, raw_samples=self.raw_samples,
            bounds=bounds,
        )
        return new_x.reshape(1, -1)


# ---------------------------------------------------------------------------
# MaternGaBO — CoRL21 Jaquier et al. / Borovitskiy et al.
# ---------------------------------------------------------------------------


class MaternGaBOSolver(_RiemannianBOBase):
    """MaternGaBO: Sphere Riemannian Matern 5/2 kernel + manifold acquisition."""

    def _build_gp_and_acq(self, X, Y):
        import gpytorch
        import botorch
        import pymanopt.manifolds as pm
        from src.external.matern_gabo_bm.kernel_utils.kernels_sphere import (
            SphereRiemannianMaternKernel,
        )

        D = X.shape[-1]
        manifold_dim = D - 1
        base_kernel = SphereRiemannianMaternKernel(
            dim=manifold_dim,
            nu=2.5,
            nu_prior=gpytorch.priors.GammaPrior(2.0, 0.1),
            lengthscale_prior=None,
        )
        kernel = gpytorch.kernels.ScaleKernel(
            base_kernel,
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
        )
        noise_prior = gpytorch.priors.GammaPrior(1.1, 0.05)
        lik = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
            initial_value=(noise_prior.concentration - 1) / noise_prior.rate,
        )
        gp = botorch.models.SingleTaskGP(X, Y, covar_module=kernel, likelihood=lik)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
        botorch.fit.fit_gpytorch_mll(mll)
        gp.eval()

        best_y = float(Y.max().item())
        acq = botorch.acquisition.ExpectedImprovement(model=gp, best_f=best_y, maximize=True)
        manifold = pm.Sphere(D)
        return gp, acq, manifold

    def _propose_on_manifold(self, acq_fct, manifold):
        from src.external.matern_gabo_bm.manifold_optimization.manifold_optimize import (
            joint_optimize_manifold,
        )
        from src.external.matern_gabo_bm.manifold_optimization.robust_trust_regions import (
            TrustRegions,
        )

        D = manifold.dim + 1
        bounds = torch.stack([
            -torch.ones(D, dtype=torch.float64),
            torch.ones(D, dtype=torch.float64),
        ])
        solver = TrustRegions(maxiter=self.max_pymanopt_iter, logverbosity=0)
        new_x = joint_optimize_manifold(
            acq_fct, manifold, solver, q=1,
            num_restarts=self.num_restarts, raw_samples=self.raw_samples,
            bounds=bounds,
        )
        return new_x.reshape(1, -1)


# ---------------------------------------------------------------------------
# HD-GaBO — NeurIPS20 Jaquier & Rozo
# ---------------------------------------------------------------------------


class HDGaBOSolver(_RiemannianBOBase):
    """HD-GaBO: nested-sphere learned latent sphere (NeurIPS20).

    Parameters
    ----------
    latent_dim : int
        Manifold dimension of the latent sphere. ``latent_dim=2`` ⇒ S² in R³
        (matches the original paper); ``latent_dim=16`` ⇒ S^16 in R^17 (matches
        SphereVO-CMA's R^16 random subspace).
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        latent_dim: int = 2,
        num_restarts: int = 3,
        raw_samples: int = 30,
        max_pymanopt_iter: int = 50,
        fit_maxiter: int = 15,
        fit_init_candidates: int = 4,
        recon_maxiter: int = 25,
        recon_init_candidates: int = 10,
    ) -> None:
        super().__init__(
            black_box, x0, y0,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            max_pymanopt_iter=max_pymanopt_iter,
        )
        # The vendored kernel takes the AMBIENT dim of the latent sphere.
        self._latent_ambient = latent_dim + 1
        # Cap manifold fit budget — see ``_build_gp_and_acq`` for rationale.
        self._fit_maxiter = fit_maxiter
        self._fit_init_candidates = fit_init_candidates
        self._recon_maxiter = recon_maxiter
        self._recon_init_candidates = recon_init_candidates

    def _build_gp_and_acq(self, X, Y):
        import gpytorch
        import botorch
        import pymanopt.manifolds as pm
        import pymanopt.solvers as ps
        from torch.autograd import Variable
        from src.external.gabotorch_bm.kernel_utils.kernels_sphere import SphereGaussianKernel
        from src.external.gabotorch_bm.kernel_utils.kernels_nested_sphere import (
            NestedSphereGaussianKernel,
        )
        from src.external.gabotorch_bm.manifold_optimization.manifold_gp_fit import (
            fit_gpytorch_manifold,
        )
        from src.external.gabotorch_bm.nested_mappings.nested_spheres_utils import (
            projection_from_sphere_to_subsphere,
        )
        from src.external.gabotorch_bm.nested_mappings.nested_spheres_optimization import (
            optimize_reconstruction_parameters_nested_sphere,
        )

        D = X.shape[-1]
        latent_D = self._latent_ambient

        # --- ambient kernel: jointly optimises GP hyperparams + nested-sphere axes
        k_amb = gpytorch.kernels.ScaleKernel(
            NestedSphereGaussianKernel(dim=D, latent_dim=latent_D, beta_min=_BETA_MIN_HIGH_D),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
        )
        # --- latent kernel: standard Sphere Gaussian on the learned subsphere
        k_lat = gpytorch.kernels.ScaleKernel(
            SphereGaussianKernel(beta_min=_BETA_MIN_HIGH_D),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
        )
        k_lat.to(X.dtype, non_blocking=False)

        noise_prior = gpytorch.priors.GammaPrior(1.1, 0.05)
        lik_amb = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
            initial_value=(noise_prior.concentration - 1) / noise_prior.rate,
        )
        lik_lat = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
            initial_value=(noise_prior.concentration - 1) / noise_prior.rate,
        )

        gp_amb = botorch.models.SingleTaskGP(X, Y, covar_module=k_amb, likelihood=lik_amb)
        mll_amb = gpytorch.mlls.ExactMarginalLogLikelihood(gp_amb.likelihood, gp_amb)
        # Manifold-aware joint fit (GP hyperparams + nested-sphere axes via pymanopt).
        # The vendored fitter mutates `mll_amb` in place; we don't go through
        # botorch.fit.fit_gpytorch_mll (its modern optimizer protocol is incompatible).
        # The vendored defaults (CG maxiter=500, 200 init candidates) are tuned
        # for D=5 in the original paper. For D=256 with 253 nested-sphere axes
        # the default needs ~17 min/iter — we cap to a budget that finishes in
        # ~1 min/iter while still doing real joint optimisation.
        cg_solver = ps.ConjugateGradient(maxiter=self._fit_maxiter, logverbosity=0)
        fit_gpytorch_manifold(
            mll_amb,
            solver=cg_solver,
            nb_init_candidates=self._fit_init_candidates,
        )

        # Snapshot the learned nested-sphere mapping for both projection and
        # reconstruction (back-projection).
        sphere_axes = [
            Variable(a.data.clone(), requires_grad=False)
            for a in k_amb.base_kernel.axes
        ]
        sphere_distances = [
            Variable(d.data.clone(), requires_grad=False)
            for d in k_amb.base_kernel.distances_to_axis
        ]
        x_sub = projection_from_sphere_to_subsphere(X, sphere_axes, sphere_distances)[-1]

        # Sync hyperparams from the ambient kernel into the latent kernel.
        k_lat.base_kernel.beta = k_amb.base_kernel.beta
        k_lat.outputscale = k_amb.outputscale
        lik_lat.noise = lik_amb.noise

        gp_lat = botorch.models.SingleTaskGP(
            x_sub, Y, covar_module=k_lat, likelihood=lik_lat,
        )

        # Optimise the back-projection (latent → ambient) parameters.
        # Default maxiter=200 + nb_init_candidates=100 burns hundreds of
        # cost evals through 253-axis nested mapping at D=256 — dominant
        # per-step cost. We cap both: 25 trust-region iters and 10 init
        # candidates keep reconstruction meaningful but tractable.
        reconstruction_solver = ps.TrustRegions(maxiter=self._recon_maxiter, logverbosity=0)
        sphere_distances = optimize_reconstruction_parameters_nested_sphere(
            X, x_sub, sphere_axes, reconstruction_solver,
            nb_init_candidates=self._recon_init_candidates,
        )

        best_y = float(Y.max().item())
        acq = botorch.acquisition.ExpectedImprovement(
            model=gp_lat, best_f=best_y, maximize=True,
        )
        latent_manifold = pm.Sphere(latent_D)
        # Stash the nested mapping for the proposal step.
        self._sphere_axes = sphere_axes
        self._sphere_distances = sphere_distances
        return gp_lat, acq, latent_manifold

    def _propose_on_manifold(self, acq_fct, manifold):
        import pymanopt.solvers as ps
        from src.external.gabotorch_bm.manifold_optimization.manifold_optimize import (
            joint_optimize_manifold,
        )
        from src.external.gabotorch_bm.nested_mappings.nested_spheres_utils import (
            projection_from_subsphere_to_sphere,
        )

        latent_D = manifold.dim + 1
        # Bounds in joint_optimize_manifold are unused for sphere manifolds (it
        # samples via manifold.rand()); we just pass a placeholder of the right
        # ambient dim.
        bounds = torch.stack([
            -torch.ones(latent_D, dtype=torch.float64),
            torch.ones(latent_D, dtype=torch.float64),
        ])
        solver = ps.TrustRegions(maxiter=self.max_pymanopt_iter, logverbosity=0)
        new_x_sub = joint_optimize_manifold(
            acq_fct, manifold, solver, q=1,
            num_restarts=self.num_restarts, raw_samples=self.raw_samples,
            bounds=bounds,
        )
        # Lift latent sphere → ambient sphere via the learned nested mapping.
        new_x_amb = projection_from_subsphere_to_sphere(
            new_x_sub, self._sphere_axes, self._sphere_distances,
        )[-1]
        return new_x_amb.reshape(1, -1)
