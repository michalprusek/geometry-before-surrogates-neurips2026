"""SphereVO solvers for molecular optimization on spherical subspaces."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import cma
import numpy as np
import torch

# Re-export task-name mappings from SSOT (src/analysis/config.py)
from src.analysis.config import GUACAMOL_TO_POLI, POLI_TO_GUACAMOL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _normalize(x: np.ndarray) -> np.ndarray:
    """Normalize rows to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def _lift_to_ambient(v: np.ndarray, A: np.ndarray, mean_norm: float) -> np.ndarray:
    """Map S^(d-1) point to ambient R^D via projection matrix A.

    Normalizes v to unit sphere in subspace, lifts via A^T, normalizes
    in ambient space, and scales by mean_norm.
    """
    norm = np.linalg.norm(v)
    v_unit = v / max(norm, 1e-8)
    u = v_unit @ A.T
    u_norm = np.linalg.norm(u)
    u_unit = u / max(u_norm, 1e-8)
    return u_unit * mean_norm


# ---------------------------------------------------------------------------
# Base solver
# ---------------------------------------------------------------------------


class StepByStepSolver(ABC):
    """Poli-compatible step-by-step solver ABC.

    Uses the poli interface: ``black_box(z) -> scores`` for evaluation.
    All solvers (ours and poli-baselines) share this unified interface
    so benchmarks are maximally fair.

    Parameters
    ----------
    black_box : callable
        Latent-space evaluator: ``black_box(z: ndarray[N, D]) -> ndarray[N, 1]``.
        Wraps VAE decode + oracle scoring.
    x0 : np.ndarray, shape ``[N, D]``
        Cold-start latent vectors.
    y0 : np.ndarray, shape ``[N, 1]``
        Cold-start objective values.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
    ) -> None:
        self._black_box = black_box
        x0 = np.asarray(x0, dtype=np.float64)
        y0 = np.asarray(y0, dtype=np.float64).reshape(-1, 1)
        self._dim = x0.shape[1]

        # Mean norm of cold-start vectors (used by subclasses for scaling)
        norms = np.linalg.norm(x0, axis=1)
        self.mean_norm = float(np.mean(norms)) if norms.mean() > 1e-6 else 1.0

        # History
        self.history: dict[str, list[np.ndarray]] = {
            "x": [x0[i : i + 1] for i in range(len(x0))],
            "y": [y0[i : i + 1] for i in range(len(y0))],
        }
        self._all_scores: list[float] = y0.flatten().tolist()
        self.iteration: int = 0

        # Expose raw arrays for subclass __init__
        self._x0 = x0
        self._y0 = y0

    # -- Abstract interface --------------------------------------------------

    @abstractmethod
    def next_candidate(self) -> np.ndarray:
        """Return the next candidate latent vector, shape ``[1, D]``."""
        ...

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Hook called after history update (override in subclasses)."""

    # -- Public API ----------------------------------------------------------

    def step(self) -> float:
        """Run one optimisation iteration: propose, evaluate via black_box, record.

        Returns
        -------
        score : float
            Objective value returned by the black box.
        """
        z = self.next_candidate()  # [1, D]
        scores = self._black_box(z)  # [1, 1]
        score = float(np.asarray(scores).flatten()[0])

        # Update history
        y = np.array([[score]])
        self.history["x"].append(z.reshape(1, -1))
        self.history["y"].append(y)
        self._all_scores.append(score)

        self.post_update(z, y)
        self.iteration += 1

        return score

    def solve(self, max_iter: int, verbose: bool = False) -> None:
        """Run ``max_iter`` steps (poli-compatible entry point)."""
        for i in range(max_iter):
            score = self.step()
            if verbose and (i + 1) % 5 == 0:
                logger.info(
                    f"  iter {i + 1}/{max_iter}  best={self.best_score:.4f}"
                )

    def get_best_performance(self) -> np.ndarray:
        """Return best score as ndarray (poli-compatible)."""
        return np.array([[self.best_score]])

    def get_history_as_arrays(
        self, penalize_nans_with: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(X, Y)`` arrays from the full history."""
        x = np.concatenate(self.history["x"], axis=0)
        y = np.concatenate(self.history["y"], axis=0)
        if penalize_nans_with is not None:
            y[np.isnan(y)] = penalize_nans_with
        return x, y

    # -- Properties ----------------------------------------------------------

    @property
    def best_score(self) -> float:
        """Best objective value seen so far."""
        return float(max(self._all_scores)) if self._all_scores else float("-inf")

    @property
    def all_scores(self) -> list[float]:
        return self._all_scores


# ---------------------------------------------------------------------------
# CMA-ES deferred-tell base
# ---------------------------------------------------------------------------


class _CMAESDeferredBase(StepByStepSolver):
    """Base for CMA-ES solvers with deferred tell (1 eval per step).

    Subclasses override ``_project_to_cma_space`` and ``_cma_to_ambient`` to
    define the mapping between CMA-ES search space and the ambient latent space.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        cma_dim: int,
        initial_mean: np.ndarray,
        initial_sigma: float = 0.5,
        population_size: int = 5,
        bounds: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(black_box, x0, y0)

        x0 = self._x0
        y0 = self._y0

        # Derive a deterministic CMA-ES seed from the global RNG state
        # (np.random.seed is set before solver init in the benchmark script)
        cma_seed = int(np.random.randint(0, 2**31))

        opts: dict = {"popsize": population_size, "verbose": -9, "seed": cma_seed}
        if bounds is not None:
            opts["bounds"] = list(bounds)
        # Subclasses can set self._extra_opts before calling super().__init__()
        if hasattr(self, "_extra_opts"):
            opts.update(self._extra_opts)

        self._cma = cma.CMAEvolutionStrategy(
            initial_mean.flatten().tolist(), initial_sigma, opts
        )
        self._pop_size = population_size

        # Initial tell with cold-start data projected to CMA space
        cs_proj = self._project_to_cma_space(x0)
        cs_scores = y0.flatten().copy()

        if len(cs_proj) < population_size:
            n_extra = population_size - len(cs_proj)
            extra = initial_mean.flatten() + np.random.randn(n_extra, cma_dim) * 0.1
            cs_proj = np.vstack([cs_proj, extra])
            cs_scores = np.concatenate(
                [cs_scores, np.full(n_extra, float(np.min(cs_scores)))]
            )
        elif len(cs_proj) > population_size:
            top_idx = np.argsort(cs_scores)[-population_size:]
            cs_proj = cs_proj[top_idx]
            cs_scores = cs_scores[top_idx]

        _ = self._cma.ask()
        self._cma.tell(cs_proj.tolist(), (-cs_scores).tolist())

        # Deferred-tell buffer
        self._current_pop: list | None = None
        self._pop_idx = 0
        self._gen_raw: list = []
        self._gen_scores: list[float] = []

    def _project_to_cma_space(self, x: np.ndarray) -> np.ndarray:
        """Project ambient points to CMA-ES search space."""
        return x

    def _cma_to_ambient(self, x_cma: np.ndarray) -> np.ndarray:
        """Convert a single CMA-ES candidate to ambient latent space."""
        raise NotImplementedError

    def next_candidate(self) -> np.ndarray:
        # Start new generation if needed
        if self._current_pop is None or self._pop_idx >= len(self._current_pop):
            if self._gen_raw:
                assert len(self._gen_raw) == len(self._gen_scores), (
                    f"Deferred-tell desync: {len(self._gen_raw)} candidates "
                    f"vs {len(self._gen_scores)} scores"
                )
                self._cma.tell(self._gen_raw, [-s for s in self._gen_scores])
                self._gen_raw = []
                self._gen_scores = []
            self._current_pop = self._cma.ask()
            self._pop_idx = 0

        raw = self._current_pop[self._pop_idx]
        self._gen_raw.append(raw)
        self._pop_idx += 1

        return self._cma_to_ambient(np.array(raw)).reshape(1, -1)

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        self._gen_scores.append(float(y.flatten()[0]))


# ---------------------------------------------------------------------------
# CMAESSubspaceSolver — the main paper method
# ---------------------------------------------------------------------------


class CMAESSubspaceSolver(_CMAESDeferredBase):
    """CMA-ES in single R^d subspace with spherical normalization.

    Single random QR subspace [D, d]. CMA-ES operates in R^d.
    Candidates are normalized to S^(d-1), lifted to S^(D-1), and
    scaled by mean_norm.

    Parameters
    ----------
    norm_mode : str
        ``"fixed"`` (default): use constant ``mean_norm`` for all candidates.
        ``"adaptive"``: add a (d+1)-th CMA-ES dimension encoding
        ``log(norm)``, letting CMA-ES co-optimise the ambient norm alongside
        direction.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        subspace_dim: int = 16,
        population_size: int = 5,
        initial_sigma: float = 0.5,
        norm_mode: str = "fixed",
    ) -> None:
        x0 = np.asarray(x0, dtype=np.float64)
        y0 = np.asarray(y0, dtype=np.float64).reshape(-1, 1)

        self.D = x0.shape[1]
        self.d = min(subspace_dim, self.D)
        self._norm_mode = norm_mode

        # Random QR subspace
        A, _ = np.linalg.qr(np.random.randn(self.D, self.d))
        self._A = A

        best_idx = int(np.argmax(y0.flatten()))
        initial_mean = x0[best_idx] @ A  # Project to R^d

        if self._norm_mode == "adaptive":
            best_norm = np.linalg.norm(x0[best_idx])
            log_norm_init = np.log(max(best_norm, 1e-8))
            initial_mean = np.append(initial_mean, log_norm_init)
            cma_dim = self.d + 1
        else:
            cma_dim = self.d

        # Norm trace for post-hoc analysis
        self._norm_trace: list[float] = []

        super().__init__(
            black_box, x0, y0,
            cma_dim=cma_dim,
            initial_mean=initial_mean,
            initial_sigma=initial_sigma,
            population_size=population_size,
        )

    def _project_to_cma_space(self, x: np.ndarray) -> np.ndarray:
        proj = x @ self._A
        if self._norm_mode == "adaptive":
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            log_norms = np.log(np.maximum(norms, 1e-8))
            proj = np.hstack([proj, log_norms])
        return proj

    def _cma_to_ambient(self, x_cma: np.ndarray) -> np.ndarray:
        if self._norm_mode == "adaptive":
            v = x_cma[:self.d].reshape(1, -1)
            log_norm = x_cma[self.d]
            ambient_norm = float(np.exp(np.clip(log_norm, -2.0, 5.0)))
        else:
            v = x_cma.reshape(1, -1)
            ambient_norm = self.mean_norm

        self._norm_trace.append(ambient_norm)

        v_norm = _normalize(v)               # S^(d-1)
        u = v_norm @ self._A.T               # lift to R^D
        u_norm = _normalize(u)               # S^(D-1)
        return (u_norm * ambient_norm).flatten()


# ---------------------------------------------------------------------------
# lq-CMA-ES — Linear Surrogate Pre-screening
# ---------------------------------------------------------------------------


class LQCMAESSubspaceSolver(CMAESSubspaceSolver):
    """CMA-ES subspace with linear surrogate pre-screening (lq-CMA-ES).

    When active, oversamples K * pop_size candidates from CMA-ES, ranks them
    with a linear model y = [1, x] @ w, and evaluates only the top pop_size.
    A Spearman gate disables the model when it provides no ranking signal,
    reverting to standard CMA-ES.

    The linear model has d+1 parameters (intercept + d slopes), fittable from
    ~2d evaluations with zero hyperparameters — qualitatively different from
    GP/RF/MLP surrogates that failed in high-D VAE latent spaces.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        subspace_dim: int = 16,
        population_size: int = 5,
        initial_sigma: float = 0.5,
        K: int = 3,
        gate_threshold: float = 0.3,
    ) -> None:
        super().__init__(
            black_box, x0, y0,
            subspace_dim=subspace_dim,
            population_size=population_size,
            initial_sigma=initial_sigma,
        )
        self._K = K
        self._gate_threshold = gate_threshold
        self._model_active = False

        cs_proj = self._project_to_cma_space(self._x0)
        self._model_X: list[np.ndarray] = [
            cs_proj[i] for i in range(len(cs_proj))
        ]
        self._model_y: list[float] = list(self._y0.flatten())

    @staticmethod
    def _fit_linear(X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        """Fit y = [1, x] @ w via least squares. Returns w or None."""
        n, d = X.shape
        if n < d + 1:
            return None
        Phi = np.column_stack([np.ones(n), X])
        try:
            w, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
            return w
        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def _predict_linear(w: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Predict scores using fitted linear model."""
        Phi = np.column_stack([np.ones(len(X)), X])
        return Phi @ w

    def next_candidate(self) -> np.ndarray:
        if self._current_pop is None or self._pop_idx >= len(self._current_pop):
            if self._gen_raw:
                assert len(self._gen_raw) == len(self._gen_scores), (
                    f"Deferred-tell desync: {len(self._gen_raw)} candidates "
                    f"vs {len(self._gen_scores)} scores"
                )
                self._update_gating()
                for raw, score in zip(self._gen_raw, self._gen_scores):
                    self._model_X.append(np.array(raw))
                    self._model_y.append(score)
                self._cma.tell(self._gen_raw, [-s for s in self._gen_scores])
                self._gen_raw = []
                self._gen_scores = []

            if self._model_active and len(self._model_X) >= self.d + 1:
                all_cands = self._cma.ask(number=self._K * self._pop_size)
                w = self._fit_linear(
                    np.array(self._model_X), np.array(self._model_y),
                )
                if w is not None:
                    preds = self._predict_linear(w, np.array(all_cands))
                    top_idx = np.argsort(preds)[-self._pop_size:]
                    self._current_pop = [all_cands[i] for i in top_idx]
                else:
                    self._current_pop = self._cma.ask()
            else:
                self._current_pop = self._cma.ask()
            self._pop_idx = 0

        raw = self._current_pop[self._pop_idx]
        self._gen_raw.append(raw)
        self._pop_idx += 1
        return self._cma_to_ambient(np.array(raw)).reshape(1, -1)

    def _update_gating(self) -> None:
        """Spearman gating: enable model if rank correlation exceeds threshold."""
        from scipy.stats import spearmanr

        n_gen = len(self._gen_scores)
        if n_gen < 3:
            return
        if len(self._model_X) < self.d + 1:
            return

        w = self._fit_linear(
            np.array(self._model_X), np.array(self._model_y),
        )
        if w is None:
            self._model_active = False
            return

        preds = self._predict_linear(
            w, np.array([np.array(r) for r in self._gen_raw]),
        )
        rho, _ = spearmanr(preds, self._gen_scores)
        self._model_active = bool(
            not np.isnan(rho) and rho >= self._gate_threshold
        )
        logger.debug(
            f"lq-CMA-ES gating: rho={rho:.3f} active={self._model_active}"
        )

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        self._gen_scores.append(float(y.flatten()[0]))


# ---------------------------------------------------------------------------
# GP-free spherical solver
# ---------------------------------------------------------------------------


class GPFreeSphericalSolver(StepByStepSolver):
    """GP-free subspace BO with geodesic trust regions on S^(d-1) subspaces.

    No GP surrogate -- pure random search within geodesic caps
    on K random orthonormal subspaces.

    Parameters
    ----------
    n_subspaces : int
        Number of random QR subspaces (K). Default 16.
    subspace_dim : int
        Dimension of each subspace (d). Default 16.
    trust_region_radius : float
        Maximum geodesic angle for the cap in radians. Default 0.5.
    n_candidates : int
        Number of candidate points to sample per step. Default 2000.
    global_fraction : float
        Fraction of candidates sampled globally (outside TR). Default 0.2.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        n_subspaces: int = 16,
        subspace_dim: int = 16,
        trust_region_radius: float = 0.5,
        n_candidates: int = 2000,
        global_fraction: float = 0.2,
    ) -> None:
        super().__init__(black_box, x0, y0)

        x0 = self._x0
        self.D = x0.shape[1]
        self.d = min(subspace_dim, self.D)
        self.K = n_subspaces
        self.n_candidates = n_candidates
        self.tr_radius = trust_region_radius
        self.global_fraction = global_fraction

        self._projections: list[np.ndarray] = []
        for _ in range(self.K):
            A, _ = np.linalg.qr(np.random.randn(self.D, self.d))
            self._projections.append(A)

        self._active_idx = 0

    def _sample_geodesic_cap(
        self, center: np.ndarray, n_samples: int, radius: float
    ) -> np.ndarray:
        """Sample uniformly in angle within a geodesic cap on S^(d-1)."""
        d = center.shape[1]
        center_flat = center.flatten()

        n_local = int(n_samples * (1 - self.global_fraction))
        n_global = n_samples - n_local

        parts = []
        if n_local > 0:
            raw = np.random.randn(n_local, d)
            dots = raw @ center_flat
            tangent = raw - dots[:, None] * center_flat[None, :]
            tangent_norms = np.linalg.norm(tangent, axis=1, keepdims=True)
            tangent = tangent / np.maximum(tangent_norms, 1e-8)
            angles = np.random.rand(n_local, 1) * radius
            local = np.cos(angles) * center_flat[None, :] + np.sin(angles) * tangent
            parts.append(_normalize(local))
        if n_global > 0:
            global_pts = np.random.randn(n_global, d)
            parts.append(_normalize(global_pts))
        return np.concatenate(parts, axis=0)

    def next_candidate(self) -> np.ndarray:
        A = self._projections[self._active_idx]
        self._active_idx = (self._active_idx + 1) % self.K

        x_history, y_history = self.get_history_as_arrays()
        best_idx = np.nanargmax(y_history.flatten())
        x_best = x_history[best_idx : best_idx + 1]

        u_best = _normalize(x_best)
        v_best = _normalize(u_best @ A)

        v_candidates = self._sample_geodesic_cap(
            v_best, self.n_candidates, self.tr_radius
        )

        idx = np.random.randint(0, len(v_candidates))
        v_opt = v_candidates[idx : idx + 1]

        u_opt = _normalize(v_opt @ A.T)
        return (u_opt * self.mean_norm).reshape(1, -1)


# ---------------------------------------------------------------------------
# GP-guided spherical solver
# ---------------------------------------------------------------------------


class GPGuidedSphericalSolver(GPFreeSphericalSolver):
    """GP-guided subspace BO on S^(D-1) with ArcCosine kernel + Thompson Sampling.

    Same architecture as GPFreeSphericalSolver (K random QR subspaces,
    geodesic caps, spherical normalization) but uses a fitted GP to
    select the best candidate via Thompson Sampling instead of random.

    GP is refitted every iteration on all data projected to the active
    subspace. ArcCosine order-0 kernel (0 hyperparams).
    """

    def next_candidate(self) -> np.ndarray:
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from gpytorch.mlls import ExactMarginalLogLikelihood

        from src.spherevo.kernels import create_kernel

        A = self._projections[self._active_idx]
        self._active_idx = (self._active_idx + 1) % self.K

        x_hist, y_hist = self.get_history_as_arrays()
        best_idx = int(np.nanargmax(y_hist.flatten()))
        x_best = x_hist[best_idx : best_idx + 1]

        # Normalize all history to S^(D-1)
        u_all = _normalize(x_hist)
        u_best = _normalize(x_best)

        # Project to S^(d-1)
        v_all = _normalize(u_all @ A)
        v_best = _normalize(u_best @ A)

        # Generate candidates in geodesic cap
        v_cand = self._sample_geodesic_cap(
            v_best, self.n_candidates, self.tr_radius
        )

        # Try GP-guided Thompson Sampling
        try:
            X = torch.tensor(v_all, dtype=torch.float64)
            Y = torch.tensor(y_hist.flatten(), dtype=torch.float64).unsqueeze(-1)
            # Standardize Y
            Y_std = (Y - Y.mean()) / (Y.std() + 1e-8)

            kernel = create_kernel("arccosine", kernel_order=0, use_scale=True)
            gp = SingleTaskGP(X, Y_std, covar_module=kernel)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()

            V_cand = torch.tensor(v_cand, dtype=torch.float64)
            with torch.no_grad():
                post = gp.posterior(V_cand)
                ts_sample = post.rsample().squeeze(0).squeeze(-1)
            best_cand_idx = int(ts_sample.argmax())
        except (RuntimeError, np.linalg.LinAlgError) as e:
            if isinstance(e, (torch.cuda.OutOfMemoryError, MemoryError)):
                raise
            self.gp_fallback_count = getattr(self, "gp_fallback_count", 0) + 1
            # Recorded for the result file (run_benchmark._surrogate_fallbacks).
            self.gp_fallback_reasons = [
                *getattr(self, "gp_fallback_reasons", []), f"{type(e).__name__}: {e}",
            ]
            logger.warning(
                f"GP-guided acquisition failed (fallback #{self.gp_fallback_count}): "
                f"{type(e).__name__}: {e}. Falling back to random selection."
            )
            best_cand_idx = np.random.randint(0, len(v_cand))

        v_opt = v_cand[best_cand_idx : best_cand_idx + 1]
        v_opt = _normalize(v_opt)
        u_opt = _normalize(v_opt @ A.T)
        return (u_opt * self.mean_norm).reshape(1, -1)


# ---------------------------------------------------------------------------
# Noise-GP ablation
# ---------------------------------------------------------------------------


class NoiseGPSphericalSolver(GPFreeSphericalSolver):
    """Noise-GP ablation: replaces GP posterior with random noise.

    Same architecture as GPFreeSphericalSolver but selects candidates
    via argmax of N(0,1) noise instead of uniform random. This tests
    whether the GP posterior carries any spatial information (it doesn't).
    """

    def next_candidate(self) -> np.ndarray:
        A = self._projections[self._active_idx]
        self._active_idx = (self._active_idx + 1) % self.K

        x_hist, y_hist = self.get_history_as_arrays()
        best_idx = int(np.nanargmax(y_hist.flatten()))
        x_best = x_hist[best_idx : best_idx + 1]

        u_best = _normalize(x_best)
        v_best = _normalize(u_best @ A)

        v_cand = self._sample_geodesic_cap(
            v_best, self.n_candidates, self.tr_radius
        )

        # KEY DIFFERENCE: random noise instead of GP posterior
        ts_sample = torch.randn(len(v_cand))
        best_cand_idx = int(ts_sample.argmax())

        v_opt = v_cand[best_cand_idx : best_cand_idx + 1]
        v_opt = _normalize(v_opt)
        u_opt = _normalize(v_opt @ A.T)
        return (u_opt * self.mean_norm).reshape(1, -1)


# ---------------------------------------------------------------------------
# ROSE — parameter-free (1+1) geodesic line search
# ---------------------------------------------------------------------------


class ROSESubspaceSolver(StepByStepSolver):
    """Parameter-free (1+1) geodesic line search on S^(d-1) in a random subspace.

    Exploits a key property of spherical geometry: parallel transport along
    geodesics is free in ambient coordinates. Given orthonormal basis
    {q_1,...,q_{d-1}} in T_p S^(d-1), after stepping along q_i via exp map,
    all other q_j (j!=i) remain tangent at the new point.

    This gives 25 center updates (vs CMA-ES's 5 with pop=5) and 0 learnable
    parameters (vs CMA-ES's ~120 covariance parameters).

    Parameters
    ----------
    subspace_dim : int
        Dimension of random QR subspace (d). Default 16.
    initial_sigma : float
        Initial geodesic step size in radians. Default 0.3 (~17 degrees).
    grow_factor : float
        Multiply sigma on success. Default 1.2.
    shrink_factor : float
        Multiply sigma on failure. Default 0.8.
    sigma_min : float
        Minimum step size. Default 0.01.
    sigma_max : float
        Maximum step size. Default 1.5.
    patience : int
        Regenerate tangent basis after this many stale evals. Default 10.
    try_both_signs : bool
        Try +q_i and -q_i before advancing to next direction. Default True.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        subspace_dim: int = 16,
        initial_sigma: float = 0.3,
        grow_factor: float = 1.2,
        shrink_factor: float = 0.8,
        sigma_min: float = 0.01,
        sigma_max: float = 1.5,
        patience: int = 10,
        try_both_signs: bool = True,
    ) -> None:
        super().__init__(black_box, x0, y0)

        x0 = self._x0
        y0 = self._y0
        self.D = x0.shape[1]
        self.d = min(subspace_dim, self.D)

        # Random QR subspace [D, d]
        rng = np.random.default_rng(np.random.randint(0, 2**31))
        A, _ = np.linalg.qr(rng.standard_normal((self.D, self.d)))
        self._A = A

        # Project best cold-start to S^(d-1)
        best_idx = int(np.argmax(y0.flatten()))
        proj = x0[best_idx] @ A
        norm = np.linalg.norm(proj)
        self._center = proj / max(norm, 1e-8)
        self._best_score_rose = float(y0.flatten()[best_idx])

        # Build initial tangent basis
        self._basis = self._random_tangent_basis(self._center, self.d, rng)

        # Step-size parameters
        self._sigma = initial_sigma
        self._grow = grow_factor
        self._shrink = shrink_factor
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

        # Search state
        self._dir_idx = 0
        self._sign = 1
        self._tried_positive = False
        self._try_both = try_both_signs
        self._patience = patience
        self._stale_count = 0
        self._rng = rng

        # Pending state for post_update
        self._pending_q: np.ndarray | None = None
        self._pending_sigma_signed = 0.0

    @staticmethod
    def _random_tangent_basis(
        p: np.ndarray, d: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Build orthonormal basis for T_p S^(d-1), shape [d-1, d]."""
        raw = rng.standard_normal((d, d))
        # Project out radial component (along p)
        raw -= np.outer(raw @ p, p)
        Q, _ = np.linalg.qr(raw.T)
        # First d-1 columns span the tangent space
        return Q[:, : d - 1].T  # [d-1, d]

    def next_candidate(self) -> np.ndarray:
        # Regenerate basis if stale or exhausted
        if self._stale_count >= self._patience or self._dir_idx >= self.d - 1:
            self._basis = self._random_tangent_basis(
                self._center, self.d, self._rng
            )
            self._dir_idx = 0
            self._sign = 1
            self._tried_positive = False
            if self._stale_count >= self._patience:
                self._stale_count = 0

        q = self._basis[self._dir_idx]

        # Geodesic step: exp_p(sigma * sign * q)
        sigma = self._sigma
        signed_q = self._sign * q
        p_cand = np.cos(sigma) * self._center + np.sin(sigma) * signed_q

        # Save for post_update
        self._pending_q = signed_q
        self._pending_sigma_signed = sigma

        return _lift_to_ambient(p_cand, self._A, self.mean_norm).reshape(1, -1)

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        score = float(y.flatten()[0])

        if score > self._best_score_rose:
            # SUCCESS: move center, parallel-transport, accelerate
            sigma = self._pending_sigma_signed
            signed_q = self._pending_q

            # New center on S^(d-1)
            p_new = np.cos(sigma) * self._center + np.sin(sigma) * signed_q
            p_new /= max(np.linalg.norm(p_new), 1e-8)

            # Parallel transport of the stepped direction
            q_transported = -np.sin(sigma) * self._center + np.cos(sigma) * signed_q
            q_transported /= max(np.linalg.norm(q_transported), 1e-8)

            # Update basis: replace current direction with transported version
            # (accounting for sign: store unsigned direction)
            self._basis[self._dir_idx] = self._sign * q_transported

            self._center = p_new
            self._best_score_rose = score
            self._sigma = min(self._sigma * self._grow, self._sigma_max)
            self._stale_count = 0
            # Don't advance dir_idx -- momentum surfing along same direction
            self._tried_positive = False
            self._sign = 1
        else:
            # FAILURE: shrink sigma, try other sign or advance direction
            self._sigma = max(self._sigma * self._shrink, self._sigma_min)
            self._stale_count += 1

            if self._try_both and not self._tried_positive:
                # First attempt was +q_i, now try -q_i
                self._tried_positive = True
                self._sign = -1
            else:
                # Advance to next basis direction
                self._dir_idx += 1
                self._sign = 1
                self._tried_positive = False


# ---------------------------------------------------------------------------
# PACE — Projected Ambient Covariance Evolution
# ---------------------------------------------------------------------------


class PACESubspaceSolver(StepByStepSolver):
    """Projected Ambient Covariance Evolution on S^(d-1) in a random subspace.

    Maintains a covariance matrix in the ambient R^d space and projects samples
    to the tangent space of the current center on S^(d-1). An 'ambient shield'
    (rank-1 correction along the radial direction) prevents covariance collapse
    from repeated tangent projection.

    Uses aggressive learning rates (c_1=0.1, c_mu=0.2) tuned for a 5-generation
    budget, extracting cross-directional signal that sequential methods like
    ROSE cannot capture.

    Parameters
    ----------
    subspace_dim : int
        Dimension of random QR subspace (d). Default 16.
    pop_size : int
        Population size per generation. Default 5.
    initial_sigma : float
        Initial step size in radians. Default 0.2.
    c_c : float
        Cumulation rate for evolution path. Default 0.3.
    c_1 : float
        Rank-1 update rate. Default 0.1.
    c_mu : float
        Rank-mu update rate. Default 0.2.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        subspace_dim: int = 16,
        pop_size: int = 5,
        initial_sigma: float = 0.2,
        c_c: float = 0.3,
        c_1: float = 0.1,
        c_mu: float = 0.2,
    ) -> None:
        super().__init__(black_box, x0, y0)

        x0 = self._x0
        y0 = self._y0
        self.D = x0.shape[1]
        self.d = min(subspace_dim, self.D)
        self._pop_size = pop_size

        # Random QR subspace [D, d]
        A, _ = np.linalg.qr(np.random.randn(self.D, self.d))
        self._A = A

        # Project best cold-start to S^(d-1)
        best_idx = int(np.argmax(y0.flatten()))
        proj = x0[best_idx] @ A
        norm = np.linalg.norm(proj)
        self._center = proj / max(norm, 1e-8)  # [d], unit vector
        self._best_score_pace = float(y0.flatten()[best_idx])

        # Recombination weights (CMA-ES style, mu = top half)
        mu = pop_size // 2 + 1
        self._mu = mu
        raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self._weights = raw_w / raw_w.sum()
        self._mueff = 1.0 / np.sum(self._weights**2)

        # Learning rates
        self._c_c = c_c
        self._c_1 = c_1
        self._c_mu = c_mu
        self._c_decay = c_1 + c_mu

        # Evolution state in ambient R^d
        self._sigma = initial_sigma
        self._p_c = np.zeros(self.d)          # Evolution path
        self._C = np.eye(self.d)              # Ambient covariance

        # Deferred-tell buffer (1 eval per step)
        self._gen_candidates: list[np.ndarray] = []  # S^(d-1) points
        self._gen_tangent_steps: list[np.ndarray] = []  # tangent vectors
        self._gen_scores: list[float] = []
        self._gen_idx = 0  # index into current generation

    def _sample_generation(self) -> None:
        """Sample pop_size candidates from ambient covariance."""
        self._gen_candidates = []
        self._gen_tangent_steps = []
        self._gen_idx = 0

        # Cholesky of ambient covariance (with jitter for stability)
        C_sym = (self._C + self._C.T) / 2.0
        try:
            L = np.linalg.cholesky(C_sym + 1e-5 * np.eye(self.d))
        except np.linalg.LinAlgError:
            self.cholesky_resets = getattr(self, "cholesky_resets", 0) + 1
            logger.warning(
                f"PACE: Cholesky failed (reset #{self.cholesky_resets}), "
                f"resetting covariance to identity"
            )
            self._C = np.eye(self.d)
            L = np.eye(self.d)

        for _ in range(self._pop_size):
            z = np.random.randn(self.d)
            u = L @ z

            # Project to tangent space of center (remove radial component)
            v = u - np.dot(u, self._center) * self._center
            self._gen_tangent_steps.append(v)

            # Exp map: p_new = cos(sigma*||v||) * center + sin(sigma*||v||) * v/||v||
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-8:
                theta = self._sigma * v_norm
                cand = np.cos(theta) * self._center + np.sin(theta) * (v / v_norm)
            else:
                cand = self._center.copy()

            cand /= max(np.linalg.norm(cand), 1e-8)
            self._gen_candidates.append(cand)

    def next_candidate(self) -> np.ndarray:
        # Sample new generation if needed
        if self._gen_idx >= len(self._gen_candidates) or not self._gen_candidates:
            # First, do PACE update if we have a complete generation
            if self._gen_scores:
                self._pace_update()
            self._sample_generation()

        cand = self._gen_candidates[self._gen_idx]
        self._gen_idx += 1
        return _lift_to_ambient(cand, self._A, self.mean_norm).reshape(1, -1)

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        score = float(y.flatten()[0])
        self._gen_scores.append(score)

        # Track global best
        if score > self._best_score_pace:
            self._best_score_pace = score

    def _pace_update(self) -> None:
        """Perform PACE covariance + center update from collected generation."""
        scores = np.array(self._gen_scores)
        self._gen_scores = []

        n = len(scores)
        if n == 0:
            return

        # Sort by score (descending)
        sort_idx = np.argsort(scores)[::-1]
        mu = min(self._mu, n)

        # Weighted mean tangent step (cross-directional signal)
        v_mean = np.zeros(self.d)
        for i in range(mu):
            idx = sort_idx[i]
            if idx < len(self._gen_tangent_steps):
                v_mean += self._weights[i] * self._gen_tangent_steps[idx]

        # --- Evolution path update ---
        self._p_c = (
            (1 - self._c_c) * self._p_c
            + np.sqrt(self._c_c * (2 - self._c_c) * self._mueff) * v_mean
        )

        # --- Rank-mu update (cross-directional covariance from batch) ---
        rank_mu = np.zeros((self.d, self.d))
        for i in range(mu):
            idx = sort_idx[i]
            if idx < len(self._gen_tangent_steps):
                vi = self._gen_tangent_steps[idx]
                rank_mu += self._weights[i] * np.outer(vi, vi)

        # --- Ambient Shield: restore lost radial variance ---
        ambient_shield = np.outer(self._center, self._center)

        # --- Covariance update ---
        self._C = (
            (1 - self._c_decay) * self._C
            + self._c_1 * np.outer(self._p_c, self._p_c)
            + self._c_mu * rank_mu
            + self._c_decay * ambient_shield
        )

        # --- Center update via weighted mean step ---
        v_mean_norm = np.linalg.norm(v_mean)
        if v_mean_norm > 1e-8:
            theta = self._sigma * v_mean_norm
            p_new = (
                np.cos(theta) * self._center
                + np.sin(theta) * (v_mean / v_mean_norm)
            )
            self._center = p_new / max(np.linalg.norm(p_new), 1e-8)

        # --- Step-size adaptation (fast momentum) ---
        batch_best = max(scores)
        if batch_best > self._best_score_pace - 1e-10:
            self._sigma = min(self._sigma * 1.3, 1.0)
        else:
            self._sigma = max(self._sigma * 0.8, 0.01)


# ---------------------------------------------------------------------------
# Jacobian SVD subspace solver
# ---------------------------------------------------------------------------


class JacobianSubspaceSolver(CMAESSubspaceSolver):
    """CMA-ES in decoder-Jacobian-informed subspace.

    Instead of random QR projection, computes the decoder Jacobian at the
    best cold-start point and uses its top-d right singular vectors as the
    projection basis.  This focuses CMA-ES on latent directions where the
    decoder output changes most — exploiting FREE structural information
    from the pretrained VAE.

    Requires a ``codec`` argument (SELFIESVAECodec) for Jacobian computation.
    Falls back to random QR if Jacobian computation fails.

    Parameters
    ----------
    codec : SELFIESVAECodec
        Pretrained VAE codec for Jacobian computation.
    n_probes : int
        Number of random probes for randomized SVD (default: 64).
    max_decode_len : int
        Max token positions for teacher-forced decode (default: 40).
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        subspace_dim: int = 16,
        population_size: int = 5,
        initial_sigma: float = 0.5,
        codec=None,
        n_probes: int = 64,
        max_decode_len: int = 40,
    ) -> None:
        # Compute Jacobian SVD subspace BEFORE calling super().__init__
        # (which uses self._A for cold-start projection)
        x0 = np.asarray(x0, dtype=np.float64)
        y0 = np.asarray(y0, dtype=np.float64).reshape(-1, 1)

        D = x0.shape[1]
        d = min(subspace_dim, D)

        best_idx = int(np.argmax(y0.flatten()))
        z_best = x0[best_idx]

        A = self._compute_jacobian_subspace(
            codec, z_best, D, d, n_probes, max_decode_len,
        )

        if A is None:
            logger.warning(
                "Jacobian SVD failed, falling back to random QR"
            )
            A, _ = np.linalg.qr(np.random.randn(D, d))

        # Store A for use after super().__init__; the parent creates its own
        # random QR matrix, which we override below along with re-initializing
        # CMA-ES to use the correct (Jacobian-informed) projection.
        self._precomputed_A = A

        super().__init__(
            black_box, x0, y0,
            subspace_dim=subspace_dim,
            population_size=population_size,
            initial_sigma=initial_sigma,
        )

        # Override parent's random A with our Jacobian A
        self._A = self._precomputed_A

        # Re-project cold-start with correct A and re-tell CMA-ES
        cs_proj = self._project_to_cma_space(self._x0)
        cs_scores = self._y0.flatten().copy()
        initial_mean = cs_proj[best_idx]

        if len(cs_proj) >= population_size:
            top_idx = np.argsort(cs_scores)[-population_size:]
            init_x = cs_proj[top_idx]
            init_f = (-cs_scores[top_idx]).tolist()
        else:
            init_x = cs_proj
            init_f = (-cs_scores).tolist()
            n_extra = population_size - len(cs_proj)
            extra = initial_mean + np.random.randn(n_extra, d) * 0.1
            init_x = np.vstack([init_x, extra])
            init_f = init_f + [float(-np.min(cs_scores))] * n_extra

        # Reinit CMA-ES with correct projection
        cma_seed = int(np.random.randint(0, 2**31))
        self._cma = cma.CMAEvolutionStrategy(
            initial_mean.flatten().tolist(),
            initial_sigma,
            {"popsize": population_size, "verbose": -9, "seed": cma_seed},
        )
        _ = self._cma.ask()
        self._cma.tell(init_x.tolist(), init_f)

    @staticmethod
    def _compute_jacobian_subspace(
        codec, z_best, D, d, n_probes, max_decode_len,
    ):
        """Compute top-d right singular vectors of decoder Jacobian at z_best.

        Uses randomized SVD via Halko et al. (2011):
        1. Sample random probe vectors v_i
        2. Compute J^T v_i via backward pass (one backprop per probe)
        3. QR of collected J^T v matrix → top-d directions

        Returns A : [D, d] orthonormal or None on failure.
        """
        import torch

        if codec is None:
            return None

        model = codec.model
        device = next(model.parameters()).device
        model.eval()

        orig_max_len = model.max_string_length
        try:
            model.max_string_length = 80

            z_t = torch.tensor(
                z_best.reshape(1, D), dtype=torch.float32, device=device
            ).requires_grad_(True)
            # Reshape for the SELFIES VAE (2 bottleneck tokens × 128 d_model)
            z_3d = z_t.reshape(1, 2, D // 2)

            # Greedy decode for tokens (no grad)
            with torch.no_grad():
                tokens = model.sample(z=z_3d.detach(), deterministic=True)

            # Teacher-forced decode WITH grad
            # model.decode drops last token internally and prepends zero,
            # so passing seq_len+1 tokens yields seq_len+1 logit positions.
            seq_len = min(tokens.shape[1], max_decode_len)
            tokens_crop = tokens[:, :seq_len + 1].detach()
            logits = model.decode(z_3d, tokens_crop)  # [1, seq_len+1, vocab]
            logits_flat = logits.reshape(-1)  # [(seq_len+1) * vocab]

            # Randomized SVD: collect J^T v for random v's
            n_out = logits_flat.shape[0]
            JtV = torch.zeros(D, n_probes, device=device)

            rng = torch.Generator(device=device)
            rng.manual_seed(42)

            for j in range(n_probes):
                v = torch.randn(n_out, device=device, generator=rng)
                logits_flat.backward(v, retain_graph=True)
                JtV[:, j] = z_t.grad.reshape(D)
                z_t.grad.zero_()

            # QR of JtV → orthonormal basis for range(J^T)
            Q, R = torch.linalg.qr(JtV.cpu().float())
            A = Q[:, :d].numpy()

            return A.astype(np.float64)

        except (torch.cuda.OutOfMemoryError, MemoryError):
            raise  # Never silently swallow OOM
        except (RuntimeError, np.linalg.LinAlgError) as e:
            logger.error(
                f"Jacobian SVD failed ({type(e).__name__}: {e}), "
                f"falling back to random QR"
            )
            return None
        finally:
            model.max_string_length = orig_max_len


# ---------------------------------------------------------------------------
# REMBO: Random EMbedding Bayesian Optimization (Wang et al. 2016)
# ---------------------------------------------------------------------------


class REMBOSolver(StepByStepSolver):
    """REMBO: random Gaussian projection R^D → R^d + GP-BO in R^d.

    Wang et al. "Bayesian Optimization in a Billion Dimensions via
    Random Embeddings" (JAIR 2016). Uses a dense random Gaussian matrix
    for dimensionality reduction, then fits SingleTaskGP + Thompson
    Sampling in the low-dimensional subspace.

    Note: uses Thompson Sampling instead of the original paper's EI
    for simplicity; at N=30 both are expected to perform comparably
    given the GP provides no signal.

    Parameters
    ----------
    subspace_dim : int
        Target dimensionality d.
    bounds_factor : float
        Search bounds in low-d space: [-bounds_factor*√d, bounds_factor*√d].
    n_candidates : int
        Number of Thompson Sampling candidates per step.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        subspace_dim: int = 16,
        bounds_factor: float = 1.0,
        n_candidates: int = 100,
    ) -> None:
        super().__init__(black_box, x0, y0)
        self.D = x0.shape[1]
        self.d = min(subspace_dim, self.D)
        self.n_candidates = n_candidates

        # Dense Gaussian projection with 1/√d normalization
        self._A = np.random.randn(self.D, self.d) / np.sqrt(self.d)

        # Bounds in low-d space: [-b, b]^d where b = bounds_factor * √d
        self._bound = bounds_factor * np.sqrt(self.d)

    def _lift(self, v: np.ndarray) -> np.ndarray:
        """Lift from R^d to R^D via transpose A^T, scaled to mean_norm."""
        z = v @ self._A.T  # (1, D)
        norm = np.linalg.norm(z)
        if norm > 1e-8:
            z = z / norm * self.mean_norm
        return z

    def next_candidate(self) -> np.ndarray:
        import torch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from gpytorch.mlls import ExactMarginalLogLikelihood

        x_hist, y_hist = self.get_history_as_arrays()
        X_low = x_hist @ self._A  # (N, d)

        # Filter NaN scores (from oracle timeouts / decode failures)
        valid = ~np.isnan(y_hist.flatten())
        X_low = X_low[valid]
        y_valid = y_hist[valid]

        # Generate random candidates in bounded subspace
        V_cand = (
            torch.rand(self.n_candidates, self.d, dtype=torch.float64) * 2 - 1
        ) * self._bound

        # Need ≥2 valid points for GP fitting (std requires ≥2)
        if valid.sum() < 2:
            best_idx = np.random.randint(0, self.n_candidates)
            v_opt = V_cand[best_idx].numpy().reshape(1, -1)
            return self._lift(v_opt).reshape(1, -1)

        X = torch.tensor(X_low, dtype=torch.float64)
        Y = torch.tensor(y_valid.flatten(), dtype=torch.float64).unsqueeze(-1)
        Y_std = (Y - Y.mean()) / (Y.std() + 1e-8)

        try:
            gp = SingleTaskGP(X, Y_std)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()

            with torch.no_grad():
                post = gp.posterior(V_cand)
                ts_sample = post.rsample().squeeze(0).squeeze(-1)
            best_idx = int(ts_sample.argmax())
        except (RuntimeError, np.linalg.LinAlgError) as e:
            if isinstance(e, (torch.cuda.OutOfMemoryError, MemoryError)):
                raise
            self.gp_fallback_count = getattr(self, "gp_fallback_count", 0) + 1
            # Recorded for the result file (run_benchmark._surrogate_fallbacks).
            self.gp_fallback_reasons = [
                *getattr(self, "gp_fallback_reasons", []), f"{type(e).__name__}: {e}",
            ]
            logger.warning(
                f"REMBO GP failed (fallback #{self.gp_fallback_count}): "
                f"{type(e).__name__}: {e}"
            )
            best_idx = np.random.randint(0, self.n_candidates)

        v_opt = V_cand[best_idx].numpy().reshape(1, -1)
        return self._lift(v_opt).reshape(1, -1)


# ---------------------------------------------------------------------------
# HeSBO: Hashing-Enhanced Subspace Bayesian Optimization (Nayebi et al. 2019)
# ---------------------------------------------------------------------------


class HeSBOSolver(StepByStepSolver):
    """HeSBO: count-sketch random hashing R^D → R^d + GP-BO in R^d.

    Nayebi et al. "A Framework for Bayesian Optimization in Embedded
    Subspaces" (ICML 2019). Each ambient dimension is hashed to exactly
    one subspace dimension with a random ±1 sign. This is the same
    sparse embedding later adopted by BAxUS (Papenmeier et al. 2022).

    Note: uses Thompson Sampling instead of the original paper's EI.

    Parameters
    ----------
    subspace_dim : int
        Target dimensionality d.
    n_candidates : int
        Number of Thompson Sampling candidates per step.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        subspace_dim: int = 16,
        n_candidates: int = 100,
    ) -> None:
        super().__init__(black_box, x0, y0)
        self.D = x0.shape[1]
        self.d = min(subspace_dim, self.D)
        self.n_candidates = n_candidates

        # Count-sketch embedding: each of D ambient dims hashed to one of d dims
        self._hash_indices = np.random.randint(0, self.d, size=self.D)
        self._hash_signs = np.random.choice([-1.0, 1.0], size=self.D)

        # Build sparse projection matrix S ∈ R^{d×D}
        self._S = np.zeros((self.d, self.D))
        for i in range(self.D):
            self._S[self._hash_indices[i], i] = self._hash_signs[i]

        # Project cold-start for search bounds estimation
        X_low = x0 @ self._S.T  # (N_cold, d)
        proj_range = np.abs(X_low).max()
        self._bound = max(proj_range * 2.0, 3.0)

    def _lift(self, v: np.ndarray) -> np.ndarray:
        """Lift from R^d to R^D via S (d×D matrix), scaled to mean_norm."""
        z = v @ self._S  # (1, D)
        norm = np.linalg.norm(z)
        if norm > 1e-8:
            z = z / norm * self.mean_norm
        return z

    def next_candidate(self) -> np.ndarray:
        import torch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from gpytorch.mlls import ExactMarginalLogLikelihood

        x_hist, y_hist = self.get_history_as_arrays()
        X_low = x_hist @ self._S.T  # (N, d)

        # Filter NaN scores (from oracle timeouts / decode failures)
        valid = ~np.isnan(y_hist.flatten())
        X_low = X_low[valid]
        y_valid = y_hist[valid]

        # Generate random candidates in bounded subspace
        V_cand = (
            torch.rand(self.n_candidates, self.d, dtype=torch.float64) * 2 - 1
        ) * self._bound

        # Need ≥2 valid points for GP fitting (std requires ≥2)
        if valid.sum() < 2:
            best_idx = np.random.randint(0, self.n_candidates)
            v_opt = V_cand[best_idx].numpy().reshape(1, -1)
            return self._lift(v_opt).reshape(1, -1)

        X = torch.tensor(X_low, dtype=torch.float64)
        Y = torch.tensor(y_valid.flatten(), dtype=torch.float64).unsqueeze(-1)
        Y_std = (Y - Y.mean()) / (Y.std() + 1e-8)

        try:
            gp = SingleTaskGP(X, Y_std)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()

            with torch.no_grad():
                post = gp.posterior(V_cand)
                ts_sample = post.rsample().squeeze(0).squeeze(-1)
            best_idx = int(ts_sample.argmax())
        except (RuntimeError, np.linalg.LinAlgError) as e:
            if isinstance(e, (torch.cuda.OutOfMemoryError, MemoryError)):
                raise
            self.gp_fallback_count = getattr(self, "gp_fallback_count", 0) + 1
            # Recorded for the result file (run_benchmark._surrogate_fallbacks).
            self.gp_fallback_reasons = [
                *getattr(self, "gp_fallback_reasons", []), f"{type(e).__name__}: {e}",
            ]
            logger.warning(
                f"HeSBO GP failed (fallback #{self.gp_fallback_count}): "
                f"{type(e).__name__}: {e}"
            )
            best_idx = np.random.randint(0, self.n_candidates)

        v_opt = V_cand[best_idx].numpy().reshape(1, -1)
        return self._lift(v_opt).reshape(1, -1)


# ---------------------------------------------------------------------------
# Random search baseline
# ---------------------------------------------------------------------------


class RandomSearchSolver(StepByStepSolver):
    """Pure random search baseline: sample random Gaussian vectors in R^D.

    Each candidate is a random Gaussian vector scaled to match the mean norm
    of the cold-start points. Provides a lower bound for all methods.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
    ) -> None:
        super().__init__(black_box, x0, y0)

    def next_candidate(self) -> np.ndarray:
        z = np.random.randn(1, self._dim)
        z = z / np.linalg.norm(z) * self.mean_norm
        return z


class RandomSearchPriorSolver(StepByStepSolver):
    """Random search drawing candidates directly from the VAE prior N(0, I_D).

    Sanity baseline complementing ``RandomSearchSolver``: candidates are i.i.d.
    samples from the standard normal prior with no normalization to the
    empirical sphere. Tests whether the typical-prior-norm shell (~ sqrt(D))
    coincides with the empirical encoded latent shell at radius mean_norm.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
    ) -> None:
        super().__init__(black_box, x0, y0)

    def next_candidate(self) -> np.ndarray:
        return np.random.randn(1, self._dim)


# The two definitions below are ported from p1-blocks commit 4780475
# (src/spherevo/solver.py, analysis-plan A10 block S2(b)). The code is
# unchanged; the docstrings drop the references to that branch's step-logging
# (E0) instrumentation, which this branch does not have.


def load_faithful_box_bounds(
    bounds_path: str | None, dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the per-coordinate ``[p01, p99]`` faithful-box bounds.

    Mirrors the loader in ``gp_competitors._BoxFaithfulMixin.__init__`` line
    for line (default path from ``src.analysis.config``, relative paths
    resolved against the repository root, ``FileNotFoundError`` when absent,
    ``ValueError`` on a dimension mismatch) so that the surrogate-free box
    control samples from *exactly* the box the box GP arms are fitted on. It
    is duplicated rather than imported so ``solver.py`` does not depend on
    the 3,000-line GP module; a change to the mixin's loader is a known drift
    point and must be mirrored here.

    Returns the raw ``(p01, p99)`` arrays; the ``1e-6`` dead-dimension floor is
    applied by the caller (as the mixin does on ``width``).
    """
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
    if p01.shape[0] != dim or p99.shape[0] != dim:
        raise ValueError(
            f"Bounds file dim {p01.shape[0]} != solver dim {dim}"
        )
    return p01, p99


class RandomBoxFaithfulSolver(StepByStepSolver):
    """Uniform-in-faithful-box random search (``random_box_faithful``).

    The surrogate-free *box* support control, paired with ``RandomSearchSolver``
    (uniform on the shell at r-bar). Every proposal is drawn uniformly per
    coordinate from the published-domain box ``[p01, p99]`` of
    ``results/precomputed/zinc_per_coord_bounds.json`` -- the same bounds file,
    the same loader semantics and the same ``1e-6`` dead-dimension floor on the
    width that ``gp_competitors._BoxFaithfulMixin`` uses for the box GP arms.
    Concretely ``z = p01 + u * max(p99 - p01, 1e-6)`` with ``u ~ U[0,1)^D``,
    which is *literally the box GP arms' random-fallback proposal*
    (``_rescale_from_unit_cube(torch.rand(1, D))``) taken on every step, so the
    difference between this arm and ``hvarfner_scaled_gp_box_argmax`` is the
    surrogate + argmax alone, and the difference to ``random_search`` is the
    support alone.

    No GP is fitted and no acquisition exists, so no ``_fallback_count`` is
    defined (the result file records ``surrogate_fallback_count = 0``, as for
    ``random_search``). Consumes NumPy's global RNG only (parity with
    ``RandomSearchSolver``). Ported unchanged from p1-blocks commit 4780475,
    where it was run on the batch runner (``results/guacamol/p1_blocks/``).

    Parameters
    ----------
    bounds_path : str, optional
        Per-codec bounds JSON; ``None`` -> ``default_bounds_path`` -> the
        SELFIES ZINC bounds from ``src.analysis.config``.
    """

    #: Overridden per codec by run_benchmark.py via the ``bounds_path`` kwarg.
    default_bounds_path: str | None = None

    #: Dead-dimension floor on the box width (identical to _BoxFaithfulMixin).
    width_floor: float = 1e-6

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        bounds_path: str | None = None,
    ) -> None:
        super().__init__(black_box, x0, y0)
        if bounds_path is None:
            bounds_path = self.default_bounds_path
        p01, p99 = load_faithful_box_bounds(bounds_path, self._dim)
        width = p99 - p01
        width = np.where(width > self.width_floor, width, self.width_floor)
        self._lo = p01
        self._hi = p99
        self._width = width

    def next_candidate(self) -> np.ndarray:
        u = np.random.rand(1, self._dim)
        return self._lo + u * self._width


class RandomBoxDirSolver(RandomBoxFaithfulSolver):
    """C2b ``random_boxdir``: [p01, p99]-box directions at the run's shell
    radius, no surrogate (control block of the camera-ready run rules,
    amendment of 2026-10-12; anisotropy control).

    Every proposal is ``random_box_faithful``'s draw ``z = p01 + u * width``
    (``u ~ U[0,1)^D``, the same bounds file, loader and 1e-6 width floor)
    renormalised to ``self.mean_norm``, the mean norm of the run's cold-start
    latents, i.e. the radius of ``random_search``:
    ``z / ||z|| * mean_norm``, in the operation order of
    ``RandomSearchSolver``. This is the candidate distribution of the GP arm
    ``hvarfner_scaled_gp_boxdir_argmax`` (C2a) with one draw per step and no
    surrogate, as ``random_box_faithful`` is to ``hvarfner_scaled_gp_box_argmax``
    and ``random_search`` to ``hvarfner_scaled_gp``. Under the same NumPy seed
    the proposal is exactly the renormalised ``random_box_faithful`` proposal.

    Random numbers: NumPy's global generator only, one ``np.random.rand(1, D)``
    per step (as ``random_box_faithful``); torch's generators are not used. No
    GP is fitted, so no fallback counter exists (the result file records
    ``surrogate_fallback_count = 0``, as for ``random_search``). A zero-norm
    draw has probability zero and is not special-cased.
    """

    def next_candidate(self) -> np.ndarray:
        z = super().next_candidate()
        return z / np.linalg.norm(z) * self.mean_norm


# ---------------------------------------------------------------------------
# GeoWalk: geodesic random walk from current best
# ---------------------------------------------------------------------------


class GeoWalkSolver(StepByStepSolver):
    """Geodesic random walk on the unit sphere with decaying angular step.

    Parameterized by an exploration ratio ``explore_ratio`` ∈ (0, 1) and
    angular endpoints ``angle_init``, ``angle_final``. From a budget of
    ``total_budget`` iterations, ``N1 = floor(explore_ratio * total_budget)``
    are spent on i.i.d. uniform sphere exploration and the rest on greedy
    geodesic refinement with a linearly decaying angular step.

    Defaults: ``explore_ratio=0.64``, ``angle_init=0.60``, ``angle_final=0.15``
    (the defaults of the method; the paper reports their sensitivity).
    At ``total_budget=25`` (the N=30 protocol with 5 cold-start): N1=16
    explore, N2=9 refine.

    No surrogate, no covariance, no subspace projection.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        explore_ratio: float = 0.64,
        explore_steps: int | None = None,
        angle_init: float = 0.60,
        angle_final: float = 0.15,
        total_budget: int | None = None,
        direction_accumulation: bool = False,
        srda_cc: float = 0.30,
        srda_lambda: float = 0.50,
        radius_override: float | None = None,
        radius_scale: float = 1.0,
        shell_eps: float = 0.0,
        radius_estimator: str = "mean",
        radius_range: tuple | None = None,
    ) -> None:
        super().__init__(black_box, x0, y0)
        self.angle_init = angle_init
        self.angle_final = angle_final
        self.total_budget = total_budget
        # Radius / support-geometry robustness (reviewer Lhbx Q1/Q2/Q3, 6qvY W2).
        # Default reproduces the paper exactly (base radius = mean_norm, no shell).
        self.shell_eps = float(shell_eps)
        # radius_range=(lo,hi): draw each candidate radius uniformly in [lo,hi]
        # (thick empirical shell, e.g. the 5-95th percentile [6.81, 9.14]).
        self.radius_range = tuple(radius_range) if radius_range is not None else None
        if radius_override is not None:
            self._base_radius = float(radius_override)
        elif radius_estimator == "median":
            self._base_radius = float(
                np.median(np.linalg.norm(self._x0, axis=1))
            ) * radius_scale
        else:  # "mean" — identical to self.mean_norm
            self._base_radius = self.mean_norm * radius_scale
        # Compute explore_steps from ratio and budget.
        # explore_steps kwarg overrides explore_ratio for backward compat
        # (ablation variants pass explore_steps directly).
        if explore_steps is not None:
            self.explore_steps = explore_steps
        elif total_budget is not None:
            self.explore_steps = int(explore_ratio * total_budget)
        else:
            self.explore_steps = int(explore_ratio * 25)  # default N=30 protocol
        # SRDA: Shrinkage-Regularized Direction Accumulation
        self.direction_accumulation = direction_accumulation
        if direction_accumulation:
            self._srda_path = np.zeros(x0.shape[1])
            self._srda_path_norm = 0.0
            self._srda_best = float(np.max(y0))
            self._srda_cc = srda_cc
            self._srda_lambda = srda_lambda

    def _angle(self, refine_step: int) -> float:
        """Linearly interpolate the geodesic step angle across the refinement phase."""
        if self.total_budget is not None:
            denom = max(self.total_budget - self.explore_steps - 1, 1)
        else:
            denom = max(25 - self.explore_steps - 1, 1)
        t = min(refine_step / denom, 1.0)
        return self.angle_init - (self.angle_init - self.angle_final) * t

    def _sample_radius(self) -> float:
        """Effective candidate radius, optionally jittered over a shell (annulus)."""
        if self.radius_range is not None:
            return float(np.random.uniform(self.radius_range[0], self.radius_range[1]))
        if self.shell_eps > 0.0:
            return self._base_radius * np.random.uniform(
                1.0 - self.shell_eps, 1.0 + self.shell_eps
            )
        return self._base_radius

    def next_candidate(self) -> np.ndarray:
        D = self._dim
        step = self.iteration

        # Phase 1: uniform random sphere samples
        if step < self.explore_steps:
            x = np.random.randn(D)
            x = x / (np.linalg.norm(x) + 1e-10) * self._sample_radius()
            return x.reshape(1, D)

        # Phase 2: greedy geodesic walk from current best.
        # self.history already contains cold-start + evaluated points; the
        # base class concatenates them on init (no extra concat needed).
        ys = np.concatenate(self.history["y"], axis=0).flatten()
        pts = np.concatenate(self.history["x"], axis=0)

        best_idx = int(np.argmax(ys))
        anc = pts[best_idx]
        anc_n = anc / (np.linalg.norm(anc) + 1e-10)

        # Update SRDA path on improvement (if enabled)
        if self.direction_accumulation and step > self.explore_steps:
            last_score = float(ys[-1])
            if last_score > self._srda_best:
                self._srda_best = last_score
                last_pt = pts[-1]
                last_n = last_pt / (np.linalg.norm(last_pt) + 1e-10)
                d_dir = last_n - np.dot(last_n, anc_n) * anc_n
                d_norm = np.linalg.norm(d_dir)
                if d_norm > 1e-10:
                    d_dir /= d_norm
                    cc = self._srda_cc
                    self._srda_path = (1 - cc) * self._srda_path + cc * d_dir
                    self._srda_path_norm = np.linalg.norm(self._srda_path)

        # Random tangent direction (orthogonalized against the anchor)
        v = np.random.randn(D)

        # SRDA: bias direction toward accumulated successful path
        if self.direction_accumulation and self._srda_path_norm > 0.05:
            p_hat = self._srda_path / self._srda_path_norm
            lam = self._srda_lambda
            bias = (1.0 - lam) / max(lam, 0.01)
            v = v + bias * np.dot(v, p_hat) * p_hat

        v -= np.dot(v, anc_n) * anc_n
        v /= (np.linalg.norm(v) + 1e-10)

        # Linearly decaying angle across the refinement phase
        refine_step = step - self.explore_steps
        angle = self._angle(refine_step)

        # Exponential map on the unit sphere, scaled to the effective radius
        x = np.cos(angle) * anc_n + np.sin(angle) * v
        x = x / (np.linalg.norm(x) + 1e-10) * self._sample_radius()
        return x.reshape(1, D)


class GeoWalkOneFifthSolver(GeoWalkSolver):
    """GeoWalk with a success-based angle rule in the refinement phase.

    The first refinement angle is 0.30 rad; after each refinement step it is
    multiplied by 1.2 if the step improved the incumbent and by 0.85 otherwise,
    then clamped to [0.05, 0.50] rad. Despite the registry name this is not a
    1/5 rule: the angle is stationary at a success rate of about 0.47 (O6).

    Registry name ``geowalk_one_fifth``: the GeoWalk variant of the
    adaptive-step sentence of the paper's Section 3.3. Everything is GeoWalk
    except the step angle of the refinement phase, so the variant differs
    from ``geowalk`` in the angle rule only:

    * exploration: GeoWalk's, unchanged -- ``explore_steps =
      int(0.64 * total_budget)`` (16 at the N = 30 protocol) i.i.d. uniform
      points on the shell of radius ``mean_norm``;
    * refinement: GeoWalk's greedy geodesic step from the current best (anchor,
      random tangent direction, exponential map, radius), with the angle set
      by the rule below instead of GeoWalk's linear schedule 0.60 -> 0.15.

    The rule, point by point:

    O1 **Start.** The first refinement step uses ``angle_start = 0.30`` rad.
       The angle changes only after a refinement step has been evaluated
       (O2), so the first refinement step uses 0.30 itself and the second
       uses 0.30 x 1.2 = 0.36 after a success or 0.30 x 0.85 = 0.255 after
       a failure. ``__init__`` checks
       ``0 < angle_min <= angle_start <= angle_max`` and raises
       ``ValueError`` otherwise.
    O2 **Update.** After every *refinement* step: ``angle *= 1.2`` if that step
       improved the incumbent, else ``angle *= 0.85``; then clamp to
       ``[0.05, 0.50]``. The factors and the clamp are the keyword arguments
       ``angle_up``, ``angle_down``, ``angle_min`` and ``angle_max`` (defaults
       1.2, 0.85, 0.05 and 0.50). Exploration steps never change the angle;
       the first update follows the first refinement step.
    O3 **"Improved"** means the evaluated score exceeds the best *finite*
       score seen before that step (cold start included) by more than
       ``1e-10`` (``improve_tol``). A NaN score (oracle timeout
       or decode failure) is never an improvement, so it contracts the angle.
    O4 **Anchor and geometry** are GeoWalk's: :meth:`next_candidate` is
       inherited and only :meth:`_angle` is overridden, so the anchor is the
       argmax over cold start + all evaluations (on exact ties the first
       maximum in that order, as ``np.argmax`` returns it) and the radius is
       GeoWalk's ``mean_norm``.
    O5 **RNG.** The angle is updated in :meth:`post_update`, after the
       evaluation, and the rule draws no random numbers, so the random stream
       is GeoWalk's: under the same seed, cold start and exploration phase are
       bit-identical to ``geowalk``.
    O6 **Naming.** The registry name refers to the 1/5th success rule of
       evolution strategies. With these multipliers the angle is stationary at
       a success rate of ln(1/0.85) / ln(1.2/0.85) ~= 0.47, not 1/5 (a
       1/5-balanced contraction for x1.2 would be 1.2**-0.25 ~= 0.955), so it
       is a success-rule adaptation of the 1/5 form with a much faster
       contraction. A printed description should give the constants.

    ``angle_trace`` holds the angle used at each refinement step, in order.
    The ``angle_init`` / ``angle_final`` arguments of :class:`GeoWalkSolver`
    are accepted but have no effect here.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        angle_start: float = 0.30,
        angle_up: float = 1.2,
        angle_down: float = 0.85,
        angle_min: float = 0.05,
        angle_max: float = 0.50,
        improve_tol: float = 1e-10,
        **geowalk_kwargs: Any,
    ) -> None:
        super().__init__(black_box, x0, y0, **geowalk_kwargs)
        if not 0.0 < angle_min <= angle_start <= angle_max:
            raise ValueError(
                f"need 0 < angle_min <= angle_start <= angle_max, got "
                f"{angle_min}, {angle_start}, {angle_max}"
            )
        self.angle_start = float(angle_start)
        self.angle_up = float(angle_up)
        self.angle_down = float(angle_down)
        self.angle_min = float(angle_min)
        self.angle_max = float(angle_max)
        self.improve_tol = float(improve_tol)
        self._adaptive_angle = self.angle_start
        self.angle_trace: list[float] = []

    def _angle(self, refine_step: int) -> float:
        """Current adaptive angle (choice O1/O2); ``refine_step`` is unused."""
        self.angle_trace.append(self._adaptive_angle)
        return self._adaptive_angle

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Adapt the angle after a refinement step (choices O2, O3, O5)."""
        super().post_update(x, y)
        # step() calls post_update before incrementing self.iteration, so
        # self.iteration is the index of the step that was just evaluated.
        if self.iteration < self.explore_steps:
            return
        score = float(np.asarray(y, dtype=np.float64).flatten()[0])
        before = np.asarray(self.all_scores[:-1], dtype=np.float64)
        before = before[np.isfinite(before)]
        best_before = float(before.max()) if before.size else -np.inf
        improved = bool(np.isfinite(score) and score > best_before + self.improve_tol)
        factor = self.angle_up if improved else self.angle_down
        self._adaptive_angle = float(
            np.clip(self._adaptive_angle * factor, self.angle_min, self.angle_max)
        )


# ---------------------------------------------------------------------------
# FlatGeoWalk: same algorithm as GeoWalk but in Euclidean R^D (no sphere)
# ---------------------------------------------------------------------------


class FlatGeoWalkSolver(StepByStepSolver):
    """GeoWalk ablation without spherical normalization.

    Phase 1: i.i.d. Gaussian samples (NOT projected to sphere).
    Phase 2: Euclidean step from anchor (NOT geodesic, NO re-normalization).
    Same explore_ratio and angular schedule as GeoWalk for fair comparison.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        explore_ratio: float = 0.64,
        total_budget: int | None = None,
        angle_init: float = 0.60,
        angle_final: float = 0.15,
    ) -> None:
        super().__init__(black_box, x0, y0)
        self.angle_init = angle_init
        self.angle_final = angle_final
        self.total_budget = total_budget
        budget = total_budget if total_budget is not None else 25
        self.explore_steps = int(explore_ratio * budget)

    def _step_size(self, refine_step: int) -> float:
        budget = self.total_budget if self.total_budget is not None else 25
        denom = max(budget - self.explore_steps - 1, 1)
        t = min(refine_step / denom, 1.0)
        return self.angle_init - (self.angle_init - self.angle_final) * t

    def next_candidate(self) -> np.ndarray:
        D = self._dim
        step = self.iteration

        # Phase 1: i.i.d. Gaussian scaled per-component (no sphere projection).
        # Each component ~ N(0, (mean_norm/sqrt(D))^2) so expected norm ≈ mean_norm,
        # but with natural Gaussian variance in norm (unlike sphere projection).
        if step < self.explore_steps:
            x = np.random.randn(D) * (self.mean_norm / np.sqrt(D))
            return x.reshape(1, D)

        # Phase 2: Euclidean step from best point (no geodesic, no renorm)
        ys = np.concatenate(self.history["y"], axis=0).flatten()
        pts = np.concatenate(self.history["x"], axis=0)
        best_idx = int(np.argmax(ys))
        anc = pts[best_idx]

        v = np.random.randn(D)
        v /= np.linalg.norm(v) + 1e-10

        refine_step = step - self.explore_steps
        step_size = self._step_size(refine_step) * self.mean_norm
        x = anc + step_size * v
        return x.reshape(1, D)


# ---------------------------------------------------------------------------
# GeoWalkNoAnchor: geodesic walk from FRESH RANDOM sphere point (no memory)
# ---------------------------------------------------------------------------


class GeoWalkNoAnchorSolver(StepByStepSolver):
    """GeoWalk ablation that removes the argmax-anchor memory.

    Phase 1 (iters 0..N1-1): uniform random sphere samples at mean_norm.
    Phase 2 (iters N1..): tangent walk at decaying angle, but the anchor is a
    FRESH uniform random sphere point each step (NOT argmax of history).

    Isolates the contribution of "memory" in GeoWalk: if no-anchor scores ≈
    full GeoWalk, argmax anchoring is not load-bearing. If no-anchor <<
    full, memory is essential.

    All other components (sphere scaffold, tangent walk, decaying schedule)
    are identical to GeoWalkSolver.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        explore_ratio: float = 0.64,
        angle_init: float = 0.60,
        angle_final: float = 0.15,
        total_budget: int | None = None,
    ) -> None:
        super().__init__(black_box, x0, y0)
        self.angle_init = angle_init
        self.angle_final = angle_final
        self.total_budget = total_budget
        budget = total_budget if total_budget is not None else 25
        self.explore_steps = int(explore_ratio * budget)

    def _angle(self, refine_step: int) -> float:
        budget = self.total_budget if self.total_budget is not None else 25
        denom = max(budget - self.explore_steps - 1, 1)
        t = min(refine_step / denom, 1.0)
        return self.angle_init - (self.angle_init - self.angle_final) * t

    def next_candidate(self) -> np.ndarray:
        D = self._dim
        step = self.iteration

        if step < self.explore_steps:
            x = np.random.randn(D)
            x = x / (np.linalg.norm(x) + 1e-10) * self.mean_norm
            return x.reshape(1, D)

        # Phase 2: fresh random anchor (no memory of past scores)
        anc_n = np.random.randn(D)
        anc_n = anc_n / (np.linalg.norm(anc_n) + 1e-10)

        v = np.random.randn(D)
        v -= np.dot(v, anc_n) * anc_n
        v /= (np.linalg.norm(v) + 1e-10)

        refine_step = step - self.explore_steps
        angle = self._angle(refine_step)

        x = np.cos(angle) * anc_n + np.sin(angle) * v
        x = x / (np.linalg.norm(x) + 1e-10) * self.mean_norm
        return x.reshape(1, D)


# ---------------------------------------------------------------------------
# GeoWalkNoTangent: uniform sphere sampling at mean_norm for ALL iterations
# ---------------------------------------------------------------------------


class GeoWalkNoTangentSolver(StepByStepSolver):
    """GeoWalk ablation that removes the geodesic tangent walk.

    All iterations: i.i.d. uniform samples on S^{D-1} at radius mean_norm.
    Equivalent to running Phase 1 of GeoWalk for the full budget.

    Isolates the contribution of tangent-walk structure: if no-tangent ≈
    full GeoWalk, geodesic walk is not load-bearing (sphere scaffold alone
    suffices). If no-tangent << full, tangent walk provides genuine refinement.

    This is stricter than "random_search": candidates are constrained to
    mean_norm (VAE training distribution), unlike random_search which samples
    from a box.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
    ) -> None:
        super().__init__(black_box, x0, y0)

    def next_candidate(self) -> np.ndarray:
        D = self._dim
        x = np.random.randn(D)
        x = x / (np.linalg.norm(x) + 1e-10) * self.mean_norm
        return x.reshape(1, D)


# ---------------------------------------------------------------------------
# Alternative-support baselines (reviewer Lhbx Q4): "beyond the hypersphere".
# The reviewer asks whether other supports — e.g. the convex hull of the
# observed latents — might work as well as the empirical shell. These two
# solvers sample candidates from (a) the convex hull of encoded ZINC training
# latents and (b) a Gaussian (Mahalanobis-ellipsoid) fit to them, so we can
# compare directly against on-sphere random search on the same 11 tasks.
# ---------------------------------------------------------------------------


def _load_zinc_latent_sample() -> np.ndarray:
    """Load the cached encoded-ZINC latent sample (project-root relative)."""
    from pathlib import Path

    from src.analysis.config import ZINC_LATENTS_SAMPLE_PATH

    rel = ZINC_LATENTS_SAMPLE_PATH
    for base in (Path.cwd(), *Path(__file__).resolve().parents):
        cand = base / rel
        if cand.exists():
            return np.load(cand)["Z"].astype(np.float64)
    raise FileNotFoundError(
        f"{rel} not found; run scripts/analysis/precompute_zinc_latents.py first."
    )


class ConvexHullRandomSearchSolver(StepByStepSolver):
    """Random search over the convex hull of encoded training latents.

    Each candidate is a convex combination ``sum_i w_i z_i`` of ``n_anchors``
    randomly chosen encoded ZINC latents, with ``w ~ Dirichlet(alpha)``. Convex
    combinations lie strictly inside the hull and (by concentration) at a radius
    below the empirical shell, so this tests the reviewer's alternative support.
    """

    _Z_CACHE = None

    def __init__(
        self, black_box, x0, y0, *, n_anchors: int = 3, alpha: float = 1.0,
    ) -> None:
        super().__init__(black_box, x0, y0)
        self.n_anchors = int(n_anchors)
        self.alpha = float(alpha)
        if ConvexHullRandomSearchSolver._Z_CACHE is None:
            ConvexHullRandomSearchSolver._Z_CACHE = _load_zinc_latent_sample()
        self._Z = ConvexHullRandomSearchSolver._Z_CACHE
        if self._Z.shape[1] != self._dim:
            raise ValueError(
                f"Cached latents dim {self._Z.shape[1]} != solver dim {self._dim} "
                f"(convex-hull baseline is SELFIES-256D only)."
            )

    def next_candidate(self) -> np.ndarray:
        idx = np.random.choice(len(self._Z), self.n_anchors, replace=False)
        w = np.random.dirichlet(np.full(self.n_anchors, self.alpha))
        x = (w[:, None] * self._Z[idx]).sum(axis=0)
        return x.reshape(1, self._dim)


class EllipsoidRandomSearchSolver(StepByStepSolver):
    """Random search from a Gaussian (Mahalanobis-ellipsoid) fit to the latents.

    Samples ``z ~ N(mean, Cov)`` where mean/Cov are estimated from the encoded
    ZINC sample. This is the "filled" interior alternative to the shell; note
    that by the Gaussian-annulus phenomenon such samples re-concentrate near the
    same radius as the data, which is itself informative for reviewer Lhbx W2.
    """

    _STATS_CACHE = None

    def __init__(self, black_box, x0, y0) -> None:
        super().__init__(black_box, x0, y0)
        if EllipsoidRandomSearchSolver._STATS_CACHE is None:
            Z = _load_zinc_latent_sample()
            mean = Z.mean(axis=0)
            cov = np.cov(Z, rowvar=False)
            # Symmetric eigendecomposition -> sampling factor L (Cov = L L^T).
            evals, evecs = np.linalg.eigh(cov)
            evals = np.clip(evals, 1e-12, None)
            L = evecs * np.sqrt(evals)
            EllipsoidRandomSearchSolver._STATS_CACHE = (mean, L)
        self._mean, self._L = EllipsoidRandomSearchSolver._STATS_CACHE
        if self._mean.shape[0] != self._dim:
            raise ValueError(
                f"Cached latents dim {self._mean.shape[0]} != solver dim {self._dim}."
            )

    def next_candidate(self) -> np.ndarray:
        eps = np.random.randn(self._dim)
        x = self._mean + self._L @ eps
        return x.reshape(1, self._dim)


class PCASubspaceSphereSolver(StepByStepSolver):
    """Uniform on a sphere of radius r-bar restricted to the top-K PCA subspace.

    Answers Lhbx W2 directly. The paper's scaffold samples uniformly on the full
    S^(D-1), which spreads energy over all D=256 coordinates even though only
    ~64 latent directions are active. Reviewer objection: that puts most of the
    norm on collapsed directions, so an isotropic sphere cannot be the right
    geometry.

    This solver samples a uniform direction *inside* the top-K principal subspace
    of the encoded training latents and scales it to the same empirical radius,
    so it concentrates 100% of the norm on active directions while holding the
    radius fixed. Comparing against the isotropic sphere isolates DIRECTION from
    NORM:
      - if it does not beat the isotropic sphere, the norm carries the effect
        (a strong positive result for the paper's framing);
      - if it does, it is simply a better method.

    Distinct from ``EllipsoidRandomSearchSolver``, which samples N(mean, Cov) and
    therefore weights directions by eigenvalue rather than spreading uniformly
    within the active subspace.
    """

    _BASIS_CACHE: dict = {}

    def __init__(self, black_box, x0, y0, *, n_components: int = 64) -> None:
        super().__init__(black_box, x0, y0)
        self._k = int(n_components)
        cached = PCASubspaceSphereSolver._BASIS_CACHE.get(self._k)
        if cached is None:
            Z = _load_zinc_latent_sample()
            Zc = Z - Z.mean(axis=0)
            cov = np.cov(Zc, rowvar=False)
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1][: self._k]
            cached = np.ascontiguousarray(evecs[:, order])  # [D, k]
            PCASubspaceSphereSolver._BASIS_CACHE[self._k] = cached
        self._basis = cached
        if self._basis.shape[0] != self._dim:
            raise ValueError(
                f"Cached latents dim {self._basis.shape[0]} != solver dim {self._dim}."
            )

    def next_candidate(self) -> np.ndarray:
        # Uniform direction on S^(k-1), lifted into the ambient space by the
        # orthonormal PCA basis, then scaled to the empirical shell radius.
        c = np.random.randn(self._k)
        c /= np.linalg.norm(c)
        x = self._basis @ c * self.mean_norm
        return x.reshape(1, self._dim)


class ProximitySelectionSolver(StepByStepSolver):
    """Select an on-shell candidate by proximity to the incumbent. No surrogate.

    This is the null model for what a fitted GP actually contributes at extreme
    low budget. Instrumenting the on-sphere Hvarfner recipe (an analysis of
    the rebuttal, not part of this release) shows that at N=30 in
    D=256 the fitted GP has no predictive power on its own observations
    (leave-one-out R^2 ~ 0), yet its acquisition value is strongly rank-ordered
    by *angular proximity to the incumbent*, so its argmax over a pool of
    ``n_candidates`` uniform on-shell points lands in the nearest ``q_select``
    fraction of that pool.

    This solver reproduces exactly that angular selection statistic and nothing
    else: it draws the same pool, ranks by angle to the incumbent, and picks
    uniformly among the nearest ``q_select`` fraction. It never fits a model and
    never uses an objective value except to identify the incumbent.

    If it matches the on-sphere GP, everything the surrogate buys at this budget
    is summarised by one number -- the selection quantile.

    ``mode="far"`` inverts the rule (pick among the *farthest* fraction) and is
    the falsification control: if proximity is what matters, this must be worse
    than uniform random search on the same shell.
    """

    def __init__(
        self,
        black_box,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        n_candidates: int = 2048,
        q_select: float = 0.06,
        mode: str = "near",
        explore_steps: int = 0,
    ) -> None:
        super().__init__(black_box, x0, y0)
        if not 0.0 < q_select <= 1.0:
            raise ValueError(f"q_select must be in (0, 1], got {q_select}")
        if mode not in ("near", "far"):
            raise ValueError(f"mode must be 'near' or 'far', got {mode!r}")
        self.n_candidates = int(n_candidates)
        self.q_select = float(q_select)
        self.mode = mode
        self.explore_steps = int(explore_steps)

    def next_candidate(self) -> np.ndarray:
        D = self._dim
        u = np.random.randn(self.n_candidates, D)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        if self.iteration < self.explore_steps:
            return (u[0] * self.mean_norm).reshape(1, D)

        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=-1.0)
        x_star = x_hist[int(np.argmax(y_hist.flatten()))]
        u_star = x_star / np.linalg.norm(x_star)

        # Rank by angle to the incumbent; arccos is monotone in -cos, so ranking
        # by cosine descending is equivalent and avoids the transcendental call.
        cos = u @ u_star
        k = max(1, int(round(self.q_select * self.n_candidates)))
        # argpartition puts the k largest (mode="near") cosines in the last k slots.
        part = np.argpartition(cos, -k)[-k:] if self.mode == "near" else np.argpartition(cos, k)[:k]
        idx = int(np.random.choice(part))
        return (u[idx] * self.mean_norm).reshape(1, D)
