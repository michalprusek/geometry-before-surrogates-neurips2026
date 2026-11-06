"""TuRBO-1 on the FIXED [p01, p99] latent box (``turbo_box_faithful``).

Why this arm exists
-------------------
The submitted TuRBO row was produced by :mod:`src.spherevo.turbo_baseline`
(class ``TuRBOBaseline``), whose search domain is the *running bounding box of
the observed latents*::

    self._z_lb = self.train_Z.min(dim=0).values      # turbo_baseline.py:199
    self._z_ub = self.train_Z.max(dim=0).values      # turbo_baseline.py:200

Every call to ``_normalize_z`` recomputes that box from the data seen so far, so

1. the unit cube the trust region lives in **moves after every evaluation** — a
   trust region of "length 0.8" is a different subset of latent space at step 1
   and at step 25, and the TR centre coordinates change even when the incumbent
   does not;
2. the domain is defined by the arm's own proposals, so TuRBO is compared
   against the other box arms on a *different* (and, with 5 cold-start points,
   much smaller) search space; and
3. the mapping from the unit cube back to latent space is history-dependent,
   which makes the arm non-comparable with every other box arm in the paper.

This module keeps TuRBO-1 and replaces that frame by the same **fixed**
per-coordinate ``[p01, p99]`` box the other faithful box arms use
(``results/precomputed/zinc_per_coord_bounds.json`` via
:class:`src.spherevo.gp_competitors._BoxFaithfulMixin`). Everything else about
the harness — cold start, budget 5 + 25, result format, GP-fallback recording,
provenance — is the shared ``StepByStepSolver`` machinery, identical to
``hvarfner_scaled_gp_box_faithful``, ``papenmeier_msr_box_faithful``,
``xu_standard_gp_box_faithful`` and ``papenmeier_msr_published``.

How this row relates to the submitted TuRBO row
-----------------------------------------------
The GP is **the submitted wrapper's GP, unchanged** (camera-ready decision D2,
reviewer option (a)), and so are the trust-region rules, so that at the level of
the method this arm differs from the submitted TuRBO row in two things only:
the **fixed** ``[p01, p99]`` search box (the reason the arm exists) and
**duplicate accounting** (below). It is not "the tutorial GP" and not "the
paper's GP"; it is the mix the wrapper used, and is labelled as such.
Implementation details below the method also differ, because the submitted
row ran through a separate harness that no longer exists
(``docs/camera_ready/RUN_RULES.md`` section 3): this arm uses the shared
in-repo harness (ZINC cold start, decoder, oracle, budget accounting) and the
choices C3-C10 listed at the end of this docstring.

Sources: Eriksson, Pearce, Gardner, Turner & Poloczek, *Scalable Global
Optimization via Local Bayesian Optimization*, NeurIPS 2019 (arXiv:1910.01739;
"the paper"), the authors' code (github.com/uber-research/TuRBO,
``turbo/turbo_1.py``, ``turbo/gp.py``; "the original code") and the BoTorch
TuRBO-1 tutorial (``tutorials/turbo_1``, v0.17.2; "the tutorial").

* GP: ``SingleTaskGP`` with ``ScaleKernel(MaternKernel(nu=2.5, ard))`` and the
  constraints of ``turbo_baseline.py`` lines 232-240: lengthscale
  ``Interval(0.005, 4.0)`` and noise ``Interval(1e-8, 1e-3)`` are the
  tutorial's; outputscale ``Interval(0.05, 20.0)`` is the paper's (appendix
  "Gaussian process regression"), which the tutorial omits. The paper itself
  uses lengthscale ``[0.005, 2.0]`` and noise ``[5e-4, 0.1]`` (the original code
  ``[5e-4, 0.2]``), so this combination matches neither the paper nor the
  tutorial. The bounds bind in practice (cr/H_review_solvers.md section 4.4: up
  to 207/256 lengthscales at the 4.0 ceiling, noise at the 1e-8 floor on 20 of
  25 steps), so a different set would give a different GP.
* Trust region: ``length`` init 0.8, ``length_min`` ``0.5**7``, ``length_max``
  1.6 (paper, original code and tutorial agree). ``success_tolerance`` 3 as in
  the paper and the original code; the tutorial uses 10 (and notes that the
  original paper uses 3). This matters here because the success branch is the
  only trust-region mechanism that can fire at this budget (C7).
  ``failure_tolerance``: the paper's appendix ("TuRBO details") gives
  ``ceil(d / q)``; the original code and the tutorial use
  ``ceil(max(4 / q, d / q))``; with ``D = 256`` and batch size ``q = 1`` both
  give 256. Doubling on ``success_tolerance`` consecutive improvements,
  halving on ``failure_tolerance`` consecutive non-improvements, restart when
  ``length < length_min``. The rules are reused verbatim by importing
  :class:`src.spherevo.turbo_baseline.TurboState`.
* Candidate set: lengthscale-weighted TR clamped to the unit cube, scrambled
  Sobol inside it, per-coordinate perturbation probability
  ``min(1, 20 / D)`` with at least one perturbed coordinate per candidate;
  ``min(100 * D, 5000)`` candidates as in the paper's appendix and the
  original code (the tutorial uses ``min(5000, max(2000, 200 * D))``, also
  5000 at ``D = 256``); selection by Thompson sampling
  (``MaxPosteriorSampling(replacement=False)``, one sample).

Duplicate accounting differs from the submitted row
---------------------------------------------------
The submitted wrapper skipped a proposal whose decoded SMILES it had already
seen (``turbo_baseline.py`` lines 386-387: ``return None, smiles``): no oracle
call, no training row and no trust-region update, but the step still counted
towards the budget. This arm, like every in-repo arm, scores and counts every
proposal through the shared ``StepByStepSolver.step``: a repeated molecule is
re-scored (served by the oracle cache), enters the GP data and counts as a
trust-region failure. In the real rano s42 cell, 30 evaluations gave only 23
distinct scores. Wherever both TuRBO rows are printed, say so.

Choices that the published description does not fix (all recorded here)
-----------------------------------------------------------------------
C1 **TR centre projection.** The tutorial assumes every observation lies in
   ``[0, 1]^D``; here cold-start encodings routinely fall outside the fixed
   ``[p01, p99]`` box (``cr/F_msr_published.md`` §4.4), and an out-of-box centre
   would (a) survive in every unperturbed coordinate of every candidate and
   (b) in every coordinate whose violation exceeds the TR half-width
   ``w*L/2``, collapse ``clamp(centre ± w*L/2, 0, 1)`` to a face of the cube.
   The centre is therefore **clamped into the box**, exactly as the
   published-MSR arm projects its RAASP centres
   (``_propose_msr_published_on_unit_box``). Candidates are consequently
   always inside the fixed box, which is what the arm claims.
C2 **GP training inputs are NOT clamped** — the GP is fitted on the unit-cube
   image of the raw history, like every other ``_BoxFaithfulMixin`` arm. Only
   the acquisition domain is the box.
C3 **y standardisation** uses the repo's ``_standardize`` (mean/sd of the
   history, NaNs penalised at -1.0 by ``get_history_as_arrays``) plus
   BoTorch's default ``Standardize`` outcome transform, as in every other arm,
   rather than the wrapper's hand-written ``(y - mean) / std``. Thompson
   selection is invariant to a monotone affine map of y.
C4 **Sobol seeding** follows the tutorial: ``SobolEngine(D, scramble=True)``
   with no explicit seed, drawn from the torch global RNG that
   ``scripts/run_benchmark.py`` seeds per cell. The wrapper instead passed
   ``seed=self.seed + len(self.history)``; the harness does not hand solvers a
   seed, and the global stream is what makes every other arm reproducible.
C5 **Cholesky.** Fit and Thompson sampling run inside
   ``gpytorch.settings.max_cholesky_size(inf)``, as the tutorial does, so the
   5000-point posterior sample is drawn from a Cholesky factor instead of
   gpytorch's Lanczos path (default ``max_cholesky_size`` is 800). The
   submitted wrapper omitted this.
C6 **Random perturbed coordinate** for candidates whose mask is empty is drawn
   from ``randint(0, D)``; the tutorial writes ``randint(0, dim - 1)``, which
   can never pick the last coordinate. The submitted wrapper already used
   ``randint(0, D)``.
C7 **Failure/restart branches are unreachable at this budget.** With ``D = 256``
   and ``b = 1`` the published formula gives ``failure_tolerance = 256``, while
   a cell runs 25 steps, so the TR can only ever double (0.8 -> 1.6 after three
   consecutive improvements) and ``restart_triggered`` can never fire. The code
   implements both branches (and the tests exercise them through a lowered
   tolerance), but the paper should state that TuRBO's shrink/restart mechanics
   are inactive inside a 30-evaluation budget.
C8 **Restart semantics**, if it ever fires, resets the trust-region state
   (length 0.8, counters 0, ``best_value`` = current incumbent) and keeps the
   observation history, which is what ``TuRBOBaseline.optimize`` does
   (``turbo_baseline.py`` lines 411-418). Full TuRBO-1 would re-initialise with
   a fresh design; there is no budget for that here.
C9 **NaN evaluations** (oracle timeout / decode failure) count as a TR failure
   and enter the state as -1.0, the value ``get_history_as_arrays`` already uses
   for NaN; the wrapper instead dropped such steps.
C10 **Fallback** on a numerical GP/acquisition failure is a uniform point in the
   fixed box, recorded in ``_fallback_count`` / ``_fallback_reasons``, exactly
   like the other box arms (the wrapper retried with a wider noise constraint
   and then reused the previous GP; that path is not reproduced).
"""

from __future__ import annotations

import logging
import math

import gpytorch
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from src.spherevo.gp_competitors import (
    _BOX_FALLBACK_ERRORS,
    HvarfnerScaledGPSolver,
    ModelFittingError,
    _BoxFaithfulMixin,
    _standardize,
)

logger = logging.getLogger(__name__)


# --- TuRBO-1 settings (the source of each is in the module docstring) -------
TURBO_LENGTH_INIT = 0.8
TURBO_LENGTH_MIN = 0.5**7
TURBO_LENGTH_MAX = 1.6
TURBO_SUCCESS_TOLERANCE = 3          # paper and original code; the tutorial uses 10
TURBO_BATCH_SIZE = 1
TURBO_CANDIDATES_PER_DIM = 100       # min(100*D, 5000): paper appendix, original code
TURBO_MAX_CANDIDATES = 5000
TURBO_PERTURB_NUMERATOR = 20.0
# GP constraints: the SUBMITTED WRAPPER's set (turbo_baseline.py:232-240), kept
# unchanged on purpose (decision D2): lengthscale and noise bounds from the
# tutorial, outputscale bound from the paper. Neither the paper's set nor the
# tutorial's set.
TURBO_LENGTHSCALE_INTERVAL = (0.005, 4.0)
TURBO_OUTPUTSCALE_INTERVAL = (0.05, 20.0)
TURBO_NOISE_INTERVAL = (1e-8, 1e-3)

#: Value a NaN evaluation contributes to the trust-region state (choice C9);
#: the same number ``get_history_as_arrays(penalize_nans_with=...)`` uses.
TURBO_NAN_SCORE = -1.0


def turbo_n_candidates(dim: int) -> int:
    """``min(100 * D, 5000)``: the paper's appendix and the original code.

    The BoTorch tutorial uses ``min(5000, max(2000, 200 * D))``; both give 5000
    at ``D = 256``.
    """
    return int(min(TURBO_CANDIDATES_PER_DIM * dim, TURBO_MAX_CANDIDATES))


def turbo_failure_tolerance(dim: int, batch_size: int = TURBO_BATCH_SIZE) -> int:
    """``ceil(max(4 / b, D / b))``: the original code and the BoTorch tutorial.

    The paper's appendix ("TuRBO details") gives ``ceil(d / q)``; the two agree
    whenever ``D >= 4``, e.g. 256 at ``D = 256``, ``b = 1``.
    """
    return int(math.ceil(max(4.0 / batch_size, dim / batch_size)))


class TurboBoxFaithfulSolver(_BoxFaithfulMixin, HvarfnerScaledGPSolver):
    """TuRBO-1 whose trust region lives in the fixed ``[p01, p99]`` unit cube.

    ``HvarfnerScaledGPSolver`` is subclassed **only** for the harness plumbing
    every GP box arm shares (``__init__`` validating the cold start and setting
    ``n_candidates``, ``_device``, ``_fallback_count``, ``_fallback_reasons``).
    None of the Hvarfner GP recipe survives: :meth:`_make_gp` is overridden with
    the submitted wrapper's TuRBO GP, and :meth:`next_candidate` with the TuRBO step. The
    box, the rescaling and the ``__init__`` bounds loading come from
    ``_BoxFaithfulMixin``, so this arm's domain is bit-for-bit the domain of
    ``hvarfner_scaled_gp_box_faithful``.
    """

    def __init__(
        self,
        black_box,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        n_candidates: int | None = None,
        bounds_path: str | None = None,
        length_init: float = TURBO_LENGTH_INIT,
        length_min: float = TURBO_LENGTH_MIN,
        length_max: float = TURBO_LENGTH_MAX,
        success_tolerance: int = TURBO_SUCCESS_TOLERANCE,
        failure_tolerance: int | None = None,
    ) -> None:
        dim = int(np.asarray(x0).shape[1])
        if n_candidates is None:
            n_candidates = turbo_n_candidates(dim)
        super().__init__(
            black_box=black_box, x0=x0, y0=y0,
            n_candidates=n_candidates, bounds_path=bounds_path,
        )
        self._turbo_cfg = {
            "length": float(length_init),
            "length_min": float(length_min),
            "length_max": float(length_max),
            "success_tolerance": int(success_tolerance),
            "failure_tolerance": failure_tolerance,
        }
        self._tr_restarts = 0
        self.turbo_state = self._new_state()
        logger.info(
            "TurboBoxFaithful: D=%d n_candidates=%d length=%.3f "
            "failure_tolerance=%d (budget is 25 steps, so the failure and "
            "restart branches are unreachable at D=%d)",
            self._dim, self.n_candidates, self.turbo_state.length,
            self.turbo_state.failure_tolerance, self._dim,
        )

    # -- trust-region state --------------------------------------------------

    def _new_state(self):
        """Fresh :class:`TurboState`, ``best_value`` = current incumbent.

        ``TurboState`` is imported here rather than at module import time
        because :mod:`src.spherevo.turbo_baseline` calls
        ``logging.basicConfig`` at import; by the time a solver is built,
        ``scripts/run_benchmark.py`` has configured logging and the call is a
        no-op.
        """
        from src.spherevo.turbo_baseline import TurboState

        cfg = self._turbo_cfg
        state = TurboState(
            dim=self._dim,
            batch_size=TURBO_BATCH_SIZE,
            length=cfg["length"],
            length_min=cfg["length_min"],
            length_max=cfg["length_max"],
            success_tolerance=cfg["success_tolerance"],
            failure_tolerance=cfg["failure_tolerance"],
        )
        state.best_value = self._finite_best()
        return state

    def _finite_best(self) -> float:
        """Best finite score so far (choice C9: NaNs never become the incumbent)."""
        scores = np.asarray(self.all_scores, dtype=np.float64)
        scores = scores[np.isfinite(scores)]
        return float(scores.max()) if scores.size else -float("inf")

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Feed the evaluated score to the TR state; restart if it collapsed."""
        score = float(np.asarray(y, dtype=np.float64).flatten()[0])
        if not np.isfinite(score):
            score = TURBO_NAN_SCORE
        self.turbo_state.update(score)
        if self.turbo_state.restart_triggered:
            self._tr_restarts += 1
            logger.info(
                "TurboBoxFaithful: trust region collapsed below %.3g at iter %d "
                "-> restart #%d (state reset, history kept; choice C8)",
                self.turbo_state.length_min, self.iteration, self._tr_restarts,
            )
            self.turbo_state = self._new_state()

    # -- GP ------------------------------------------------------------------

    def _make_gp(self, X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
        """The submitted wrapper's TuRBO GP (``turbo_baseline.py`` lines 232-240).

        Matern-5/2 ARD; lengthscale and noise bounds from the BoTorch tutorial,
        outputscale bound from the paper. Kept unchanged so this arm differs
        from the submitted TuRBO row only in the box and in duplicate
        accounting (module docstring).
        """
        D = X.shape[-1]
        likelihood = GaussianLikelihood(
            noise_constraint=Interval(*TURBO_NOISE_INTERVAL),
        )
        covar = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=D,
                lengthscale_constraint=Interval(*TURBO_LENGTHSCALE_INTERVAL),
            ),
            outputscale_constraint=Interval(*TURBO_OUTPUTSCALE_INTERVAL),
        )
        gp = SingleTaskGP(
            train_X=X, train_Y=Y, likelihood=likelihood, covar_module=covar,
        )
        gp.to(X)
        return gp

    # -- TuRBO step ----------------------------------------------------------

    def _tr_centre(self, X_unit: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Incumbent in unit-cube coordinates, clamped into the box (choice C1)."""
        best = int(torch.argmax(Y.view(-1)))
        return X_unit[best].clamp(0.0, 1.0)

    def trust_region_bounds(
        self, gp: SingleTaskGP, centre: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lengthscale-weighted TR, clamped to the unit cube."""
        ls = gp.covar_module.base_kernel.lengthscale.detach().reshape(-1)
        if not torch.isfinite(ls).all() or float(ls.min()) <= 0.0:
            raise ModelFittingError(f"non-usable lengthscales: {ls}")
        weights = ls / ls.mean()
        weights = weights / torch.prod(weights.pow(1.0 / weights.numel()))
        half = weights * self.turbo_state.length / 2.0
        return (
            torch.clamp(centre - half, 0.0, 1.0),
            torch.clamp(centre + half, 0.0, 1.0),
        )

    def generate_candidates(
        self, gp: SingleTaskGP, centre: torch.Tensor,
    ) -> torch.Tensor:
        """``n_candidates`` perturbations of the centre inside the TR."""
        D = centre.numel()
        tr_lb, tr_ub = self.trust_region_bounds(gp, centre)

        sobol = SobolEngine(D, scramble=True)  # choice C4
        pert = sobol.draw(self.n_candidates).to(
            dtype=torch.float64, device=self._device,
        )
        pert = tr_lb + (tr_ub - tr_lb) * pert

        prob_perturb = min(TURBO_PERTURB_NUMERATOR / D, 1.0)
        mask = (
            torch.rand(
                self.n_candidates, D, dtype=torch.float64, device=self._device,
            )
            <= prob_perturb
        )
        empty = torch.where(mask.sum(dim=1) == 0)[0]
        if empty.numel() > 0:  # choice C6
            mask[
                empty,
                torch.randint(0, D, size=(empty.numel(),), device=self._device),
            ] = True

        cand = centre.expand(self.n_candidates, D).clone()
        cand[mask] = pert[mask]
        return cand

    def thompson_select(
        self, gp: SingleTaskGP, cand: torch.Tensor,
    ) -> torch.Tensor:
        """One Thompson sample over the candidate set (TuRBO's acquisition)."""
        sampler = MaxPosteriorSampling(model=gp, replacement=False)
        with torch.no_grad():
            chosen = sampler(cand, num_samples=TURBO_BATCH_SIZE)
        return chosen.reshape(1, -1)

    def _fallback(self, D: int) -> np.ndarray:
        """Uniform point in the fixed box (choice C10)."""
        u = torch.rand(1, D, dtype=torch.float64, device=self._device)
        return self._rescale_from_unit_cube(u).cpu().numpy().reshape(1, -1)

    def next_candidate(self) -> np.ndarray:
        x_hist, y_hist = self.get_history_as_arrays(
            penalize_nans_with=TURBO_NAN_SCORE,
        )
        D = x_hist.shape[-1]

        X_unit = self._rescale_to_unit_cube(
            torch.tensor(x_hist, dtype=torch.float64, device=self._device)
        )
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
            return self._fallback(D)

        try:
            with gpytorch.settings.max_cholesky_size(float("inf")):  # choice C5
                gp = self._make_gp(X_unit, Y_std)
                fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
                gp.eval()
                centre = self._tr_centre(X_unit, Y_std)
                cand = self.generate_candidates(gp, centre)
                x_next_unit = self.thompson_select(gp, cand)
            if not torch.isfinite(x_next_unit).all():
                raise ModelFittingError(
                    "Thompson sampling returned a non-finite candidate"
                )
            x_next = self._rescale_from_unit_cube(x_next_unit)
            return x_next.detach().cpu().numpy().reshape(1, -1)
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            raise
        except _BOX_FALLBACK_ERRORS as e:
            self._fallback_count += 1
            reason = f"{type(e).__name__}: {e}"
            self._fallback_reasons.append(reason)
            logger.error(
                f"{type(self).__name__} GP/TS FAILED at iter {self.iteration} "
                f"(fallback #{self._fallback_count}): {reason}"
            )
            return self._fallback(D)
