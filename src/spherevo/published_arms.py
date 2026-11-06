"""GuacaMol arms that run the PUBLISHED GP-BO recipes (camera-ready, published-recipe block).

Each arm runs one recipe of :mod:`src.spherevo.published_recipes` unchanged; only the domain is
ours. The harness (5 cold-start ZINC molecules, 25 steps, deterministic decoding, result format)
is the shared ``StepByStepSolver`` machinery of every GuacaMol arm.

==============================  ===========  ===========================================
solver name                     recipe       domain
==============================  ===========  ===========================================
``hvarfner_published``          hvarfner     fixed per-coordinate ``[p01, p99]`` box
``papenmeier_msr_reference``    msr          fixed per-coordinate ``[p01, p99]`` box
``papenmeier_mle_scaled_reference``  mle_scaled  fixed per-coordinate ``[p01, p99]`` box
``xu_published``                xu           fixed per-coordinate ``[p01, p99]`` box
``linear_bo_box8``              doumont      Doumont's molecular box ``[-8, 8]^256``
``turbo_published``             turbo1       fixed per-coordinate ``[p01, p99]`` box
``hvarfner_published_allobs``   hvarfner_allobs  as ``hvarfner_published``, every observation in
                                             the qLogNEI baseline (round 2, R2-5)
``papenmeier_msr_reference_learned``  msr_learned  as ``papenmeier_msr_reference``, noise learned (R2-6)
``papenmeier_mle_scaled_reference_learned``  mle_scaled_learned  MLE (scaled), noise learned (R2-6)
==============================  ===========  ===========================================

``turbo_published`` (``TurboPublishedSolver``, end of module) runs the external optimiser
``published_external.Turbo1Published`` instead of a GP recipe.

The ``[p01, p99]`` arms load the same bounds file and use the same rescaling as
``papenmeier_msr_published`` (``_BoxFaithfulMixin``; codec-specific bounds through the same
dispatch in ``scripts/run_benchmark.py``). Observations are NOT clamped before fitting: 191 of the
200 real cold-start encodings have a coordinate outside ``[p01, p99]`` (F_msr_published.md 4.4).
Where a recipe seeds its acquisition optimiser around the best observations (Hvarfner, MSR), the
seeding centres are the OBSERVED points and BoTorch's starting points are clamped onto the box, as
the authors' BoTorch does silently (round-1 fix C3; ``published_recipes.clamped_default_ic_generator``);
Hvarfner's qLogNEI baseline holds only the in-box observations, as Ax's bounds filter (C2). Xu does
not seed locally (raw Sobol points only). Doumont's kernel requires in-box inputs, so
``linear_bo_box8`` clamps its inputs onto ``[-8, 8]`` (counted per step; the cold starts were
inside the box in every check so far), and an invalid molecule enters its GP as -0.01, Doumont's
value (``guacamol_task.py:649``), not the -1.0 of the other arms (C4); ``n_nan_scores`` counts
them.

Every step appends a record to ``published_recipe_record["steps"]`` (fallback reason, events, fit
and acquisition time); ``scripts/run_benchmark.py`` writes the whole record to the result file
under ``published_recipe``. Proposal fallbacks (a uniform point in the box) are also counted in
``_fallback_count`` / ``_fallback_reasons`` like in every other GP arm.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch

from src.spherevo.gp_competitors import HvarfnerScaledGPSolver, _BoxFaithfulMixin
from src.spherevo.published_recipes import (
    DOUMONT_MOLECULAR_BOX,
    DTYPE,
    propose,
    recipe_settings,
)

logger = logging.getLogger(__name__)

PUBLISHED_ARMS: dict[str, str] = {
    "hvarfner_published": "hvarfner",
    "papenmeier_msr_reference": "msr",
    "papenmeier_mle_scaled_reference": "mle_scaled",
    "xu_published": "xu",
    "linear_bo_box8": "doumont",
}


class _PublishedRecipeStepMixin:
    """``next_candidate`` = one step of ``self.recipe`` on unit-cube coordinates.

    The host provides ``_rescale_to_unit_cube`` / ``_rescale_from_unit_cube`` and the harness
    attributes of ``HvarfnerScaledGPSolver`` (``_device``, ``_fallback_count``,
    ``_fallback_reasons``); none of the Hvarfner-scaled GP survives.
    """

    recipe: str = ""
    domain: str = ""
    #: GP training value of an invalid molecule (NaN score); the recorded score stays NaN
    nan_score_gp_value: float = -1.0

    def _init_record(self) -> None:
        self.published_recipe_record: dict[str, Any] = {
            "recipe": self.recipe,
            "domain": self.domain,
            "settings": recipe_settings(self.recipe, D=self._dim),
            "nan_score_gp_value": self.nan_score_gp_value,
            "n_nan_scores": self._count_nan_scores(),
            "steps": [],
        }

    def _count_nan_scores(self) -> int:
        """NaN scores (invalid molecules) in the history, cold starts included."""
        return int(np.isnan(np.concatenate(self.history["y"], axis=0)).sum())

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        super().post_update(x, y)
        self.published_recipe_record["n_nan_scores"] = self._count_nan_scores()

    def next_candidate(self) -> np.ndarray:
        x_hist, y_hist = self.get_history_as_arrays(penalize_nans_with=self.nan_score_gp_value)
        D = x_hist.shape[-1]
        X_raw = torch.tensor(x_hist, dtype=DTYPE, device=self._device)
        X_unit = self._rescale_to_unit_cube(X_raw)
        Y = torch.tensor(y_hist.flatten(), dtype=DTYPE, device=self._device).unsqueeze(-1)
        x_unit, rec = propose(self.recipe, X_unit, Y)
        n_out = int(((X_unit < 0.0) | (X_unit > 1.0)).any(dim=-1).sum())
        self.published_recipe_record["steps"].append(
            rec.to_dict(iteration=int(self.iteration), n_obs_outside_box=n_out))
        if rec.fallback is not None:
            self._fallback_count += 1
            self._fallback_reasons.append(rec.fallback)
            logger.error(
                f"{type(self).__name__} fallback at iter {self.iteration} "
                f"(#{self._fallback_count}): {rec.fallback}"
            )
        x_raw = self._rescale_from_unit_cube(x_unit.to(device=self._device, dtype=DTYPE))
        return x_raw.detach().cpu().numpy().reshape(1, D)


class _PublishedP01P99Solver(_PublishedRecipeStepMixin, _BoxFaithfulMixin, HvarfnerScaledGPSolver):
    """Published recipe on the fixed per-coordinate ``[p01, p99]`` box (``_BoxFaithfulMixin``)."""

    domain = "[p01, p99] per-coordinate box (bounds file of *_box_faithful)"

    def __init__(self, black_box: Any, x0: np.ndarray, y0: np.ndarray, *,
                 n_candidates: int = 2048, bounds_path: Optional[str] = None):
        super().__init__(black_box=black_box, x0=x0, y0=y0, n_candidates=n_candidates,
                         bounds_path=bounds_path)
        self._init_record()


class HvarfnerPublishedSolver(_PublishedP01P99Solver):
    """Vanilla BO (Hvarfner et al., ICML 2024), ``default`` model and Ax acquisition settings."""

    recipe = "hvarfner"


class PapenmeierMSRReferenceSolver(_PublishedP01P99Solver):
    """MSR (Papenmeier et al., ICML 2025): reference GP and fit + published acquisition call."""

    recipe = "msr"


class PapenmeierMLEScaledReferenceSolver(_PublishedP01P99Solver):
    """MLE (scaled) (Papenmeier et al., ICML 2025): MSR without RAASP."""

    recipe = "mle_scaled"


class XuPublishedSolver(_PublishedP01P99Solver):
    """SBO-Matern (RI) (Xu et al., ICLR 2025): GP_ARD_setls, UCB(beta=1.5), 10 x 1000."""

    recipe = "xu"


class LinearBOBox8Solver(_PublishedRecipeStepMixin, HvarfnerScaledGPSolver):
    """Spherical linear kernel (Doumont et al.) in the authors' molecular box ``[-8, 8]^256``.

    ``guacamol_task.py:611-616`` of linear-bo; inputs normalised to ``[0,1]^D`` as in
    ``optimize.py:117-120``. Unlike ``linear_bo`` (``[p01, p99]`` box, plain fit, retry on
    OptimizationWarning), the fit and ``optimize_acqf`` call are Doumont's.
    """

    recipe = "doumont"
    domain = "[-8, 8]^D (Doumont et al., guacamol_task.py:611-616)"
    nan_score_gp_value = -0.01      # linear-bo src/benchmarks/guacamol_task.py:646-649

    def __init__(self, black_box: Any, x0: np.ndarray, y0: np.ndarray, *,
                 n_candidates: int = 2048):
        super().__init__(black_box=black_box, x0=x0, y0=y0, n_candidates=n_candidates)
        lo, hi = DOUMONT_MOLECULAR_BOX
        self._lo = torch.full((self._dim,), lo, dtype=DTYPE, device=self._device)
        self._hi = torch.full((self._dim,), hi, dtype=DTYPE, device=self._device)
        self._width = self._hi - self._lo
        self._init_record()

    def _rescale_to_unit_cube(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._lo) / self._width

    def _rescale_from_unit_cube(self, x_norm: torch.Tensor) -> torch.Tensor:
        return self._lo + x_norm * self._width


PUBLISHED_ARM_CLASSES: dict[str, type] = {
    "hvarfner_published": HvarfnerPublishedSolver,
    "papenmeier_msr_reference": PapenmeierMSRReferenceSolver,
    "papenmeier_mle_scaled_reference": PapenmeierMLEScaledReferenceSolver,
    "xu_published": XuPublishedSolver,
    "linear_bo_box8": LinearBOBox8Solver,
}


class TurboPublishedSolver(_BoxFaithfulMixin, HvarfnerScaledGPSolver):
    """TuRBO-1 (Eriksson et al., NeurIPS 2019) as released, on the fixed ``[p01, p99]`` box.

    The optimiser is ``published_external.Turbo1Published`` (a port of ``turbo/turbo_1.py`` and
    ``turbo/gp.py``): the paper's GP and its Adam fit, failure tolerance ceil(max(4, D)) = 256,
    min(100 D, 5000) = 5000 Sobol perturbations with a fresh numpy-seeded Sobol sequence per step,
    perturbation probability min(1, 20/D), Thompson sampling, q = 1. The 5 cold-start encodings are
    its initial design (TuRBO accepts any initial data). The official code asserts that its data
    lie in the unit cube, so observations are projected onto the box before they enter TuRBO
    (the projected cold-start coordinates are counted once, at construction:
    ``cold_start_coordinates_projected``); proposals are always inside the box. Every proposal is evaluated and
    counted, repeated molecules included, as in the reference. ``turbo_box_faithful`` (the
    submitted wrapper's GP and choices C1-C10) is left untouched.
    """

    recipe = "turbo1"
    domain = "[p01, p99] per-coordinate box (bounds file of *_box_faithful)"

    def __init__(self, black_box: Any, x0: np.ndarray, y0: np.ndarray, *,
                 n_candidates: int = 2048, bounds_path: Optional[str] = None):
        from src.spherevo.published_external import Turbo1Published, external_settings

        super().__init__(black_box=black_box, x0=x0, y0=y0, n_candidates=n_candidates,
                         bounds_path=bounds_path)
        x0 = np.asarray(x0, dtype=np.float64)
        y0 = np.asarray(y0, dtype=np.float64).reshape(-1)
        U0 = self._to_unit_np(x0)
        n_out = int(((U0 < 0.0) | (U0 > 1.0)).sum())
        self.turbo = Turbo1Published(dim=self._dim, n_init=len(x0), batch_size=1)
        y0 = np.where(np.isfinite(y0), y0, -1.0)            # get_history_as_arrays convention
        self.turbo.seed_data(np.clip(U0, 0.0, 1.0), -y0)    # TuRBO minimises
        self.published_recipe_record = {
            "recipe": self.recipe, "domain": self.domain,
            "settings": external_settings("turbo1", dim=self._dim),
            "cold_start_coordinates_projected": n_out, "steps": [],
        }

    def _to_unit_np(self, x: np.ndarray) -> np.ndarray:
        X = torch.tensor(np.asarray(x, dtype=np.float64).reshape(-1, self._dim),
                         dtype=DTYPE, device=self._device)
        return self._rescale_to_unit_cube(X).cpu().numpy()

    def next_candidate(self) -> np.ndarray:
        u, info = self.turbo.ask()
        step = info.to_dict()
        step.update(iteration=int(self.iteration), failcount=int(self.turbo.failcount),
                    succcount=int(self.turbo.succcount), restarts=int(self.turbo.n_restarts))
        self.published_recipe_record["steps"].append(step)
        if info.fallback is not None:
            self._fallback_count += 1
            self._fallback_reasons.append(info.fallback)
            logger.error(f"TurboPublishedSolver fallback at iter {self.iteration}: {info.fallback}")
        x = self._rescale_from_unit_cube(torch.tensor(u, dtype=DTYPE, device=self._device))
        return x.detach().cpu().numpy().reshape(1, self._dim)

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        score = float(np.asarray(y, dtype=np.float64).reshape(-1)[0])
        if not np.isfinite(score):
            score = -1.0
        u = np.clip(self._to_unit_np(x), 0.0, 1.0)
        self.turbo.tell(u, -score)


PUBLISHED_ARM_CLASSES["turbo_published"] = TurboPublishedSolver
PUBLISHED_ARMS["turbo_published"] = "turbo1"


class HvarfnerPublishedAllObsSolver(_PublishedP01P99Solver):
    """Vanilla BO as ``hvarfner_published``, except that the qLogNEI baseline holds every observation,
    out-of-box cold starts included (the pre-C2 behaviour; round 2, R2-5). Ax 0.3.7 keeps only in-box
    points in the baseline in both ``fit_out_of_design`` modes; this variant is reported beside the
    Ax-filtered arm, and the two coincide whenever every observation lies in the box."""

    recipe = "hvarfner_allobs"


PUBLISHED_ARM_CLASSES["hvarfner_published_allobs"] = HvarfnerPublishedAllObsSolver
PUBLISHED_ARMS["hvarfner_published_allobs"] = "hvarfner_allobs"


class PapenmeierMSRReferenceLearnedSolver(_PublishedP01P99Solver):
    """MSR as ``papenmeier_msr_reference`` with the Gaussian noise LEARNED (the release's
    ``--no-noiseless`` path; round 2, R2-6)."""

    recipe = "msr_learned"


class PapenmeierMLEScaledReferenceLearnedSolver(_PublishedP01P99Solver):
    """MLE (scaled) as ``papenmeier_mle_scaled_reference`` with the Gaussian noise LEARNED (R2-6)."""

    recipe = "mle_scaled_learned"


PUBLISHED_ARM_CLASSES["papenmeier_msr_reference_learned"] = PapenmeierMSRReferenceLearnedSolver
PUBLISHED_ARM_CLASSES["papenmeier_mle_scaled_reference_learned"] = (
    PapenmeierMLEScaledReferenceLearnedSolver)
PUBLISHED_ARMS["papenmeier_msr_reference_learned"] = "msr_learned"
PUBLISHED_ARMS["papenmeier_mle_scaled_reference_learned"] = "mle_scaled_learned"


class BaxusCRSolver(_BoxFaithfulMixin, HvarfnerScaledGPSolver):
    """BAxUS (Papenmeier, Nardi & Poloczek, NeurIPS 2022) on the fixed ``[p01, p99]``
    box mapped to the tutorial's own domain ``[-1, 1]^D``.

    The optimiser is ``published_external.BaxusPublished``, the port of the BAxUS
    tutorial the authors contributed to BoTorch, unchanged: count-sketch embedding,
    ``d_init`` and the number of splits from D, three new bins per split, trust region
    ``x_center +- weights * length`` clamped to [-1, 1] with length 0.8 / 0.5**7 / 1.6
    and success tolerance 3, ``min(5000, max(2000, 200 D))`` candidates drawn from the
    INPUT dimension (tutorial cell 21), ``SingleTaskGP`` with the installed BoTorch's
    default kernel and outcome transform and ``GaussianLikelihood(noise Interval(1e-8,
    1e-3))``, ``fit_gpytorch_mll`` with an Adam fallback, Thompson sampling.

    What this arm defines, because the harness does and the tutorial does not:

    * **the domain.** The latent box of the other faithful-domain arms,
      ``[p01, p99]`` per coordinate, is mapped affinely onto ``[-1, 1]^D``.
    * **the initial design.** The five shared cold-start molecules of every other row,
      not a private Sobol design: they are drug-like, and a private random design
      would handicap the baseline in our favour.  They are entered into the target
      space by the least-squares projection ``baxus_project_down`` and clamped to
      [-1, 1], and after each split the history is re-projected the same way -- the
      identity on the optimiser's own points, so this touches only the seeded rows.
      Coordinates clamped on entry are counted in the record.
    * **the split schedule.** ``eval_budget`` is the true proposal budget, 25, so the
      split budget and the failure tolerance follow from it (at D = 256 this gives
      ``d_init = 1``, 4 splits, failure tolerance 1).
    * **an invalid molecule** enters the GP as -1.0, the convention of every other arm
      (``nan_score_gp_value``); the recorded score stays NaN.

    Inherited from the verified port and NOT changed here: the per-step failure
    handling (a numerical failure gives a uniform point in the target space for that
    step, recorded in ``fallback`` and counted in ``surrogate_fallback_count``), the
    candidate count from the input dimension, and ``state.best_value`` starting at
    ``-inf``, so the first BO step always counts as a failure -- exactly how the
    tutorial treats its own initial design.

    Every proposal is evaluated and counted, repeated molecules included, as in the
    reference.  The record also keeps, per cell, the number of DISTINCT proposed
    latents and how many proposals are exact corners of the box: the submitted
    wrapper's missing normalisation collapsed all 25 onto one corner.
    """

    recipe = "baxus"
    domain = "[p01, p99] per-coordinate box mapped to [-1, 1]^D (bounds file of *_box_faithful)"
    #: GP training value of an invalid molecule (NaN score); the recorded score stays NaN
    nan_score_gp_value: float = -1.0

    def __init__(self, black_box: Any, x0: np.ndarray, y0: np.ndarray, *,
                 n_candidates: int = 2048, bounds_path: Optional[str] = None,
                 seed: int = 0, eval_budget: int):
        from src.spherevo.published_external import BaxusPublished, external_settings

        super().__init__(black_box=black_box, x0=x0, y0=y0, n_candidates=n_candidates,
                         bounds_path=bounds_path)
        x0 = np.asarray(x0, dtype=np.float64)
        y0 = np.asarray(y0, dtype=np.float64).reshape(-1)
        y0 = np.where(np.isfinite(y0), y0, self.nan_score_gp_value)
        V0 = self._to_pm1_np(x0)
        self.baxus = BaxusPublished(dim=self._dim, n_init=len(x0),
                                    eval_budget=int(eval_budget), seed=int(seed))
        seeding = self.baxus.seed_external(V0, y0)       # BAxUS maximises
        self._proposed_latents: set[bytes] = set()
        st = self.baxus.state
        self.published_recipe_record: dict[str, Any] = {
            "recipe": self.recipe,
            "domain": self.domain,
            "settings": {
                **external_settings("baxus", dim=self._dim),
                "eval_budget": int(eval_budget),
                "d_init": int(st.d_init),
                "n_splits": int(st.n_splits),
                "split_budget": int(st.split_budget),
                "failure_tolerance_at_d_init": int(st.failure_tolerance),
                "trust_region_half_width": "x_center +- weights * length (tutorial cell 19; "
                                           "NOT the /2 of the TuRBO tutorial)",
                "initial_design": "the 5 shared cold-start molecules, least-squares "
                                  "projected into the target space and clamped to [-1, 1]",
                "duplicates": "every proposal is evaluated and counted (as the reference)",
            },
            "deviations_ours": [
                "domain: the [p01, p99] latent box of the other faithful-domain arms, "
                "mapped affinely onto the tutorial's [-1, 1]^D",
                "initial design: the 5 shared cold-start molecules instead of a private "
                "Sobol design, entered by least-squares projection through the embedding "
                "and clamped to [-1, 1]",
                "after each split the history is re-projected the same way (the identity "
                "on the optimiser's own points; only the seeded rows differ)",
                "eval_budget = 25, the true proposal budget, which sets the split "
                "schedule and the failure tolerance",
                "an invalid molecule enters the GP as -1.0 (harness convention); the "
                "recorded score stays NaN",
                "no duplicate or undecodable suppression: such a proposal spends one of "
                "the 25 proposals, as in every other arm",
            ],
            "deviations_inherited": [
                "per-step failure handling: a numerical failure gives a uniform point in "
                "the target space for that step and is recorded (the reference has no "
                "per-step handling)",
                "candidate count min(5000, max(2000, 200 D)) from the INPUT dimension "
                "(tutorial cell 21)",
                "state.best_value starts at -inf, so the first BO step counts as a "
                "failure (the tutorial treats its own initial design the same way)",
                "SingleTaskGP with the installed BoTorch's default kernel and outcome "
                "transform (the tutorial's; differs from the BAxUS pip package)",
            ],
            **seeding,
            "n_distinct_proposed_latents": 0,
            "n_box_corner_proposals": 0,
            "splits": [],
            "steps": [],
        }

    # -- domain ------------------------------------------------------------
    def _to_pm1_np(self, x: np.ndarray) -> np.ndarray:
        """Raw latents -> the tutorial's [-1, 1]^D coordinates of the [p01, p99] box."""
        X = torch.tensor(np.asarray(x, dtype=np.float64).reshape(-1, self._dim),
                         dtype=DTYPE, device=self._device)
        return (2.0 * self._rescale_to_unit_cube(X) - 1.0).detach().cpu().numpy()

    def _from_pm1(self, v: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`_to_pm1_np`."""
        V = torch.tensor(np.asarray(v, dtype=np.float64).reshape(-1, self._dim),
                         dtype=DTYPE, device=self._device)
        return self._rescale_from_unit_cube(
            (V + 1.0) / 2.0).detach().cpu().numpy().reshape(-1, self._dim)

    # -- step --------------------------------------------------------------
    def next_candidate(self) -> np.ndarray:
        v, info = self.baxus.ask()
        v = np.asarray(v, dtype=np.float64).reshape(1, self._dim)
        is_corner = bool(np.all(np.abs(v) == 1.0))
        self._proposed_latents.add(v.tobytes())
        rec = self.published_recipe_record
        rec["n_distinct_proposed_latents"] = len(self._proposed_latents)
        rec["n_box_corner_proposals"] += int(is_corner)
        rec["steps"].append({
            "iteration": int(self.iteration),
            "kind": info["kind"],
            "fallback": info["fallback"],
            "events": info["events"],
            "target_dim": int(info["target_dim"]),
            "tr_length": float(info["tr_length"]),
            "t_acq_s": info["t_acq_s"],
            "is_box_corner": is_corner,
            "n_splits_done": int(self.baxus.n_splits_done),
            "failure_counter": int(self.baxus.state.failure_counter),
            "success_counter": int(self.baxus.state.success_counter),
        })
        if info["fallback"] is not None:
            self._fallback_count += 1
            self._fallback_reasons.append(info["fallback"])
            logger.error(f"BaxusCRSolver fallback at iter {self.iteration} "
                         f"(#{self._fallback_count}): {info['fallback']}")
        for event in info["events"]:                  # the tutorial's Adam fallback of the GP fit
            self._fallback_count += 1
            self._fallback_reasons.append(f"iter {self.iteration}: {event}")
            logger.error(f"BaxusCRSolver fit fallback at iter {self.iteration}: {event}")
        return self._from_pm1(v)

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        score = float(np.asarray(y, dtype=np.float64).reshape(-1)[0])
        if not np.isfinite(score):
            score = self.nan_score_gp_value
            self._fallback_count += 1
            self._fallback_reasons.append(
                f"iter {self.iteration}: non-finite score (invalid molecule) entered the GP "
                f"as {self.nan_score_gp_value}")
        target_dim_before = int(self.baxus.S.shape[0])
        self.baxus.tell(score)                        # BAxUS maximises
        target_dim_after = int(self.baxus.S.shape[0])
        if target_dim_after != target_dim_before:
            self.published_recipe_record["splits"].append({
                "iteration": int(self.iteration),
                "target_dim_before": target_dim_before,
                "target_dim_after": target_dim_after,
            })


PUBLISHED_ARM_CLASSES["baxus_cr"] = BaxusCRSolver
PUBLISHED_ARMS["baxus_cr"] = "baxus"
